# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Analyze one eight-arm DAPO load-alignment scheduler replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nemo_rl.algorithms.async_utils.dapo_load_alignment import (
    DapoLoadAlignmentArm,
    DapoLoadAlignmentPlan,
    load_dapo_load_alignment_plan,
)
from nemo_rl.algorithms.async_utils.scheduler_trace import (
    SchedulerEventType,
    iter_scheduler_trace,
)
from tools.analyze_dapo_operational_mixture_shadows import analyze_run
from tools.analyze_dapo_load_alignment_references import _spearman
from tools.analyze_scheduler_selection_opportunity import _current_clean_commit


EXPECTED_GROUPS = 32


class DapoLoadAlignmentShadowError(ValueError):
    """A signed-control scheduler input or invariant is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DapoLoadAlignmentShadowError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_fixed_reference(
    path: Path, *, pool_seed: int, expected_sha256: str
) -> tuple[dict[str, bool], dict[str, bool], dict[str, float], dict[str, float]]:
    _require(_sha(path) == expected_sha256, "private reference manifest hash mismatch")
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise DapoLoadAlignmentShadowError("invalid reference manifest") from error
    _require(
        isinstance(value, dict)
        and value.get("analysis_status")
        == "frozen_private_dapo_load_alignment_reference",
        "reference manifest labels mismatch",
    )
    pools = value.get("pools")
    _require(isinstance(pools, list), "reference pools missing")
    matches = [
        item
        for item in pools
        if isinstance(item, dict) and item.get("pool_seed") == pool_seed
    ]
    _require(len(matches) == 1, "reference pool identity mismatch")
    prompts = matches[0].get("prompts")
    _require(
        isinstance(prompts, list) and len(prompts) == 32, "reference prompts missing"
    )
    lower_load: dict[str, bool] = {}
    easier: dict[str, bool] = {}
    load_scores: dict[str, float] = {}
    difficulty_scores: dict[str, float] = {}
    for ordinal, item in enumerate(prompts):
        _require(
            isinstance(item, dict)
            and item.get("source_pool_ordinal") == ordinal
            and isinstance(item.get("source_prompt_id"), str)
            and isinstance(item.get("fixed_lower_load"), bool)
            and isinstance(item.get("fixed_harder"), bool)
            and isinstance(item.get("generated_load_score"), (int, float))
            and isinstance(item.get("difficulty_score"), (int, float)),
            "reference prompt record mismatch",
        )
        source_id = str(item["source_prompt_id"])
        lower_load[source_id] = bool(item["fixed_lower_load"])
        easier[source_id] = not bool(item["fixed_harder"])
        load_scores[source_id] = float(item["generated_load_score"])
        difficulty_scores[source_id] = float(item["difficulty_score"])
    _require(
        len(lower_load) == 32
        and sum(lower_load.values()) == 16
        and sum(easier.values()) == 16,
        "fixed half mismatch",
    )
    return lower_load, easier, load_scores, difficulty_scores


def _delay_contract(
    run_dir: Path, arm: DapoLoadAlignmentArm, delayed_prompt_ids: frozenset[str]
) -> bool:
    events = tuple(iter_scheduler_trace(run_dir / "scheduler_trace.v1.jsonl"))
    completed = [
        event
        for event in events
        if event.event_type is SchedulerEventType.ROLLOUT_COMPLETED
    ]
    if len(completed) != EXPECTED_GROUPS:
        return False
    observed_ids = {event.source_prompt_id for event in completed}
    if not delayed_prompt_ids <= observed_ids:
        return False
    for event in completed:
        expected = (
            arm.release_delay_seconds
            if event.source_prompt_id in delayed_prompt_ids
            else 0.0
        )
        if (
            event.scalar_summaries.get("scheduler_assay_delay_target")
            != arm.delay_target
            or event.scalar_summaries.get("scheduler_assay_release_delay_seconds")
            != expected
        ):
            return False
    return True


def _comparison(
    in_order: Mapping[str, Any],
    ready_first: Mapping[str, Any],
    lower_load: Mapping[str, bool],
    easier: Mapping[str, bool],
    load_scores: Mapping[str, float],
    difficulty_scores: Mapping[str, float],
) -> dict[str, object]:
    in_order_order = in_order["selected_order"]
    ready_order = ready_first["selected_order"]
    _require(
        set(in_order_order) == set(ready_order) == set(lower_load) == set(easier),
        "cross-arm prompt identity parity mismatch",
    )
    in_steps = {source_id: index // 4 for index, source_id in enumerate(in_order_order)}
    ready_steps = {source_id: index // 4 for index, source_id in enumerate(ready_order)}

    def promotion(labels: Mapping[str, bool]) -> float:
        return statistics.fmean(
            (in_steps[source_id] - ready_steps[source_id]) / 7
            for source_id, included in labels.items()
            if included
        )

    horizons = {}
    for horizon in (4, 8, 12, 16, 24, 32):
        horizons[str(horizon)] = {
            sampler: {
                "fixed_lower_load_share": sum(
                    lower_load[item] for item in order[:horizon]
                )
                / horizon,
                "fixed_easier_share": sum(easier[item] for item in order[:horizon])
                / horizon,
                "generated_load_score_mean": statistics.fmean(
                    load_scores[item] for item in order[:horizon]
                ),
                "difficulty_score_mean": statistics.fmean(
                    difficulty_scores[item] for item in order[:horizon]
                ),
            }
            for sampler, order in (
                ("in_order", in_order_order),
                ("ready_first", ready_order),
            )
        }
    return {
        "fixed_lower_load_mean_normalized_selection_step_promotion": promotion(
            lower_load
        ),
        "fixed_easier_mean_normalized_selection_step_promotion": promotion(easier),
        "same_selected_step_sets": sum(
            set(in_order_order[index : index + 4])
            == set(ready_order[index : index + 4])
            for index in range(0, 32, 4)
        ),
        "fixed_lower_load_easier_overlap": sum(
            lower_load[source_id] and easier[source_id] for source_id in lower_load
        ),
        "fixed_generated_load_difficulty_spearman": _spearman(
            [load_scores[source_id] for source_id in lower_load],
            [difficulty_scores[source_id] for source_id in lower_load],
        ),
        "horizons": horizons,
    }


def analyze_block(
    *,
    plan: DapoLoadAlignmentPlan,
    order_seed: int,
    runs: Mapping[str, Path],
    materialization_manifest_path: Path,
    private_reference_path: Path,
) -> dict[str, object]:
    pool = plan.pool(order_seed)
    _require(set(runs) == {arm.arm_id for arm in plan.arms}, "all eight arms required")
    lower_load, easier, load_scores, difficulty_scores = _load_fixed_reference(
        private_reference_path,
        pool_seed=order_seed,
        expected_sha256=pool.private_reference_manifest_sha256,
    )
    _require(
        frozenset(source_id for source_id, value in lower_load.items() if value)
        == frozenset(pool.fixed_lower_load_prompt_ids),
        "plan lower-load identities differ from frozen reference",
    )
    results: dict[str, dict[str, Any]] = {}
    for arm_id in pool.arm_execution_order:
        arm = plan.arm(arm_id)
        result, _ = analyze_run(  # type: ignore[arg-type]
            run_dir=runs[arm_id],
            plan=plan,
            arm=arm,
            order_seed=order_seed,
            materialization_manifest_path=materialization_manifest_path,
        )
        delayed_ids = plan.delayed_prompt_ids(arm_id, order_seed)
        result["validity_checks"]["signed_delay_contract_exact"] = _delay_contract(
            runs[arm_id], arm, delayed_ids
        )
        result["all_validity_gates_passed"] = all(result["validity_checks"].values())
        results[arm_id] = result
    conditions = {
        "l0_natural": ("l0_natural_in_order", "l0_natural_ready_first"),
        "l3_natural": ("l3_natural_in_order", "l3_natural_ready_first"),
        "l3_high_load_delayed": (
            "l3_high_load_delayed_in_order",
            "l3_high_load_delayed_ready_first",
        ),
        "l3_low_load_delayed": (
            "l3_low_load_delayed_in_order",
            "l3_low_load_delayed_ready_first",
        ),
    }
    comparisons = {
        condition: _comparison(
            results[in_order],
            results[ready_first],
            lower_load,
            easier,
            load_scores,
            difficulty_scores,
        )
        for condition, (in_order, ready_first) in conditions.items()
    }
    checks = {
        "all_eight_arms_valid": all(
            result["all_validity_gates_passed"] for result in results.values()
        ),
        "cross_arm_exact_source_prompt_identity_parity_each_pair": all(
            set(results[left]["source_prompt_ids"])
            == set(results[right]["source_prompt_ids"])
            for left, right in conditions.values()
        ),
        "l0_selected_step_sets_match_all_8_steps": comparisons["l0_natural"][
            "same_selected_step_sets"
        ]
        == 8,
        "l0_fixed_reference_promotions_equal_zero": comparisons["l0_natural"][
            "fixed_lower_load_mean_normalized_selection_step_promotion"
        ]
        == 0
        and comparisons["l0_natural"][
            "fixed_easier_mean_normalized_selection_step_promotion"
        ]
        == 0,
    }
    for result in results.values():
        result.pop("source_prompt_ids")
        result.pop("selected_order")
    return {
        "schema_version": 1,
        "analysis_status": plan.analysis_status,
        "calibration_only": True,
        "confirmatory_eligible": False,
        "population_claim_authorized": False,
        "counterfactual_replay_authorized": False,
        "training_authorized": False,
        "plan_id": plan.plan_id,
        "order_seed": order_seed,
        "arms": results,
        "comparisons": comparisons,
        "replication_checks": checks,
        "decision": "pass_to_next_replication"
        if all(checks.values())
        else "stop_and_diagnose",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--order-seed", type=int, required=True)
    parser.add_argument("--materialization-manifest", type=Path, required=True)
    parser.add_argument("--private-reference", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, metavar="ARM_ID=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path)
    args = parser.parse_args()
    plan = load_dapo_load_alignment_plan(args.plan)
    if args.repo_root is not None:
        _require(
            _current_clean_commit(args.repo_root) == plan.analysis_code_commit,
            "analysis repository does not match frozen commit",
        )
    runs = {}
    for assignment in args.run:
        arm_id, separator, path = assignment.partition("=")
        _require(
            bool(separator and arm_id and path) and arm_id not in runs, "invalid --run"
        )
        runs[arm_id] = Path(path)
    result = analyze_block(
        plan=plan,
        order_seed=args.order_seed,
        runs=runs,
        materialization_manifest_path=args.materialization_manifest,
        private_reference_path=args.private_reference,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
