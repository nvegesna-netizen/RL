# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fail-closed analysis for one DAPO operational-mixture shadow replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nemo_rl.algorithms.async_utils.dapo_operational_mixture import (
    DapoOperationalMixtureArm,
    DapoOperationalMixturePlan,
    load_dapo_operational_mixture_plan,
)
from nemo_rl.algorithms.async_utils.fixed_pool import (
    load_fixed_pool_manifest,
    validate_fixed_pool_manifest_design,
    validate_fixed_pool_materialization,
)
from nemo_rl.algorithms.async_utils.scheduler_trace import (
    SchedulerEventType,
    iter_scheduler_trace,
    validate_scheduler_trace,
)
from tools.analyze_scheduler_pressure_response import (
    _maximum_active_generation_concurrency,
    _maximum_unreleased_group_concurrency,
    _replay_parity,
)
from tools.analyze_scheduler_selection_opportunity import (
    TraceAnalysis,
    _arm_summary,
    _current_clean_commit,
    analyze_trace_events,
)


EXPECTED_GROUPS = 32
EXPECTED_COMPLETIONS = 16
EXPECTED_SELECTIONS = 8


class DapoOperationalMixtureShadowError(ValueError):
    """A scheduler-shadow input or frozen invariant is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DapoOperationalMixtureShadowError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_fixed_reference(
    path: Path, *, pool_seed: int, expected_sha256: str
) -> tuple[dict[str, bool], dict[str, float]]:
    _require(_sha(path) == expected_sha256, "private reference manifest hash mismatch")
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise DapoOperationalMixtureShadowError("invalid reference manifest") from error
    _require(
        isinstance(value, dict)
        and value.get("schema_version") == 1
        and value.get("analysis_status")
        == "frozen_private_dapo_operational_mixture_reference",
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
        isinstance(prompts, list) and len(prompts) == EXPECTED_GROUPS,
        "reference prompt count mismatch",
    )
    harder: dict[str, bool] = {}
    scores: dict[str, float] = {}
    for ordinal, item in enumerate(prompts):
        _require(
            isinstance(item, dict)
            and item.get("source_pool_ordinal") == ordinal
            and isinstance(item.get("source_prompt_id"), str)
            and isinstance(item.get("fixed_harder"), bool)
            and isinstance(item.get("reference_score"), (int, float))
            and not isinstance(item.get("reference_score"), bool),
            "reference prompt record mismatch",
        )
        source_id = str(item["source_prompt_id"])
        harder[source_id] = bool(item["fixed_harder"])
        scores[source_id] = float(item["reference_score"])
    _require(
        len(harder) == EXPECTED_GROUPS and sum(harder.values()) == 16,
        "fixed harder-half mismatch",
    )
    return harder, scores


def analyze_run(
    *,
    run_dir: Path,
    plan: DapoOperationalMixturePlan,
    arm: DapoOperationalMixtureArm,
    order_seed: int,
    materialization_manifest_path: Path,
) -> tuple[dict[str, Any], TraceAnalysis]:
    pool = plan.pool(order_seed)
    trace_path = run_dir / "scheduler_trace.v1.jsonl"
    copied_manifest = run_dir / "fixed_pool_manifest.v1.json"
    _require(
        trace_path.is_file() and copied_manifest.is_file(), "run artifacts missing"
    )
    _require(
        copied_manifest.read_bytes() == materialization_manifest_path.read_bytes(),
        "run manifest is not byte-identical to materialization",
    )
    manifest = load_fixed_pool_manifest(materialization_manifest_path)
    validate_fixed_pool_materialization(manifest)
    validate_fixed_pool_manifest_design(manifest, plan.source_design_id)
    _require(
        (manifest.order_seed, manifest.pool_id, manifest.manifest_sha256)
        == (pool.order_seed, pool.pool_id, pool.manifest_sha256),
        "run manifest does not match frozen pool",
    )
    report = validate_scheduler_trace(trace_path)
    _require(
        not report.incomplete_attempt_ids
        and not report.administratively_censored_group_ids,
        "trace is incomplete or administratively censored",
    )
    events = tuple(iter_scheduler_trace(trace_path))
    starts = [
        event for event in events if event.event_type is SchedulerEventType.RUN_STARTED
    ]
    ends = [
        event for event in events if event.event_type is SchedulerEventType.RUN_ENDED
    ]
    _require(len(starts) == len(ends) == 1, "trace boundaries mismatch")
    _require(
        starts[0].run_mode == "scheduler_assay"
        and starts[0].pool_id == pool.pool_id
        and starts[0].scalar_summaries.get("scheduler_assay_plan_id") == plan.plan_id
        and starts[0].scalar_summaries.get("scheduler_assay_arm_id") == arm.arm_id
        and starts[0].scalar_summaries.get("generation_study_seed")
        == pool.scheduler_generation_seed,
        "run identity mismatch",
    )
    _require(
        ends[0].terminal_reason == "scheduler_assay_complete"
        and ends[0].scalar_summaries.get("completed_train_steps") == 0
        and ends[0].scalar_summaries.get("final_physical_weight_version") == 0,
        "run failed or changed learner state",
    )
    analysis = analyze_trace_events(events, trace_sha256=_sha(trace_path))
    _require(
        len(analysis.groups) == EXPECTED_GROUPS
        and len(analysis.decisions) == EXPECTED_SELECTIONS
        and all(decision.selected_count == 4 for decision in analysis.decisions),
        "scheduler selection geometry mismatch",
    )
    completed = [
        event
        for event in events
        if event.event_type is SchedulerEventType.ROLLOUT_COMPLETED
    ]
    _require(len(completed) == EXPECTED_GROUPS, "completion group count mismatch")
    length_rates = []
    source_ids = set()
    for event in completed:
        _require(event.source_prompt_id is not None, "completion identity missing")
        source_ids.add(event.source_prompt_id)
        count = event.scalar_summaries.get("completion_count")
        rate = event.scalar_summaries.get("backend_length_termination_rate")
        _require(count == EXPECTED_COMPLETIONS, "completion count mismatch")
        _require(
            isinstance(rate, (int, float)) and not isinstance(rate, bool),
            "length rate missing",
        )
        length_rates.append(float(rate))
    active = _maximum_active_generation_concurrency(events)
    unreleased = _maximum_unreleased_group_concurrency(events)
    checks = {
        "exactly_32_groups_and_512_completions": len(completed) == EXPECTED_GROUPS,
        "exactly_8_nonempty_four_group_selections": True,
        "maximum_active_generation_groups_equals_4": active == 4,
        "native_reconstruction_exact_group_id_parity": _replay_parity(
            events, manifest, arm
        ),
        "zero_administrative_censoring": True,
        "zero_learner_steps_and_physical_weight_version_zero": True,
        "backend_length_termination_rate_le_0_2": statistics.fmean(length_rates) <= 0.2,
    }
    if arm.pressure_level == "l0":
        checks["maximum_unreleased_groups_equals_4"] = unreleased == 4
        checks["ready_but_ineligible_zero_each_decision"] = all(
            decision.ready_count == decision.eligible_count
            for decision in analysis.decisions
        )
    return {
        "arm_id": arm.arm_id,
        "pressure_level": arm.pressure_level,
        "sampler": arm.sampler,
        "prompt_groups": len(completed),
        "completions": len(completed) * EXPECTED_COMPLETIONS,
        "selection_steps": len(analysis.decisions),
        "maximum_active_generation_groups": active,
        "maximum_unreleased_groups": unreleased,
        "backend_length_termination_rate": statistics.fmean(length_rates),
        "selection_pressure": _arm_summary(analysis),
        "source_prompt_ids": sorted(source_ids),
        "selected_order": list(analysis.selected_order),
        "validity_checks": checks,
        "all_validity_gates_passed": all(checks.values()),
        "source_artifacts": {
            "trace_sha256": _sha(trace_path),
            "manifest_sha256": _sha(copied_manifest),
        },
    }, analysis


def _comparison(
    in_order: dict[str, Any],
    ready_first: dict[str, Any],
    harder: Mapping[str, bool],
    scores: Mapping[str, float],
) -> dict[str, object]:
    in_order_order = in_order["selected_order"]
    ready_order = ready_first["selected_order"]
    _require(
        set(in_order_order) == set(ready_order) == set(harder),
        "cross-arm prompt identity parity mismatch",
    )
    in_steps = {source_id: index // 4 for index, source_id in enumerate(in_order_order)}
    ready_steps = {source_id: index // 4 for index, source_id in enumerate(ready_order)}
    promotions = [
        (in_steps[source_id] - ready_steps[source_id]) / 7
        for source_id, is_harder in harder.items()
        if is_harder
    ]
    horizons = {}
    for horizon in (4, 8, 12, 16, 24, 32):
        horizons[str(horizon)] = {
            sampler: {
                "fixed_harder_share": sum(
                    harder[source_id] for source_id in order[:horizon]
                )
                / horizon,
                "fixed_reference_score_mean": statistics.fmean(
                    scores[source_id] for source_id in order[:horizon]
                ),
            }
            for sampler, order in (
                ("in_order", in_order_order),
                ("ready_first", ready_order),
            )
        }
    return {
        "fixed_harder_mean_normalized_selection_step_promotion": statistics.fmean(
            promotions
        ),
        "fixed_harder_net_selection_step_promotion": sum(
            value * 7 for value in promotions
        ),
        "fixed_harder_mean_absolute_selection_step_displacement": statistics.fmean(
            abs(value * 7) for value in promotions
        ),
        "same_selected_step_sets": sum(
            set(in_order_order[index : index + 4])
            == set(ready_order[index : index + 4])
            for index in range(0, EXPECTED_GROUPS, 4)
        ),
        "horizons": horizons,
    }


def analyze_block(
    *,
    plan: DapoOperationalMixturePlan,
    order_seed: int,
    runs: Mapping[str, Path],
    materialization_manifest_path: Path,
    private_reference_path: Path,
) -> dict[str, object]:
    pool = plan.pool(order_seed)
    _require(
        set(runs) == {arm.arm_id for arm in plan.arms},
        "block must contain all six arms",
    )
    harder, scores = _load_fixed_reference(
        private_reference_path,
        pool_seed=order_seed,
        expected_sha256=pool.private_reference_manifest_sha256,
    )
    results: dict[str, dict[str, Any]] = {}
    traces: dict[str, TraceAnalysis] = {}
    for arm_id in pool.arm_execution_order:
        results[arm_id], traces[arm_id] = analyze_run(
            run_dir=runs[arm_id],
            plan=plan,
            arm=plan.arm(arm_id),
            order_seed=order_seed,
            materialization_manifest_path=materialization_manifest_path,
        )
    comparisons = {
        level: _comparison(
            results[f"{level}_in_order"],
            results[f"{level}_ready_first"],
            harder,
            scores,
        )
        for level in ("l0", "l1", "l3")
    }
    checks = {
        "all_six_arms_valid": all(
            result["all_validity_gates_passed"] for result in results.values()
        ),
        "cross_arm_exact_source_prompt_identity_parity_each_pair": all(
            set(results[f"{level}_in_order"]["source_prompt_ids"])
            == set(results[f"{level}_ready_first"]["source_prompt_ids"])
            for level in ("l0", "l1", "l3")
        ),
        "l0_selected_step_sets_match_all_8_steps": comparisons["l0"][
            "same_selected_step_sets"
        ]
        == 8,
        "l0_fixed_harder_prompt_promotion_equals_zero": comparisons["l0"][
            "fixed_harder_mean_normalized_selection_step_promotion"
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
        else "stop_no_replay_or_training",
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
    plan = load_dapo_operational_mixture_plan(args.plan)
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
