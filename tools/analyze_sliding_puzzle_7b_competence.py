# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Analyze the frozen exposed-pool 7B sliding-puzzle competence screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Final

from nemo_rl.algorithms.async_utils.fixed_pool import (
    load_fixed_pool_manifest,
    validate_fixed_pool_manifest_design,
    validate_fixed_pool_materialization,
    validate_fixed_pool_trace,
)
from nemo_rl.algorithms.async_utils.scheduler_trace import (
    SchedulerEventType,
    iter_scheduler_trace,
)
from tools.analyze_sliding_puzzle_latency_feasibility import (
    DesignItem,
    GroupObservation,
    observations_from_events,
)


PLAN_ID: Final[str] = "3b2b0dbbbc0e799c9668d3649155eca0f97b8b4fc4436d0c2e430d3c700208eb"
PLAN_SHA256: Final[str] = (
    "81912121165377eb2de725f71e61a2fef96af74ea9a1d92085695f338e029494"
)
EXPECTED_THRESHOLDS: Final[dict[str, object]] = {
    "action_format_valid_rate_min": 0.9,
    "all_expected_completions_required": True,
    "easy_max_turn_rate_max": 0.25,
    "easy_solve_rate_min": 0.5,
    "easy_truncation_rate_max": 0.25,
    "movement_legal_rate_min": 0.75,
}
EXPECTED_RUNTIME: Final[dict[str, object]] = {
    "fixed_pool_design_id": "sliding_puzzle_7b_competence_v1",
    "generation_backend": "vllm",
    "max_total_sequence_length": 2048,
    "configured_max_new_tokens": 128,
    "generation_context_length": 2048,
    "generation_temperature": 0.7,
    "generation_top_p": 0.8,
    "generation_top_k": 20,
    "generation_study_seed": 63001,
    "grpo_seed": 20260901,
    "num_generations_per_prompt": 2,
    "max_rollout_turns": 12,
    "num_prompts_per_step": 4,
    "max_inflight_prompts": 4,
    "max_buffered_rollouts": 8,
    "generation_use_async_rollouts": True,
    "tokenizer_eos_token_present": True,
    "effective_stop_token_count": 1,
    "effective_stop_string_count": 0,
    "generation_ignore_eos": False,
    "vllm_skip_tokenizer_init": True,
    "vllm_include_stop_str_in_output": True,
    "finish_reason_code_schema_version": 1,
}


class PuzzleCompetenceAnalysisError(ValueError):
    """The competence materialization or trace violates the frozen plan."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PuzzleCompetenceAnalysisError(f"cannot read {path.name}") from error
    if not isinstance(value, dict):
        raise PuzzleCompetenceAnalysisError(f"{path.name} must contain an object")
    return value


def _load_plan(path: Path) -> dict[str, Any]:
    if _sha256_path(path) != PLAN_SHA256:
        raise PuzzleCompetenceAnalysisError("competence plan byte hash mismatch")
    plan = _load_json(path)
    without_id = {key: value for key, value in plan.items() if key != "plan_id"}
    if (
        plan.get("plan_id") != PLAN_ID
        or hashlib.sha256(_canonical(without_id)).hexdigest() != PLAN_ID
        or plan.get("analysis_status")
        != "calibration_only_exposed_pool_model_competence"
        or plan.get("calibration_only") is not True
        or plan.get("confirmatory_eligible") is not False
        or plan.get("latency_or_readiness_inference_allowed") is not False
        or plan.get("hard_board_metrics_gate") is not False
        or plan.get("replay_authorized") is not False
        or plan.get("training_authorized") is not False
        or plan.get("thresholds") != EXPECTED_THRESHOLDS
    ):
        raise PuzzleCompetenceAnalysisError("competence plan mismatch")
    return plan


def _source_rows(manifest: Any) -> dict[str, list[dict[str, object]]]:
    values: dict[str, list[dict[str, object]]] = {}
    for source in manifest.sources:
        path = manifest.manifest_path.parent / source.materialized_file
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if any(not isinstance(row, dict) for row in rows):
            raise PuzzleCompetenceAnalysisError(
                "materialized source row is not an object"
            )
        values[source.source_id] = rows
    return values


def _design_items(manifest: Any) -> tuple[DesignItem, ...]:
    rows = _source_rows(manifest)
    items = []
    for item in manifest.items:
        record = rows[item.source_id][item.materialized_source_row]
        stratum = item.task_name.removeprefix("sliding_puzzle_")
        if stratum not in {"easy", "hard"}:
            raise PuzzleCompetenceAnalysisError("invalid competence stratum")
        items.append(
            DesignItem(
                source_pool_ordinal=item.ordinal,
                source_prompt_id=item.source_prompt_id,
                task_name=item.task_name,
                pair_id=item.matching_pair_id,
                stratum=stratum,
                optimal_distance=int(record["optimal_distance"]),
                rendered_prompt_tokens=item.input_token_count,
                board_state_sha256=str(record["board_state_sha256"]),
            )
        )
    return tuple(items)


def _validate_lineage(
    *,
    manifest: Any,
    plan: dict[str, Any],
    lineage_path: Path,
    items: tuple[DesignItem, ...],
) -> dict[str, Any]:
    lineage = _load_json(lineage_path)
    source = plan["source_pool"]
    expected = {
        "schema_version": 1,
        "analysis_status": plan["analysis_status"],
        "plan_id": PLAN_ID,
        "plan_sha256": manifest.design_protocol_sha256,
        "source_manifest_sha256": source["fixed_pool_manifest_sha256"],
        "source_pool_id": source["pool_id"],
        "source_selection_design_sha256": source["selection_design_sha256"],
        "board_state_sha256": sorted(item.board_state_sha256 for item in items),
        "materialized_manifest_sha256": manifest.manifest_sha256,
        "materialized_pool_id": manifest.pool_id,
    }
    if lineage != expected:
        raise PuzzleCompetenceAnalysisError("source-lineage binding mismatch")
    return lineage


def _summarize(observations: tuple[GroupObservation, ...]) -> dict[str, object]:
    if len(observations) != 16:
        raise PuzzleCompetenceAnalysisError("competence analysis requires 16 boards")
    by_stratum: dict[str, list[GroupObservation]] = defaultdict(list)
    for observation in observations:
        by_stratum[observation.item.stratum].append(observation)
    if {key: len(value) for key, value in by_stratum.items()} != {
        "easy": 8,
        "hard": 8,
    }:
        raise PuzzleCompetenceAnalysisError("competence strata are incomplete")
    action_turns = sum(item.action_turns for item in observations)
    format_valid = sum(item.format_valid_actions for item in observations)
    legal_moves = sum(item.legal_moves for item in observations)
    invalid_moves = sum(item.invalid_moves for item in observations)
    movement_attempts = legal_moves + invalid_moves
    action_format_rate = format_valid / action_turns if action_turns else 0.0
    movement_legal_rate = legal_moves / movement_attempts if movement_attempts else 0.0
    solve_rates = {
        stratum: sum(item.solved_completions for item in values) / (len(values) * 2)
        for stratum, values in by_stratum.items()
    }
    truncation_rates = {
        stratum: sum(item.truncations for item in values) / (len(values) * 2)
        for stratum, values in by_stratum.items()
    }
    max_turn_rates = {
        stratum: sum(item.max_turns_reached for item in values) / (len(values) * 2)
        for stratum, values in by_stratum.items()
    }
    checks = {
        "all_expected_completions": True,
        "action_format_valid_rate_ge_0_9": action_format_rate >= 0.9,
        "movement_legal_rate_ge_0_75": movement_legal_rate >= 0.75,
        "easy_solve_rate_ge_0_5": solve_rates["easy"] >= 0.5,
        "easy_truncation_rate_le_0_25": truncation_rates["easy"] <= 0.25,
        "easy_max_turn_rate_le_0_25": max_turn_rates["easy"] <= 0.25,
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "analysis_status": "calibration_only_exposed_pool_model_competence",
        "calibration_only": True,
        "confirmatory_eligible": False,
        "latency_or_readiness_inference_allowed": False,
        "replay_authorized": False,
        "training_authorized": False,
        "plan_id": PLAN_ID,
        "boards": 16,
        "completions": 32,
        "actions": {
            "turn_count": action_turns,
            "format_valid_count": format_valid,
            "format_valid_rate": action_format_rate,
            "legal_move_count": legal_moves,
            "invalid_move_count": invalid_moves,
            "movement_attempt_count": movement_attempts,
            "movement_legal_rate": movement_legal_rate,
            "invalid_format_count": sum(
                item.invalid_format_actions for item in observations
            ),
            "view_count": sum(item.view_actions for item in observations),
        },
        "solve_rate_by_stratum": solve_rates,
        "truncation_rate_by_stratum": truncation_rates,
        "max_turn_rate_by_stratum": max_turn_rates,
        "locked_checks": checks,
        "decision": "pass_competence_to_fresh_disjoint_latency_design"
        if passed
        else "stop_model_not_competent",
        "interpretation": "model-competence calibration on an exposed board pool; no scheduler-latency inference",
    }


def analyze(
    *, trace_path: Path, manifest_path: Path, plan_path: Path, lineage_path: Path
) -> dict[str, object]:
    """Validate the frozen run and compute competence-only metrics."""
    plan = _load_plan(plan_path)
    manifest = load_fixed_pool_manifest(manifest_path)
    validate_fixed_pool_materialization(manifest)
    validate_fixed_pool_manifest_design(manifest, "sliding_puzzle_7b_competence_v1")
    if _sha256_path(plan_path) != manifest.design_protocol_sha256:
        raise PuzzleCompetenceAnalysisError("manifest/plan hash mismatch")
    items = _design_items(manifest)
    lineage = _validate_lineage(
        manifest=manifest, plan=plan, lineage_path=lineage_path, items=items
    )
    report = validate_fixed_pool_trace(
        trace_path, manifest, expected_completions_per_group=2
    )
    if report.physical_weight_version != 0:
        raise PuzzleCompetenceAnalysisError("competence requires weight version zero")
    events = tuple(iter_scheduler_trace(trace_path))
    observations = observations_from_events(
        events,
        items,
        expected_runtime=EXPECTED_RUNTIME,
        generation_seed=63001,
    )
    starts = [
        event for event in events if event.event_type is SchedulerEventType.RUN_STARTED
    ]
    runtime = starts[0].scalar_summaries
    expected_path = str(manifest_path.resolve().parent / "model_snapshot")
    expected_path_sha = hashlib.sha256(_canonical(expected_path)).hexdigest()
    if (
        runtime.get("policy_model_name_sha256") != expected_path_sha
        or runtime.get("policy_tokenizer_name_sha256") != expected_path_sha
    ):
        raise PuzzleCompetenceAnalysisError("runtime model snapshot binding mismatch")
    result = _summarize(observations)
    result["runtime_binding"] = dict(runtime)
    result["source_lineage"] = lineage
    result["source_artifacts"] = {
        "trace_sha256": _sha256_path(trace_path),
        "manifest_sha256": manifest.manifest_sha256,
        "plan_sha256": _sha256_path(plan_path),
        "lineage_sha256": _sha256_path(lineage_path),
        "pool_id": manifest.pool_id,
        "physical_weight_version": report.physical_weight_version,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = analyze(
        trace_path=args.trace,
        manifest_path=args.manifest,
        plan_path=args.plan,
        lineage_path=args.lineage,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
