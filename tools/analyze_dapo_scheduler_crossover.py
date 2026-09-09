# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fail-closed analysis for the paired DAPO natural-latency crossover."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from nemo_rl.algorithms.async_utils.fixed_pool import (
    FixedPoolManifest,
    load_fixed_pool_manifest,
    validate_fixed_pool_manifest_design,
    validate_fixed_pool_materialization,
)
from nemo_rl.algorithms.async_utils.scheduler_replay import (
    ReplayGroup,
    ReplayPolicy,
    ReplayTick,
    natural_releases,
    replay_schedule,
)
from nemo_rl.algorithms.async_utils.scheduler_trace import (
    SchedulerEventType,
    SchedulerTraceEvent,
    iter_scheduler_trace,
    validate_scheduler_trace,
)
from nemo_rl.algorithms.async_utils.structured_scheduler_crossover import (
    DapoSchedulerCrossoverPlan,
    StructuredSchedulerCrossoverArm,
    load_dapo_scheduler_crossover_plan,
)
from tools import materialize_dapo_scheduler_crossover as materializer


EXPECTED_GROUPS = 16
EXPECTED_COMPLETIONS = 16
EXPECTED_SELECTIONS = 4
GROUPS_PER_SELECTION = 4


class DapoSchedulerCrossoverAnalysisError(ValueError):
    """A trace, immutable input, or frozen requirement is inconsistent."""


@dataclass(frozen=True, slots=True)
class DesignItem:
    source_pool_ordinal: int
    source_prompt_id: str
    task_name: str
    canonical_prompt_sha256: str
    source_dataset_index: int
    source_duplicate_count: int
    rendered_prompt_tokens: int


@dataclass(frozen=True, slots=True)
class Observation:
    item: DesignItem
    logical_group_id: str
    dispatch_ns: int
    ready_ns: int
    reward_mean: float
    reward_min: float
    reward_max: float
    mean_generated_tokens: float
    min_generated_tokens: int
    max_generated_tokens: int
    length_terminations: int

    @property
    def ready_latency_ns(self) -> int:
        return self.ready_ns - self.dispatch_ns


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DapoSchedulerCrossoverAnalysisError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _number(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DapoSchedulerCrossoverAnalysisError(f"missing numeric summary {key}")
    number = float(value)
    _require(math.isfinite(number), f"non-finite summary {key}")
    return number


def _current_clean_commit(repo_root: Path) -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(not status, "analysis worktree must be clean")
    return commit


def _load_design(path: Path, expected_seed: int) -> tuple[str, tuple[DesignItem, ...]]:
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise DapoSchedulerCrossoverAnalysisError(
            "selection design is unreadable"
        ) from error
    required = {
        "schema_version",
        "analysis_status",
        "calibration_only",
        "confirmatory_eligible",
        "protocol_sha256",
        "fixed_pool_id",
        "selection_seed",
        "generation_seed",
        "items",
    }
    expected_generation = next(
        generation
        for selection, _, generation in materializer.POOL_SPECS
        if selection == expected_seed
    )
    _require(
        isinstance(raw, dict)
        and set(raw) == required
        and raw["schema_version"] == 1
        and raw["analysis_status"] == "controlled_dapo_scheduler_crossover_pool"
        and raw["calibration_only"] is True
        and raw["confirmatory_eligible"] is False
        and raw["protocol_sha256"] == materializer.PROTOCOL_SHA256
        and raw["selection_seed"] == expected_seed
        and raw["generation_seed"] == expected_generation,
        "selection design contract mismatch",
    )
    fields = set(DesignItem.__dataclass_fields__)
    items = []
    for value in raw["items"]:
        _require(isinstance(value, dict) and set(value) == fields, "item mismatch")
        items.append(DesignItem(**value))
    _require(
        len(items) == EXPECTED_GROUPS
        and [item.source_pool_ordinal for item in items]
        == list(range(EXPECTED_GROUPS))
        and len({item.canonical_prompt_sha256 for item in items}) == EXPECTED_GROUPS,
        "selection design pool geometry mismatch",
    )
    return str(raw["fixed_pool_id"]), tuple(items)


def _load_manifest(
    *, copied: Path, materialized: Path, plan: DapoSchedulerCrossoverPlan
) -> FixedPoolManifest:
    _require(
        copied.read_bytes() == materialized.read_bytes(),
        "run manifest is not byte-identical to materialization",
    )
    manifest = load_fixed_pool_manifest(materialized)
    validate_fixed_pool_materialization(manifest)
    validate_fixed_pool_manifest_design(manifest, plan.source_design_id)
    return manifest


def _successful_decisions(
    events: Sequence[SchedulerTraceEvent],
) -> tuple[SchedulerTraceEvent, ...]:
    decisions = tuple(
        event
        for event in events
        if event.event_type is SchedulerEventType.SELECT_DECISION
        and event.selected_logical_group_ids
    )
    _require(len(decisions) == EXPECTED_SELECTIONS, "requires four selections")
    for index, decision in enumerate(decisions):
        _require(
            len(decision.selected_logical_group_ids) == GROUPS_PER_SELECTION
            and decision.scalar_summaries.get("selected_prompt_groups")
            == GROUPS_PER_SELECTION
            and decision.scalar_summaries.get("scheduler_assay_step") == index,
            "selection size or logical step mismatch",
        )
    return decisions


def _maximum_concurrency(observations: Sequence[Observation]) -> int:
    changes = [
        change
        for item in observations
        for change in ((item.dispatch_ns, 1), (item.ready_ns, -1))
    ]
    active = maximum = 0
    for _, change in sorted(changes, key=lambda item: (item[0], item[1])):
        active += change
        maximum = max(maximum, active)
    return maximum


def _observations_and_order(
    events: Sequence[SchedulerTraceEvent],
    manifest: FixedPoolManifest,
    items: Sequence[DesignItem],
    arm: StructuredSchedulerCrossoverArm,
) -> tuple[tuple[Observation, ...], tuple[str, ...], bool]:
    starts = [event for event in events if event.event_type is SchedulerEventType.RUN_STARTED]
    _require(len(starts) == 1, "trace requires one run_started")
    runtime = starts[0].scalar_summaries
    _require(
        runtime.get("generation_backend") == "vllm"
        and runtime.get("max_total_sequence_length") == 6144
        and runtime.get("configured_max_new_tokens") == 4096
        and runtime.get("generation_context_length") == 6144,
        "runtime generation bounds mismatch",
    )
    forbidden = {
        SchedulerEventType.ATTEMPT_FAILED,
        SchedulerEventType.ATTEMPT_REMOVED,
        SchedulerEventType.GROUP_EVICTED,
        SchedulerEventType.GROUP_ARCHIVED,
        SchedulerEventType.PROMPT_SKIPPED,
        SchedulerEventType.GROUP_REPLACED,
        SchedulerEventType.GROUP_PROMOTED,
    }
    _require(not any(event.event_type in forbidden for event in events), "forbidden event")
    by_ordinal = {item.source_pool_ordinal: item for item in items}
    maps: dict[SchedulerEventType, dict[int, SchedulerTraceEvent]] = {
        kind: {}
        for kind in (
            SchedulerEventType.ATTEMPT_DISPATCHED,
            SchedulerEventType.ROLLOUT_COMPLETED,
            SchedulerEventType.GROUP_READY,
        )
    }
    for event in events:
        if event.event_type not in maps:
            continue
        ordinal = event.source_pool_ordinal
        _require(
            ordinal in by_ordinal and ordinal not in maps[event.event_type],
            "unknown or duplicate lifecycle ordinal",
        )
        maps[event.event_type][cast(int, ordinal)] = event
    expected = set(by_ordinal)
    _require(all(set(values) == expected for values in maps.values()), "incomplete lifecycle")
    observations = []
    replay_groups = []
    manifest_by_ordinal = {item.ordinal: item for item in manifest.items}
    dispatch_ordinals = sorted(
        expected,
        key=lambda ordinal: maps[SchedulerEventType.ATTEMPT_DISPATCHED][ordinal].event_seq,
    )
    for slot_order, ordinal in enumerate(dispatch_ordinals):
        item = by_ordinal[ordinal]
        dispatch = maps[SchedulerEventType.ATTEMPT_DISPATCHED][ordinal]
        completed = maps[SchedulerEventType.ROLLOUT_COMPLETED][ordinal]
        ready = maps[SchedulerEventType.GROUP_READY][ordinal]
        _require(
            all(
                event.source_prompt_id == item.source_prompt_id
                and event.task_name == item.task_name
                and event.logical_group_id == dispatch.logical_group_id
                for event in (dispatch, completed, ready)
            ),
            "trace/design identity mismatch",
        )
        _require(
            dispatch.monotonic_ns < completed.monotonic_ns <= ready.monotonic_ns,
            "non-monotonic lifecycle",
        )
        summary = completed.scalar_summaries
        _require(_number(summary, "completion_count") == EXPECTED_COMPLETIONS, "completion count mismatch")
        mean_tokens = _number(summary, "mean_gen_tokens_per_sample")
        min_tokens = _number(summary, "gen_tokens_per_sample/min")
        max_tokens = _number(summary, "gen_tokens_per_sample/max")
        length_rate = _number(summary, "backend_length_termination_rate")
        length_count = round(length_rate * EXPECTED_COMPLETIONS)
        _require(
            min_tokens.is_integer()
            and max_tokens.is_integer()
            and 0 <= min_tokens <= mean_tokens <= max_tokens <= 4096
            and math.isclose(length_count, length_rate * EXPECTED_COMPLETIONS, abs_tol=1e-9)
            and _number(summary, "backend_finish_reason_availability_rate") == 1.0,
            "completion summaries mismatch",
        )
        group_id = dispatch.logical_group_id
        _require(group_id is not None, "missing logical group ID")
        observations.append(
            Observation(
                item=item,
                logical_group_id=cast(str, group_id),
                dispatch_ns=dispatch.monotonic_ns,
                ready_ns=ready.monotonic_ns,
                reward_mean=_number(summary, "reward_mean"),
                reward_min=_number(summary, "reward_min"),
                reward_max=_number(summary, "reward_max"),
                mean_generated_tokens=mean_tokens,
                min_generated_tokens=int(min_tokens),
                max_generated_tokens=int(max_tokens),
                length_terminations=length_count,
            )
        )
        manifest_item = manifest_by_ordinal[ordinal]
        replay_groups.append(
            ReplayGroup(
                logical_group_id=cast(str, group_id),
                prompt_uid=manifest_item.pool_item_id,
                source_prompt_id=manifest_item.source_prompt_id,
                task_stratum=manifest_item.task_name,
                repeated_prompt_cluster_uid=manifest_item.repeated_prompt_cluster_id,
                dispatch_cohort=str(manifest_item.dispatch_cohort),
                decorrelation_block=manifest_item.decorrelation_block,
                slot_order=slot_order,
                dispatch_ns=dispatch.monotonic_ns,
                ready_ns=ready.monotonic_ns,
                nominal_start_version=0,
                target_step=dispatch.target_step,
                physical_start_version=dispatch.start_weight_version,
                physical_end_version=ready.end_weight_version,
                scalar_summaries={},
            )
        )
    decisions = _successful_decisions(events)
    native_order = tuple(
        group_id
        for decision in decisions
        for group_id in decision.selected_logical_group_ids
    )
    replay = replay_schedule(
        tuple(replay_groups),
        tuple(
            ReplayTick(
                tick_id=index,
                monotonic_ns=decision.monotonic_ns,
                nominal_trainer_version=index,
                min_prompt_groups=GROUPS_PER_SELECTION,
                max_prompt_groups=GROUPS_PER_SELECTION,
            )
            for index, decision in enumerate(decisions)
        ),
        ReplayPolicy(name=arm.sampler, max_staleness_versions=1),
        natural_releases(tuple(replay_groups)),
    )
    parity = len(replay.decisions) == len(decisions) and all(
        native.selected_logical_group_ids == reconstructed.selected_group_ids
        and native.eligible_logical_group_ids == reconstructed.eligible_group_ids
        for native, reconstructed in zip(decisions, replay.decisions, strict=True)
    )
    _require(not replay.evicted_group_ids and not replay.live_group_ids, "replay did not drain")
    return tuple(observations), native_order, parity


def analyze_run(
    *,
    run_dir: Path,
    plan: DapoSchedulerCrossoverPlan,
    arm: StructuredSchedulerCrossoverArm,
    order_seed: int,
    materialization_root: Path,
) -> dict[str, Any]:
    manifest_source = materialization_root / f"fixed_pool_manifest.v1.{order_seed}.json"
    manifest = _load_manifest(
        copied=run_dir / "fixed_pool_manifest.v1.json",
        materialized=manifest_source,
        plan=plan,
    )
    pool = plan.pool(order_seed)
    design_pool_id, items = _load_design(
        materialization_root / f"selection_design.v1.{order_seed}.json", order_seed
    )
    _require(
        (manifest.pool_id, manifest.manifest_sha256, design_pool_id)
        == (pool.pool_id, pool.manifest_sha256, pool.pool_id),
        "artifacts do not match frozen pool",
    )
    trace_path = run_dir / "scheduler_trace.v1.jsonl"
    report = validate_scheduler_trace(trace_path)
    _require(
        not report.incomplete_attempt_ids and not report.administratively_censored_group_ids,
        "trace is incomplete or administratively censored",
    )
    events = tuple(iter_scheduler_trace(trace_path))
    started = [event for event in events if event.event_type is SchedulerEventType.RUN_STARTED]
    ended = [event for event in events if event.event_type is SchedulerEventType.RUN_ENDED]
    _require(len(started) == len(ended) == 1, "trace requires one start and end")
    _require(
        started[0].run_mode == "scheduler_assay"
        and started[0].pool_id == pool.pool_id
        and started[0].scalar_summaries.get("scheduler_assay_plan_id") == plan.plan_id
        and started[0].scalar_summaries.get("scheduler_assay_arm_id") == arm.arm_id,
        "run identity mismatch",
    )
    _require(ended[0].terminal_reason == "scheduler_assay_complete", "run failed")
    _require(
        ended[0].scalar_summaries.get("completed_train_steps") == 0
        and ended[0].scalar_summaries.get("final_physical_weight_version") == 0,
        "learner update or physical weight change detected",
    )
    observations, selected_ids, parity = _observations_and_order(events, manifest, items, arm)
    by_group = {item.logical_group_id: item for item in observations}
    _require(set(selected_ids) == set(by_group), "selection did not drain exact groups")
    selected_prompt_ids = [by_group[group_id].item.source_prompt_id for group_id in selected_ids]
    length_rate = sum(item.length_terminations for item in observations) / (
        EXPECTED_GROUPS * EXPECTED_COMPLETIONS
    )
    validity = {
        "complete_16_groups_256_completions": len(observations) == EXPECTED_GROUPS,
        "four_exact_selection_decisions": len(selected_ids) == EXPECTED_GROUPS,
        "zero_administrative_censoring": True,
        "zero_learner_steps_and_physical_version_zero": True,
        "native_reconstruction_exact_group_id_parity": parity,
        "backend_length_termination_rate_le_0_2": (
            length_rate <= plan.thresholds.backend_length_termination_rate_max_each_arm
        ),
        "maximum_concurrent_groups_equals_4": _maximum_concurrency(observations) == 4,
    }
    return {
        "arm_id": arm.arm_id,
        "order_seed": order_seed,
        "prompt_groups": len(observations),
        "completions": len(observations) * EXPECTED_COMPLETIONS,
        "selection_steps": EXPECTED_SELECTIONS,
        "train_steps": 0,
        "final_physical_weight_version": 0,
        "selection_order_source_prompt_ids": selected_prompt_ids,
        "observations": [
            {
                "source_prompt_id": item.item.source_prompt_id,
                "mean_generated_tokens": item.mean_generated_tokens,
                "ready_latency_ns": item.ready_latency_ns,
                "reward_mean": item.reward_mean,
                "reward_min": item.reward_min,
                "reward_max": item.reward_max,
                "length_terminations": item.length_terminations,
            }
            for item in observations
        ],
        "backend_length_termination_rate": length_rate,
        "maximum_concurrent_groups": _maximum_concurrency(observations),
        "validity_checks": validity,
        "all_arm_validity_gates_passed": all(validity.values()),
        "source_artifacts": {
            "trace_sha256": _sha(trace_path),
            "manifest_sha256": _sha(manifest_source),
            "selection_design_sha256": _sha(
                materialization_root / f"selection_design.v1.{order_seed}.json"
            ),
        },
    }


def analyze_replication(
    *,
    plan: DapoSchedulerCrossoverPlan,
    order_seed: int,
    runs: Mapping[str, Path],
    materialization_root: Path,
) -> dict[str, Any]:
    pool = plan.pool(order_seed)
    _require(set(runs) == {"ready_first", "in_order"}, "both exact arms required")
    results = {
        arm_id: analyze_run(
            run_dir=runs[arm_id],
            plan=plan,
            arm=plan.arm(arm_id),
            order_seed=order_seed,
            materialization_root=materialization_root,
        )
        for arm_id in pool.arm_execution_order
    }
    ready = results["ready_first"]
    ordered = results["in_order"]
    ready_ids = ready["selection_order_source_prompt_ids"]
    ordered_ids = ordered["selection_order_source_prompt_ids"]
    _require(set(ready_ids) == set(ordered_ids), "cross-arm prompt identity mismatch")
    in_order_observations = {
        item["source_prompt_id"]: item for item in ordered["observations"]
    }
    lower_half = sorted(
        in_order_observations,
        key=lambda prompt_id: (
            in_order_observations[prompt_id]["mean_generated_tokens"], prompt_id
        ),
    )[:8]
    ready_ranks = {prompt_id: rank for rank, prompt_id in enumerate(ready_ids, 1)}
    ordered_ranks = {prompt_id: rank for rank, prompt_id in enumerate(ordered_ids, 1)}
    advantages = {
        prompt_id: (ordered_ranks[prompt_id] - ready_ranks[prompt_id]) / 15
        for prompt_id in lower_half
    }
    distinct_loads = len(
        {item["mean_generated_tokens"] for item in ordered["observations"]}
    )
    replication_validity = {
        "both_arms_valid": all(
            result["all_arm_validity_gates_passed"] for result in results.values()
        ),
        "cross_arm_exact_source_prompt_id_parity": True,
        "distinct_in_order_reference_loads_ge_12": (
            distinct_loads
            >= plan.thresholds.distinct_in_order_group_reference_loads_min_each_replication
        ),
    }
    return {
        "schema_version": 1,
        "analysis_status": plan.analysis_status,
        "calibration_only": True,
        "confirmatory_eligible": False,
        "population_claim_authorized": False,
        "counterfactual_replay_authorized": False,
        "training_authorized": False,
        "plan_id": plan.plan_id,
        "replication_id": pool.replication_id,
        "order_seed": order_seed,
        "arm_execution_order": list(pool.arm_execution_order),
        "lower_load_source_prompt_ids": lower_half,
        "distinct_in_order_group_reference_loads": distinct_loads,
        "lower_load_normalized_selection_rank_advantage": statistics.fmean(
            advantages.values()
        ),
        "lower_load_per_group_normalized_rank_advantage": advantages,
        "validity_checks": replication_validity,
        "all_replication_validity_gates_passed": all(replication_validity.values()),
        "runs": results,
        "decision": "replication_recorded_for_final_aggregate",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--order-seed", type=int, required=True)
    parser.add_argument("--run", action="append", required=True, metavar="ARM=DIR")
    parser.add_argument("--materialization-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = load_dapo_scheduler_crossover_plan(args.plan)
    _require(
        _current_clean_commit(args.repo_root) == plan.analysis_code_commit,
        "analysis code commit does not match frozen plan",
    )
    runs: dict[str, Path] = {}
    for assignment in args.run:
        arm_id, separator, path = assignment.partition("=")
        _require(bool(separator and arm_id and path), f"invalid --run {assignment!r}")
        _require(arm_id not in runs, f"duplicate arm {arm_id!r}")
        runs[arm_id] = Path(path)
    result = analyze_replication(
        plan=plan,
        order_seed=args.order_seed,
        runs=runs,
        materialization_root=args.materialization_root,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
