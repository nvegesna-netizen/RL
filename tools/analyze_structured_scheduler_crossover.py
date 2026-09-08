# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fail-closed analysis for the structured natural-latency scheduler crossover."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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
    CONFIRMED_CANDIDATE_SHA256,
    StructuredSchedulerCrossoverArm,
    StructuredSchedulerCrossoverPlan,
    load_structured_scheduler_crossover_plan,
)


EXPECTED_GROUPS = 16
EXPECTED_COMPLETIONS = 2
EXPECTED_SELECTIONS = 4
GROUPS_PER_SELECTION = 4


class StructuredSchedulerCrossoverAnalysisError(ValueError):
    """An artifact or locked protocol invariant is invalid."""


@dataclass(frozen=True, slots=True)
class DesignItem:
    source_pool_ordinal: int
    source_prompt_id: str
    task_name: str
    pair_id: str
    stratum: Literal["short", "long"]
    left_operand: int
    right_operand: int
    expected_answer: int
    required_check_lines: int
    rendered_prompt_tokens: int


@dataclass(frozen=True, slots=True)
class Observation:
    item: DesignItem
    logical_group_id: str
    dispatch_ns: int
    completed_ns: int
    ready_ns: int
    reward_mean: float
    min_generated_tokens: int
    max_generated_tokens: int
    length_terminations: int

    @property
    def ready_latency_ns(self) -> int:
        return self.ready_ns - self.dispatch_ns


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StructuredSchedulerCrossoverAnalysisError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _number(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StructuredSchedulerCrossoverAnalysisError(
            f"missing numeric summary {key}"
        )
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


def load_design(path: Path) -> tuple[str, tuple[DesignItem, ...]]:
    raw = json.loads(path.read_text())
    required = {
        "schema_version",
        "analysis_status",
        "calibration_only",
        "confirmatory_eligible",
        "protocol_id",
        "fixed_pool_id",
        "items",
    }
    _require(isinstance(raw, dict) and set(raw) == required, "design fields mismatch")
    _require(
        raw["schema_version"] == 1
        and raw["analysis_status"]
        == "controlled_natural_latency_scheduler_crossover_pool"
        and raw["calibration_only"] is True
        and raw["confirmatory_eligible"] is False
        and raw["protocol_id"] == CONFIRMED_CANDIDATE_SHA256,
        "design labels mismatch",
    )
    fields = set(DesignItem.__dataclass_fields__)
    items = []
    for value in raw["items"]:
        _require(
            isinstance(value, dict) and set(value) == fields, "item fields mismatch"
        )
        items.append(DesignItem(**value))
    _require(len(items) == EXPECTED_GROUPS, "expected exactly 16 prompt groups")
    _require(
        [item.source_pool_ordinal for item in items] == list(range(EXPECTED_GROUPS)),
        "ordinals are not contiguous",
    )
    by_pair: dict[str, list[DesignItem]] = defaultdict(list)
    by_cohort: dict[int, list[DesignItem]] = defaultdict(list)
    for item in items:
        by_pair[item.pair_id].append(item)
        by_cohort[item.source_pool_ordinal // GROUPS_PER_SELECTION].append(item)
        expected_lines = 2 if item.stratum == "short" else 16
        _require(
            item.task_name == f"structured_{item.stratum}"
            and item.required_check_lines == expected_lines
            and item.expected_answer == item.left_operand + item.right_operand
            and item.rendered_prompt_tokens > 0,
            "selection item value mismatch",
        )
    _require(
        len(by_pair) == 8
        and all(
            len(pair) == 2 and {item.stratum for item in pair} == {"short", "long"}
            for pair in by_pair.values()
        ),
        "pairing is incomplete",
    )
    _require(
        len(by_cohort) == 4
        and all(
            len(cohort) == 4 and sum(item.stratum == "short" for item in cohort) == 2
            for cohort in by_cohort.values()
        ),
        "dispatch cohorts are not balanced 2+2",
    )
    return str(raw["fixed_pool_id"]), tuple(items)


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


def _observations_and_replay(
    events: Sequence[SchedulerTraceEvent],
    manifest: FixedPoolManifest,
    items: Sequence[DesignItem],
    arm: StructuredSchedulerCrossoverArm,
) -> tuple[tuple[Observation, ...], tuple[str, ...], bool]:
    starts = [
        event for event in events if event.event_type is SchedulerEventType.RUN_STARTED
    ]
    _require(len(starts) == 1, "trace requires one run_started")
    runtime = starts[0].scalar_summaries
    _require(
        runtime.get("generation_backend") == "vllm"
        and runtime.get("max_total_sequence_length") == 768
        and runtime.get("configured_max_new_tokens") == 768
        and runtime.get("generation_context_length") == 768,
        "runtime generation bounds mismatch",
    )
    by_ordinal = {item.source_pool_ordinal: item for item in items}
    manifest_by_ordinal = {item.ordinal: item for item in manifest.items}
    maps = {
        kind: {}
        for kind in (
            SchedulerEventType.ATTEMPT_DISPATCHED,
            SchedulerEventType.ROLLOUT_COMPLETED,
            SchedulerEventType.GROUP_READY,
        )
    }
    forbidden = {
        SchedulerEventType.ATTEMPT_FAILED,
        SchedulerEventType.ATTEMPT_REMOVED,
        SchedulerEventType.GROUP_EVICTED,
        SchedulerEventType.GROUP_ARCHIVED,
    }
    _require(
        not any(event.event_type in forbidden for event in events),
        "trace contains failure, removal, eviction, or archive",
    )
    for event in events:
        if event.event_type not in maps:
            continue
        ordinal = event.source_pool_ordinal
        _require(
            ordinal in by_ordinal and ordinal not in maps[event.event_type],
            "unknown or duplicate lifecycle ordinal",
        )
        maps[event.event_type][ordinal] = event
    expected = set(by_ordinal)
    _require(
        all(set(event_map) == expected for event_map in maps.values()),
        "lifecycle coverage is incomplete",
    )
    observations = []
    replay_groups = []
    for slot_order, ordinal in enumerate(
        sorted(
            expected,
            key=lambda value: (
                maps[SchedulerEventType.ATTEMPT_DISPATCHED][value].event_seq
            ),
        )
    ):
        item = by_ordinal[ordinal]
        manifest_item = manifest_by_ordinal[ordinal]
        dispatch = maps[SchedulerEventType.ATTEMPT_DISPATCHED][ordinal]
        completed = maps[SchedulerEventType.ROLLOUT_COMPLETED][ordinal]
        ready = maps[SchedulerEventType.GROUP_READY][ordinal]
        identity_events = (dispatch, completed, ready)
        _require(
            all(
                event.source_prompt_id == item.source_prompt_id
                and event.task_name == item.task_name
                and event.logical_group_id == dispatch.logical_group_id
                for event in identity_events
            ),
            "trace/design identity mismatch",
        )
        _require(
            dispatch.monotonic_ns < completed.monotonic_ns <= ready.monotonic_ns,
            "non-monotonic lifecycle",
        )
        _require(
            "scheduler_assay_release_delay_seconds" not in completed.scalar_summaries,
            "artificial release-delay metadata is forbidden",
        )
        summaries = completed.scalar_summaries
        _require(
            _number(summaries, "completion_count") == EXPECTED_COMPLETIONS,
            "completion count mismatch",
        )
        mean_tokens = _number(summaries, "mean_gen_tokens_per_sample")
        min_tokens = _number(summaries, "gen_tokens_per_sample/min")
        max_tokens = _number(summaries, "gen_tokens_per_sample/max")
        length_rate = _number(summaries, "backend_length_termination_rate")
        length_count = round(length_rate * EXPECTED_COMPLETIONS)
        _require(
            min_tokens.is_integer()
            and max_tokens.is_integer()
            and 0 <= min_tokens <= mean_tokens <= max_tokens
            and math.isclose(
                EXPECTED_COMPLETIONS * mean_tokens,
                min_tokens + max_tokens,
                abs_tol=1e-9,
            )
            and math.isclose(
                length_count,
                length_rate * EXPECTED_COMPLETIONS,
                abs_tol=1e-9,
            )
            and _number(summaries, "backend_finish_reason_availability_rate") == 1.0,
            "completion summaries mismatch",
        )
        group_id = dispatch.logical_group_id
        if group_id is None:
            raise StructuredSchedulerCrossoverAnalysisError(
                "logical group ID is missing"
            )
        observations.append(
            Observation(
                item=item,
                logical_group_id=group_id,
                dispatch_ns=dispatch.monotonic_ns,
                completed_ns=completed.monotonic_ns,
                ready_ns=ready.monotonic_ns,
                reward_mean=_number(summaries, "reward_mean"),
                min_generated_tokens=int(min_tokens),
                max_generated_tokens=int(max_tokens),
                length_terminations=length_count,
            )
        )
        replay_groups.append(
            ReplayGroup(
                logical_group_id=group_id,
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
    ticks = tuple(
        ReplayTick(
            tick_id=index,
            monotonic_ns=decision.monotonic_ns,
            nominal_trainer_version=index,
            min_prompt_groups=GROUPS_PER_SELECTION,
            max_prompt_groups=GROUPS_PER_SELECTION,
        )
        for index, decision in enumerate(decisions)
    )
    replay = replay_schedule(
        tuple(replay_groups),
        ticks,
        ReplayPolicy(name=arm.sampler, max_staleness_versions=1),
        natural_releases(tuple(replay_groups)),
    )
    parity = len(replay.decisions) == len(decisions) and all(
        native.selected_logical_group_ids == reconstructed.selected_group_ids
        and native.eligible_logical_group_ids == reconstructed.eligible_group_ids
        for native, reconstructed in zip(decisions, replay.decisions, strict=True)
    )
    _require(
        not replay.evicted_group_ids and not replay.live_group_ids,
        "replay did not drain",
    )
    return tuple(observations), replay.selected_group_ids, parity


def _maximum_concurrency(observations: Sequence[Observation]) -> int:
    changes = [
        change
        for observation in observations
        for change in ((observation.dispatch_ns, 1), (observation.ready_ns, -1))
    ]
    active = maximum = 0
    for _, change in sorted(changes, key=lambda value: (value[0], value[1])):
        active += change
        maximum = max(maximum, active)
    return maximum


def analyze_run(
    *,
    run_dir: Path,
    plan: StructuredSchedulerCrossoverPlan,
    arm: StructuredSchedulerCrossoverArm,
    order_seed: int,
) -> dict[str, Any]:
    trace_path = run_dir / "scheduler_trace.v1.jsonl"
    manifest_path = run_dir / "fixed_pool_manifest.v1.json"
    design_path = run_dir / "selection_design.v1.json"
    _require(trace_path.is_file(), f"missing trace: {trace_path}")
    _require(manifest_path.is_file(), f"missing manifest: {manifest_path}")
    _require(design_path.is_file(), f"missing design: {design_path}")
    manifest = load_fixed_pool_manifest(manifest_path)
    validate_fixed_pool_materialization(manifest)
    validate_fixed_pool_manifest_design(manifest, plan.source_design_id)
    pool = plan.pool(order_seed)
    design_pool_id, items = load_design(design_path)
    _require(
        (manifest.order_seed, manifest.pool_id, manifest.manifest_sha256)
        == (pool.order_seed, pool.pool_id, pool.manifest_sha256)
        and design_pool_id == pool.pool_id,
        "run artifacts do not match the frozen pool",
    )
    report = validate_scheduler_trace(trace_path)
    _require(
        not report.incomplete_attempt_ids
        and not report.administratively_censored_group_ids,
        "trace is incomplete or administratively censored",
    )
    events = tuple(iter_scheduler_trace(trace_path))
    started = [
        event for event in events if event.event_type is SchedulerEventType.RUN_STARTED
    ]
    ended = [
        event for event in events if event.event_type is SchedulerEventType.RUN_ENDED
    ]
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
    observations, selected_ids, parity = _observations_and_replay(
        events, manifest, items, arm
    )
    by_stratum: dict[str, list[Observation]] = defaultdict(list)
    for observation in observations:
        by_stratum[observation.item.stratum].append(observation)
    _require(
        {key: len(value) for key, value in by_stratum.items()}
        == {"short": 8, "long": 8},
        "observations are not balanced 8+8",
    )
    generated_ratio = statistics.median(
        [
            value
            for item in by_stratum["long"]
            for value in (item.min_generated_tokens, item.max_generated_tokens)
        ]
    ) / statistics.median(
        [
            value
            for item in by_stratum["short"]
            for value in (item.min_generated_tokens, item.max_generated_tokens)
        ]
    )
    ready_ratio = statistics.median(
        item.ready_latency_ns for item in by_stratum["long"]
    ) / statistics.median(item.ready_latency_ns for item in by_stratum["short"])
    reward_means = {
        stratum: statistics.fmean(item.reward_mean for item in values)
        for stratum, values in by_stratum.items()
    }
    length_rates = {
        stratum: sum(item.length_terminations for item in values)
        / (len(values) * EXPECTED_COMPLETIONS)
        for stratum, values in by_stratum.items()
    }
    by_id = {item.logical_group_id: item for item in observations}
    _require(len(selected_ids) == EXPECTED_GROUPS, "selection did not drain 16 groups")
    shares = {
        str(horizon): sum(
            by_id[group_id].item.stratum == "short"
            for group_id in selected_ids[:horizon]
        )
        / horizon
        for horizon in (4, 8, 12)
    }
    ranks = {group_id: rank for rank, group_id in enumerate(selected_ids, start=1)}
    rank_difference = statistics.fmean(
        ranks[item.logical_group_id] for item in by_stratum["long"]
    ) - statistics.fmean(ranks[item.logical_group_id] for item in by_stratum["short"])
    thresholds = plan.thresholds
    validity_checks = {
        "complete_16_groups_32_completions": len(observations) == EXPECTED_GROUPS,
        "zero_administrative_censoring": True,
        "zero_learner_steps_and_physical_version_zero": True,
        "native_reconstruction_exact_group_id_parity": parity,
        "reward_mean_ge_min_each_stratum": all(
            value >= thresholds.reward_mean_min_each_stratum_each_arm
            for value in reward_means.values()
        ),
        "length_termination_rate_le_max_each_stratum": all(
            value
            <= thresholds.backend_length_termination_rate_max_each_stratum_each_arm
            for value in length_rates.values()
        ),
        "generated_token_median_ratio_ge_min": (
            generated_ratio
            >= thresholds.long_short_generated_token_median_ratio_min_each_arm
        ),
        "ready_latency_median_ratio_ge_min": (
            ready_ratio >= thresholds.long_short_ready_latency_median_ratio_min_each_arm
        ),
        "maximum_concurrent_groups_ge_min": (
            _maximum_concurrency(observations)
            >= thresholds.maximum_concurrent_groups_min_each_arm
        ),
    }
    return {
        "arm_id": arm.arm_id,
        "order_seed": order_seed,
        "prompt_groups": len(observations),
        "completions": len(observations) * EXPECTED_COMPLETIONS,
        "selection_steps": EXPECTED_SELECTIONS,
        "train_steps": 0,
        "final_physical_weight_version": 0,
        "short_share_by_horizon": shares,
        "short_minus_long_selection_rank_advantage": rank_difference,
        "reward_mean_by_stratum": reward_means,
        "backend_length_termination_rate_by_stratum": length_rates,
        "generated_output_token_median_long_short_ratio": generated_ratio,
        "ready_latency_median_long_short_ratio": ready_ratio,
        "maximum_concurrent_groups": _maximum_concurrency(observations),
        "validity_checks": validity_checks,
        "all_validity_gates_passed": all(validity_checks.values()),
        "source_artifacts": {
            "trace_sha256": _sha(trace_path),
            "manifest_sha256": _sha(manifest_path),
            "selection_design_sha256": _sha(design_path),
        },
    }


def analyze_block(
    *,
    plan: StructuredSchedulerCrossoverPlan,
    order_seed: int,
    runs: Mapping[str, Path],
) -> dict[str, Any]:
    pool = plan.pool(order_seed)
    expected = {arm.arm_id for arm in plan.arms}
    _require(set(runs) == expected, "block must contain both frozen arms exactly once")
    results = {
        arm_id: analyze_run(
            run_dir=runs[arm_id],
            plan=plan,
            arm=plan.arm(arm_id),
            order_seed=order_seed,
        )
        for arm_id in pool.arm_execution_order
    }
    ready_share = results["ready_first"]["short_share_by_horizon"]["8"]
    ordered_share = results["in_order"]["short_share_by_horizon"]["8"]
    contrast = ready_share - ordered_share
    all_valid = all(result["all_validity_gates_passed"] for result in results.values())
    progression_checks = {
        "both_arms_valid": all_valid,
        "in_order_short_share_equals_0_5": (
            ordered_share == plan.thresholds.in_order_short_share_at_primary_horizon
        ),
        "ready_first_short_share_ge_0_75": (
            ready_share >= plan.thresholds.ready_first_short_share_min
        ),
        "contrast_ge_0_25": contrast >= plan.thresholds.primary_contrast_min,
    }
    passed = all(progression_checks.values())
    decision = (
        "pass_to_replications"
        if order_seed == 46001 and passed
        else "stop_no_training_or_broader_claim"
        if order_seed == 46001
        else "replication_recorded_for_final_aggregate"
        if all_valid
        else "stop_invalid_replication"
    )
    return {
        "schema_version": 1,
        "analysis_status": plan.analysis_status,
        "calibration_only": True,
        "confirmatory_eligible": False,
        "natural_benchmark_claim_authorized": False,
        "counterfactual_replay_authorized": False,
        "training_authorized": False,
        "plan_id": plan.plan_id,
        "replication_id": pool.replication_id,
        "order_seed": order_seed,
        "arm_execution_order": list(pool.arm_execution_order),
        "primary_horizon": plan.primary_horizon,
        "ready_first_minus_in_order_short_share": contrast,
        "progression_checks": progression_checks,
        "all_progression_gates_passed": passed,
        "decision": decision,
        "runs": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--order-seed", type=int, required=True)
    parser.add_argument("--run", action="append", required=True, metavar="ARM=DIR")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = load_structured_scheduler_crossover_plan(args.plan)
    _require(
        _current_clean_commit(args.repo_root) == plan.analysis_code_commit,
        "analysis code commit does not match the frozen plan",
    )
    runs = {}
    for assignment in args.run:
        arm_id, separator, path = assignment.partition("=")
        _require(bool(separator and arm_id and path), f"invalid --run {assignment!r}")
        _require(arm_id not in runs, f"duplicate run arm {arm_id!r}")
        runs[arm_id] = Path(path)
    result = analyze_block(plan=plan, order_seed=args.order_seed, runs=runs)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
