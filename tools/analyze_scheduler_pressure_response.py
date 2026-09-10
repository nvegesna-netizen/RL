# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fail-closed analysis for one scheduler pressure-response replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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
    PRESSURE_RESPONSE_CONFIRMED_CANDIDATE_SHA256,
    SchedulerPressureResponseArm,
    SchedulerPressureResponsePlan,
    load_scheduler_pressure_response_plan,
)
from tools.analyze_scheduler_selection_opportunity import (
    TraceAnalysis,
    _arm_summary,
    _current_clean_commit,
    analyze_comparison,
    analyze_trace_events,
)


EXPECTED_GROUPS = 32
EXPECTED_COMPLETIONS = 2
EXPECTED_SELECTIONS = 8
GROUPS_PER_SELECTION = 4


class SchedulerPressureResponseAnalysisError(ValueError):
    """An artifact or locked protocol invariant is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SchedulerPressureResponseAnalysisError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _number(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchedulerPressureResponseAnalysisError(f"missing numeric summary {key}")
    number = float(value)
    _require(math.isfinite(number), f"non-finite summary {key}")
    return number


def _validated_manifest(
    *, copied_path: Path, materialization_path: Path, design_id: str
) -> FixedPoolManifest:
    _require(
        copied_path.read_bytes() == materialization_path.read_bytes(),
        "run manifest is not byte-identical to materialization",
    )
    manifest = load_fixed_pool_manifest(materialization_path)
    validate_fixed_pool_materialization(manifest)
    validate_fixed_pool_manifest_design(manifest, design_id)
    return manifest


def _load_design(path: Path) -> tuple[str, dict[str, str]]:
    value = json.loads(path.read_text())
    _require(
        isinstance(value, dict)
        and value.get("schema_version") == 1
        and value.get("analysis_status")
        == "controlled_zero_update_scheduler_pressure_response_pool"
        and value.get("calibration_only") is True
        and value.get("confirmatory_eligible") is False
        and value.get("protocol_id") == PRESSURE_RESPONSE_CONFIRMED_CANDIDATE_SHA256,
        "selection design labels mismatch",
    )
    items = value.get("items")
    _require(
        isinstance(items, list) and len(items) == EXPECTED_GROUPS,
        "design size mismatch",
    )
    strata: dict[str, str] = {}
    by_cohort: dict[int, list[str]] = defaultdict(list)
    pairs: dict[str, list[str]] = defaultdict(list)
    for ordinal, item in enumerate(items):
        _require(
            isinstance(item, dict)
            and item.get("source_pool_ordinal") == ordinal
            and item.get("stratum") in {"short", "long"}
            and item.get("task_name") == f"structured_{item.get('stratum')}",
            "selection design item mismatch",
        )
        source_id = item.get("source_prompt_id")
        pair_id = item.get("pair_id")
        _require(
            isinstance(source_id, str) and isinstance(pair_id, str),
            "design identity missing",
        )
        strata[source_id] = str(item["stratum"])
        by_cohort[ordinal // GROUPS_PER_SELECTION].append(str(item["stratum"]))
        pairs[pair_id].append(str(item["stratum"]))
    _require(
        len(strata) == EXPECTED_GROUPS
        and len(by_cohort) == 8
        and all(
            sorted(values) == ["long", "long", "short", "short"]
            for values in by_cohort.values()
        )
        and len(pairs) == 16
        and all(sorted(values) == ["long", "short"] for values in pairs.values()),
        "selection design balance mismatch",
    )
    return str(value.get("fixed_pool_id")), strata


def _replay_parity(
    events: Sequence[SchedulerTraceEvent],
    manifest: FixedPoolManifest,
    arm: SchedulerPressureResponseArm,
) -> bool:
    dispatched = {
        event.logical_group_id: event
        for event in events
        if event.event_type is SchedulerEventType.ATTEMPT_DISPATCHED
    }
    ready = {
        event.logical_group_id: event
        for event in events
        if event.event_type is SchedulerEventType.GROUP_READY
    }
    manifest_by_ordinal = {item.ordinal: item for item in manifest.items}
    groups = []
    for slot, (group_id, dispatch) in enumerate(
        sorted(dispatched.items(), key=lambda item: item[1].event_seq)
    ):
        if group_id is None or dispatch.source_pool_ordinal is None:
            raise SchedulerPressureResponseAnalysisError("dispatch identity missing")
        item = manifest_by_ordinal[dispatch.source_pool_ordinal]
        release = ready[group_id]
        groups.append(
            ReplayGroup(
                logical_group_id=group_id,
                prompt_uid=item.pool_item_id,
                source_prompt_id=item.source_prompt_id,
                task_stratum=item.task_name,
                repeated_prompt_cluster_uid=item.repeated_prompt_cluster_id,
                dispatch_cohort=str(item.dispatch_cohort),
                decorrelation_block=item.decorrelation_block,
                slot_order=slot,
                dispatch_ns=dispatch.monotonic_ns,
                ready_ns=release.monotonic_ns,
                nominal_start_version=0,
                target_step=dispatch.target_step,
                physical_start_version=dispatch.start_weight_version,
                physical_end_version=release.end_weight_version,
                scalar_summaries={},
            )
        )
    decisions = [
        event
        for event in events
        if event.event_type is SchedulerEventType.SELECT_DECISION
        and event.selected_logical_group_ids
    ]
    _require(
        len(groups) == EXPECTED_GROUPS and len(decisions) == EXPECTED_SELECTIONS,
        "lifecycle size mismatch",
    )
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
        tuple(groups),
        ticks,
        ReplayPolicy(
            name=arm.sampler, max_staleness_versions=arm.sampler_lookahead_versions
        ),
        natural_releases(tuple(groups)),
    )
    return (
        not replay.evicted_group_ids
        and not replay.live_group_ids
        and len(replay.decisions) == len(decisions)
        and all(
            native.selected_logical_group_ids == reconstructed.selected_group_ids
            and native.eligible_logical_group_ids == reconstructed.eligible_group_ids
            for native, reconstructed in zip(decisions, replay.decisions, strict=True)
        )
    )


def _maximum_concurrency(events: Sequence[SchedulerTraceEvent]) -> int:
    changes = []
    for event in events:
        if event.event_type is SchedulerEventType.ATTEMPT_DISPATCHED:
            changes.append((event.monotonic_ns, 1))
        elif event.event_type is SchedulerEventType.GROUP_READY:
            changes.append((event.monotonic_ns, -1))
    active = maximum = 0
    for _, change in sorted(changes, key=lambda item: (item[0], item[1])):
        active += change
        maximum = max(maximum, active)
    return maximum


def analyze_run(
    *,
    run_dir: Path,
    plan: SchedulerPressureResponsePlan,
    arm: SchedulerPressureResponseArm,
    order_seed: int,
    materialization_manifest_path: Path,
) -> tuple[dict[str, Any], TraceAnalysis]:
    trace_path = run_dir / "scheduler_trace.v1.jsonl"
    copied_manifest = run_dir / "fixed_pool_manifest.v1.json"
    design_path = run_dir / "selection_design.v1.json"
    _require(
        trace_path.is_file() and copied_manifest.is_file() and design_path.is_file(),
        "run artifacts missing",
    )
    manifest = _validated_manifest(
        copied_path=copied_manifest,
        materialization_path=materialization_manifest_path,
        design_id=plan.source_design_id,
    )
    pool = plan.pool(order_seed)
    design_pool_id, strata = _load_design(design_path)
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
        "trace is incomplete or censored",
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
        and starts[0].scalar_summaries.get("scheduler_assay_arm_id") == arm.arm_id,
        "run identity mismatch",
    )
    _require(
        ends[0].terminal_reason == "scheduler_assay_complete"
        and ends[0].scalar_summaries.get("completed_train_steps") == 0
        and ends[0].scalar_summaries.get("final_physical_weight_version") == 0,
        "run failed or changed learner state",
    )
    trace_analysis = analyze_trace_events(events, trace_sha256=_sha(trace_path))
    _require(
        len(trace_analysis.groups) == EXPECTED_GROUPS,
        "trace does not contain 32 groups",
    )
    _require(
        len(trace_analysis.decisions) == EXPECTED_SELECTIONS,
        "trace does not contain eight selections",
    )
    _require(
        all(item.selected_count == 4 for item in trace_analysis.decisions),
        "selection size mismatch",
    )
    completed = [
        event
        for event in events
        if event.event_type is SchedulerEventType.ROLLOUT_COMPLETED
    ]
    ready_by_id = {
        event.logical_group_id: event
        for event in events
        if event.event_type is SchedulerEventType.GROUP_READY
    }
    by_stratum: dict[str, list[SchedulerTraceEvent]] = defaultdict(list)
    holds: dict[str, list[float]] = defaultdict(list)
    for event in completed:
        if (
            event.source_prompt_id is None
            or event.logical_group_id is None
            or event.source_prompt_id not in strata
            or event.logical_group_id not in ready_by_id
        ):
            raise SchedulerPressureResponseAnalysisError("completion identity mismatch")
        stratum = strata[event.source_prompt_id]
        by_stratum[stratum].append(event)
        holds[stratum].append(
            (ready_by_id[event.logical_group_id].monotonic_ns - event.monotonic_ns)
            / 1e9
        )
        _require(
            _number(event.scalar_summaries, "completion_count") == EXPECTED_COMPLETIONS,
            "completion count mismatch",
        )
    _require(
        {key: len(value) for key, value in by_stratum.items()}
        == {"short": 16, "long": 16},
        "completion balance mismatch",
    )
    token_values = {
        key: [
            _number(event.scalar_summaries, field)
            for event in values
            for field in ("gen_tokens_per_sample/min", "gen_tokens_per_sample/max")
        ]
        for key, values in by_stratum.items()
    }
    ready_latencies = {
        key: [
            ready_by_id[event.logical_group_id].monotonic_ns
            - next(
                dispatch.monotonic_ns
                for dispatch in events
                if dispatch.event_type is SchedulerEventType.ATTEMPT_DISPATCHED
                and dispatch.logical_group_id == event.logical_group_id
            )
            for event in values
        ]
        for key, values in by_stratum.items()
    }
    reward_means = {
        key: statistics.fmean(
            _number(event.scalar_summaries, "reward_mean") for event in values
        )
        for key, values in by_stratum.items()
    }
    length_rates = {
        key: statistics.fmean(
            _number(event.scalar_summaries, "backend_length_termination_rate")
            for event in values
        )
        for key, values in by_stratum.items()
    }
    delayed = "short" if arm.delayed_task == "structured_short" else "long"
    release_metadata = {
        key: [
            _number(event.scalar_summaries, "scheduler_assay_release_delay_seconds")
            for event in values
        ]
        for key, values in by_stratum.items()
    }
    if arm.latency_condition == "natural":
        hold_valid = all(
            value == 0.0 for values in release_metadata.values() for value in values
        )
    else:
        undelayed = "long" if delayed == "short" else "short"
        hold_valid = (
            min(holds[delayed]) >= 30.0
            and all(value == 30.0 for value in release_metadata[delayed])
            and all(value == 0.0 for value in release_metadata[undelayed])
        )
    thresholds = plan.thresholds
    natural = arm.latency_condition == "natural"
    generated_ratio = statistics.median(token_values["long"]) / statistics.median(
        token_values["short"]
    )
    latency_ratio = statistics.median(ready_latencies["long"]) / statistics.median(
        ready_latencies["short"]
    )
    checks = {
        "complete_32_groups_64_completions": len(completed) == EXPECTED_GROUPS,
        "eight_nonempty_four_group_selections": True,
        "zero_administrative_censoring_failures_removals_evictions_replacements_or_promotions": True,
        "zero_learner_steps_and_physical_weight_version_zero": True,
        "native_reconstruction_exact_group_id_parity": _replay_parity(
            events, manifest, arm
        ),
        "maximum_concurrent_groups_equals_4": _maximum_concurrency(events) == 4,
        "backend_length_termination_rate_le_max_each_stratum": all(
            value
            <= thresholds.backend_length_termination_rate_max_each_stratum_each_arm
            for value in length_rates.values()
        ),
        "reward_mean_ge_min_each_stratum": all(
            value >= thresholds.reward_mean_min_each_stratum_each_arm
            for value in reward_means.values()
        ),
        "release_delay_contract": hold_valid,
        "natural_generated_token_ratio_ge_min": (
            not natural
            or generated_ratio
            >= thresholds.natural_long_short_generated_token_median_ratio_min_each_arm
        ),
        "natural_ready_latency_ratio_ge_min": (
            not natural
            or latency_ratio
            >= thresholds.natural_long_short_ready_latency_median_ratio_min_each_arm
        ),
    }
    order = trace_analysis.selected_order
    shares = {
        str(horizon): sum(strata[source_id] == "short" for source_id in order[:horizon])
        / horizon
        for horizon in (8, 16, 24)
    }
    return (
        {
            "arm_id": arm.arm_id,
            "pressure_level": arm.pressure_level,
            "latency_condition": arm.latency_condition,
            "delay_mapping": arm.delay_mapping,
            "prompt_groups": len(completed),
            "completions": len(completed) * EXPECTED_COMPLETIONS,
            "selection_steps": len(trace_analysis.decisions),
            "short_share_by_horizon": shares,
            "undelayed_share_at_horizon_16": (
                shares["16"]
                if arm.delayed_task == "structured_long"
                else 1 - shares["16"]
            )
            if not natural
            else None,
            "generated_output_token_median_long_short_ratio": generated_ratio,
            "ready_latency_median_long_short_ratio": latency_ratio,
            "reward_mean_by_stratum": reward_means,
            "backend_length_termination_rate_by_stratum": length_rates,
            "maximum_concurrent_groups": _maximum_concurrency(events),
            "selection_pressure": _arm_summary(trace_analysis),
            "validity_checks": checks,
            "all_validity_gates_passed": all(checks.values()),
            "source_artifacts": {
                "trace_sha256": _sha(trace_path),
                "manifest_sha256": _sha(copied_manifest),
                "selection_design_sha256": _sha(design_path),
            },
        },
        trace_analysis,
    )


def analyze_block(
    *,
    plan: SchedulerPressureResponsePlan,
    order_seed: int,
    runs: Mapping[str, Path],
    materialization_manifest_path: Path,
) -> dict[str, Any]:
    pool = plan.pool(order_seed)
    _require(
        set(runs) == {arm.arm_id for arm in plan.arms},
        "block must contain all ten arms",
    )
    results = {}
    traces = {}
    first_run = next(iter(runs.values()))
    _, fixed_strata = _load_design(first_run / "selection_design.v1.json")
    for arm_id in pool.arm_execution_order:
        results[arm_id], traces[arm_id] = analyze_run(
            run_dir=runs[arm_id],
            plan=plan,
            arm=plan.arm(arm_id),
            order_seed=order_seed,
            materialization_manifest_path=materialization_manifest_path,
        )
    comparisons = {}
    for arm in plan.arms:
        if arm.sampler != "ready_first":
            continue
        peer = next(
            candidate
            for candidate in plan.arms
            if candidate.sampler == "in_order"
            and (
                candidate.pressure_level,
                candidate.latency_condition,
                candidate.delay_mapping,
            )
            == (arm.pressure_level, arm.latency_condition, arm.delay_mapping)
        )
        comparison_id = arm.arm_id.removesuffix("_ready_first")
        comparison = analyze_comparison(
            comparison_id=comparison_id,
            ready_first=traces[arm.arm_id],
            in_order=traces[peer.arm_id],
            reference_summary_key="mean_gen_tokens_per_sample",
        )
        ready_ranks = {
            source_id: rank
            for rank, source_id in enumerate(traces[arm.arm_id].selected_order, 1)
        }
        ordered_ranks = {
            source_id: rank
            for rank, source_id in enumerate(traces[peer.arm_id].selected_order, 1)
        }
        short_promotions = [
            ordered_ranks[source_id] - ready_ranks[source_id]
            for source_id, stratum in fixed_strata.items()
            if stratum == "short"
        ]
        comparison["fixed_stratum_reference"] = {
            "lower_load_stratum": "structured_short",
            "definition_uses_realized_arm_outcomes": False,
            "lower_reference_net_rank_promotion": sum(short_promotions),
            "lower_reference_mean_normalized_rank_promotion": (
                statistics.fmean(short_promotions) / (EXPECTED_GROUPS - 1)
            ),
        }
        comparison["secondary_continuous_reference"] = comparison.pop("reference")
        comparison["cross_arm"].pop("lower_reference_net_rank_promotion")
        comparison["cross_arm"].pop("lower_reference_mean_normalized_rank_promotion")
        comparisons[comparison_id] = comparison
    l0 = comparisons["natural_l0"]
    positive_checks = {}
    for mapping in ("short_delayed", "long_delayed"):
        comparison = comparisons[f"controlled_positive_control_l3_{mapping}"]
        ready_id = f"controlled_positive_control_l3_{mapping}_ready_first"
        ordered_id = f"controlled_positive_control_l3_{mapping}_in_order"
        ready_share = results[ready_id]["undelayed_share_at_horizon_16"]
        ordered_share = results[ordered_id]["undelayed_share_at_horizon_16"]
        positive_checks[mapping] = {
            "ready_first_undelayed_share_ge_min": ready_share
            >= plan.thresholds.positive_control_undelayed_share_ready_first_min,
            "in_order_undelayed_share_equals_required": ordered_share
            == plan.thresholds.positive_control_undelayed_share_in_order_required,
            "contrast_ge_min": ready_share - ordered_share
            >= plan.thresholds.positive_control_contrast_min_each_mapping,
        }
    checks = {
        "all_ten_arms_valid": all(
            result["all_validity_gates_passed"] for result in results.values()
        ),
        "l0_ready_but_ineligible_zero_each_decision": all(
            item["ready_but_ineligible"] == 0
            for sampler in ("ready_first", "in_order")
            for item in l0["arms"][sampler]["per_decision_counts"]
        ),
        "l0_selected_step_sets_match": l0["cross_arm"]["same_selected_step_sets"]
        == EXPECTED_SELECTIONS,
        "both_positive_controls_pass": all(
            all(values.values()) for values in positive_checks.values()
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
        "order_seed": order_seed,
        "arms": results,
        "comparisons": comparisons,
        "replication_checks": checks,
        "positive_control_checks": positive_checks,
        "decision": "pass_to_next_replication"
        if all(checks.values())
        else "stop_no_replay_or_training",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--order-seed", type=int, required=True)
    parser.add_argument("--materialization-manifest", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, metavar="ARM_ID=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=False)
    args = parser.parse_args()
    plan = load_scheduler_pressure_response_plan(args.plan)
    if args.repo_root is not None:
        _require(
            _current_clean_commit(args.repo_root) == plan.analysis_code_commit,
            "analysis repository does not match the frozen commit",
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
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
