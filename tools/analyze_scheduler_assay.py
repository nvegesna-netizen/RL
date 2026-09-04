# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fail-closed analysis for the controlled zero-update live scheduler assay."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nemo_rl.algorithms.async_utils.fixed_pool import (
    FixedPoolManifest,
    load_fixed_pool_manifest,
    validate_fixed_pool_manifest_design,
)
from nemo_rl.algorithms.async_utils.scheduler_assay import (
    SchedulerAssayArm,
    SchedulerAssayPlan,
    load_scheduler_assay_plan,
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


class SchedulerAssayAnalysisError(ValueError):
    """An assay artifact or locked progression gate is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SchedulerAssayAnalysisError(message)


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


def _one(events: Sequence[SchedulerTraceEvent], kind: SchedulerEventType):
    matches = [event for event in events if event.event_type is kind]
    _require(len(matches) == 1, f"trace requires exactly one {kind.value}")
    return matches[0]


def _groups_and_ticks(
    events: Sequence[SchedulerTraceEvent], manifest: FixedPoolManifest
) -> tuple[tuple[ReplayGroup, ...], tuple[ReplayTick, ...]]:
    manifest_by_ordinal = {item.ordinal: item for item in manifest.items}
    dispatched = {
        event.logical_group_id: event
        for event in events
        if event.event_type is SchedulerEventType.ATTEMPT_DISPATCHED
    }
    completed = {
        event.logical_group_id: event
        for event in events
        if event.event_type is SchedulerEventType.ROLLOUT_COMPLETED
    }
    ready = {
        event.logical_group_id: event
        for event in events
        if event.event_type is SchedulerEventType.GROUP_READY
    }
    _require(
        len(dispatched) == len(completed) == len(ready) == 48,
        "assay requires 48 unique dispatch/completion/ready groups",
    )
    _require(set(dispatched) == set(completed) == set(ready), "lifecycle mismatch")
    groups = []
    for slot_order, (group_id, dispatch) in enumerate(
        sorted(dispatched.items(), key=lambda item: item[1].event_seq)
    ):
        assert group_id is not None and dispatch.source_pool_ordinal is not None
        item = manifest_by_ordinal.get(dispatch.source_pool_ordinal)
        _require(item is not None, "source pool ordinal absent from manifest")
        assert item is not None
        _require(
            (
                dispatch.source_prompt_id,
                dispatch.task_name,
                dispatch.repeated_prompt_cluster_id,
                dispatch.dispatch_cohort,
            )
            == (
                item.source_prompt_id,
                item.task_name,
                item.repeated_prompt_cluster_id,
                item.dispatch_cohort,
            ),
            f"trace/manifest identity mismatch at ordinal {item.ordinal}",
        )
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
                slot_order=slot_order,
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
    ]
    _require(len(decisions) == 12, "assay requires exactly 12 selection decisions")
    ticks = []
    for index, decision in enumerate(decisions):
        logical_step = decision.scalar_summaries.get("scheduler_assay_step")
        _require(logical_step == index, "scheduler assay steps must be contiguous")
        assert decision.min_prompt_groups is not None
        assert decision.max_prompt_groups is not None
        ticks.append(
            ReplayTick(
                tick_id=index,
                monotonic_ns=decision.monotonic_ns,
                nominal_trainer_version=index,
                min_prompt_groups=decision.min_prompt_groups,
                max_prompt_groups=decision.max_prompt_groups,
            )
        )
    return tuple(groups), tuple(ticks)


def analyze_run(
    *,
    run_dir: Path,
    plan: SchedulerAssayPlan,
    arm: SchedulerAssayArm,
    order_seed: int,
) -> dict[str, Any]:
    trace_path = run_dir / "scheduler_trace.v1.jsonl"
    manifest_path = run_dir / "fixed_pool_manifest.v1.json"
    _require(trace_path.is_file(), f"missing trace: {trace_path}")
    _require(manifest_path.is_file(), f"missing manifest: {manifest_path}")
    manifest = load_fixed_pool_manifest(manifest_path)
    validate_fixed_pool_manifest_design(manifest, plan.source_design_id)
    pool = plan.pool(order_seed)
    _require(
        (manifest.order_seed, manifest.pool_id, manifest.manifest_sha256)
        == (pool.order_seed, pool.pool_id, pool.manifest_sha256),
        "run manifest does not match the frozen assay plan",
    )
    report = validate_scheduler_trace(trace_path)
    _require(
        not report.incomplete_attempt_ids
        and not report.administratively_censored_group_ids,
        "assay trace is incomplete or administratively censored",
    )
    events = tuple(iter_scheduler_trace(trace_path))
    started = _one(events, SchedulerEventType.RUN_STARTED)
    ended = _one(events, SchedulerEventType.RUN_ENDED)
    _require(started.run_mode == "scheduler_assay", "wrong run mode")
    _require(started.pool_id == pool.pool_id, "wrong run pool ID")
    _require(
        started.scalar_summaries.get("scheduler_assay_plan_id") == plan.plan_id,
        "wrong assay plan ID",
    )
    _require(
        started.scalar_summaries.get("scheduler_assay_arm_id") == arm.arm_id,
        "wrong assay arm ID",
    )
    _require(ended.terminal_reason == "scheduler_assay_complete", "run failed")
    _require(
        ended.scalar_summaries.get("completed_train_steps") == 0
        and ended.scalar_summaries.get("final_physical_weight_version") == 0,
        "assay performed a learner update or changed physical weight version",
    )
    forbidden = {
        SchedulerEventType.ATTEMPT_FAILED,
        SchedulerEventType.ATTEMPT_REMOVED,
        SchedulerEventType.GROUP_EVICTED,
        SchedulerEventType.GROUP_ARCHIVED,
    }
    _require(
        not any(event.event_type in forbidden for event in events),
        "assay contains a failure, removal, eviction, or archive",
    )
    _require(
        all(
            event.trainer_version in {None, 0}
            and event.start_weight_version in {None, 0}
            and event.end_weight_version in {None, 0}
            for event in events
        ),
        "physical weight version changed from zero",
    )

    groups, ticks = _groups_and_ticks(events, manifest)
    completed = {
        event.logical_group_id: event
        for event in events
        if event.event_type is SchedulerEventType.ROLLOUT_COMPLETED
    }
    ready = {
        event.logical_group_id: event
        for event in events
        if event.event_type is SchedulerEventType.GROUP_READY
    }
    delayed_holds = []
    undelayed_holds = []
    for group in groups:
        hold_seconds = (
            ready[group.logical_group_id].monotonic_ns
            - completed[group.logical_group_id].monotonic_ns
        ) / 1_000_000_000
        target = (
            delayed_holds if group.task_stratum == arm.delayed_task else undelayed_holds
        )
        target.append(hold_seconds)
    _require(len(delayed_holds) == len(undelayed_holds) == 24, "task balance failed")
    _require(
        min(delayed_holds) >= plan.thresholds.minimum_delayed_hold_seconds,
        "delayed hold missed its locked minimum",
    )
    _require(
        max(undelayed_holds) <= plan.thresholds.maximum_undelayed_commit_seconds,
        "undelayed commit exceeded its locked maximum",
    )

    policy = ReplayPolicy(name=arm.sampler, max_staleness_versions=3)
    replay = replay_schedule(groups, ticks, policy, natural_releases(groups))
    native_decisions = [
        event
        for event in events
        if event.event_type is SchedulerEventType.SELECT_DECISION
    ]
    for native, replayed in zip(native_decisions, replay.decisions):
        _require(
            native.selected_logical_group_ids == replayed.selected_group_ids
            and native.eligible_logical_group_ids == replayed.eligible_group_ids,
            f"native/replay group-ID mismatch at decision {replayed.tick_id}",
        )
    _require(
        not replay.evicted_group_ids and not replay.live_group_ids,
        "replay did not drain",
    )
    selected = replay.selected_group_ids
    _require(
        len(selected) == 48 and len(set(selected)) == 48,
        "selection did not drain 48 groups",
    )
    by_id = {group.logical_group_id: group for group in groups}
    horizon = selected[: plan.primary_horizon]
    undelayed_share = sum(
        by_id[group_id].task_stratum != arm.delayed_task for group_id in horizon
    ) / len(horizon)
    threshold = (
        plan.thresholds.ready_first_undelayed_share_min
        if arm.sampler == "ready_first"
        else plan.thresholds.in_order_undelayed_share
    )
    _require(
        undelayed_share == threshold
        if arm.sampler == "in_order"
        else undelayed_share >= threshold,
        "arm progression threshold failed",
    )
    return {
        "arm_id": arm.arm_id,
        "order_seed": order_seed,
        "prompt_groups": len(groups),
        "selection_steps": len(ticks),
        "train_steps": 0,
        "final_physical_weight_version": 0,
        "minimum_delayed_hold_seconds": min(delayed_holds),
        "maximum_undelayed_hold_seconds": max(undelayed_holds),
        "native_replay_group_id_parity": True,
        "primary_horizon": plan.primary_horizon,
        "primary_undelayed_share": undelayed_share,
    }


def analyze_block(
    *, plan: SchedulerAssayPlan, order_seed: int, runs: Mapping[str, Path]
) -> dict[str, Any]:
    expected = {arm.arm_id for arm in plan.arms}
    _require(set(runs) == expected, "block must contain each frozen arm exactly once")
    results = {
        arm.arm_id: analyze_run(
            run_dir=runs[arm.arm_id], plan=plan, arm=arm, order_seed=order_seed
        )
        for arm in plan.arms
    }
    contrasts = {}
    for delayed in ("aime", "gsm8k"):
        ready = results[f"ready_first_{delayed}_delayed"]["primary_undelayed_share"]
        ordered = results[f"in_order_{delayed}_delayed"]["primary_undelayed_share"]
        difference = ready - ordered
        _require(
            difference >= plan.thresholds.ready_first_minus_in_order_min,
            f"{delayed} crossover contrast missed its progression threshold",
        )
        contrasts[delayed] = difference
    return {
        "schema_version": 1,
        "analysis_status": plan.analysis_status,
        "calibration_only": plan.calibration_only,
        "confirmatory_eligible": plan.confirmatory_eligible,
        "natural_latency_claim_authorized": plan.natural_latency_claim_authorized,
        "training_authorized": plan.training_authorized,
        "plan_id": plan.plan_id,
        "order_seed": order_seed,
        "all_progression_gates_passed": True,
        "runs": results,
        "ready_first_minus_in_order": contrasts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--order-seed", type=int, required=True)
    parser.add_argument("--run", action="append", required=True, metavar="ARM=DIR")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = load_scheduler_assay_plan(args.plan)
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
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
