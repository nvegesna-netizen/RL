# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pytest

from nemo_rl.algorithms.async_utils.scheduler_replay import (
    ReleaseSchedule,
    ReplayGroup,
    ReplayPolicy,
    ReplayTick,
    SchedulerReplayError,
    decorrelated_releases,
    groups_from_trace,
    latency_diagnostics,
    mixture_metrics,
    natural_releases,
    replay_schedule,
    scheduler_release_interaction,
    validate_replay_fidelity,
)
from nemo_rl.algorithms.async_utils.scheduler_trace import (
    SchedulerEventType,
    SchedulerTraceEvent,
)


@dataclass(frozen=True)
class ManifestItem:
    ordinal: int
    pool_item_id: str
    source_prompt_id: str
    task_name: str
    repeated_prompt_cluster_id: str
    dispatch_cohort: int
    decorrelation_block: str


@dataclass(frozen=True)
class Manifest:
    pool_id: str
    manifest_sha256: str
    items: tuple[ManifestItem, ...]


def group(
    group_id: str,
    task: str,
    *,
    slot: int,
    dispatch: int = 0,
    ready: int = 0,
    nominal: int = 0,
    target: int | None = None,
    block: str = "block",
) -> ReplayGroup:
    return ReplayGroup(
        logical_group_id=group_id,
        prompt_uid=f"prompt-{group_id}",
        source_prompt_id=f"source-{group_id}",
        task_stratum=task,
        repeated_prompt_cluster_uid=f"cluster-{group_id}",
        dispatch_cohort=f"cohort-{nominal}",
        decorrelation_block=block,
        slot_order=slot,
        dispatch_ns=dispatch,
        ready_ns=ready,
        nominal_start_version=nominal,
        target_step=target,
    )


def tick(
    tick_id: int,
    clock: int,
    version: int,
    minimum: int = 1,
    maximum: int = 1,
) -> ReplayTick:
    return ReplayTick(tick_id, clock, version, minimum, maximum)


def event(
    seq: int,
    clock: int,
    kind: SchedulerEventType,
    *,
    group_id: str,
    prompt_idx: int | None = None,
    pool_ordinal: int | None = 0,
    summaries=None,
) -> SchedulerTraceEvent:
    return SchedulerTraceEvent(
        schema_version=1,
        trace_run_id="run",
        process_epoch="epoch",
        event_seq=seq,
        monotonic_ns=clock,
        event_type=kind,
        logical_group_id=group_id,
        attempt_id=f"attempt-{group_id}",
        admission_id="admission",
        prompt_idx=prompt_idx,
        task_name="a",
        source_prompt_id="1" * 64,
        repeated_prompt_cluster_id="2" * 64,
        source_pool_ordinal=pool_ordinal,
        dispatch_cohort=0,
        start_weight_version=0,
        end_weight_version=0 if kind is SchedulerEventType.GROUP_READY else None,
        scalar_summaries=summaries or {},
    )


def test_in_order_preserves_unsuccessful_ticks_and_batch_order():
    groups = (
        group("slow", "a", slot=0, ready=10, target=0),
        group("fast", "b", slot=1, ready=5, target=0),
    )
    result = replay_schedule(
        groups,
        (tick(0, 5, 0, 2, 2), tick(1, 10, 0, 2, 2)),
        ReplayPolicy("in_order"),
    )

    assert result.decisions[0].eligible_group_ids == ("fast",)
    assert result.decisions[0].selected_group_ids == ()
    assert result.decisions[1].selected_group_ids == ("slow", "fast")


def test_ready_first_changes_finite_prefix_but_never_evicts():
    groups = (
        group("slow", "a", slot=0, ready=10, nominal=0),
        group("fast", "b", slot=1, ready=5, nominal=0),
    )
    result = replay_schedule(
        groups,
        (tick(0, 5, 0), tick(1, 10, 0)),
        ReplayPolicy("ready_first"),
    )

    assert result.selected_group_ids == ("fast", "slow")
    assert result.evicted_group_ids == ()


def test_fidelity_checks_exact_ids_including_unsuccessful_decisions():
    groups = (
        group("slow", "a", slot=0, ready=10),
        group("fast", "b", slot=1, ready=5),
    )
    result = replay_schedule(
        groups,
        (tick(0, 5, 0), tick(1, 10, 0)),
        ReplayPolicy("ready_first"),
    )

    def selection(seq, eligible, selected):
        return SchedulerTraceEvent(
            schema_version=1,
            trace_run_id="run",
            process_epoch="epoch",
            event_seq=seq,
            monotonic_ns=seq,
            event_type=SchedulerEventType.SELECT_DECISION,
            min_prompt_groups=1,
            max_prompt_groups=1,
            ready_prompt_groups=len(eligible),
            eligible_prompt_groups=len(eligible),
            eligible_logical_group_ids=eligible,
            selected_logical_group_ids=selected,
            scalar_summaries={"selected_prompt_groups": len(selected)},
        )

    native = (
        selection(0, ("fast",), ("fast",)),
        selection(1, ("slow",), ("slow",)),
    )
    validate_replay_fidelity(native, result)
    with pytest.raises(SchedulerReplayError, match="decision 0"):
        validate_replay_fidelity((selection(0, ("fast",), ()), native[1]), result)


def test_release_at_tick_is_visible_before_selection():
    groups = (group("boundary", "a", slot=0, dispatch=2, ready=5),)
    result = replay_schedule(groups, (tick(0, 5, 0),), ReplayPolicy("ready_first"))
    assert result.selected_group_ids == ("boundary",)


def test_windowed_evicts_only_ready_stale_groups():
    groups = (
        group("old-ready", "a", slot=0, ready=2, nominal=0),
        group("old-late", "a", slot=1, ready=8, nominal=0),
        group("current", "b", slot=2, ready=2, nominal=2),
    )
    result = replay_schedule(
        groups,
        (tick(0, 5, 2), tick(1, 8, 2)),
        ReplayPolicy("windowed", max_staleness_versions=1),
    )

    assert result.decisions[0].evicted_group_ids == ("old-ready",)
    assert result.decisions[0].selected_group_ids == ("current",)
    assert result.decisions[1].evicted_group_ids == ("old-late",)


def test_windowed_zero_and_freshest_first_ordering():
    groups = (
        group("v1", "a", slot=0, ready=1, nominal=1),
        group("v2-first", "b", slot=1, ready=1, nominal=2),
        group("v2-second", "a", slot=2, ready=1, nominal=2),
    )
    result = replay_schedule(
        groups,
        (tick(0, 1, 2, 1, 2),),
        ReplayPolicy("windowed", max_staleness_versions=2, sample_freshest_first=True),
    )
    assert result.selected_group_ids == ("v2-first", "v2-second")

    zero = replay_schedule(
        groups,
        (tick(0, 1, 2, 1, 3),),
        ReplayPolicy("windowed", max_staleness_versions=0),
    )
    assert zero.evicted_group_ids == ("v1",)
    assert zero.selected_group_ids == ("v2-first", "v2-second")


def test_decorrelation_is_deterministic_and_preserves_each_block_multiset():
    groups = (
        group("a0", "a", slot=0, dispatch=1, ready=2, block="x"),
        group("b0", "b", slot=1, dispatch=2, ready=12, block="x"),
        group("a1", "a", slot=2, dispatch=3, ready=103, block="x"),
        group("a2", "a", slot=3, dispatch=4, ready=6, block="y"),
        group("b2", "b", slot=4, dispatch=5, ready=25, block="y"),
    )
    left = decorrelated_releases(groups, seed=17)
    right = decorrelated_releases(groups, seed=17)
    other = decorrelated_releases(groups, seed=18)

    assert left.ready_ns_by_group == right.ready_ns_by_group
    assert left.ready_ns_by_group != other.ready_ns_by_group
    for block in ("x", "y"):
        members = [item for item in groups if item.decorrelation_block == block]
        natural = Counter(item.natural_latency_ns for item in members)
        shuffled = Counter(
            left.ready_ns_by_group[item.logical_group_id] - item.dispatch_ns
            for item in members
        )
        assert shuffled == natural


def test_decorrelation_fails_closed_without_task_overlap():
    groups = (
        group("a0", "a", slot=0, ready=1),
        group("a1", "a", slot=1, ready=2),
    )
    with pytest.raises(SchedulerReplayError, match="lacks task overlap"):
        decorrelated_releases(groups, seed=1)


def test_mixture_metrics_use_full_pool_target_and_fixed_prefix():
    groups = tuple(group(f"a{i}", "a", slot=i, ready=1) for i in range(2)) + tuple(
        group(f"b{i}", "b", slot=i + 2, ready=1) for i in range(2)
    )
    result = replay_schedule(
        groups,
        tuple(tick(i, i + 1, 0) for i in range(4)),
        ReplayPolicy("ready_first"),
    )
    metrics = mixture_metrics(groups, result, horizon=2)

    assert metrics["target_share"] == {"a": 0.5, "b": 0.5}
    assert metrics["selected_share"] == {"a": 1.0, "b": 0.0}
    assert metrics["signed_share_shift"] == {"a": 0.5, "b": -0.5}
    assert metrics["total_variation"] == 0.5
    assert metrics["worst_representation_ratio"] == 0.0

    limited = replay_schedule(
        groups,
        tuple(tick(i, i + 1, 0) for i in range(2)),
        ReplayPolicy("ready_first"),
    )
    with pytest.raises(SchedulerReplayError, match="reached only"):
        mixture_metrics(groups, limited, horizon=3)


def test_latency_diagnostics_and_interaction_are_exact():
    groups = (
        group("a0", "a", slot=0, ready=1),
        group("a1", "a", slot=1, ready=1),
        group("b0", "b", slot=2, ready=100),
        group("b1", "b", slot=3, ready=100),
    )
    diagnostics = latency_diagnostics(groups, natural_releases(groups))
    assert diagnostics["task_eta_squared"] == pytest.approx(1.0)
    assert diagnostics["max_pairwise_smd"] is None
    assert diagnostics["degenerate_smd"] is True
    assert scheduler_release_interaction(
        natural_policy=0.4,
        natural_in_order=0.1,
        decorrelated_policy=(0.2, 0.3),
        decorrelated_in_order=(0.1, 0.1),
    ) == pytest.approx(0.15)


def test_trace_join_requires_stable_prompt_index_and_exact_coverage():
    manifest = Manifest(
        pool_id="3" * 64,
        manifest_sha256="7" * 64,
        items=(
            ManifestItem(
                ordinal=0,
                pool_item_id="6" * 64,
                source_prompt_id="1" * 64,
                task_name="a",
                repeated_prompt_cluster_id="2" * 64,
                dispatch_cohort=0,
                decorrelation_block="input-length-bin-0",
            ),
        ),
    )
    events = (
        SchedulerTraceEvent(
            schema_version=1,
            trace_run_id="run",
            process_epoch="epoch",
            event_seq=0,
            monotonic_ns=0,
            event_type=SchedulerEventType.RUN_STARTED,
            run_mode="fixed_pool",
            pool_id=manifest.pool_id,
            pool_manifest_sha256=manifest.manifest_sha256,
        ),
        event(1, 1, SchedulerEventType.ATTEMPT_DISPATCHED, group_id="g", prompt_idx=0),
        event(
            2,
            2,
            SchedulerEventType.ROLLOUT_COMPLETED,
            group_id="g",
            summaries={"reward_mean": 0.5},
        ),
        event(3, 3, SchedulerEventType.GROUP_READY, group_id="g"),
    )
    groups = groups_from_trace(events, manifest)
    assert groups[0].prompt_uid == "6" * 64
    assert groups[0].scalar_summaries == {"reward_mean": 0.5}

    missing_identity = (
        events[0],
        event(
            1,
            1,
            SchedulerEventType.ATTEMPT_DISPATCHED,
            group_id="g",
            pool_ordinal=None,
        ),
        *events[2:],
    )
    with pytest.raises(SchedulerReplayError, match="lacks source_pool_ordinal"):
        groups_from_trace(missing_identity, manifest)


def test_release_coverage_and_negative_counterfactual_latency_fail():
    groups = (group("g", "a", slot=0, dispatch=2, ready=3),)
    with pytest.raises(SchedulerReplayError, match="coverage mismatch"):
        replay_schedule(
            groups,
            (tick(0, 3, 0),),
            ReplayPolicy("ready_first"),
            ReleaseSchedule("natural", {}),
        )
    with pytest.raises(SchedulerReplayError, match="release precedes dispatch"):
        latency_diagnostics(groups, ReleaseSchedule("natural", {"g": 1}))
