# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Deterministic, pure-CPU replay of SingleController scheduling decisions.

This module deliberately has no dependency on the live replay buffer.  It
replays a fixed, completed prompt-group pool against an exogenous learner-tick
plan, so its results are conditional queue effects rather than estimates of a
native scheduler's throughput or of a trained-policy counterfactual.
"""

from __future__ import annotations

import math
import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Protocol

from nemo_rl.algorithms.async_utils.scheduler_trace import (
    SchedulerEventType,
    SchedulerTraceEvent,
    iter_scheduler_trace,
    validate_scheduler_trace,
)


ReplayPolicyName = Literal["in_order", "ready_first", "windowed"]


class SchedulerReplayError(ValueError):
    """Replay input is incomplete, inconsistent, or outside the assay contract."""


class FixedPoolItemLike(Protocol):
    ordinal: int
    pool_item_id: str
    source_prompt_id: str
    task_name: str
    repeated_prompt_cluster_id: str
    dispatch_cohort: int
    decorrelation_block: str


class FixedPoolManifestLike(Protocol):
    pool_id: str
    manifest_sha256: str
    items: Sequence[FixedPoolItemLike]


@dataclass(frozen=True, slots=True)
class ReplayGroup:
    logical_group_id: str
    prompt_uid: str
    source_prompt_id: str
    task_stratum: str
    repeated_prompt_cluster_uid: str
    dispatch_cohort: str
    decorrelation_block: str
    slot_order: int
    dispatch_ns: int
    ready_ns: int
    nominal_start_version: int
    target_step: Optional[int]
    archived_ns: Optional[int] = None
    physical_start_version: Optional[int] = None
    physical_end_version: Optional[int] = None
    scalar_summaries: Mapping[str, float | int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.logical_group_id or not self.prompt_uid or not self.task_stratum:
            raise SchedulerReplayError(
                "group identity and task stratum must be nonempty"
            )
        if (
            self.slot_order < 0
            or self.dispatch_ns < 0
            or self.ready_ns < self.dispatch_ns
            or (self.archived_ns is not None and self.archived_ns < self.ready_ns)
        ):
            raise SchedulerReplayError("invalid group order or controller timestamps")
        if self.nominal_start_version < 0:
            raise SchedulerReplayError("nominal_start_version must be non-negative")

    @property
    def natural_latency_ns(self) -> int:
        return self.ready_ns - self.dispatch_ns


@dataclass(frozen=True, slots=True)
class ReplayTick:
    tick_id: int
    monotonic_ns: int
    nominal_trainer_version: int
    min_prompt_groups: int
    max_prompt_groups: int

    def __post_init__(self) -> None:
        if (
            self.tick_id < 0
            or self.monotonic_ns < 0
            or self.nominal_trainer_version < 0
        ):
            raise SchedulerReplayError(
                "tick identifiers, clocks, and versions must be non-negative"
            )
        if (
            self.min_prompt_groups < 1
            or self.max_prompt_groups < self.min_prompt_groups
        ):
            raise SchedulerReplayError("invalid tick selection bounds")


@dataclass(frozen=True, slots=True)
class ReplayPolicy:
    name: ReplayPolicyName
    max_staleness_versions: int = 0
    sample_freshest_first: bool = False

    def __post_init__(self) -> None:
        if self.name not in {"in_order", "ready_first", "windowed"}:
            raise SchedulerReplayError(f"unsupported replay policy {self.name!r}")
        if self.max_staleness_versions < 0:
            raise SchedulerReplayError("max_staleness_versions must be non-negative")
        if self.sample_freshest_first and self.name != "windowed":
            raise SchedulerReplayError("freshest-first applies only to windowed")


@dataclass(frozen=True, slots=True)
class ReleaseSchedule:
    kind: Literal["natural", "decorrelated"]
    ready_ns_by_group: Mapping[str, int]
    seed: Optional[int] = None
    assignments: tuple["ReleaseAssignment", ...] = ()
    assignment_fixed_points: int = 0
    unchanged_latency_values: int = 0
    blocks: int = 0


@dataclass(frozen=True, slots=True)
class ReleaseAssignment:
    """One source-latency assignment in a blockwise release intervention."""

    block_id: str
    destination_group_id: str
    destination_prompt_uid: str
    source_group_id: str
    source_prompt_uid: str
    latency_ns: int
    assignment_fixed_point: bool
    unchanged_latency_value: bool


@dataclass(frozen=True, slots=True)
class ReplayDecision:
    tick_id: int
    monotonic_ns: int
    trainer_version: int
    ready_group_ids: tuple[str, ...]
    eligible_group_ids: tuple[str, ...]
    selected_group_ids: tuple[str, ...]
    evicted_group_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReplayResult:
    policy: ReplayPolicy
    release_kind: str
    release_seed: Optional[int]
    decisions: tuple[ReplayDecision, ...]
    selected_group_ids: tuple[str, ...]
    evicted_group_ids: tuple[str, ...]
    live_group_ids: tuple[str, ...]


def groups_from_trace(
    events: Iterable[SchedulerTraceEvent],
    manifest: FixedPoolManifestLike,
) -> tuple[ReplayGroup, ...]:
    """Join a complete source trace to its strict fixed-pool manifest.

    The fixed-policy mechanism assay uses ``dispatch_cohort`` as the explicit
    virtual start/target version.  Physical weight versions remain separate
    provenance fields and are never silently substituted.
    """
    manifest_by_ordinal = {item.ordinal: item for item in manifest.items}
    dispatched: dict[str, SchedulerTraceEvent] = {}
    completed: dict[str, SchedulerTraceEvent] = {}
    ready: dict[str, SchedulerTraceEvent] = {}
    archived: dict[str, SchedulerTraceEvent] = {}
    saw_matching_run = False
    for event in events:
        group_id = event.logical_group_id
        if event.event_type is SchedulerEventType.RUN_STARTED:
            if (
                event.run_mode != "fixed_pool"
                or event.pool_id != manifest.pool_id
                or event.pool_manifest_sha256 != manifest.manifest_sha256
            ):
                raise SchedulerReplayError("trace/manifest run identity mismatch")
            saw_matching_run = True
        elif event.event_type is SchedulerEventType.ATTEMPT_DISPATCHED:
            if group_id is None or group_id in dispatched:
                raise SchedulerReplayError("missing or duplicate dispatch group ID")
            if event.source_pool_ordinal is None:
                raise SchedulerReplayError("source dispatch lacks source_pool_ordinal")
            dispatched[group_id] = event
        elif event.event_type is SchedulerEventType.ROLLOUT_COMPLETED:
            if group_id is None or group_id in completed:
                raise SchedulerReplayError("missing or duplicate completed group ID")
            completed[group_id] = event
        elif event.event_type is SchedulerEventType.GROUP_READY:
            if group_id is None or group_id in ready:
                raise SchedulerReplayError("missing or duplicate ready group ID")
            ready[group_id] = event
        elif event.event_type is SchedulerEventType.GROUP_ARCHIVED:
            if group_id is None or group_id in archived:
                raise SchedulerReplayError("missing or duplicate archived group ID")
            archived[group_id] = event
        elif event.event_type in {
            SchedulerEventType.ATTEMPT_FAILED,
            SchedulerEventType.ATTEMPT_REMOVED,
            SchedulerEventType.GROUP_EVICTED,
        }:
            raise SchedulerReplayError(
                f"source pool is not complete/non-censoring: {event.event_type.value}"
            )

    if not saw_matching_run:
        raise SchedulerReplayError(
            "source trace lacks matching fixed-pool run identity"
        )
    if (
        set(dispatched) != set(ready)
        or set(dispatched) != set(completed)
        or set(dispatched) != set(archived)
    ):
        raise SchedulerReplayError(
            "every dispatched source group must complete, become ready, and archive"
        )
    observed_ordinals = [event.source_pool_ordinal for event in dispatched.values()]
    if len(observed_ordinals) != len(set(observed_ordinals)) or set(
        observed_ordinals
    ) != set(manifest_by_ordinal):
        raise SchedulerReplayError("trace/manifest pool-ordinal coverage mismatch")

    groups: list[ReplayGroup] = []
    for slot_order, (group_id, dispatch) in enumerate(
        sorted(dispatched.items(), key=lambda item: item[1].event_seq)
    ):
        assert dispatch.source_pool_ordinal is not None
        entry = manifest_by_ordinal[dispatch.source_pool_ordinal]
        completion = completed[group_id]
        release = ready[group_id]
        archive = archived[group_id]
        _validate_trace_manifest_identity(dispatch, entry)
        groups.append(
            ReplayGroup(
                logical_group_id=group_id,
                prompt_uid=entry.pool_item_id,
                source_prompt_id=entry.source_prompt_id,
                task_stratum=entry.task_name,
                repeated_prompt_cluster_uid=entry.repeated_prompt_cluster_id,
                dispatch_cohort=str(entry.dispatch_cohort),
                decorrelation_block=(
                    f"cohort:{entry.dispatch_cohort}|{entry.decorrelation_block}"
                ),
                slot_order=slot_order,
                dispatch_ns=dispatch.monotonic_ns,
                ready_ns=release.monotonic_ns,
                archived_ns=archive.monotonic_ns,
                nominal_start_version=entry.dispatch_cohort,
                target_step=entry.dispatch_cohort,
                physical_start_version=dispatch.start_weight_version,
                physical_end_version=release.end_weight_version,
                scalar_summaries={
                    key: value
                    for key, value in completion.scalar_summaries.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                },
            )
        )
    return tuple(groups)


def _validate_trace_manifest_identity(
    event: SchedulerTraceEvent, item: FixedPoolItemLike
) -> None:
    observed = (
        event.source_prompt_id,
        event.task_name,
        event.repeated_prompt_cluster_id,
        event.dispatch_cohort,
    )
    expected = (
        item.source_prompt_id,
        item.task_name,
        item.repeated_prompt_cluster_id,
        item.dispatch_cohort,
    )
    if observed != expected:
        raise SchedulerReplayError(
            f"trace/manifest identity mismatch at pool ordinal {item.ordinal}"
        )


def load_groups_from_trace(
    trace_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_completions_per_group: int,
) -> tuple[ReplayGroup, ...]:
    """Validate and load a complete source-pool trace."""
    # Keep the analysis kernel importable without model/data dependencies.
    from nemo_rl.algorithms.async_utils.fixed_pool import (
        load_fixed_pool_manifest,
        validate_fixed_pool_trace,
    )

    manifest = load_fixed_pool_manifest(manifest_path)
    validate_fixed_pool_trace(
        trace_path,
        manifest,
        expected_completions_per_group=expected_completions_per_group,
    )
    report = validate_scheduler_trace(trace_path)
    if report.administratively_censored_group_ids:
        raise SchedulerReplayError("source trace has administratively censored groups")
    return groups_from_trace(iter_scheduler_trace(trace_path), manifest)


def ticks_from_trace(events: Iterable[SchedulerTraceEvent]) -> tuple[ReplayTick, ...]:
    """Build a physical-version fidelity plan from recorded select decisions."""
    ticks = []
    for event in events:
        if event.event_type is not SchedulerEventType.SELECT_DECISION:
            continue
        if event.trainer_version is None:
            raise SchedulerReplayError("select decision lacks trainer_version")
        assert (
            event.min_prompt_groups is not None and event.max_prompt_groups is not None
        )
        ticks.append(
            ReplayTick(
                tick_id=len(ticks),
                monotonic_ns=event.monotonic_ns,
                nominal_trainer_version=event.trainer_version,
                min_prompt_groups=event.min_prompt_groups,
                max_prompt_groups=event.max_prompt_groups,
            )
        )
    if not ticks:
        raise SchedulerReplayError("trace contains no selection decisions")
    return tuple(ticks)


def natural_releases(groups: Sequence[ReplayGroup]) -> ReleaseSchedule:
    _validate_groups(groups)
    return ReleaseSchedule(
        kind="natural",
        ready_ns_by_group={g.logical_group_id: g.ready_ns for g in groups},
    )


def decorrelated_releases(
    groups: Sequence[ReplayGroup], *, seed: int, namespace: str = ""
) -> ReleaseSchedule:
    """Uniformly permute latencies within task-overlapping design blocks.

    Identity assignments and fixed points are intentionally allowed. Source
    ordering is derived from SHA-256 rather than a Python RNG implementation,
    making a plan stable across supported Python versions.
    """
    _validate_groups(groups)
    by_block: dict[str, list[ReplayGroup]] = defaultdict(list)
    for group in groups:
        by_block[group.decorrelation_block].append(group)
    releases: dict[str, int] = {}
    assignments: list[ReleaseAssignment] = []
    assignment_fixed_points = 0
    unchanged_latency_values = 0
    for block_id in sorted(by_block):
        block = sorted(by_block[block_id], key=lambda group: group.slot_order)
        if len(block) < 2 or len({group.task_stratum for group in block}) < 2:
            raise SchedulerReplayError(
                f"decorrelation block {block_id!r} lacks task overlap"
            )
        sources = sorted(
            block,
            key=lambda group: hashlib.sha256(
                (f"{namespace}\0{seed}\0{block_id}\0{group.prompt_uid}").encode()
            ).digest(),
        )
        for destination, source in zip(block, sources):
            latency = source.natural_latency_ns
            is_assignment_fixed = (
                destination.logical_group_id == source.logical_group_id
            )
            is_latency_unchanged = destination.natural_latency_ns == latency
            assignment_fixed_points += int(is_assignment_fixed)
            unchanged_latency_values += int(is_latency_unchanged)
            releases[destination.logical_group_id] = destination.dispatch_ns + latency
            assignments.append(
                ReleaseAssignment(
                    block_id=block_id,
                    destination_group_id=destination.logical_group_id,
                    destination_prompt_uid=destination.prompt_uid,
                    source_group_id=source.logical_group_id,
                    source_prompt_uid=source.prompt_uid,
                    latency_ns=latency,
                    assignment_fixed_point=is_assignment_fixed,
                    unchanged_latency_value=is_latency_unchanged,
                )
            )
    return ReleaseSchedule(
        kind="decorrelated",
        ready_ns_by_group=releases,
        seed=seed,
        assignments=tuple(assignments),
        assignment_fixed_points=assignment_fixed_points,
        unchanged_latency_values=unchanged_latency_values,
        blocks=len(by_block),
    )


def _validate_groups(groups: Sequence[ReplayGroup]) -> None:
    if not groups:
        raise SchedulerReplayError("replay group pool is empty")
    ids = [group.logical_group_id for group in groups]
    if len(ids) != len(set(ids)):
        raise SchedulerReplayError("duplicate logical group ID")
    orders = [group.slot_order for group in groups]
    if len(orders) != len(set(orders)):
        raise SchedulerReplayError("duplicate slot order")


def replay_schedule(
    groups: Sequence[ReplayGroup],
    ticks: Sequence[ReplayTick],
    policy: ReplayPolicy,
    releases: Optional[ReleaseSchedule] = None,
) -> ReplayResult:
    """Replay one policy against fixed dispatches, releases, and learner ticks."""
    _validate_groups(groups)
    if not ticks:
        raise SchedulerReplayError("learner tick plan is empty")
    if [tick.tick_id for tick in ticks] != list(range(len(ticks))):
        raise SchedulerReplayError("learner tick IDs must be contiguous from zero")
    if any(b.monotonic_ns < a.monotonic_ns for a, b in zip(ticks, ticks[1:])):
        raise SchedulerReplayError("learner tick clocks must be monotonic")
    if any(
        b.nominal_trainer_version < a.nominal_trainer_version
        for a, b in zip(ticks, ticks[1:])
    ):
        raise SchedulerReplayError("nominal trainer versions must be nondecreasing")
    release_schedule = releases or natural_releases(groups)
    expected_ids = {group.logical_group_id for group in groups}
    if set(release_schedule.ready_ns_by_group) != expected_ids:
        raise SchedulerReplayError("release schedule/group coverage mismatch")
    if any(
        release_schedule.ready_ns_by_group[group.logical_group_id] < group.dispatch_ns
        for group in groups
    ):
        raise SchedulerReplayError("release schedule contains a pre-dispatch release")
    if release_schedule.kind == "natural" and release_schedule.seed is not None:
        raise SchedulerReplayError("natural release schedules cannot have a seed")
    if release_schedule.kind == "decorrelated" and release_schedule.seed is None:
        raise SchedulerReplayError("decorrelated release schedules require a seed")
    by_id = {group.logical_group_id: group for group in groups}
    ordered = sorted(groups, key=lambda group: group.slot_order)
    terminal: set[str] = set()
    selected_all: list[str] = []
    evicted_all: list[str] = []
    decisions: list[ReplayDecision] = []

    for tick in ticks:
        live = [
            group
            for group in ordered
            if group.dispatch_ns <= tick.monotonic_ns
            and group.logical_group_id not in terminal
        ]
        ready = [
            group
            for group in live
            if release_schedule.ready_ns_by_group[group.logical_group_id]
            <= tick.monotonic_ns
        ]
        version = tick.nominal_trainer_version
        evicted: list[ReplayGroup] = []
        if policy.name == "in_order":
            evicted = [
                group
                for group in ready
                if group.target_step is not None and group.target_step < version
            ]
        elif policy.name == "windowed":
            minimum = max(0, version - policy.max_staleness_versions)
            evicted = [
                group for group in ready if group.nominal_start_version < minimum
            ]
        for group in evicted:
            terminal.add(group.logical_group_id)
            evicted_all.append(group.logical_group_id)

        ready_after_evict = [
            group for group in ready if group.logical_group_id not in terminal
        ]
        if policy.name == "in_order":
            eligible = [
                group for group in ready_after_evict if group.target_step == version
            ]
        elif policy.name == "ready_first":
            eligible = [
                group
                for group in ready_after_evict
                if group.nominal_start_version <= version
            ]
        else:
            minimum = max(0, version - policy.max_staleness_versions)
            eligible = [
                group
                for group in ready_after_evict
                if minimum <= group.nominal_start_version <= version
            ]
            if policy.sample_freshest_first:
                eligible.sort(
                    key=lambda group: (
                        version - group.nominal_start_version,
                        group.slot_order,
                    )
                )

        selected = (
            eligible[: tick.max_prompt_groups]
            if len(eligible) >= tick.min_prompt_groups
            else []
        )
        for group in selected:
            terminal.add(group.logical_group_id)
            selected_all.append(group.logical_group_id)
        decisions.append(
            ReplayDecision(
                tick_id=tick.tick_id,
                monotonic_ns=tick.monotonic_ns,
                trainer_version=version,
                ready_group_ids=tuple(
                    group.logical_group_id for group in ready_after_evict
                ),
                eligible_group_ids=tuple(group.logical_group_id for group in eligible),
                selected_group_ids=tuple(group.logical_group_id for group in selected),
                evicted_group_ids=tuple(group.logical_group_id for group in evicted),
            )
        )

    live_ids = tuple(
        group.logical_group_id
        for group in ordered
        if group.logical_group_id not in terminal
    )
    # Ensure an accidental unknown mapping cannot hide behind set validation.
    assert all(group_id in by_id for group_id in selected_all + evicted_all)
    return ReplayResult(
        policy=policy,
        release_kind=release_schedule.kind,
        release_seed=release_schedule.seed,
        decisions=tuple(decisions),
        selected_group_ids=tuple(selected_all),
        evicted_group_ids=tuple(evicted_all),
        live_group_ids=live_ids,
    )


def validate_replay_fidelity(
    events: Iterable[SchedulerTraceEvent], result: ReplayResult
) -> None:
    """Require group-ID-level agreement with a native scheduler trace."""
    recorded: list[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], int]] = []
    pending_evictions: list[str] = []
    for event in events:
        if event.event_type is SchedulerEventType.GROUP_EVICTED:
            assert event.logical_group_id is not None
            pending_evictions.append(event.logical_group_id)
        elif event.event_type is SchedulerEventType.SELECT_DECISION:
            recorded.append(
                (
                    event.eligible_logical_group_ids,
                    event.selected_logical_group_ids,
                    tuple(pending_evictions),
                    event.ready_prompt_groups or 0,
                )
            )
            pending_evictions = []
    if pending_evictions:
        raise SchedulerReplayError("native trace ends with unmatched evictions")
    if len(recorded) != len(result.decisions):
        raise SchedulerReplayError(
            "fidelity mismatch: replay/native decision counts differ"
        )
    for index, (native, replayed) in enumerate(zip(recorded, result.decisions)):
        observed = (
            replayed.eligible_group_ids,
            replayed.selected_group_ids,
            replayed.evicted_group_ids,
            len(replayed.ready_group_ids),
        )
        if observed != native:
            raise SchedulerReplayError(
                f"fidelity mismatch at decision {index}: "
                f"native={native!r}, replay={observed!r}"
            )


def mixture_metrics(
    groups: Sequence[ReplayGroup],
    result: ReplayResult,
    *,
    horizon: int,
) -> dict[str, object]:
    """Compute fixed-prefix task-mixture and window-censoring estimands."""
    _validate_groups(groups)
    if horizon < 1 or horizon > len(groups):
        raise SchedulerReplayError("horizon must be within the predeclared pool")
    by_id = {group.logical_group_id: group for group in groups}
    selected_ids = result.selected_group_ids[:horizon]
    if len(selected_ids) != horizon:
        raise SchedulerReplayError(
            f"policy reached only {len(selected_ids)}/{horizon} selected groups"
        )
    cumulative_selected = 0
    evicted_ids_at_horizon: list[str] = []
    horizon_tick_id: Optional[int] = None
    for decision in result.decisions:
        evicted_ids_at_horizon.extend(decision.evicted_group_ids)
        cumulative_selected += len(decision.selected_group_ids)
        if cumulative_selected >= horizon:
            horizon_tick_id = decision.tick_id
            break
    if horizon_tick_id is None:
        raise SchedulerReplayError("selected prefix has no corresponding horizon tick")
    tasks = sorted({group.task_stratum for group in groups})
    target_counts = {task: 0 for task in tasks}
    selected_counts = {task: 0 for task in tasks}
    evicted_counts = {task: 0 for task in tasks}
    for group in groups:
        target_counts[group.task_stratum] += 1
    for group_id in selected_ids:
        selected_counts[by_id[group_id].task_stratum] += 1
    for group_id in evicted_ids_at_horizon:
        evicted_counts[by_id[group_id].task_stratum] += 1

    target_share = {task: target_counts[task] / len(groups) for task in tasks}
    denominator = len(selected_ids)
    selected_share = {
        task: selected_counts[task] / denominator if denominator else 0.0
        for task in tasks
    }
    signed_shift = {task: selected_share[task] - target_share[task] for task in tasks}
    representation = {task: selected_share[task] / target_share[task] for task in tasks}
    inclusion = {task: selected_counts[task] / target_counts[task] for task in tasks}
    eviction = {task: evicted_counts[task] / target_counts[task] for task in tasks}
    return {
        "pool_groups": len(groups),
        "requested_horizon": horizon,
        "horizon_reached": True,
        "horizon_tick_id": horizon_tick_id,
        "observed_prefix_groups": denominator,
        "target_share": target_share,
        "selected_share": selected_share,
        "signed_share_shift": signed_shift,
        "total_variation": 0.5 * sum(abs(value) for value in signed_shift.values()),
        "worst_representation_ratio": min(representation.values()),
        "worst_signed_deficit": min(signed_shift.values()),
        "inclusion_rate_by_task": inclusion,
        "inclusion_disparity": max(inclusion.values()) - min(inclusion.values()),
        "eviction_fraction": len(evicted_ids_at_horizon) / len(groups),
        "eviction_rate_by_task": eviction,
        "eviction_disparity": max(eviction.values()) - min(eviction.values()),
    }


def latency_diagnostics(
    groups: Sequence[ReplayGroup], releases: ReleaseSchedule
) -> dict[str, object]:
    """Report residual task association in log latency for a release schedule."""
    _validate_groups(groups)
    if set(releases.ready_ns_by_group) != {g.logical_group_id for g in groups}:
        raise SchedulerReplayError("release schedule/group coverage mismatch")
    by_task: dict[str, list[float]] = defaultdict(list)
    raw_by_task: dict[str, list[int]] = defaultdict(list)
    all_values: list[float] = []
    for group in groups:
        latency = releases.ready_ns_by_group[group.logical_group_id] - group.dispatch_ns
        if latency < 0:
            raise SchedulerReplayError("release precedes dispatch")
        value = math.log1p(latency)
        by_task[group.task_stratum].append(value)
        raw_by_task[group.task_stratum].append(latency)
        all_values.append(value)
    means = {
        task: sum(values) / len(values) for task, values in sorted(by_task.items())
    }
    grand = sum(all_values) / len(all_values)
    between = sum(
        len(values) * (means[task] - grand) ** 2 for task, values in by_task.items()
    )
    total = sum((value - grand) ** 2 for value in all_values)
    max_smd = 0.0
    degenerate_smd = False
    task_names = sorted(by_task)
    medians = {task: _median(raw_by_task[task]) for task in task_names}
    rank_biserial: dict[str, float] = {}
    for i, left in enumerate(task_names):
        for right in task_names[i + 1 :]:
            a, b = by_task[left], by_task[right]
            va = sum((x - means[left]) ** 2 for x in a) / max(1, len(a) - 1)
            vb = sum((x - means[right]) ** 2 for x in b) / max(1, len(b) - 1)
            degrees = len(a) + len(b) - 2
            pooled = (
                math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / degrees)
                if degrees > 0
                else 0.0
            )
            difference = abs(means[left] - means[right])
            if pooled:
                max_smd = max(max_smd, difference / pooled)
            elif difference:
                degenerate_smd = True
            comparisons = [
                (left_value > right_value) - (left_value < right_value)
                for left_value in raw_by_task[left]
                for right_value in raw_by_task[right]
            ]
            rank_biserial[f"{left}__vs__{right}"] = sum(comparisons) / len(comparisons)
    positive_medians = [value for value in medians.values() if value > 0]
    return {
        "latency_median_ns_by_task": medians,
        "max_median_latency_ratio": (
            max(positive_medians) / min(positive_medians)
            if len(positive_medians) == len(medians) and positive_medians
            else None
        ),
        "rank_biserial_by_task_pair": rank_biserial,
        "log_latency_mean_by_task": means,
        "task_eta_squared": between / total if total else 0.0,
        "max_pairwise_smd": None if degenerate_smd else max_smd,
        "degenerate_smd": degenerate_smd,
        "assignment_fixed_points": releases.assignment_fixed_points,
        "unchanged_latency_values": releases.unchanged_latency_values,
        "blocks": releases.blocks,
    }


def _median(values: Sequence[int]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


def scheduler_release_interaction(
    *,
    natural_policy: float,
    natural_in_order: float,
    decorrelated_policy: Sequence[float],
    decorrelated_in_order: Sequence[float],
) -> float:
    """Difference-in-differences versus in-order, averaging release seeds."""
    if not decorrelated_policy or len(decorrelated_policy) != len(
        decorrelated_in_order
    ):
        raise SchedulerReplayError(
            "decorrelated contrasts require paired nonempty seeds"
        )
    decorated_effect = sum(
        policy_value - baseline_value
        for policy_value, baseline_value in zip(
            decorrelated_policy, decorrelated_in_order
        )
    ) / len(decorrelated_policy)
    return (natural_policy - natural_in_order) - decorated_effect
