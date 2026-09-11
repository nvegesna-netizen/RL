# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from typing import cast

from nemo_rl.algorithms.async_utils.scheduler_trace import (
    SchedulerEventType,
    SchedulerTraceEvent,
)
from tools.analyze_scheduler_pressure_response import (
    _maximum_active_generation_concurrency,
    _maximum_unreleased_group_concurrency,
)


def _event(event_type: SchedulerEventType, monotonic_ns: int) -> SchedulerTraceEvent:
    return cast(
        SchedulerTraceEvent,
        SimpleNamespace(event_type=event_type, monotonic_ns=monotonic_ns),
    )


def test_generation_concurrency_excludes_post_completion_release_hold() -> None:
    events = [
        _event(SchedulerEventType.ATTEMPT_DISPATCHED, 1),
        _event(SchedulerEventType.ATTEMPT_DISPATCHED, 2),
        _event(SchedulerEventType.ROLLOUT_COMPLETED, 3),
        _event(SchedulerEventType.ATTEMPT_DISPATCHED, 4),
        _event(SchedulerEventType.ROLLOUT_COMPLETED, 5),
        _event(SchedulerEventType.ROLLOUT_COMPLETED, 6),
        _event(SchedulerEventType.GROUP_READY, 10),
        _event(SchedulerEventType.GROUP_READY, 11),
        _event(SchedulerEventType.GROUP_READY, 12),
    ]

    assert _maximum_active_generation_concurrency(events) == 2
    assert _maximum_unreleased_group_concurrency(events) == 3


def test_failed_attempt_releases_both_concurrency_intervals() -> None:
    events = [
        _event(SchedulerEventType.ATTEMPT_DISPATCHED, 1),
        _event(SchedulerEventType.ATTEMPT_FAILED, 2),
        _event(SchedulerEventType.ATTEMPT_DISPATCHED, 3),
        _event(SchedulerEventType.ROLLOUT_COMPLETED, 4),
        _event(SchedulerEventType.GROUP_READY, 5),
    ]

    assert _maximum_active_generation_concurrency(events) == 1
    assert _maximum_unreleased_group_concurrency(events) == 1
