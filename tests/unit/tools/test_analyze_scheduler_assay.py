# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for controlled live scheduler assay analysis."""

import pytest

from nemo_rl.algorithms.async_utils.scheduler_trace import (
    SchedulerEventType,
    SchedulerTraceEvent,
)
from tools.analyze_scheduler_assay import (
    SchedulerAssayAnalysisError,
    _successful_selection_decisions,
)


def _decision(event_seq: int, step: int, *, selected: bool) -> SchedulerTraceEvent:
    selected_ids = (
        tuple(f"group-{step}-{index}" for index in range(4)) if selected else ()
    )
    return SchedulerTraceEvent(
        schema_version=1,
        trace_run_id="run",
        process_epoch="epoch",
        event_seq=event_seq,
        monotonic_ns=event_seq,
        event_type=SchedulerEventType.SELECT_DECISION,
        min_prompt_groups=4,
        max_prompt_groups=4,
        selected_logical_group_ids=selected_ids,
        scalar_summaries={
            "scheduler_assay_step": step,
            "selected_prompt_groups": len(selected_ids),
        },
    )


def test_successful_selection_decisions_ignore_non_mutating_live_polls() -> None:
    events = []
    event_seq = 0
    for step in range(12):
        events.append(_decision(event_seq, step, selected=False))
        event_seq += 1
        events.append(_decision(event_seq, step, selected=True))
        event_seq += 1

    successful = _successful_selection_decisions(events)

    assert len(successful) == 12
    assert [event.event_seq for event in successful] == list(range(1, 24, 2))


def test_successful_selection_decisions_require_all_logical_ticks() -> None:
    events = [_decision(step, step, selected=True) for step in range(11)]

    with pytest.raises(
        SchedulerAssayAnalysisError,
        match="exactly 12 successful selections",
    ):
        _successful_selection_decisions(events)


def test_successful_selection_decisions_require_contiguous_steps() -> None:
    events = [
        _decision(step, step if step < 7 else step + 1, selected=True)
        for step in range(12)
    ]

    with pytest.raises(SchedulerAssayAnalysisError, match="steps must be contiguous"):
        _successful_selection_decisions(events)
