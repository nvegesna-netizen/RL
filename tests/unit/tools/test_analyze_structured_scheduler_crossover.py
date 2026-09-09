# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused tests for structured scheduler-crossover analysis."""

from pathlib import Path

import pytest

from nemo_rl.algorithms.async_utils.scheduler_trace import (
    SchedulerEventType,
    SchedulerTraceEvent,
)
from tools import analyze_structured_scheduler_crossover as analyzer
from tools.analyze_structured_scheduler_crossover import (
    StructuredSchedulerCrossoverAnalysisError,
    _load_validated_pool_manifest,
    _successful_decisions,
)


def _decision(event_seq: int, step: int, selected_count: int) -> SchedulerTraceEvent:
    selected = tuple(f"group-{step}-{index}" for index in range(selected_count))
    return SchedulerTraceEvent(
        schema_version=1,
        trace_run_id="run",
        process_epoch="epoch",
        event_seq=event_seq,
        monotonic_ns=event_seq,
        event_type=SchedulerEventType.SELECT_DECISION,
        min_prompt_groups=4,
        max_prompt_groups=4,
        eligible_prompt_groups=len(selected),
        eligible_logical_group_ids=selected,
        selected_logical_group_ids=selected,
        scalar_summaries={
            "scheduler_assay_step": step,
            "selected_prompt_groups": len(selected),
        },
    )


def test_successful_decisions_ignore_empty_live_polls() -> None:
    events = []
    for step in range(4):
        events.append(_decision(2 * step, step, 0))
        events.append(_decision(2 * step + 1, step, 4))

    decisions = _successful_decisions(events)

    assert len(decisions) == 4
    assert [event.event_seq for event in decisions] == [1, 3, 5, 7]


def test_successful_decisions_reject_non_exact_selection() -> None:
    events = [_decision(step, step, 4) for step in range(4)]
    events[-1] = _decision(3, 3, 3)

    with pytest.raises(
        StructuredSchedulerCrossoverAnalysisError,
        match="selection size or logical step mismatch",
    ):
        _successful_decisions(events)


def test_pool_validation_uses_materialization_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied = tmp_path / "run" / "fixed_pool_manifest.v1.json"
    materialized = tmp_path / "pool" / "fixed_pool_manifest.v1.46001.json"
    copied.parent.mkdir()
    materialized.parent.mkdir()
    copied.write_bytes(b"same immutable manifest\n")
    materialized.write_bytes(copied.read_bytes())
    sentinel = object()
    loaded_paths = []
    validated = []
    monkeypatch.setattr(
        analyzer,
        "load_fixed_pool_manifest",
        lambda path: loaded_paths.append(path) or sentinel,
    )
    monkeypatch.setattr(
        analyzer,
        "validate_fixed_pool_materialization",
        lambda manifest: validated.append(manifest),
    )

    assert (
        _load_validated_pool_manifest(
            copied_manifest_path=copied,
            materialization_manifest_path=materialized,
        )
        is sentinel
    )
    assert loaded_paths == [materialized]
    assert validated == [sentinel]


def test_pool_validation_rejects_nonidentical_run_copy(tmp_path: Path) -> None:
    copied = tmp_path / "copied.json"
    materialized = tmp_path / "materialized.json"
    copied.write_bytes(b"copied")
    materialized.write_bytes(b"materialized")

    with pytest.raises(
        StructuredSchedulerCrossoverAnalysisError,
        match="not byte-identical",
    ):
        _load_validated_pool_manifest(
            copied_manifest_path=copied,
            materialization_manifest_path=materialized,
        )
