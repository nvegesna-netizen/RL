"""Unit tests for controller-local rollout lifecycle events."""

from __future__ import annotations

import json

import pytest

from nemo_rl.algorithms.async_utils.rollout_lifecycle import (
    RolloutLifecycleRecorder,
    RolloutLifecycleStage,
    RolloutRemovalReason,
)


def test_records_one_monotonic_clock_domain_and_mixed_version_status():
    timestamps = iter((10, 20, 30))
    recorder = RolloutLifecycleRecorder(
        run_id="run",
        clock_domain_id="controller",
        clock_ns=lambda: next(timestamps),
    )

    recorder.record(
        group_id="group",
        stage=RolloutLifecycleStage.RESERVED,
        start_weight_version=3,
        target_step=4,
    )
    recorder.record(
        group_id="group",
        stage=RolloutLifecycleStage.GROUP_READY,
        start_weight_version=3,
        end_weight_version=4,
        target_step=4,
        sample_ids=("group_g0", "group_g1"),
    )
    recorder.record(
        group_id="group",
        stage=RolloutLifecycleStage.REMOVED,
        start_weight_version=3,
        end_weight_version=4,
        target_step=4,
        learner_weight_version=5,
        sample_ids=("group_g0", "group_g1"),
        removal_reason=RolloutRemovalReason.SELECTED,
    )

    events = recorder.snapshot()
    assert [event.sequence for event in events] == [0, 1, 2]
    assert [event.timestamp_ns for event in events] == [10, 20, 30]
    assert {event.clock_domain_id for event in events} == {"controller"}
    assert events[0].mixed_generation_versions is None
    assert events[1].mixed_generation_versions is True
    assert events[2].learner_weight_version == 5


def test_rejects_reason_on_non_removal_and_missing_reason_on_removal():
    recorder = RolloutLifecycleRecorder(clock_ns=lambda: 0)

    with pytest.raises(ValueError, match="valid only for removed"):
        recorder.record(
            group_id="group",
            stage=RolloutLifecycleStage.RESERVED,
            start_weight_version=0,
            removal_reason=RolloutRemovalReason.FAILED,
        )
    with pytest.raises(ValueError, match="require removal_reason"):
        recorder.record(
            group_id="group",
            stage=RolloutLifecycleStage.REMOVED,
            start_weight_version=0,
        )


def test_flush_jsonl_is_machine_readable(tmp_path):
    recorder = RolloutLifecycleRecorder(
        run_id="run",
        clock_domain_id="controller",
        clock_ns=lambda: 123,
    )
    recorder.record(
        group_id="group",
        stage=RolloutLifecycleStage.RESERVED,
        start_weight_version=2,
    )

    output_path = tmp_path / "nested" / "events.jsonl"
    recorder.flush_jsonl(output_path)

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rows == [
        {
            "clock_domain_id": "controller",
            "end_weight_version": None,
            "group_id": "group",
            "learner_weight_version": None,
            "mixed_generation_versions": None,
            "removal_reason": None,
            "run_id": "run",
            "sample_ids": [],
            "schema_version": 1,
            "sequence": 0,
            "stage": "reserved",
            "start_weight_version": 2,
            "target_step": None,
            "timestamp_ns": 123,
        }
    ]
