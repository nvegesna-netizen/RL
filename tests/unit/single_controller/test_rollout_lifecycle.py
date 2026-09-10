"""Unit tests for controller-local rollout lifecycle events."""

from __future__ import annotations

import json

import pytest

from nemo_rl.algorithms.async_utils.rollout_lifecycle import (
    ControllerEventSequencer,
    RolloutLifecycleRecorder,
    RolloutLifecycleStage,
    RolloutRemovalReason,
)


def test_shared_controller_sequencer_orders_equal_clock_events() -> None:
    sequencer = ControllerEventSequencer(clock_ns=lambda: 10)
    first = RolloutLifecycleRecorder(
        run_id="run",
        clock_domain_id="controller",
        controller_sequencer=sequencer,
    )
    second = RolloutLifecycleRecorder(
        run_id="run",
        clock_domain_id="controller",
        controller_sequencer=sequencer,
    )

    event0 = first.record(
        group_id="g0",
        stage=RolloutLifecycleStage.RESERVED,
        start_weight_version=0,
    )
    event1 = second.record(
        group_id="g1",
        stage=RolloutLifecycleStage.RESERVED,
        start_weight_version=0,
    )

    assert event0.schema_version == event1.schema_version == 4
    assert event0.timestamp_ns == event1.timestamp_ns == 10
    assert (event0.controller_sequence, event1.controller_sequence) == (0, 1)
    assert event0.to_dict()["controller_sequence"] == 0


def test_recorder_wraps_the_complete_record_body_in_duty_meter() -> None:
    class DutyMeter:
        def __init__(self) -> None:
            self.entered = 0
            self.exited = 0

        def observe(self):
            meter = self

            class Observation:
                def __enter__(self):
                    meter.entered += 1

                def __exit__(self, *args):
                    meter.exited += 1

            return Observation()

    meter = DutyMeter()
    recorder = RolloutLifecycleRecorder(clock_ns=lambda: 1, duty_meter=meter)
    recorder.record(
        group_id="group",
        stage=RolloutLifecycleStage.RESERVED,
        start_weight_version=0,
    )

    assert (meter.entered, meter.exited) == (1, 1)


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


def test_records_learner_version_transition_on_controller_clock():
    recorder = RolloutLifecycleRecorder(
        run_id="run",
        clock_domain_id="controller",
        clock_ns=lambda: 123,
    )

    event = recorder.record_learner_version_advanced(
        previous_version=4,
        learner_weight_version=5,
    )

    assert event.stage is RolloutLifecycleStage.LEARNER_VERSION_ADVANCED
    assert event.group_id == "__learner__"
    assert event.start_weight_version == 4
    assert event.learner_weight_version == 5
    assert event.timestamp_ns == 123

    with pytest.raises(ValueError, match="advance by exactly one"):
        recorder.record_learner_version_advanced(
            previous_version=5,
            learner_weight_version=7,
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
            "assistant_tokens": None,
            "end_weight_version": None,
            "env_tokens": None,
            "environment_duration_ns": None,
            "generation_duration_ns": None,
            "group_id": "group",
            "learner_weight_version": None,
            "mixed_generation_versions": None,
            "removal_reason": None,
            "run_id": "run",
            "sample_ids": [],
            "schema_version": 3,
            "sequence": 0,
            "sibling_idx": None,
            "stage": "reserved",
            "start_weight_version": 2,
            "target_step": None,
            "timestamp_ns": 123,
            "terminated": None,
            "trajectory_id": None,
            "truncated": None,
            "turn_count": None,
            "reward": None,
            "release_arm": None,
            "release_delay_seconds": None,
            "release_arm_mass": None,
            "release_total_mass": None,
            "release_global_ordinal": None,
            "release_draw": None,
            "release_nonce": None,
            "generation_inflight": None,
            "active_release_holds": None,
            "reserved_buffer_occupancy": None,
            "ready_buffer_depth": None,
            "buffer_admission_stalls": None,
        }
    ]


def test_release_delay_stage_requires_and_serializes_assignment_fields() -> None:
    recorder = RolloutLifecycleRecorder(clock_ns=lambda: 10)
    assignment_fields = {
        "release_arm": "d30",
        "release_delay_seconds": 30.0,
        "release_arm_mass": 1,
        "release_total_mass": 6,
        "release_global_ordinal": 4,
        "release_draw": 4,
        "release_nonce": 0,
    }

    event = recorder.record(
        group_id="group",
        stage=RolloutLifecycleStage.RELEASE_DELAY_STARTED,
        start_weight_version=2,
        generation_inflight=3,
        active_release_holds=1,
        reserved_buffer_occupancy=5,
        ready_buffer_depth=2,
        buffer_admission_stalls=7,
        **assignment_fields,
    )

    assert event.schema_version == 3
    assert event.release_arm == "d30"
    assert event.release_draw == 4
    assert event.active_release_holds == 1
    assert event.to_dict()["release_global_ordinal"] == 4

    with pytest.raises(ValueError, match="complete assignment fields"):
        recorder.record(
            group_id="missing",
            stage=RolloutLifecycleStage.RELEASE_DELAY_ASSIGNED,
            start_weight_version=0,
        )

    with pytest.raises(ValueError, match="only for release-delay"):
        recorder.record(
            group_id="wrong-stage",
            stage=RolloutLifecycleStage.RESERVED,
            start_weight_version=0,
            **assignment_fields,
        )


def test_bounded_shutdown_reason_serializes_in_schema_v3() -> None:
    recorder = RolloutLifecycleRecorder(clock_ns=lambda: 10)

    event = recorder.record(
        group_id="ready-at-stop",
        stage=RolloutLifecycleStage.REMOVED,
        start_weight_version=127,
        end_weight_version=127,
        learner_weight_version=128,
        sample_ids=("sample",),
        removal_reason=RolloutRemovalReason.BOUNDED_SHUTDOWN,
    )

    assert event.schema_version == 3
    assert event.removal_reason is RolloutRemovalReason.BOUNDED_SHUTDOWN
    assert event.to_dict()["removal_reason"] == "bounded_shutdown"
