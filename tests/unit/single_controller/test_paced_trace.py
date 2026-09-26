"""Synthetic full-lifecycle tests, independent of Ray and model generation."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "nemo_rl/algorithms/async_utils/scheduler_trace.py"
)
SPEC = importlib.util.spec_from_file_location("paced_trace_under_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def records() -> list[dict]:
    output: list[dict] = []

    def emit(kind: str, **fields: object) -> None:
        event = MODULE.SchedulerTraceEvent(
            schema_version=1,
            trace_run_id="test",
            process_epoch="test",
            event_seq=len(output),
            monotonic_ns=len(output) * 1_000_000_000,
            event_type=MODULE.SchedulerEventType(kind),
            trainer_version=0,
            **fields,
        )
        output.append(event.to_record())

    emit("run_started")
    emit(
        "admission_granted",
        admission_id="a",
        sampler_dispatch_index=0,
        scalar_summaries={"expected_prompt_groups": 4},
    )
    groups = tuple(f"g{i}" for i in range(4))
    for group in groups:
        for kind in ("attempt_dispatched", "rollout_completed", "group_ready"):
            emit(kind, admission_id="a", attempt_id=group, logical_group_id=group)
    emit(
        "select_decision",
        min_prompt_groups=4,
        max_prompt_groups=4,
        ready_prompt_groups=4,
        eligible_prompt_groups=4,
        eligible_logical_group_ids=groups,
        selected_logical_group_ids=groups,
        scalar_summaries={"selected_prompt_groups": 4, "scheduler_assay_step": 0},
    )
    for kind in ("consumer_buffer_released", "consumer_started", "consumer_completed"):
        summaries = {
            "scheduler_assay_step": 0,
            "physical_weight_version": 0,
            "selected_prompt_groups": 4,
            "requested_consumer_seconds": 1.0,
        }
        if kind == "consumer_completed":
            summaries["actual_consumer_seconds"] = 1.0
        emit(kind, selected_logical_group_ids=groups, scalar_summaries=summaries)
    emit(
        "consumer_drained",
        scalar_summaries={
            "scheduler_assay_step": 1,
            "physical_weight_version": 0,
            "selected_prompt_groups": 4,
            "buffered_prompt_groups": 0,
        },
    )
    emit(
        "run_ended",
        terminal_reason="scheduler_assay_complete",
        scalar_summaries={
            "completed_train_steps": 0,
            "final_physical_weight_version": 0,
            "final_scheduler_assay_step": 1,
            "assay_selection_steps": 1,
            "assay_selected_prompt_groups": 4,
        },
    )
    return output


class PacedTraceTests(unittest.TestCase):
    def validate(self, events: list[dict]) -> None:
        # Renumber edited fixtures so failures exercise lifecycle, not sequence gaps.
        for index, event in enumerate(events):
            event["event_seq"] = index
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "trace.jsonl"
            filename.write_text("".join(json.dumps(event) + "\n" for event in events))
            MODULE.validate_paced_consumer_trace(
                filename, consumer_seconds=1.0, expected_groups=4
            )

    def test_complete_trace(self) -> None:
        self.validate(records())

    def test_missing_lifecycle_event_rejected(self) -> None:
        for kind in (
            "consumer_buffer_released",
            "consumer_started",
            "consumer_completed",
            "consumer_drained",
        ):
            with (
                self.subTest(kind=kind),
                self.assertRaises(MODULE.SchedulerTraceValidationError),
            ):
                self.validate(
                    [event for event in records() if event["event_type"] != kind]
                )

    def test_changed_step_version_batch_and_timing_rejected(self) -> None:
        for field, value in (
            ("scheduler_assay_step", 1),
            ("physical_weight_version", 1),
            ("selected_prompt_groups", 3),
            ("actual_consumer_seconds", 0.5),
            ("actual_consumer_seconds", 2.0),
            ("requested_consumer_seconds", 2.0),
        ):
            events = records()
            event = next(
                item for item in events if item["event_type"] == "consumer_completed"
            )
            event["scalar_summaries"][field] = value
            with (
                self.subTest(field=field, value=value),
                self.assertRaises(MODULE.SchedulerTraceValidationError),
            ):
                self.validate(events)

    def test_duplicate_drain_rejected(self) -> None:
        events = records()
        events.insert(-1, dict(events[-2]))
        with self.assertRaises(MODULE.SchedulerTraceValidationError):
            self.validate(events)

    def test_wrong_admission_index_rejected(self) -> None:
        events = records()
        events[1]["sampler_dispatch_index"] = 4
        with self.assertRaises(MODULE.SchedulerTraceValidationError):
            self.validate(events)

    def test_nonzero_training_or_error_terminal_rejected(self) -> None:
        for change in ("training", "error", "weight"):
            events = records()
            if change == "training":
                events[-1]["scalar_summaries"]["completed_train_steps"] = 1
            elif change == "error":
                events[-1]["terminal_reason"] = "error"
            else:
                events[2]["trainer_version"] = 1
            with (
                self.subTest(change=change),
                self.assertRaises(MODULE.SchedulerTraceValidationError),
            ):
                self.validate(events)


if __name__ == "__main__":
    unittest.main()
