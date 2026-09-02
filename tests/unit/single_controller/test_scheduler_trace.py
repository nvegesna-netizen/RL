# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from nemo_rl.algorithms.async_utils.scheduler_trace import (
    JsonlSchedulerTraceSink,
    NoopSchedulerTraceSink,
    SchedulerEventType,
    SchedulerTraceValidationError,
    SchedulerTraceWriteError,
    iter_scheduler_trace,
    validate_scheduler_trace,
)


class SchedulerTraceTests(unittest.TestCase):
    @staticmethod
    def _start_run(sink: JsonlSchedulerTraceSink) -> None:
        sink.emit(SchedulerEventType.RUN_STARTED)

    @staticmethod
    def _end_run(
        sink: JsonlSchedulerTraceSink, live_group_ids: tuple[str, ...] = ()
    ) -> None:
        sink.emit(
            SchedulerEventType.RUN_ENDED,
            terminal_reason="test",
            live_logical_group_ids=live_group_ids,
        )

    def test_noop_creates_no_file_or_task(self) -> None:
        async def exercise(path: Path) -> None:
            sink = NoopSchedulerTraceSink()
            await sink.start()
            sink.emit(SchedulerEventType.SELECT_DECISION, arbitrary="ignored")
            await sink.close()
            self.assertFalse(path.exists())

        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(exercise(Path(directory) / "trace.jsonl"))

    def test_distinct_attempts_and_ready_lifecycle_round_trip(self) -> None:
        async def exercise(path: Path) -> None:
            sink = JsonlSchedulerTraceSink(
                path, trace_run_id="run", process_epoch="epoch", flush_every=2
            )
            await sink.start()
            self._start_run(sink)
            for attempt in ("g/0", "g/1"):
                group_id = attempt.replace("/", "-")
                admission_id = f"admit-{attempt}"
                sink.emit(
                    SchedulerEventType.ADMISSION_GRANTED,
                    admission_id=admission_id,
                    scalar_summaries={"expected_prompt_groups": 1},
                )
                common = {
                    "logical_group_id": group_id,
                    "attempt_id": attempt,
                    "admission_id": admission_id,
                }
                sink.emit(SchedulerEventType.ATTEMPT_DISPATCHED, **common)
                sink.emit(
                    SchedulerEventType.ROLLOUT_COMPLETED,
                    **common,
                    scalar_summaries={"reward_mean": 0.5},
                )
                sink.emit(SchedulerEventType.GROUP_READY, **common)
            self._end_run(sink, ("g-0", "g-1"))
            await sink.close()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            asyncio.run(exercise(path))
            events = list(iter_scheduler_trace(path))
            self.assertEqual([event.event_seq for event in events], list(range(10)))
            self.assertNotEqual(events[2].attempt_id, events[6].attempt_id)
            report = validate_scheduler_trace(path)
            self.assertEqual(report.events, 10)
            self.assertEqual(
                report.administratively_censored_group_ids, ("g-0", "g-1")
            )

    def test_existing_file_is_never_appended(self) -> None:
        async def exercise(path: Path) -> None:
            sink = JsonlSchedulerTraceSink(path)
            with self.assertRaises(SchedulerTraceWriteError):
                await sink.start()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text("existing\n")
            asyncio.run(exercise(path))

    def test_incomplete_lifecycle_is_reported(self) -> None:
        async def exercise(path: Path) -> None:
            sink = JsonlSchedulerTraceSink(path, trace_run_id="r", process_epoch="p")
            await sink.start()
            self._start_run(sink)
            sink.emit(
                SchedulerEventType.ADMISSION_GRANTED,
                admission_id="admit",
                scalar_summaries={"expected_prompt_groups": 1},
            )
            sink.emit(
                SchedulerEventType.ATTEMPT_DISPATCHED,
                logical_group_id="g",
                attempt_id="a",
                admission_id="admit",
            )
            self._end_run(sink)
            await sink.close()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            asyncio.run(exercise(path))
            with self.assertRaises(SchedulerTraceValidationError):
                validate_scheduler_trace(path)
            report = validate_scheduler_trace(path, require_complete_attempts=False)
            self.assertEqual(report.incomplete_attempt_ids, ("a",))

    def test_only_truncated_final_record_can_be_tolerated(self) -> None:
        async def exercise(path: Path) -> None:
            sink = JsonlSchedulerTraceSink(path, trace_run_id="r", process_epoch="p")
            await sink.start()
            self._start_run(sink)
            sink.emit(
                SchedulerEventType.SELECT_DECISION,
                min_prompt_groups=1,
                max_prompt_groups=2,
                eligible_prompt_groups=0,
                ready_prompt_groups=0,
                scalar_summaries={"selected_prompt_groups": 0},
            )
            self._end_run(sink)
            await sink.close()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            asyncio.run(exercise(path))
            with path.open("ab") as handle:
                handle.write(b'{"schema_version":')
            with self.assertRaises(SchedulerTraceValidationError):
                list(iter_scheduler_trace(path))
            self.assertEqual(
                len(
                    list(
                        iter_scheduler_trace(
                            path, tolerate_truncated_final_record=True
                        )
                    )
                ),
                1,
            )


if __name__ == "__main__":
    unittest.main()
