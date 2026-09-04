# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Strict controller-local scheduler lifecycle traces.

The schema is metadata-only. It intentionally rejects prompt/completion text,
arbitrary rollout metric objects, non-finite values, silent queue loss, trace
append, and cross-process monotonic-clock comparisons.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import Final, Iterator, Mapping, Optional, Protocol, TypeAlias


TRACE_SCHEMA_VERSION: Final[int] = 1
MAX_IDENTIFIER_LENGTH: Final[int] = 256
MAX_SUMMARY_FIELDS: Final[int] = 64
Scalar: TypeAlias = str | int | float | bool | None


class SchedulerTraceError(RuntimeError):
    """Base trace error."""


class SchedulerTraceWriteError(SchedulerTraceError):
    """A trace event cannot be accepted or flushed without loss."""


class SchedulerTraceValidationError(SchedulerTraceError):
    """A trace violates its versioned schema or lifecycle contract."""


class SchedulerEventType(StrEnum):
    RUN_STARTED = "run_started"
    RUN_ENDED = "run_ended"
    ADMISSION_GRANTED = "admission_granted"
    ATTEMPT_DISPATCHED = "attempt_dispatched"
    ROLLOUT_COMPLETED = "rollout_completed"
    GROUP_READY = "group_ready"
    GROUP_ARCHIVED = "group_archived"
    ATTEMPT_FAILED = "attempt_failed"
    ATTEMPT_REMOVED = "attempt_removed"
    SELECT_DECISION = "select_decision"
    GROUP_EVICTED = "group_evicted"
    ABORT_REQUESTED = "abort_requested"
    PROMPT_SKIPPED = "prompt_skipped"
    GROUP_REPLACED = "group_replaced"
    GROUP_PROMOTED = "group_promoted"


_ATTEMPT_EVENTS: Final = frozenset(
    {
        SchedulerEventType.ATTEMPT_DISPATCHED,
        SchedulerEventType.ROLLOUT_COMPLETED,
        SchedulerEventType.GROUP_READY,
        SchedulerEventType.GROUP_ARCHIVED,
        SchedulerEventType.ATTEMPT_FAILED,
        SchedulerEventType.ATTEMPT_REMOVED,
        SchedulerEventType.ABORT_REQUESTED,
    }
)


def _identifier(name: str, value: Optional[str], *, required: bool = False) -> None:
    if value is None:
        if required:
            raise ValueError(f"{name} is required")
        return
    if not value or len(value) > MAX_IDENTIFIER_LENGTH or "\n" in value:
        raise ValueError(
            f"{name} must be a nonempty single-line value no longer than "
            f"{MAX_IDENTIFIER_LENGTH} characters"
        )


def _summaries(values: Mapping[str, Scalar]) -> None:
    if len(values) > MAX_SUMMARY_FIELDS:
        raise ValueError(f"at most {MAX_SUMMARY_FIELDS} scalar summaries are allowed")
    for key, value in values.items():
        _identifier("summary key", key, required=True)
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise TypeError(f"summary {key!r} is not a JSON scalar")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"summary {key!r} must be finite")


@dataclass(frozen=True, slots=True)
class SchedulerTraceEvent:
    schema_version: int
    trace_run_id: str
    process_epoch: str
    event_seq: int
    monotonic_ns: int
    event_type: SchedulerEventType
    logical_group_id: Optional[str] = None
    attempt_id: Optional[str] = None
    admission_id: Optional[str] = None
    prompt_idx: Optional[int] = None
    task_name: Optional[str] = None
    source_prompt_id: Optional[str] = None
    repeated_prompt_cluster_id: Optional[str] = None
    source_pool_ordinal: Optional[int] = None
    dispatch_cohort: Optional[int] = None
    pool_id: Optional[str] = None
    pool_manifest_sha256: Optional[str] = None
    model_revision: Optional[str] = None
    model_weights_sha256: Optional[str] = None
    run_mode: Optional[str] = None
    sampler_name: Optional[str] = None
    sampler_fingerprint: Optional[str] = None
    trainer_version: Optional[int] = None
    sampler_dispatch_index: Optional[int] = None
    target_step: Optional[int] = None
    start_weight_version: Optional[int] = None
    end_weight_version: Optional[int] = None
    min_prompt_groups: Optional[int] = None
    max_prompt_groups: Optional[int] = None
    ready_prompt_groups: Optional[int] = None
    eligible_prompt_groups: Optional[int] = None
    selected_logical_group_ids: tuple[str, ...] = ()
    eligible_logical_group_ids: tuple[str, ...] = ()
    live_logical_group_ids: tuple[str, ...] = ()
    terminal_reason: Optional[str] = None
    failure_class: Optional[str] = None
    exception_class: Optional[str] = None
    scalar_summaries: Mapping[str, Scalar] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.scalar_summaries is None:
            object.__setattr__(self, "scalar_summaries", {})
        if self.schema_version != TRACE_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version={self.schema_version}")
        _identifier("trace_run_id", self.trace_run_id, required=True)
        _identifier("process_epoch", self.process_epoch, required=True)
        if self.event_seq < 0 or self.monotonic_ns < 0:
            raise ValueError("event sequence and monotonic time must be non-negative")
        if not isinstance(self.event_type, SchedulerEventType):
            raise TypeError("event_type must be SchedulerEventType")
        needs_attempt = self.event_type in _ATTEMPT_EVENTS
        _identifier("logical_group_id", self.logical_group_id, required=needs_attempt)
        _identifier("attempt_id", self.attempt_id, required=needs_attempt)
        for name in (
            "admission_id",
            "task_name",
            "source_prompt_id",
            "repeated_prompt_cluster_id",
            "pool_id",
            "pool_manifest_sha256",
            "model_revision",
            "model_weights_sha256",
            "run_mode",
            "sampler_name",
            "sampler_fingerprint",
            "terminal_reason",
            "failure_class",
            "exception_class",
        ):
            _identifier(name, getattr(self, name))
        for field_name in (
            "selected_logical_group_ids",
            "eligible_logical_group_ids",
            "live_logical_group_ids",
        ):
            for group_id in getattr(self, field_name):
                _identifier(field_name, group_id, required=True)
        _summaries(self.scalar_summaries)
        for name in (
            "prompt_idx",
            "source_pool_ordinal",
            "dispatch_cohort",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer")
        if self.event_type is SchedulerEventType.SELECT_DECISION:
            if self.min_prompt_groups is None or self.max_prompt_groups is None:
                raise ValueError("select_decision requires min/max prompt groups")
            if (
                self.min_prompt_groups < 1
                or self.max_prompt_groups < self.min_prompt_groups
            ):
                raise ValueError("invalid select_decision prompt-group bounds")
            if self.eligible_prompt_groups != len(self.eligible_logical_group_ids):
                raise ValueError("eligible count does not match eligible IDs")
        if self.event_type is SchedulerEventType.GROUP_EVICTED:
            _identifier("logical_group_id", self.logical_group_id, required=True)
        if self.event_type is SchedulerEventType.GROUP_ARCHIVED:
            if self.terminal_reason != "fixed_pool_archive":
                raise ValueError(
                    "group_archived requires terminal_reason=fixed_pool_archive"
                )
        if self.event_type is SchedulerEventType.ADMISSION_GRANTED:
            _identifier("admission_id", self.admission_id, required=True)
            expected = self.scalar_summaries.get("expected_prompt_groups")
            if (
                not isinstance(expected, int)
                or isinstance(expected, bool)
                or expected < 1
            ):
                raise ValueError(
                    "admission_granted requires positive integer expected_prompt_groups"
                )
        if self.event_type is SchedulerEventType.RUN_ENDED:
            _identifier("terminal_reason", self.terminal_reason, required=True)

    def to_record(self) -> dict[str, object]:
        record = asdict(self)
        record["event_type"] = self.event_type.value
        record["selected_logical_group_ids"] = list(self.selected_logical_group_ids)
        record["eligible_logical_group_ids"] = list(self.eligible_logical_group_ids)
        record["live_logical_group_ids"] = list(self.live_logical_group_ids)
        return {key: value for key, value in record.items() if value is not None}

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "SchedulerTraceEvent":
        unknown = set(record) - {field.name for field in fields(cls)}
        if unknown:
            raise SchedulerTraceValidationError(f"unknown fields: {sorted(unknown)}")
        values = dict(record)
        try:
            values["event_type"] = SchedulerEventType(values["event_type"])
            for name in (
                "selected_logical_group_ids",
                "eligible_logical_group_ids",
                "live_logical_group_ids",
            ):
                values[name] = tuple(values.get(name, ()))
            return cls(**values)  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError) as error:
            raise SchedulerTraceValidationError(str(error)) from error


class SchedulerTraceSink(Protocol):
    @property
    def enabled(self) -> bool: ...
    async def start(self) -> None: ...
    def emit(self, event_type: SchedulerEventType, **event_fields: object) -> None: ...
    async def close(self) -> None: ...


class NoopSchedulerTraceSink:
    """Disabled path: no file, directory, queue, or task."""

    @property
    def enabled(self) -> bool:
        return False

    async def start(self) -> None:
        return None

    def emit(self, event_type: SchedulerEventType, **event_fields: object) -> None:
        return None

    async def close(self) -> None:
        return None


_STOP: Final = object()


def _write_and_flush(handle, payloads: list[bytes]) -> None:
    handle.writelines(payloads)
    handle.flush()


class JsonlSchedulerTraceSink:
    """Single-writer bounded JSONL sink that fails rather than dropping events."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_queue_events: int = 4096,
        flush_every: int = 64,
        trace_run_id: Optional[str] = None,
        process_epoch: Optional[str] = None,
    ) -> None:
        if max_queue_events < 1 or flush_every < 1:
            raise ValueError("queue and flush bounds must be positive")
        self.path = Path(path)
        self.max_queue_events = max_queue_events
        self.flush_every = flush_every
        self.trace_run_id = trace_run_id or str(uuid.uuid4())
        self.process_epoch = process_epoch or str(uuid.uuid4())
        # One reserved slot guarantees shutdown never blocks behind event data.
        self._queue: asyncio.Queue[SchedulerTraceEvent | object] = asyncio.Queue(
            maxsize=max_queue_events + 1
        )
        self._task: Optional[asyncio.Task[None]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._next_seq = 0
        self._error: Optional[BaseException] = None
        self._closed = False

    @property
    def enabled(self) -> bool:
        return True

    async def start(self) -> None:
        if self._task is not None or self._closed:
            raise SchedulerTraceWriteError("trace sink is already started or closed")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = self.path.open("xb", buffering=0)
        except FileExistsError as error:
            raise SchedulerTraceWriteError(
                f"refusing to append to {self.path}"
            ) from error
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.create_task(self._writer(handle), name="scheduler-trace")

    def emit(self, event_type: SchedulerEventType, **event_fields: object) -> None:
        if self._task is None or self._loop is None or self._closed:
            raise SchedulerTraceWriteError("trace sink is not active")
        if asyncio.get_running_loop() is not self._loop:
            raise SchedulerTraceWriteError("events must use the owning controller loop")
        self._raise_error()
        event = SchedulerTraceEvent(
            schema_version=TRACE_SCHEMA_VERSION,
            trace_run_id=self.trace_run_id,
            process_epoch=self.process_epoch,
            event_seq=self._next_seq,
            monotonic_ns=time.monotonic_ns(),
            event_type=event_type,
            **event_fields,  # type: ignore[arg-type]
        )
        try:
            if self._queue.qsize() >= self.max_queue_events:
                raise asyncio.QueueFull
            self._queue.put_nowait(event)
        except asyncio.QueueFull as error:
            raise SchedulerTraceWriteError("trace queue overflow") from error
        self._next_seq += 1

    async def close(self) -> None:
        if self._closed:
            self._raise_error()
            return
        self._closed = True
        if self._task is None:
            return
        self._raise_error()
        self._queue.put_nowait(_STOP)
        try:
            await self._task
        finally:
            self._task = None
        self._raise_error()

    def _raise_error(self) -> None:
        if self._error is not None:
            raise SchedulerTraceWriteError("trace writer failed") from self._error

    async def _writer(self, handle) -> None:
        batch: list[bytes] = []
        try:
            while True:
                item = await self._queue.get()
                if item is _STOP:
                    if batch:
                        await asyncio.to_thread(_write_and_flush, handle, batch)
                    return
                assert isinstance(item, SchedulerTraceEvent)
                batch.append(
                    json.dumps(
                        item.to_record(),
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode()
                    + b"\n"
                )
                if len(batch) >= self.flush_every:
                    await asyncio.to_thread(_write_and_flush, handle, batch)
                    batch = []
        except BaseException as error:
            self._error = error
            raise
        finally:
            await asyncio.to_thread(handle.close)


@dataclass(frozen=True, slots=True)
class SchedulerTraceValidationReport:
    events: int
    truncated_final_record: bool
    incomplete_attempt_ids: tuple[str, ...]
    administratively_censored_group_ids: tuple[str, ...]


def iter_scheduler_trace(
    path: str | Path, *, tolerate_truncated_final_record: bool = False
) -> Iterator[SchedulerTraceEvent]:
    trace_path = Path(path)
    size = trace_path.stat().st_size
    prior_seq: Optional[int] = None
    prior_clock: dict[str, int] = {}
    trace_run_id: Optional[str] = None
    process_epoch: Optional[str] = None
    with trace_path.open("rb") as handle:
        while line := handle.readline():
            offset = handle.tell() - len(line)
            if not line.endswith(b"\n"):
                if tolerate_truncated_final_record and handle.tell() == size:
                    break
                raise SchedulerTraceValidationError(
                    f"unterminated record at byte {offset}"
                )
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise SchedulerTraceValidationError(
                    f"invalid JSON at byte {offset}"
                ) from error
            if not isinstance(record, dict):
                raise SchedulerTraceValidationError(
                    f"non-object record at byte {offset}"
                )
            event = SchedulerTraceEvent.from_record(record)
            if prior_seq is None and event.event_seq != 0:
                raise SchedulerTraceValidationError("event sequence must start at zero")
            if prior_seq is not None and event.event_seq != prior_seq + 1:
                raise SchedulerTraceValidationError("non-contiguous event sequence")
            prior_seq = event.event_seq
            if trace_run_id is None:
                trace_run_id, process_epoch = event.trace_run_id, event.process_epoch
            elif (event.trace_run_id, event.process_epoch) != (
                trace_run_id,
                process_epoch,
            ):
                raise SchedulerTraceValidationError(
                    "trace contains multiple run identities"
                )
            previous = prior_clock.get(event.process_epoch)
            if previous is not None and event.monotonic_ns < previous:
                raise SchedulerTraceValidationError(
                    "clock moved backwards within an epoch"
                )
            prior_clock[event.process_epoch] = event.monotonic_ns
            yield event


def validate_scheduler_trace(
    path: str | Path,
    *,
    require_complete_attempts: bool = True,
    tolerate_truncated_final_record: bool = False,
) -> SchedulerTraceValidationReport:
    events = list(
        iter_scheduler_trace(
            path,
            tolerate_truncated_final_record=tolerate_truncated_final_record,
        )
    )
    if not events or events[0].event_type is not SchedulerEventType.RUN_STARTED:
        raise SchedulerTraceValidationError("trace must begin with run_started")
    if events[-1].event_type is not SchedulerEventType.RUN_ENDED:
        raise SchedulerTraceValidationError("trace must end with run_ended")
    if (
        sum(e.event_type is SchedulerEventType.RUN_STARTED for e in events) != 1
        or sum(e.event_type is SchedulerEventType.RUN_ENDED for e in events) != 1
    ):
        raise SchedulerTraceValidationError(
            "trace requires exactly one run boundary pair"
        )

    dispatched: dict[str, str] = {}
    attempt_identity: dict[str, tuple[object, ...]] = {}
    group_attempt: dict[str, str] = {}
    admissions: dict[str, int] = {}
    admission_dispatches: dict[str, int] = {}
    live_dispatched_groups: set[str] = set()
    attempt_state: dict[str, str] = {}
    ready_groups: set[str] = set()
    terminal_groups: set[str] = set()
    for event in events:
        if event.event_type is SchedulerEventType.ADMISSION_GRANTED:
            assert event.admission_id is not None
            if event.admission_id in admissions:
                raise SchedulerTraceValidationError("duplicate admission ID")
            admissions[event.admission_id] = int(
                event.scalar_summaries["expected_prompt_groups"]
            )
            admission_dispatches[event.admission_id] = 0
        elif event.event_type is SchedulerEventType.ATTEMPT_DISPATCHED:
            assert event.attempt_id is not None and event.logical_group_id is not None
            if event.admission_id not in admissions:
                raise SchedulerTraceValidationError("attempt has no prior admission")
            if event.attempt_id in dispatched:
                raise SchedulerTraceValidationError("duplicate attempt dispatch")
            if event.logical_group_id in live_dispatched_groups | terminal_groups:
                raise SchedulerTraceValidationError("logical group ID was reused")
            assert event.admission_id is not None
            admission_dispatches[event.admission_id] += 1
            live_dispatched_groups.add(event.logical_group_id)
            group_attempt[event.logical_group_id] = event.attempt_id
            dispatched[event.attempt_id] = event.logical_group_id
            attempt_state[event.attempt_id] = "dispatched"
            attempt_identity[event.attempt_id] = (
                event.admission_id,
                event.prompt_idx,
                event.task_name,
                event.source_prompt_id,
                event.repeated_prompt_cluster_id,
                event.source_pool_ordinal,
                event.dispatch_cohort,
            )
        elif event.event_type in _ATTEMPT_EVENTS - {
            SchedulerEventType.ATTEMPT_DISPATCHED
        }:
            assert event.attempt_id is not None and event.logical_group_id is not None
            if dispatched.get(event.attempt_id) != event.logical_group_id:
                raise SchedulerTraceValidationError("attempt/group identity mismatch")
            if attempt_identity[event.attempt_id] != (
                event.admission_id,
                event.prompt_idx,
                event.task_name,
                event.source_prompt_id,
                event.repeated_prompt_cluster_id,
                event.source_pool_ordinal,
                event.dispatch_cohort,
            ):
                raise SchedulerTraceValidationError(
                    "attempt source identity changed across lifecycle events"
                )
            state = attempt_state.get(event.attempt_id)
            if event.event_type is SchedulerEventType.ROLLOUT_COMPLETED:
                if state != "dispatched":
                    raise SchedulerTraceValidationError(
                        "illegal rollout completion order"
                    )
                attempt_state[event.attempt_id] = "completed"
            elif event.event_type is SchedulerEventType.GROUP_READY:
                if state != "completed":
                    raise SchedulerTraceValidationError("group ready before completion")
                attempt_state[event.attempt_id] = "ready"
                ready_groups.add(event.logical_group_id)
            elif event.event_type is SchedulerEventType.GROUP_ARCHIVED:
                if state != "ready" or event.logical_group_id not in ready_groups:
                    raise SchedulerTraceValidationError(
                        "group archived before becoming live and ready"
                    )
                attempt_state[event.attempt_id] = "archived"
                ready_groups.remove(event.logical_group_id)
                live_dispatched_groups.discard(event.logical_group_id)
                terminal_groups.add(event.logical_group_id)
            elif event.event_type is SchedulerEventType.ATTEMPT_FAILED:
                if state not in {"dispatched", "completed"}:
                    raise SchedulerTraceValidationError("illegal attempt failure order")
                attempt_state[event.attempt_id] = "failed"
            elif event.event_type is SchedulerEventType.ATTEMPT_REMOVED:
                if state not in {"dispatched", "completed", "failed"}:
                    raise SchedulerTraceValidationError("illegal attempt removal order")
                attempt_state[event.attempt_id] = "removed"
                live_dispatched_groups.discard(event.logical_group_id)
                terminal_groups.add(event.logical_group_id)
            elif event.event_type is SchedulerEventType.ABORT_REQUESTED:
                raise SchedulerTraceValidationError(
                    "abort_requested is unsupported by this pinned runtime"
                )
        elif event.event_type is SchedulerEventType.SELECT_DECISION:
            eligible = set(event.eligible_logical_group_ids)
            if not eligible <= ready_groups or eligible & terminal_groups:
                raise SchedulerTraceValidationError(
                    "eligible IDs are not live ready groups"
                )
            if not set(event.selected_logical_group_ids) <= eligible:
                raise SchedulerTraceValidationError("selected IDs were not eligible")
            selected_count = len(event.selected_logical_group_ids)
            if selected_count > (event.max_prompt_groups or 0) or (
                selected_count and selected_count < (event.min_prompt_groups or 0)
            ):
                raise SchedulerTraceValidationError("selected count violates bounds")
            if event.ready_prompt_groups != len(ready_groups):
                raise SchedulerTraceValidationError(
                    "ready count does not match lifecycle"
                )
            reported_selected = event.scalar_summaries.get("selected_prompt_groups")
            if reported_selected != selected_count:
                raise SchedulerTraceValidationError("selected count summary mismatch")
            for group_id in event.selected_logical_group_ids:
                if group_id in terminal_groups:
                    raise SchedulerTraceValidationError("group selected more than once")
                terminal_groups.add(group_id)
                ready_groups.discard(group_id)
                live_dispatched_groups.discard(group_id)
                attempt_state[group_attempt[group_id]] = "selected"
        elif event.event_type is SchedulerEventType.GROUP_EVICTED:
            assert event.logical_group_id is not None
            if event.logical_group_id not in ready_groups:
                raise SchedulerTraceValidationError(
                    "evicted group was not live and ready"
                )
            ready_groups.remove(event.logical_group_id)
            live_dispatched_groups.discard(event.logical_group_id)
            terminal_groups.add(event.logical_group_id)
    incomplete_admissions = {
        admission_id: (admission_dispatches[admission_id], expected)
        for admission_id, expected in admissions.items()
        if admission_dispatches[admission_id] != expected
    }
    if incomplete_admissions:
        raise SchedulerTraceValidationError(
            f"admission dispatch accounting mismatch: {incomplete_admissions}"
        )
    incomplete = tuple(
        sorted(
            attempt_id
            for attempt_id, state in attempt_state.items()
            if state not in {"ready", "removed", "archived", "selected"}
        )
    )
    if require_complete_attempts and incomplete:
        raise SchedulerTraceValidationError(
            f"attempts lack terminal events: {list(incomplete)}"
        )
    raw = Path(path).read_bytes()
    administrative = tuple(events[-1].live_logical_group_ids)
    if set(administrative) != ready_groups:
        raise SchedulerTraceValidationError(
            "run-end administrative-censoring IDs do not match live groups"
        )
    return SchedulerTraceValidationReport(
        events=len(events),
        truncated_final_record=bool(raw and not raw.endswith(b"\n")),
        incomplete_attempt_ids=incomplete,
        administratively_censored_group_ids=administrative,
    )
