# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fixed-event-stream scheduler resimulation for the DAPO diagnostic."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Mapping, Sequence, TypeAlias


SamplerName: TypeAlias = Literal["in_order", "ready_first"]
EXPECTED_GROUPS = 32
GROUPS_PER_STEP = 4
EXPECTED_STEPS = 8


class CommonInputResimulationError(RuntimeError):
    """A frozen input or simulated scheduler lifecycle is invalid."""


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """The strict subset of a frozen scheduler event required here."""

    event_seq: int
    monotonic_ns: int
    event_type: str
    logical_group_id: str | None
    source_prompt_id: str | None
    source_pool_ordinal: int | None
    dispatch_cohort: int | None
    selected_logical_group_ids: tuple[str, ...]


def iter_trace(path: str | Path) -> Iterator[TraceEvent]:
    """Read a contiguous single-clock scheduler trace without heavy imports."""
    prior_seq: int | None = None
    prior_clock: int | None = None
    trace_identity: tuple[object, object] | None = None
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise CommonInputResimulationError(
                    f"unterminated trace record at line {line_number}"
                )
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise CommonInputResimulationError(
                    f"invalid trace JSON at line {line_number}"
                ) from error
            if not isinstance(record, dict):
                raise CommonInputResimulationError(
                    f"trace record must be an object at line {line_number}"
                )
            event_seq = record.get("event_seq")
            monotonic_ns = record.get("monotonic_ns")
            event_type = record.get("event_type")
            if (
                not isinstance(event_seq, int)
                or isinstance(event_seq, bool)
                or not isinstance(monotonic_ns, int)
                or isinstance(monotonic_ns, bool)
                or not isinstance(event_type, str)
            ):
                raise CommonInputResimulationError(
                    f"invalid trace event header at line {line_number}"
                )
            if event_seq != (0 if prior_seq is None else prior_seq + 1):
                raise CommonInputResimulationError(
                    "non-contiguous trace event sequence"
                )
            if prior_clock is not None and monotonic_ns < prior_clock:
                raise CommonInputResimulationError("trace clock moved backwards")
            identity = (record.get("trace_run_id"), record.get("process_epoch"))
            if trace_identity is None:
                trace_identity = identity
            elif identity != trace_identity:
                raise CommonInputResimulationError("trace identity changed")
            selected = record.get("selected_logical_group_ids", [])
            if not isinstance(selected, list) or not all(
                isinstance(value, str) and value for value in selected
            ):
                raise CommonInputResimulationError("invalid selected group IDs")
            yield TraceEvent(
                event_seq=event_seq,
                monotonic_ns=monotonic_ns,
                event_type=event_type,
                logical_group_id=(
                    str(record["logical_group_id"])
                    if isinstance(record.get("logical_group_id"), str)
                    else None
                ),
                source_prompt_id=(
                    str(record["source_prompt_id"])
                    if isinstance(record.get("source_prompt_id"), str)
                    else None
                ),
                source_pool_ordinal=(
                    int(record["source_pool_ordinal"])
                    if isinstance(record.get("source_pool_ordinal"), int)
                    and not isinstance(record.get("source_pool_ordinal"), bool)
                    else None
                ),
                dispatch_cohort=(
                    int(record["dispatch_cohort"])
                    if isinstance(record.get("dispatch_cohort"), int)
                    and not isinstance(record.get("dispatch_cohort"), bool)
                    else None
                ),
                selected_logical_group_ids=tuple(selected),
            )
            prior_seq = event_seq
            prior_clock = monotonic_ns


@dataclass(frozen=True, slots=True)
class FixedStreamGroup:
    """One prompt group and its fixed trace-local lifecycle times."""

    logical_group_id: str
    source_prompt_id: str
    source_pool_ordinal: int
    dispatch_cohort: int
    dispatch_order: int
    dispatch_ns: int
    rollout_completed_ns: int
    observed_ready_ns: int


@dataclass(frozen=True, slots=True)
class SelectionRecord:
    """One nonempty exact scheduler selection."""

    step: int
    selected_group_ids: tuple[str, ...]
    ready_groups: int
    eligible_groups: int


@dataclass(frozen=True, slots=True)
class FixedStreamResult:
    """Complete selection order for one sampler over one fixed stream."""

    sampler: SamplerName
    selections: tuple[SelectionRecord, ...]
    source_prompt_step: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Aggregate-only paired sampler comparison."""

    lower_load_mean_normalized_promotion: float
    easier_mean_normalized_promotion: float
    lower_load_net_step_displacement: int
    lower_load_absolute_step_displacement: int
    changed_selected_step_sets: int
    changed_prompt_selection_steps: int
    in_order_ready_surplus_fraction: float


def _required_string(event: TraceEvent, field_name: str) -> str:
    value = getattr(event, field_name)
    if not isinstance(value, str) or not value:
        raise CommonInputResimulationError(
            f"{event.event_type} requires nonempty {field_name}"
        )
    return value


def _required_int(event: TraceEvent, field_name: str) -> int:
    value = getattr(event, field_name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CommonInputResimulationError(
            f"{event.event_type} requires nonnegative integer {field_name}"
        )
    return value


def load_fixed_stream(path: str | Path) -> tuple[FixedStreamGroup, ...]:
    """Load exactly 32 complete prompt lifecycles from one scheduler trace."""
    events = tuple(iter_trace(path))
    dispatched = {
        _required_string(event, "logical_group_id"): event
        for event in events
        if event.event_type == "attempt_dispatched"
    }
    completed = {
        _required_string(event, "logical_group_id"): event
        for event in events
        if event.event_type == "rollout_completed"
    }
    ready = {
        _required_string(event, "logical_group_id"): event
        for event in events
        if event.event_type == "group_ready"
    }
    if not (
        len(dispatched) == len(completed) == len(ready) == EXPECTED_GROUPS
        and set(dispatched) == set(completed) == set(ready)
    ):
        raise CommonInputResimulationError(
            "trace requires 32 exact dispatch/completion/ready lifecycles"
        )
    groups: list[FixedStreamGroup] = []
    for group_id, dispatch in sorted(
        dispatched.items(), key=lambda item: item[1].event_seq
    ):
        completion = completed[group_id]
        ready_event = ready[group_id]
        if not (
            dispatch.monotonic_ns < completion.monotonic_ns <= ready_event.monotonic_ns
        ):
            raise CommonInputResimulationError("non-monotonic prompt lifecycle")
        source_prompt_id = _required_string(dispatch, "source_prompt_id")
        if (
            completion.source_prompt_id != source_prompt_id
            or ready_event.source_prompt_id != source_prompt_id
        ):
            raise CommonInputResimulationError(
                "prompt identity changed across lifecycle"
            )
        source_pool_ordinal = _required_int(dispatch, "source_pool_ordinal")
        groups.append(
            FixedStreamGroup(
                logical_group_id=group_id,
                source_prompt_id=source_prompt_id,
                source_pool_ordinal=source_pool_ordinal,
                dispatch_cohort=_required_int(dispatch, "dispatch_cohort"),
                dispatch_order=source_pool_ordinal,
                dispatch_ns=dispatch.monotonic_ns,
                rollout_completed_ns=completion.monotonic_ns,
                observed_ready_ns=ready_event.monotonic_ns,
            )
        )
    if (
        len({group.source_prompt_id for group in groups}) != EXPECTED_GROUPS
        or len({group.source_pool_ordinal for group in groups}) != EXPECTED_GROUPS
        or {group.dispatch_cohort for group in groups} != set(range(EXPECTED_STEPS))
        or any(
            sum(group.dispatch_cohort == cohort for group in groups) != GROUPS_PER_STEP
            for cohort in range(EXPECTED_STEPS)
        )
    ):
        raise CommonInputResimulationError("fixed stream identity or cohort mismatch")
    return tuple(groups)


def observed_nonempty_selections(
    path: str | Path,
) -> tuple[tuple[str, ...], ...]:
    """Read the exact native nonempty selected-group sequence."""
    selected = tuple(
        event.selected_logical_group_ids
        for event in iter_trace(path)
        if event.event_type == "select_decision" and event.selected_logical_group_ids
    )
    if len(selected) != EXPECTED_STEPS or any(
        len(step) != GROUPS_PER_STEP for step in selected
    ):
        raise CommonInputResimulationError("native trace has nonexact selections")
    return selected


def observed_selection_tick_times(path: str | Path) -> tuple[int, ...]:
    """Read every native selection opportunity, including empty polls."""
    ticks = tuple(
        event.monotonic_ns
        for event in iter_trace(path)
        if event.event_type == "select_decision"
    )
    if len(ticks) < EXPECTED_STEPS:
        raise CommonInputResimulationError("native trace has too few selection ticks")
    return ticks


def trace_start_ns(path: str | Path) -> int:
    """Return the trace-local run-start time."""
    first = next(iter_trace(path), None)
    if first is None or first.event_type != "run_started":
        raise CommonInputResimulationError("trace does not start with run_started")
    return first.monotonic_ns


def shifted_ready_times(
    groups: Sequence[FixedStreamGroup],
    *,
    delayed_prompt_ids: frozenset[str],
    delay_seconds: float,
) -> dict[str, int]:
    """Build completion-plus-delay virtual readiness times."""
    if delay_seconds < 0:
        raise ValueError("delay_seconds must be nonnegative")
    observed_prompt_ids = {group.source_prompt_id for group in groups}
    if not delayed_prompt_ids <= observed_prompt_ids:
        raise CommonInputResimulationError("delay target is not in fixed stream")
    delay_ns = round(delay_seconds * 1_000_000_000)
    return {
        group.logical_group_id: group.rollout_completed_ns
        + (delay_ns if group.source_prompt_id in delayed_prompt_ids else 0)
        for group in groups
    }


def observed_ready_times(
    groups: Sequence[FixedStreamGroup],
) -> dict[str, int]:
    """Return the native observed readiness time for every group."""
    return {group.logical_group_id: group.observed_ready_ns for group in groups}


def simulate_fixed_stream(
    groups: Sequence[FixedStreamGroup],
    *,
    sampler: SamplerName,
    ready_ns_by_group: Mapping[str, int],
    cohort_restricted_ready_first: bool = False,
    selection_tick_ns: Sequence[int] | None = None,
) -> FixedStreamResult:
    """Run exact-batch selection over a fixed dispatch/readiness event stream."""
    if len(groups) != EXPECTED_GROUPS:
        raise CommonInputResimulationError("simulation requires exactly 32 groups")
    group_by_id = {group.logical_group_id: group for group in groups}
    if set(group_by_id) != set(ready_ns_by_group):
        raise CommonInputResimulationError("readiness mapping does not match stream")
    events: list[tuple[int, int, int, str]] = []
    for group in groups:
        ready_ns = ready_ns_by_group[group.logical_group_id]
        if ready_ns < group.dispatch_ns:
            raise CommonInputResimulationError("group became ready before dispatch")
        events.append(
            (group.dispatch_ns, 0, group.dispatch_order, group.logical_group_id)
        )
        events.append((ready_ns, 1, group.dispatch_order, group.logical_group_id))
    if selection_tick_ns is not None:
        if any(
            left > right
            for left, right in zip(selection_tick_ns, selection_tick_ns[1:])
        ):
            raise CommonInputResimulationError("selection ticks moved backwards")
        events.extend(
            (tick_ns, 2, tick_index, "")
            for tick_index, tick_ns in enumerate(selection_tick_ns)
        )
    events.sort()

    buffer_order: list[str] = []
    ready_ids: set[str] = set()
    selected_ids: set[str] = set()
    selections: list[SelectionRecord] = []
    current_step = 0

    def drain_selectable() -> None:
        nonlocal current_step
        while current_step < EXPECTED_STEPS:
            ready_unselected = sorted(
                (
                    candidate
                    for candidate in buffer_order
                    if candidate in ready_ids and candidate not in selected_ids
                ),
                key=lambda candidate: group_by_id[candidate].dispatch_order,
            )
            if sampler == "in_order" or cohort_restricted_ready_first:
                eligible = [
                    candidate
                    for candidate in ready_unselected
                    if group_by_id[candidate].dispatch_cohort == current_step
                ]
            else:
                eligible = ready_unselected
            if len(eligible) < GROUPS_PER_STEP:
                return
            selected = tuple(eligible[:GROUPS_PER_STEP])
            selections.append(
                SelectionRecord(
                    step=current_step,
                    selected_group_ids=selected,
                    ready_groups=len(ready_unselected),
                    eligible_groups=len(eligible),
                )
            )
            selected_ids.update(selected)
            current_step += 1

    for _, event_kind, _, group_id in events:
        if event_kind == 0:
            buffer_order.append(group_id)
            continue
        if event_kind == 1:
            ready_ids.add(group_id)
            if selection_tick_ns is None:
                drain_selectable()
            continue
        drain_selectable()

    if (
        current_step != EXPECTED_STEPS
        or len(selected_ids) != EXPECTED_GROUPS
        or len(selections) != EXPECTED_STEPS
    ):
        raise CommonInputResimulationError(
            f"incomplete selection: steps={current_step}, groups={len(selected_ids)}"
        )
    source_prompt_step = {
        group_by_id[group_id].source_prompt_id: selection.step
        for selection in selections
        for group_id in selection.selected_group_ids
    }
    if len(source_prompt_step) != EXPECTED_GROUPS:
        raise CommonInputResimulationError("source prompt selection is not one-to-one")
    return FixedStreamResult(
        sampler=sampler,
        selections=tuple(selections),
        source_prompt_step=source_prompt_step,
    )


def compare_samplers(
    in_order: FixedStreamResult,
    ready_first: FixedStreamResult,
    *,
    lower_load_prompt_ids: frozenset[str],
    easier_prompt_ids: frozenset[str],
) -> ComparisonResult:
    """Calculate the frozen aggregate paired-sampler diagnostics."""
    if (
        in_order.sampler != "in_order"
        or ready_first.sampler != "ready_first"
        or set(in_order.source_prompt_step) != set(ready_first.source_prompt_step)
    ):
        raise CommonInputResimulationError("paired sampler result mismatch")
    observed = set(in_order.source_prompt_step)
    if (
        len(lower_load_prompt_ids) != 16
        or len(easier_prompt_ids) != 16
        or not lower_load_prompt_ids <= observed
        or not easier_prompt_ids <= observed
    ):
        raise CommonInputResimulationError("fixed reference halves mismatch")
    displacement = {
        prompt_id: in_order.source_prompt_step[prompt_id]
        - ready_first.source_prompt_step[prompt_id]
        for prompt_id in observed
    }

    def promotion(prompt_ids: frozenset[str]) -> float:
        return sum(displacement[prompt_id] / 7 for prompt_id in prompt_ids) / len(
            prompt_ids
        )

    lower_values = [displacement[prompt_id] for prompt_id in lower_load_prompt_ids]
    return ComparisonResult(
        lower_load_mean_normalized_promotion=promotion(lower_load_prompt_ids),
        easier_mean_normalized_promotion=promotion(easier_prompt_ids),
        lower_load_net_step_displacement=sum(lower_values),
        lower_load_absolute_step_displacement=sum(abs(value) for value in lower_values),
        changed_selected_step_sets=sum(
            set(left.selected_group_ids) != set(right.selected_group_ids)
            for left, right in zip(
                in_order.selections, ready_first.selections, strict=True
            )
        ),
        changed_prompt_selection_steps=sum(
            value != 0 for value in displacement.values()
        ),
        in_order_ready_surplus_fraction=sum(
            selection.ready_groups > selection.eligible_groups
            for selection in in_order.selections
        )
        / EXPECTED_STEPS,
    )


def comparison_to_dict(result: ComparisonResult) -> dict[str, int | float]:
    """Serialize a privacy-safe paired comparison."""
    return {
        "lower_load_mean_normalized_promotion": (
            result.lower_load_mean_normalized_promotion
        ),
        "easier_mean_normalized_promotion": result.easier_mean_normalized_promotion,
        "lower_load_net_step_displacement": result.lower_load_net_step_displacement,
        "lower_load_absolute_step_displacement": (
            result.lower_load_absolute_step_displacement
        ),
        "changed_selected_step_sets": result.changed_selected_step_sets,
        "changed_prompt_selection_steps": result.changed_prompt_selection_steps,
        "in_order_ready_surplus_fraction": result.in_order_ready_surplus_fraction,
    }
