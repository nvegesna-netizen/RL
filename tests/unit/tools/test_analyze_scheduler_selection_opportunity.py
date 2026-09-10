# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json

from nemo_rl.algorithms.async_utils.scheduler_trace import (
    SchedulerEventType,
    SchedulerTraceEvent,
)
from tools import analyze_scheduler_selection_opportunity as analyzer


def _event(sequence: int, kind: SchedulerEventType, **fields):
    return SchedulerTraceEvent(
        schema_version=1,
        trace_run_id="run",
        process_epoch="epoch",
        event_seq=sequence,
        monotonic_ns=sequence,
        event_type=kind,
        **fields,
    )


def _events(order: tuple[str, ...]):
    events = [
        _event(
            0,
            SchedulerEventType.RUN_STARTED,
            run_mode="scheduler_assay",
        )
    ]
    sequence = 1
    source_ids = tuple(f"private-{item}" for item in "abcd")
    group_ids = {source_id: f"group-{source_id}" for source_id in source_ids}
    for cohort, source_id in enumerate(source_ids):
        group_id = group_ids[source_id]
        events.extend(
            (
                _event(
                    sequence,
                    SchedulerEventType.ATTEMPT_DISPATCHED,
                    logical_group_id=group_id,
                    attempt_id=f"attempt-{source_id}",
                    source_prompt_id=source_id,
                    dispatch_cohort=cohort,
                ),
                _event(
                    sequence + 1,
                    SchedulerEventType.ROLLOUT_COMPLETED,
                    logical_group_id=group_id,
                    attempt_id=f"attempt-{source_id}",
                    source_prompt_id=source_id,
                    dispatch_cohort=cohort,
                    scalar_summaries={"mean_gen_tokens_per_sample": cohort + 1},
                ),
            )
        )
        sequence += 2
    for step in range(2):
        selected = order[step * 2 : step * 2 + 2]
        eligible = order[step * 2 :] if step == 0 else selected
        events.append(
            _event(
                sequence,
                SchedulerEventType.SELECT_DECISION,
                min_prompt_groups=2,
                max_prompt_groups=2,
                ready_prompt_groups=len(eligible),
                eligible_prompt_groups=len(eligible),
                eligible_logical_group_ids=tuple(group_ids[item] for item in eligible),
                selected_logical_group_ids=tuple(group_ids[item] for item in selected),
                scalar_summaries={
                    "scheduler_assay_step": step,
                    "buffered_prompt_groups": len(eligible),
                    "selected_prompt_groups": 2,
                },
            )
        )
        sequence += 1
    events.append(
        _event(
            sequence,
            SchedulerEventType.RUN_ENDED,
            terminal_reason="scheduler_assay_complete",
            scalar_summaries={
                "completed_train_steps": 0,
                "final_physical_weight_version": 0,
            },
        )
    )
    return tuple(events)


def test_comparison_reports_choice_and_displacement_without_identities() -> None:
    ready = analyzer.analyze_trace_events(
        _events(("private-b", "private-c", "private-a", "private-d")),
        trace_sha256="1" * 64,
    )
    ordered = analyzer.analyze_trace_events(
        _events(("private-a", "private-b", "private-c", "private-d")),
        trace_sha256="2" * 64,
    )

    result = analyzer.analyze_comparison(
        comparison_id="fixture",
        ready_first=ready,
        in_order=ordered,
        reference_summary_key="mean_gen_tokens_per_sample",
    )

    assert (
        result["arms"]["ready_first"]["decisions_with_eligible_greater_than_selected"]
        == 1
    )
    assert result["arms"]["ready_first"]["eligible_to_selected_ratio"] == {
        "minimum": 1.0,
        "median": 1.5,
        "maximum": 2.0,
    }
    assert result["arms"]["ready_first"]["total_ready_but_ineligible_groups"] == 0
    assert result["cross_arm"]["differing_global_selection_positions"] == 3
    assert result["cross_arm"]["same_selected_step_sets"] == 0
    assert result["cross_arm"]["groups_crossing_selection_step_boundaries"] == 2
    assert result["cross_arm"][
        "groups_crossing_selection_step_boundaries_by_dispatch_cohort"
    ] == {"0": 1, "2": 1}
    assert result["cross_arm"]["absolute_rank_displacement"] == 4
    assert result["cross_arm"]["lower_reference_net_rank_promotion"] == -1
    serialized = json.dumps(result)
    assert all(f"private-{item}" not in serialized for item in "abcd")
    assert all(f"group-private-{item}" not in serialized for item in "abcd")


def test_trace_rejects_nonzero_learner_updates() -> None:
    events = list(_events(tuple(f"private-{item}" for item in "abcd")))
    ended = events[-1]
    events[-1] = _event(
        ended.event_seq,
        SchedulerEventType.RUN_ENDED,
        terminal_reason="scheduler_assay_complete",
        scalar_summaries={
            "completed_train_steps": 1,
            "final_physical_weight_version": 0,
        },
    )

    try:
        analyzer.analyze_trace_events(tuple(events), trace_sha256="3" * 64)
    except analyzer.SelectionOpportunityAnalysisError as error:
        assert "learner update" in str(error)
    else:
        raise AssertionError("nonzero learner updates must fail closed")
