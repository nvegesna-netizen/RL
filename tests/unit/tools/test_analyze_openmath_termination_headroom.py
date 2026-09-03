# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
from dataclasses import replace

from nemo_rl.algorithms.async_utils.scheduler_trace import (
    SchedulerEventType,
    SchedulerTraceEvent,
)
from tools.analyze_openmath_latency_feasibility import DesignItem, GroupObservation
from tools.analyze_openmath_termination_headroom import (
    FeasibilityAnalysisError,
    TerminationDiagnostics,
    _runtime,
    analyze_observations,
    observations_from_events,
)


def _runtime_event(*, context_length: int = 1024) -> SchedulerTraceEvent:
    return SchedulerTraceEvent(
        schema_version=1,
        trace_run_id="run",
        process_epoch="epoch",
        event_seq=0,
        monotonic_ns=1,
        event_type=SchedulerEventType.RUN_STARTED,
        scalar_summaries={
            "generation_backend": "vllm",
            "max_total_sequence_length": context_length,
            "configured_max_new_tokens": context_length,
            "generation_context_length": context_length,
            "tokenizer_eos_token_present": True,
            "tokenizer_eos_token_id": 151643,
            "effective_stop_token_count": 1,
            "effective_stop_token_ids_sha256": (
                "7e1aeac81a03df7990cb619bb968771c93cb562129636439c0cae6159040dbc7"
            ),
            "effective_single_stop_token_id": 151643,
            "effective_stop_string_count": 0,
            "effective_stop_strings_sha256": (
                "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
            ),
            "vllm_skip_tokenizer_init": True,
            "generation_ignore_eos": False,
            "generation_temperature": 1.0,
            "generation_top_p": 1.0,
            "generation_top_k": -1,
            "generation_use_async_rollouts": True,
            "generation_study_seed": 52001,
            "generation_speculative_config_sha256": (
                "74234e98afe7498fb5daf1f36ac2d78acc339464f950703b8c019892f982b90b"
            ),
            "policy_tokenizer_name_sha256": "c" * 64,
            "policy_model_name_sha256": "d" * 64,
            "grpo_seed": 20260901,
            "num_generations_per_prompt": 2,
            "max_rollout_turns": 1,
            "num_prompts_per_step": 4,
            "max_inflight_prompts": 4,
            "max_buffered_rollouts": 8,
            "vllm_include_stop_str_in_output": True,
            "fixed_pool_design_id": "openmath_termination_headroom_v1",
            "finish_reason_code_schema_version": 1,
        },
    )


def _observations() -> tuple[GroupObservation, ...]:
    output = []
    for pair in range(8):
        for stratum in ("short", "long"):
            ordinal = pair * 2 + (stratum == "long")
            is_long = stratum == "long"
            dispatch = (ordinal // 4) * 1_000 + (ordinal % 4) * 10
            latency = 150 if is_long else 100
            tokens = 160 if is_long else 100
            item = DesignItem(
                source_pool_ordinal=ordinal,
                source_prompt_id=f"prompt-{ordinal}",
                task_name=f"openmath_{stratum}",
                pair_id=f"pair-{pair}",
                stratum=stratum,
                problem_source="math",
                rendered_prompt_tokens=100,
                reference_solution_tokens=320 if is_long else 80,
                reference_answer_token_count=10,
            )
            output.append(
                GroupObservation(
                    item=item,
                    dispatch_ns=dispatch,
                    completed_ns=dispatch + latency - 10,
                    ready_ns=dispatch + latency,
                    archived_ns=dispatch + latency + 1,
                    mean_generated_tokens=tokens,
                    min_generated_tokens=tokens,
                    max_generated_tokens=tokens,
                    cap_hits=0,
                    natural_terminations=2,
                    effective_output_cap=924,
                )
            )
    return tuple(output)


def _diagnostics(*, length_finishes: int = 0) -> dict[int, TerminationDiagnostics]:
    return {
        ordinal: TerminationDiagnostics(
            finish_available=2,
            stop_finishes=2 - length_finishes,
            length_finishes=length_finishes,
            abort_finishes=0,
            other_finishes=0,
            context_exhausted=0,
            token_stop_reasons=0,
            eos_matching_stop_reasons=0,
            cap_hits=length_finishes,
            near_cap=length_finishes,
        )
        for ordinal in range(16)
    }


def test_runtime_accepts_only_the_separate_1024_contract() -> None:
    assert _runtime((_runtime_event(),))["tokenizer_eos_token_id"] == 151643

    with pytest.raises(FeasibilityAnalysisError, match="max_total_sequence_length"):
        _runtime((_runtime_event(context_length=512),))


def test_clean_headroom_and_proxy_signal_pass_to_fresh_design() -> None:
    result = analyze_observations(_observations(), _diagnostics())

    assert result["decision"] == (
        "pass_to_fresh_disjoint_replicated_calibration_design"
    )
    assert all(result["locked_checks"].values())
    assert result["calibration_only"] is True
    assert result["confirmatory_eligible"] is False
    assert result["authorizes_replay"] is False
    assert result["authorizes_training"] is False


def test_residual_length_censoring_retires_the_model_or_prompt() -> None:
    result = analyze_observations(_observations(), _diagnostics(length_finishes=1))

    assert result["decision"] == "stop_model_or_prompt_overlong"
    assert not result["locked_checks"]["length_rate_lt_0_2_each"]


def test_asymmetric_near_cap_censoring_fails_headroom_gate() -> None:
    diagnostics = _diagnostics()
    for ordinal in (1, 3, 5):
        diagnostics[ordinal] = replace(diagnostics[ordinal], near_cap=1)

    result = analyze_observations(_observations(), diagnostics)

    assert result["decision"] == "stop_model_or_prompt_overlong"
    assert result["locked_checks"]["near_cap_rate_lt_0_2_each"] is True
    assert result["locked_checks"]["near_cap_rate_difference_le_0_1"] is False


def test_scheduler_latency_is_descriptive_not_a_stage0_gate() -> None:
    reversed_latency = []
    for observation in _observations():
        latency = 90 if observation.item.stratum == "long" else 150
        reversed_latency.append(
            replace(
                observation,
                completed_ns=observation.dispatch_ns + latency - 10,
                ready_ns=observation.dispatch_ns + latency,
                archived_ns=observation.dispatch_ns + latency + 1,
            )
        )

    result = analyze_observations(tuple(reversed_latency), _diagnostics())

    assert result["ready_latency"]["median_long_short_ratio"] < 1
    assert result["decision"] == (
        "pass_to_fresh_disjoint_replicated_calibration_design"
    )


def _lifecycle_events() -> tuple[SchedulerTraceEvent, ...]:
    events = [_runtime_event()]
    sequence = 1
    for observation in _observations():
        identity = {
            "logical_group_id": f"group-{observation.item.source_pool_ordinal}",
            "attempt_id": f"attempt-{observation.item.source_pool_ordinal}",
            "source_pool_ordinal": observation.item.source_pool_ordinal,
            "source_prompt_id": observation.item.source_prompt_id,
            "task_name": observation.item.task_name,
        }
        for event_type, timestamp in (
            (SchedulerEventType.ATTEMPT_DISPATCHED, observation.dispatch_ns),
            (SchedulerEventType.ROLLOUT_COMPLETED, observation.completed_ns),
            (SchedulerEventType.GROUP_READY, observation.ready_ns),
            (SchedulerEventType.GROUP_ARCHIVED, observation.archived_ns),
        ):
            summaries = {}
            if event_type is SchedulerEventType.ROLLOUT_COMPLETED:
                summaries = {
                    "completion_count": 2,
                    "mean_gen_tokens_per_sample": observation.mean_generated_tokens,
                    "gen_tokens_per_sample/min": observation.min_generated_tokens,
                    "gen_tokens_per_sample/max": observation.max_generated_tokens,
                    "effective_max_new_tokens/min": 924,
                    "effective_max_new_tokens/max": 924,
                    "effective_engine_seed/min": 52001,
                    "effective_engine_seed/max": 52001,
                    "backend_finish_reason_availability_rate": 1.0,
                    "backend_stop_termination_rate": 1.0,
                    "backend_length_termination_rate": 0.0,
                    "backend_abort_termination_rate": 0.0,
                    "backend_other_termination_rate": 0.0,
                    "backend_context_exhausted_rate": 0.0,
                    "backend_stop_reason_token_rate": 0.0,
                    "backend_stop_reason_matches_tokenizer_eos_rate": 0.0,
                    "generated_at_effective_cap_rate": 0.0,
                    "generated_near_effective_cap_rate": 0.0,
                    "natural_termination_rate": 1.0,
                }
            events.append(
                SchedulerTraceEvent(
                    schema_version=1,
                    trace_run_id="run",
                    process_epoch="epoch",
                    event_seq=sequence,
                    monotonic_ns=timestamp,
                    event_type=event_type,
                    terminal_reason=(
                        "fixed_pool_archive"
                        if event_type is SchedulerEventType.GROUP_ARCHIVED
                        else None
                    ),
                    scalar_summaries=summaries,
                    **identity,
                )
            )
            sequence += 1
    return tuple(events)


def test_lifecycle_join_requires_the_new_cap_and_finish_summaries() -> None:
    items = tuple(value.item for value in _observations())
    observations, diagnostics, _ = observations_from_events(_lifecycle_events(), items)
    assert len(observations) == 16
    assert all(value.finish_available == 2 for value in diagnostics.values())

    broken = list(_lifecycle_events())
    completed_index = next(
        index
        for index, event in enumerate(broken)
        if event.event_type is SchedulerEventType.ROLLOUT_COMPLETED
    )
    event = broken[completed_index]
    summaries = dict(event.scalar_summaries)
    summaries.pop("effective_engine_seed/max")
    broken[completed_index] = replace(event, scalar_summaries=summaries)
    with pytest.raises(FeasibilityAnalysisError, match="effective_engine_seed/max"):
        observations_from_events(broken, items)
