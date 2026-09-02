# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import tools.analyze_openmath_latency_feasibility as analyzer
from nemo_rl.algorithms.async_utils.scheduler_trace import (
    SchedulerEventType,
    SchedulerTraceEvent,
)
from tools.analyze_openmath_latency_feasibility import (
    DesignItem,
    FeasibilityAnalysisError,
    GroupObservation,
    analyze_observations,
    observations_from_events,
)


def _observations(
    *, output_ratio: float = 1.6, ready_long_ns: int = 130
) -> tuple[GroupObservation, ...]:
    observations = []
    for pair in range(8):
        for stratum in ("short", "long"):
            ordinal = pair * 2 + (stratum == "long")
            dispatch = ordinal * 10
            is_long = stratum == "long"
            ready_latency = ready_long_ns if is_long and pair < 6 else 100
            if is_long and pair >= 6:
                ready_latency = 90
            mean_tokens = 100 * output_ratio if is_long else 100
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
            observations.append(
                GroupObservation(
                    item=item,
                    dispatch_ns=dispatch,
                    completed_ns=dispatch + ready_latency - 10,
                    ready_ns=dispatch + ready_latency,
                    archived_ns=dispatch + ready_latency + 1,
                    mean_generated_tokens=mean_tokens,
                    min_generated_tokens=int(mean_tokens),
                    max_generated_tokens=int(mean_tokens),
                    cap_hits=0,
                    natural_terminations=2,
                    effective_output_cap=412,
                )
            )
    return tuple(observations)


def test_locked_gate_does_not_substitute_stronger_descriptive_screen() -> None:
    result = analyze_observations(_observations())

    assert result["decision"] == "pass_to_fresh_replicated_design"
    assert all(result["locked_stage_a_checks"].values())
    assert result["ready_latency"]["pair_sign_count"] == 6
    assert result["ready_latency"]["matched_pairs_rank_biserial"] == pytest.approx(
        5 / 6
    )
    assert result["stronger_descriptive_robustness"] == {
        "output_median_ratio_ge_1_75": False,
        "ready_latency_median_ratio_ge_1_5": False,
        "ready_pair_sign_count_ge_7": False,
    }
    assert result["calibration_only"] is True
    assert result["confirmatory_eligible"] is False


def test_bootstrap_is_deterministic_and_stop_rule_is_fail_closed() -> None:
    observations = _observations(ready_long_ns=110)

    first = analyze_observations(observations)
    second = analyze_observations(observations)

    assert (
        first["ready_latency"]["median_ratio_bootstrap"]
        == second["ready_latency"]["median_ratio_bootstrap"]
    )
    assert first["decision"] == "stop_no_replay"
    assert not first["locked_stage_a_checks"]["ready_latency_median_ratio_ge_1_25"]


def test_trace_rejects_missing_resolved_runtime_binding() -> None:
    event = SchedulerTraceEvent(
        schema_version=1,
        trace_run_id="run",
        process_epoch="epoch",
        event_seq=0,
        monotonic_ns=0,
        event_type=SchedulerEventType.RUN_STARTED,
        scalar_summaries={},
    )

    with pytest.raises(FeasibilityAnalysisError, match="exact cap semantics"):
        observations_from_events((event,), tuple(item.item for item in _observations()))


def test_zero_pair_denominator_is_rejected() -> None:
    observations = list(_observations())
    original = observations[0]
    observations[0] = GroupObservation(
        item=original.item,
        dispatch_ns=original.dispatch_ns,
        completed_ns=original.completed_ns,
        ready_ns=original.ready_ns,
        archived_ns=original.archived_ns,
        mean_generated_tokens=0,
        min_generated_tokens=0,
        max_generated_tokens=0,
        cap_hits=0,
        natural_terminations=2,
        effective_output_cap=original.effective_output_cap,
    )

    with pytest.raises(FeasibilityAnalysisError, match="ratio denominator"):
        analyze_observations(observations)


def test_near_cap_completion_is_reported_but_not_an_extra_locked_gate() -> None:
    observations = list(_observations())
    original = observations[1]
    observations[1] = GroupObservation(
        item=original.item,
        dispatch_ns=original.dispatch_ns,
        completed_ns=original.completed_ns,
        ready_ns=original.ready_ns,
        archived_ns=original.archived_ns,
        mean_generated_tokens=250,
        min_generated_tokens=100,
        max_generated_tokens=400,
        cap_hits=0,
        natural_terminations=2,
        effective_output_cap=original.effective_output_cap,
    )

    result = analyze_observations(observations)

    assert result["backend_length_termination"]["near_cap_rate_by_stratum"][
        "long"
    ] == pytest.approx(1 / 16)
    assert result["backend_length_termination"][
        "near_cap_is_diagnostic_not_a_locked_gate"
    ]
    assert result["decision"] == "pass_to_fresh_replicated_design"


def test_analyze_validates_materialization_and_requires_weight_version_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    manifest = SimpleNamespace(pool_id="pool", manifest_sha256="manifest", items=())
    materialization_validator = MagicMock()
    design_validator = MagicMock()
    monkeypatch.setattr(analyzer, "load_design", lambda _: ("pool", "manifest", ()))
    monkeypatch.setattr(analyzer, "load_fixed_pool_manifest", lambda _: manifest)
    monkeypatch.setattr(
        analyzer, "validate_fixed_pool_materialization", materialization_validator
    )
    monkeypatch.setattr(
        analyzer, "validate_fixed_pool_manifest_design", design_validator
    )
    monkeypatch.setattr(
        analyzer,
        "validate_fixed_pool_trace",
        lambda *args, **kwargs: SimpleNamespace(physical_weight_version=1),
    )

    with pytest.raises(FeasibilityAnalysisError, match="physical_weight_version=0"):
        analyzer.analyze(
            trace_path=tmp_path / "trace.jsonl",
            manifest_path=tmp_path / "manifest.json",
            design_path=tmp_path / "selection.json",
        )

    materialization_validator.assert_called_once_with(manifest)
    design_validator.assert_called_once_with(
        manifest, "openmath_latency_feasibility_v1"
    )
