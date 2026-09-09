"""Static checks for the prospective Llama qualification repair."""

from __future__ import annotations

import json
from pathlib import Path


_REPO = Path(__file__).resolve().parents[3]
_AMENDMENT = (
    _REPO / "reports/auto_research/2026-09-08-m4-llama-family-transport/"
    "qualification_repair_amendment.json"
)
_FINAL_AMENDMENT = (
    _REPO / "reports/auto_research/2026-09-08-m4-llama-family-transport/"
    "qualification_repair_r2_amendment.json"
)


def test_repair_preserves_threshold_scope_and_historical_results() -> None:
    amendment = json.loads(_AMENDMENT.read_bytes())

    assert amendment["status"] == ("FROZEN_LOCAL_REPAIR_PENDING_NO_TRAINING_PREFLIGHT")
    assert amendment["observer_repair"] == {
        "arithmetic_contract": (
            "Preserve the existing row-wise float64 sibling reductions and every "
            "serialized summary value."
        ),
        "implementation": (
            "Vectorize sibling valid-token and absolute-opportunity reductions and "
            "replace per-sibling scalar extraction with batched tolist conversion."
        ),
        "measurement_scope_changed": False,
        "observer_duty_maximum": 0.01,
        "threshold_changed": False,
        "workload_slowdown_as_metric_repair_allowed": False,
    }
    assert amendment["causal_scope"] == {
        "acquisition_allowed": False,
        "causal_estimate_allowed": False,
        "qualification_observations_enter_estimators": False,
    }
    assert amendment["requalification"]["required_cells"] == [
        "llama3p2_1b_openmath",
        "llama3p2_1b_gsm8k",
    ]
    assert amendment["requalification"]["submission_authorized"] is False


def test_repair_lifecycle_contract_fails_closed() -> None:
    lifecycle = json.loads(_AMENDMENT.read_bytes())["lifecycle_repair"]

    assert lifecycle["all_recorded_release_fields_must_remain"] == {
        "release_arm": "neutral",
        "release_arm_mass": 1,
        "release_delay_seconds": 0.0,
        "release_total_mass": 1,
    }
    assert lifecycle["opportunity_group_ids_must_equal_started_group_ids"] is True
    assert lifecycle["unexplained_missing_completion_allowed"] is False


def test_final_repair_preserves_scientific_gate_and_freezes_stop_rule() -> None:
    amendment = json.loads(_FINAL_AMENDMENT.read_bytes())

    assert amendment["status"] == "FROZEN_LOCAL_FINAL_REPAIR_PENDING_IMPLEMENTATION"
    assert amendment["observer_repair"]["observer_duty_maximum"] == 0.01
    assert amendment["observer_repair"]["threshold_changed"] is False
    assert amendment["observer_repair"]["measurement_scope_changed"] is False
    assert (
        amendment["observer_repair"]["workload_slowdown_as_metric_repair_allowed"]
        is False
    )
    assert amendment["no_training_preflight"] == {
        "absolute_mean_callback_ms_maximum": 1.0,
        "automatic_retry": False,
        "benchmark_shape": {"sequence_length": 2048, "siblings": 8},
        "minimum_speedup_fraction_against_expanded_reference": 0.25,
        "qualification_allowed": False,
        "submission_attempt_limit": 1,
        "training_allowed": False,
    }
    assert amendment["requalification"]["permanent_close_after_any_failure"] is True
    assert amendment["authorization"]["final_paired_qualification_authorized"] is False
    assert amendment["authorization"]["acquisition_authorized"] is False
