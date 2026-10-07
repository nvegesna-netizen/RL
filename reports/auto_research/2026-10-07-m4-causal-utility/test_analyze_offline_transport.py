"""Dependency-light behavioral tests for the M4 offline transport analyzer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "analyze_offline_transport", HERE / "analyze_offline_transport.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def assignment(index: int, *, arm: str, lost: bool, opportunity: float):
    """Build a deterministic synthetic assignment."""
    base = float(index % 7)
    features = (
        base / 7.0,
        float(index % 3),
        np.log1p(100 + index),
        float(index % 2),
        0.25 if index % 2 else 0.0,
        0.0,
        np.log1p(1000 + index),
        float(index % 5),
        float(index % 11),
        float(index % 4),
        float(index % 13),
        0.0,
        np.log1p(opportunity),
        float(opportunity == 0.0),
    )
    return MODULE.Assignment(
        assignment_id=f"g{index}",
        acquisition=f"a{index % 3}",
        family="qwen" if index % 2 else "llama",
        model="synthetic",
        workload="gsm8k",
        arm=arm,
        opportunity=opportunity,
        delivered=not lost,
        features=features,
    )


def test_capture_prefers_high_value_score() -> None:
    target = np.asarray([0.0, 1.0, 10.0, 0.0])
    good = np.asarray([0.0, 1.0, 2.0, 0.0])
    bad = np.asarray([2.0, 1.0, 0.0, 3.0])
    assert MODULE._capture(target, good, 0.25) == 10.0 / 11.0
    assert MODULE._capture(target, bad, 0.25) == 0.0


def test_schema_specific_pre_hold_ordering() -> None:
    assert MODULE.opportunity_precedes_hold(
        {"schema_version": 1, "timestamp_ns": 9},
        {"timestamp_ns": 10, "controller_sequence": 1},
    )
    assert MODULE.opportunity_precedes_hold(
        {"schema_version": 2, "source_controller_sequence_max": 9},
        {"timestamp_ns": 1, "controller_sequence": 10},
    )
    assert not MODULE.opportunity_precedes_hold(
        {"schema_version": 2, "source_controller_sequence_max": 10},
        {"timestamp_ns": 20, "controller_sequence": 10},
    )


def test_arm_models_and_evaluation_are_finite() -> None:
    training = [
        assignment(
            index,
            arm="d5" if index % 2 else "control",
            lost=index % 4 in (1, 2),
            opportunity=float((index % 5) * 10),
        )
        for index in range(120)
    ]
    testing = [
        assignment(
            index + 1000,
            arm="d5" if index % 2 else "control",
            lost=index % 3 == 0,
            opportunity=float((index % 7) * 8),
        )
        for index in range(60)
    ]
    result = MODULE.evaluate_split("synthetic", training, testing)
    assert result["training_assignments"] == 120
    assert result["testing_assignments"] == 60
    for model in ("conventional_prediction", "m4_augmented_prediction"):
        assert all(np.isfinite(value) for value in result[model].values())
    assert set(result["ranking"]) == set(MODULE.SCORES)


def test_summary_applies_frozen_gate() -> None:
    acquisitions = ["a", "b"]

    def result(name: str, m4: float, reward_variance: float):
        return {
            "name": name,
            "ranking": {
                "m4_at_risk": {"observed_lost_m4_capture_top_10pct": m4},
                "reward_variance_at_risk": {
                    "observed_lost_m4_capture_top_10pct": reward_variance
                },
            },
        }

    results = [
        result("family-qwen-to-llama", 0.4, 0.3),
        result("family-llama-to-qwen", 0.5, 0.4),
        result("acquisition-holdout-a", 0.5, 0.3),
        result("acquisition-holdout-b", 0.4, 0.3),
    ]
    summary = MODULE.summarize(results, acquisitions)
    assert summary["decision"] == "PASS_AUTHORIZE_GRADIENT_UTILITY_AUDIT"
    assert not summary["authorizes_quality_acquisition"]
