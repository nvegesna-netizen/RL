# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import pytest

from tools.m4_llama3b_qualification import (
    Llama3BQualificationError,
    assess_llama3b_qualification,
)


def counts(value: int = 16) -> dict[int, int]:
    return {version: value for version in range(8, 56)}


def test_joint_assignment_and_timing_gate_passes() -> None:
    result = assess_llama3b_qualification(
        counts(),
        [20.0] * 48,
        total_qualification_runtime_seconds=1_080.0,
        assignment_bootstrap_seed=20261111,
        timing_bootstrap_seed=20261113,
    )
    assert result.assignment_projection["projected_primary_assignments_lower_95"] == pytest.approx(6_400.0)
    assert result.projected_total_runtime_seconds_upper_95 == pytest.approx(8_120.0)
    assert result.qualified is True


def test_assignment_failure_cannot_be_rescued_by_timing() -> None:
    result = assess_llama3b_qualification(
        counts(12),
        [20.0] * 48,
        total_qualification_runtime_seconds=1_080.0,
        assignment_bootstrap_seed=20261112,
        timing_bootstrap_seed=20261114,
    )
    assert result.assignment_support_passed is False
    assert result.timing_support_passed is True
    assert result.qualified is False


def test_timing_failure_cannot_be_rescued_by_assignments() -> None:
    result = assess_llama3b_qualification(
        counts(),
        [32.0] * 48,
        total_qualification_runtime_seconds=1_650.0,
        assignment_bootstrap_seed=20261111,
        timing_bootstrap_seed=20261113,
    )
    assert result.assignment_support_passed is True
    assert result.timing_support_passed is False
    assert result.qualified is False


def test_incomplete_or_nonpositive_timing_series_is_rejected() -> None:
    with pytest.raises(Llama3BQualificationError):
        assess_llama3b_qualification(
            counts(),
            [20.0] * 47,
            total_qualification_runtime_seconds=1_080.0,
            assignment_bootstrap_seed=20261111,
            timing_bootstrap_seed=20261113,
        )
