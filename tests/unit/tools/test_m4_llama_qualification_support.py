# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the prospective V4 Llama qualification support gate."""

from __future__ import annotations

import pytest

from tools.m4_llama_qualification_support import (
    QualificationSupportError,
    assess_qualification_support,
)


def _counts(value: int) -> dict[int, int]:
    return {version: value for version in range(8, 56)}


def test_constant_support_passes_at_or_above_registered_lower_bound() -> None:
    result = assess_qualification_support(_counts(18), bootstrap_seed=20261017)
    assert result.projected_primary_assignments == pytest.approx(7200.0)
    assert result.projected_primary_assignments_lower_95 == pytest.approx(7200.0)
    assert result.joined_group_count == 864
    assert result.qualification_support_passed is True


def test_constant_support_fails_below_registered_lower_bound() -> None:
    result = assess_qualification_support(_counts(17), bootstrap_seed=20261018)
    assert result.projected_primary_assignments == pytest.approx(6800.0)
    assert result.projected_primary_assignments_lower_95 == pytest.approx(6800.0)
    assert result.qualification_support_passed is False


def test_seeded_circular_bootstrap_is_deterministic() -> None:
    counts = {version: 12 + version % 11 for version in range(8, 56)}
    first = assess_qualification_support(counts, bootstrap_seed=20261017)
    second = assess_qualification_support(counts, bootstrap_seed=20261017)
    assert first == second
    assert first.projected_primary_assignments_lower_95 == pytest.approx(
        6583.333333333333
    )


@pytest.mark.parametrize(
    "counts",
    [
        {version: 18 for version in range(8, 55)},
        {**_counts(18), 56: 18},
        {**_counts(18), 17: 0},
    ],
)
def test_incomplete_extra_or_empty_version_support_is_rejected(
    counts: dict[int, int],
) -> None:
    with pytest.raises(QualificationSupportError):
        assess_qualification_support(counts, bootstrap_seed=20261017)


def test_block_geometry_must_partition_the_common_window() -> None:
    with pytest.raises(QualificationSupportError, match="divisible"):
        assess_qualification_support(_counts(18), block_size=7, bootstrap_seed=20261017)
