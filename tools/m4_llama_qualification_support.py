# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Prospectively frozen support gate for the V4 Llama M4 qualification."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass


class QualificationSupportError(ValueError):
    """Raised when version support cannot satisfy the frozen V4 contract."""


@dataclass(frozen=True)
class QualificationSupportResult:
    """Point and lower-bound acquisition-assignment projections."""

    bootstrap_draws: int
    circular_block_size_versions: int
    evaluable_end_version: int
    evaluable_start_version: int
    evaluable_version_count: int
    joined_group_count: int
    minimum_joined_groups_in_any_version: int
    projected_primary_assignments: float
    projected_primary_assignments_lower_95: float
    qualification_support_passed: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible canonical field mapping."""
        return asdict(self)


def _lower_empirical_quantile(values: list[float], alpha: float) -> float:
    """Return order statistic ``ceil(alpha * n) - 1`` after sorting."""
    if not values or not 0.0 < alpha < 1.0:
        raise QualificationSupportError("invalid empirical quantile inputs")
    ordered = sorted(values)
    index = max(0, math.ceil(alpha * len(ordered)) - 1)
    return ordered[index]


def assess_qualification_support(
    groups_by_start_version: Mapping[int, int],
    *,
    start_version: int = 8,
    end_version: int = 55,
    acquisition_primary_version_count: int = 400,
    block_size: int = 8,
    bootstrap_draws: int = 20_000,
    bootstrap_seed: int,
    minimum_lower_projection: float = 6_900.0,
) -> QualificationSupportResult:
    """Assess the frozen common-window throughput-support requirement.

    Each bootstrap draw concatenates uniformly sampled circular blocks until it
    contains exactly the registered number of versions. The registered 48 by 8
    geometry is exact and therefore requires no partial final block.
    """
    if (
        isinstance(start_version, bool)
        or isinstance(end_version, bool)
        or not isinstance(start_version, int)
        or not isinstance(end_version, int)
        or end_version < start_version
    ):
        raise QualificationSupportError("invalid evaluable version window")
    version_count = end_version - start_version + 1
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size <= 0
        or version_count % block_size != 0
    ):
        raise QualificationSupportError(
            "evaluable version count must be divisible by block size"
        )
    if (
        isinstance(bootstrap_draws, bool)
        or not isinstance(bootstrap_draws, int)
        or bootstrap_draws <= 0
        or isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or isinstance(acquisition_primary_version_count, bool)
        or not isinstance(acquisition_primary_version_count, int)
        or acquisition_primary_version_count <= 0
        or not math.isfinite(minimum_lower_projection)
        or minimum_lower_projection <= 0.0
    ):
        raise QualificationSupportError("invalid projection contract")

    expected_versions = set(range(start_version, end_version + 1))
    if set(groups_by_start_version) != expected_versions:
        raise QualificationSupportError(
            "group counts must cover exactly the common evaluable window"
        )
    counts: list[int] = []
    for version in range(start_version, end_version + 1):
        count = groups_by_start_version[version]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise QualificationSupportError(
                "every evaluable version must contain a joined group"
            )
        counts.append(count)

    scale = acquisition_primary_version_count / version_count
    point = scale * sum(counts)
    rng = random.Random(bootstrap_seed)
    block_count = version_count // block_size
    draws: list[float] = []
    for _ in range(bootstrap_draws):
        total = 0
        for _ in range(block_count):
            block_start = rng.randrange(version_count)
            total += sum(
                counts[(block_start + offset) % version_count]
                for offset in range(block_size)
            )
        draws.append(scale * total)
    lower = _lower_empirical_quantile(draws, 0.05)
    return QualificationSupportResult(
        bootstrap_draws=bootstrap_draws,
        circular_block_size_versions=block_size,
        evaluable_end_version=end_version,
        evaluable_start_version=start_version,
        evaluable_version_count=version_count,
        joined_group_count=sum(counts),
        minimum_joined_groups_in_any_version=min(counts),
        projected_primary_assignments=point,
        projected_primary_assignments_lower_95=lower,
        qualification_support_passed=lower >= minimum_lower_projection,
    )
