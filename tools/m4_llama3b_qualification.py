# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Frozen outcome-neutral support and timing gates for Llama 3.2 3B M4."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from tools.m4_llama_qualification_support import assess_qualification_support


class Llama3BQualificationError(ValueError):
    """Raised when qualification inputs violate the prospective contract."""


@dataclass(frozen=True)
class Llama3BQualificationResult:
    """Outcome-neutral feasibility decision for one workload."""

    assignment_projection: dict[str, object]
    bootstrap_draws: int
    timing_block_size_versions: int
    measured_non_active_overhead_seconds: float
    projected_400_version_active_seconds_upper_95: float
    projected_total_runtime_seconds_upper_95: float
    maximum_projected_runtime_seconds: float
    assignment_support_passed: bool
    timing_support_passed: bool
    qualified: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _type7(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def assess_llama3b_qualification(
    groups_by_start_version: Mapping[int, int],
    update_intervals_seconds: Sequence[float],
    *,
    total_qualification_runtime_seconds: float,
    assignment_bootstrap_seed: int,
    timing_bootstrap_seed: int,
    bootstrap_draws: int = 20_000,
    block_size: int = 8,
    minimum_assignment_projection_each_replicate: float = 5_000.0,
    maximum_projected_runtime_seconds: float = 12_600.0,
) -> Llama3BQualificationResult:
    """Project one 400-version acquisition from versions 8--55.

    Timing resamples circular eight-version blocks from the 48 evaluable update
    intervals and projects their sum to 400 versions. Measured non-active
    overhead is added once, rather than multiplied seven times.
    """
    support = assess_qualification_support(
        groups_by_start_version,
        bootstrap_seed=assignment_bootstrap_seed,
        bootstrap_draws=bootstrap_draws,
        block_size=block_size,
        minimum_lower_projection=minimum_assignment_projection_each_replicate,
    )
    intervals = tuple(float(value) for value in update_intervals_seconds)
    if (
        len(intervals) != 48
        or any(not math.isfinite(value) or value <= 0.0 for value in intervals)
        or not math.isfinite(total_qualification_runtime_seconds)
        or total_qualification_runtime_seconds <= 0.0
        or 48 % block_size != 0
        or 400 % block_size != 0
        or bootstrap_draws <= 0
        or maximum_projected_runtime_seconds <= 0.0
    ):
        raise Llama3BQualificationError("invalid timing qualification inputs")
    measured_active = math.fsum(intervals)
    overhead = total_qualification_runtime_seconds - measured_active
    if overhead < 0.0:
        raise Llama3BQualificationError("total runtime is shorter than active intervals")
    rng = random.Random(timing_bootstrap_seed)
    projected = []
    for _ in range(bootstrap_draws):
        total = 0.0
        for _ in range(400 // block_size):
            start = rng.randrange(48)
            total += math.fsum(intervals[(start + offset) % 48] for offset in range(block_size))
        projected.append(total)
    active_upper = _type7(projected, 0.95)
    total_upper = overhead + active_upper
    timing_passed = total_upper <= maximum_projected_runtime_seconds
    return Llama3BQualificationResult(
        assignment_projection=support.to_dict(),
        bootstrap_draws=bootstrap_draws,
        timing_block_size_versions=block_size,
        measured_non_active_overhead_seconds=overhead,
        projected_400_version_active_seconds_upper_95=active_upper,
        projected_total_runtime_seconds_upper_95=total_upper,
        maximum_projected_runtime_seconds=maximum_projected_runtime_seconds,
        assignment_support_passed=support.qualification_support_passed,
        timing_support_passed=timing_passed,
        qualified=support.qualification_support_passed and timing_passed,
    )
