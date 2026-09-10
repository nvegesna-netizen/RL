# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Prospectively frozen within-Llama size-contrast synthesis helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import NormalDist

from tools.m4_llama_v5_analysis import ReplicateInference


class LlamaSizeExtensionError(ValueError):
    """Raised when the two-size replicated contrast is malformed."""


@dataclass(frozen=True)
class SizeContrast:
    estimate: float
    hac_interval: tuple[float, float]
    bootstrap_interval: tuple[float, float]
    confidence_envelope: tuple[float, float]
    conclusion: str
    combination_rule: str = "3b_equal_replicate_mean_minus_1b_equal_replicate_mean"

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


def infer_size_contrast(
    one_b: Mapping[str, ReplicateInference],
    three_b: Mapping[str, ReplicateInference],
    *,
    confidence: float = 0.95,
) -> SizeContrast:
    """Estimate 3B minus 1B while retaining all four run uncertainties."""
    if set(one_b) != {"r1", "r2"} or set(three_b) != {"r1", "r2"}:
        raise LlamaSizeExtensionError("each size requires r1 and r2")
    runs = [one_b["r1"], one_b["r2"], three_b["r1"], three_b["r2"]]
    draws = {len(run.lower.bootstrap_shifts) for run in runs}
    if len(draws) != 1 or 0 in draws or not 0.0 < confidence < 1.0:
        raise LlamaSizeExtensionError("replicate draw counts or confidence disagree")
    estimate = (
        three_b["r1"].lower.estimate
        + three_b["r2"].lower.estimate
        - one_b["r1"].lower.estimate
        - one_b["r2"].lower.estimate
    ) / 2.0
    se = math.sqrt(sum(run.lower.hac_standard_error**2 for run in runs)) / 2.0
    shifts = [
        (
            three_b["r1"].lower.bootstrap_shifts[index]
            + three_b["r2"].lower.bootstrap_shifts[index]
            - one_b["r1"].lower.bootstrap_shifts[index]
            - one_b["r2"].lower.bootstrap_shifts[index]
        )
        / 2.0
        for index in range(next(iter(draws)))
    ]
    tail = (1.0 - confidence) / 2.0
    z = NormalDist().inv_cdf(1.0 - tail)
    hac = (estimate - z * se, estimate + z * se)
    bootstrap = (estimate - _type7(shifts, 1.0 - tail), estimate - _type7(shifts, tail))
    envelope = (min(hac[0], bootstrap[0]), max(hac[1], bootstrap[1]))
    conclusion = "POSITIVE" if envelope[0] > 0.0 else "NEGATIVE" if envelope[1] < 0.0 else "INCONCLUSIVE"
    return SizeContrast(estimate, hac, bootstrap, envelope, conclusion)
