# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Frozen joint inference for the two-cell M4 Llama family-transport study."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any


CELL_NAMES = ("llama3p2_1b_openmath", "llama3p2_1b_gsm8k")
COMMON_START_VERSION = 8
COMMON_END_VERSION = 407


class FamilyTransportAnalysisError(ValueError):
    """Raised when prospective cell inputs violate the frozen contract."""


@dataclass(frozen=True)
class FamilyTransportCellInference:
    """One authenticated common-window cell result."""

    estimate: float
    hac_standard_error: float
    bootstrap_shifts: tuple[float, ...]
    confidence_envelope: tuple[float, float]
    materiality_conclusion: str
    primary_start_version: int = COMMON_START_VERSION
    primary_end_version: int = COMMON_END_VERSION
    terminal_scoring_complete: bool = True


@dataclass(frozen=True)
class FamilyTransportInference:
    """Registered Llama OpenMath-minus-GSM8K contrast."""

    cell_estimates: dict[str, float]
    cell_materiality: dict[str, str]
    openmath_family_transport: str
    gsm8k_effect_direction: str
    workload_contrast_estimate: float
    workload_contrast_hac_standard_error: float
    workload_contrast_hac_interval: tuple[float, float]
    workload_contrast_bootstrap_interval: tuple[float, float]
    workload_contrast_confidence_envelope: tuple[float, float]
    workload_pattern_conclusion: str
    confidence: float
    bootstrap_draws: int
    role: str = "prospective_cross_family_transport_and_workload_pattern"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible result."""
        return asdict(self)


def _probability(value: float, *, name: str) -> float:
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise FamilyTransportAnalysisError(f"{name} must be in (0, 1)")
    return value


def _type7(values: Sequence[float], probability: float) -> float:
    if not values:
        raise FamilyTransportAnalysisError("quantile requires values")
    _probability(probability, name="quantile probability")
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def infer_family_transport(
    cells: Mapping[str, FamilyTransportCellInference],
    *,
    confidence: float = 0.95,
    expected_bootstrap_draws: int = 20_000,
) -> FamilyTransportInference:
    """Combine independently acquired cells under the frozen direction rule."""
    if set(cells) != set(CELL_NAMES):
        raise FamilyTransportAnalysisError("family-transport cell identities disagree")
    if expected_bootstrap_draws <= 0:
        raise FamilyTransportAnalysisError("bootstrap draws must be positive")
    confidence = _probability(confidence, name="confidence")
    allowed_materiality = {"MATERIAL", "NOT_MATERIAL", "INCONCLUSIVE"}
    for name in CELL_NAMES:
        cell = cells[name]
        if len(cell.bootstrap_shifts) != expected_bootstrap_draws:
            raise FamilyTransportAnalysisError(f"{name} bootstrap draws disagree")
        if (
            not math.isfinite(cell.estimate)
            or not math.isfinite(cell.hac_standard_error)
            or cell.hac_standard_error < 0.0
            or len(cell.confidence_envelope) != 2
            or any(not math.isfinite(value) for value in cell.confidence_envelope)
            or cell.confidence_envelope[0] > cell.confidence_envelope[1]
            or cell.materiality_conclusion not in allowed_materiality
            or cell.primary_start_version != COMMON_START_VERSION
            or cell.primary_end_version != COMMON_END_VERSION
            or not cell.terminal_scoring_complete
            or any(not math.isfinite(value) for value in cell.bootstrap_shifts)
        ):
            raise FamilyTransportAnalysisError(f"{name} violates cell contract")
        lower, upper = cell.confidence_envelope
        expected_materiality = (
            "MATERIAL"
            if lower > 0.2
            else "NOT_MATERIAL"
            if upper <= 0.2
            else "INCONCLUSIVE"
        )
        if cell.materiality_conclusion != expected_materiality:
            raise FamilyTransportAnalysisError(
                f"{name} materiality label disagrees with its envelope"
            )

    openmath = cells["llama3p2_1b_openmath"]
    gsm8k = cells["llama3p2_1b_gsm8k"]
    contrast = openmath.estimate - gsm8k.estimate
    standard_error = math.hypot(openmath.hac_standard_error, gsm8k.hac_standard_error)
    z_value = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    hac_interval = (
        contrast - z_value * standard_error,
        contrast + z_value * standard_error,
    )
    contrast_shifts = [
        openmath.bootstrap_shifts[index] - gsm8k.bootstrap_shifts[index]
        for index in range(expected_bootstrap_draws)
    ]
    tail = (1.0 - confidence) / 2.0
    bootstrap_interval = (
        contrast - _type7(contrast_shifts, 1.0 - tail),
        contrast - _type7(contrast_shifts, tail),
    )
    envelope = (
        min(hac_interval[0], bootstrap_interval[0]),
        max(hac_interval[1], bootstrap_interval[1]),
    )
    if envelope[0] > 0.0:
        workload_conclusion = "SAME_DIRECTION"
    elif envelope[1] < 0.0:
        workload_conclusion = "OPPOSITE_DIRECTION"
    else:
        workload_conclusion = "INCONCLUSIVE"
    gsm8k_direction = (
        "POSITIVE"
        if gsm8k.confidence_envelope[0] > 0.0
        else "NEGATIVE"
        if gsm8k.confidence_envelope[1] < 0.0
        else "INCONCLUSIVE"
    )
    return FamilyTransportInference(
        cell_estimates={name: cells[name].estimate for name in CELL_NAMES},
        cell_materiality={
            name: cells[name].materiality_conclusion for name in CELL_NAMES
        },
        openmath_family_transport=openmath.materiality_conclusion,
        gsm8k_effect_direction=gsm8k_direction,
        workload_contrast_estimate=contrast,
        workload_contrast_hac_standard_error=standard_error,
        workload_contrast_hac_interval=hac_interval,
        workload_contrast_bootstrap_interval=bootstrap_interval,
        workload_contrast_confidence_envelope=envelope,
        workload_pattern_conclusion=workload_conclusion,
        confidence=confidence,
        bootstrap_draws=expected_bootstrap_draws,
    )
