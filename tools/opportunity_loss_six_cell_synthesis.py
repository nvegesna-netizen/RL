# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Dependency-aware retrospective synthesis for the six-cell M4 evidence map."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any

from tools.opportunity_loss_grid_analysis import GridCellInference


WORKLOADS = ("openmath", "gsm8k", "numinamath")
SCALES = ("0p6b", "1p7b")
CELL_NAMES = tuple(
    f"qwen3_{scale}_{workload}" for workload in WORKLOADS for scale in SCALES
)
REFERENCE_INTERACTIONS = (
    "gsm8k_minus_openmath",
    "gsm8k_minus_numinamath",
)


class SixCellSynthesisError(ValueError):
    """Raised when the retrospective six-cell contract is violated."""


@dataclass(frozen=True)
class Interval:
    """One estimate with HAC, marginal bootstrap, and simultaneous intervals."""

    estimate: float
    hac_standard_error: float
    hac_interval: tuple[float, float]
    bootstrap_interval: tuple[float, float]
    simultaneous_bootstrap_interval: tuple[float, float]


@dataclass(frozen=True)
class SixCellSynthesis:
    """Joint result that retains dependence induced by shared GSM8K cells."""

    cell_estimates: dict[str, float]
    model_scale_effects: dict[str, float]
    reference_interactions: dict[str, Interval]
    hac_covariance: tuple[tuple[float, float], tuple[float, float]]
    hac_correlation: float
    bootstrap_covariance: tuple[tuple[float, float], tuple[float, float]]
    bootstrap_correlation: float
    global_heterogeneity_wald_chi_square: float
    global_heterogeneity_degrees_of_freedom: int
    global_heterogeneity_p_value: float
    simultaneous_bootstrap_critical_value: float
    confidence: float
    bootstrap_draws: int
    role: str = "retrospective_secondary_cannot_override_registered_results"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def _type7(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 < probability < 1.0:
        raise SixCellSynthesisError("invalid quantile input")
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _covariance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise SixCellSynthesisError("covariance inputs disagree")
    left_mean = math.fsum(left) / len(left)
    right_mean = math.fsum(right) / len(right)
    return math.fsum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True)
    ) / (len(left) - 1)


def _correlation(
    covariance: float, left_variance: float, right_variance: float
) -> float:
    denominator = math.sqrt(left_variance * right_variance)
    if denominator <= 0.0:
        raise SixCellSynthesisError("correlation is not identified")
    return covariance / denominator


def infer_six_cell_synthesis(
    cells: Mapping[str, GridCellInference],
    *,
    confidence: float = 0.95,
    expected_bootstrap_draws: int = 20_000,
) -> SixCellSynthesis:
    """Synthesize three workload-specific scale effects from six independent cells.

    The two reference interactions share both GSM8K cell estimates. Their HAC
    covariance includes the shared GSM8K variances, and their bootstrap shifts
    reuse the same GSM8K resamples. This is the dependency that is lost if the
    two historical four-cell analyses are treated as independent replications.
    """
    if set(cells) != set(CELL_NAMES):
        raise SixCellSynthesisError("six-cell identities disagree")
    if not 0.0 < confidence < 1.0 or expected_bootstrap_draws <= 1:
        raise SixCellSynthesisError("invalid confidence or draw count")
    for name in CELL_NAMES:
        cell = cells[name]
        if (
            not math.isfinite(cell.estimate)
            or not math.isfinite(cell.hac_standard_error)
            or cell.hac_standard_error < 0.0
            or len(cell.bootstrap_shifts) != expected_bootstrap_draws
            or any(not math.isfinite(value) for value in cell.bootstrap_shifts)
        ):
            raise SixCellSynthesisError(f"{name} violates the cell contract")

    estimates = {name: cells[name].estimate for name in CELL_NAMES}
    scale_effects = {
        workload: (
            estimates[f"qwen3_1p7b_{workload}"] - estimates[f"qwen3_0p6b_{workload}"]
        )
        for workload in WORKLOADS
    }
    interaction_estimates = {
        "gsm8k_minus_openmath": scale_effects["gsm8k"] - scale_effects["openmath"],
        "gsm8k_minus_numinamath": scale_effects["gsm8k"] - scale_effects["numinamath"],
    }

    def interaction_shift(index: int, comparison: str) -> float:
        workload = comparison.removeprefix("gsm8k_minus_")
        return (
            cells["qwen3_1p7b_gsm8k"].bootstrap_shifts[index]
            - cells["qwen3_0p6b_gsm8k"].bootstrap_shifts[index]
            - cells[f"qwen3_1p7b_{workload}"].bootstrap_shifts[index]
            + cells[f"qwen3_0p6b_{workload}"].bootstrap_shifts[index]
        )

    shifts = {
        comparison: tuple(
            interaction_shift(index, comparison)
            for index in range(expected_bootstrap_draws)
        )
        for comparison in REFERENCE_INTERACTIONS
    }

    gsm_variance = (
        cells["qwen3_0p6b_gsm8k"].hac_standard_error ** 2
        + cells["qwen3_1p7b_gsm8k"].hac_standard_error ** 2
    )
    hac_variances = {
        comparison: gsm_variance
        + cells[
            f"qwen3_0p6b_{comparison.removeprefix('gsm8k_minus_')}"
        ].hac_standard_error
        ** 2
        + cells[
            f"qwen3_1p7b_{comparison.removeprefix('gsm8k_minus_')}"
        ].hac_standard_error
        ** 2
        for comparison in REFERENCE_INTERACTIONS
    }
    hac_covariance = (
        (hac_variances[REFERENCE_INTERACTIONS[0]], gsm_variance),
        (gsm_variance, hac_variances[REFERENCE_INTERACTIONS[1]]),
    )
    hac_correlation = _correlation(
        gsm_variance,
        hac_variances[REFERENCE_INTERACTIONS[0]],
        hac_variances[REFERENCE_INTERACTIONS[1]],
    )

    bootstrap_variances = {
        name: _covariance(shifts[name], shifts[name]) for name in REFERENCE_INTERACTIONS
    }
    bootstrap_cross_covariance = _covariance(
        shifts[REFERENCE_INTERACTIONS[0]], shifts[REFERENCE_INTERACTIONS[1]]
    )
    bootstrap_covariance = (
        (bootstrap_variances[REFERENCE_INTERACTIONS[0]], bootstrap_cross_covariance),
        (bootstrap_cross_covariance, bootstrap_variances[REFERENCE_INTERACTIONS[1]]),
    )
    bootstrap_correlation = _correlation(
        bootstrap_cross_covariance,
        bootstrap_variances[REFERENCE_INTERACTIONS[0]],
        bootstrap_variances[REFERENCE_INTERACTIONS[1]],
    )

    shift_means = {
        name: math.fsum(shifts[name]) / expected_bootstrap_draws
        for name in REFERENCE_INTERACTIONS
    }
    standardized_maxima = []
    for index in range(expected_bootstrap_draws):
        standardized_maxima.append(
            max(
                abs(shifts[name][index] - shift_means[name])
                / math.sqrt(bootstrap_variances[name])
                for name in REFERENCE_INTERACTIONS
            )
        )
    simultaneous_critical = _type7(standardized_maxima, confidence)
    tail = (1.0 - confidence) / 2.0
    z_value = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    intervals = {}
    for name in REFERENCE_INTERACTIONS:
        estimate = interaction_estimates[name]
        se = math.sqrt(hac_variances[name])
        bootstrap_se = math.sqrt(bootstrap_variances[name])
        intervals[name] = Interval(
            estimate=estimate,
            hac_standard_error=se,
            hac_interval=(estimate - z_value * se, estimate + z_value * se),
            bootstrap_interval=(
                estimate - _type7(shifts[name], 1.0 - tail),
                estimate - _type7(shifts[name], tail),
            ),
            simultaneous_bootstrap_interval=(
                estimate - simultaneous_critical * bootstrap_se,
                estimate + simultaneous_critical * bootstrap_se,
            ),
        )

    a, b = hac_covariance[0]
    _, d = hac_covariance[1]
    determinant = a * d - b * b
    if determinant <= 0.0:
        raise SixCellSynthesisError("HAC covariance matrix is singular")
    first = interaction_estimates[REFERENCE_INTERACTIONS[0]]
    second = interaction_estimates[REFERENCE_INTERACTIONS[1]]
    wald = (
        d * first * first - 2.0 * b * first * second + a * second * second
    ) / determinant
    p_value = math.exp(-wald / 2.0)  # exact chi-square(2) survival function

    return SixCellSynthesis(
        cell_estimates=estimates,
        model_scale_effects=scale_effects,
        reference_interactions=intervals,
        hac_covariance=hac_covariance,
        hac_correlation=hac_correlation,
        bootstrap_covariance=bootstrap_covariance,
        bootstrap_correlation=bootstrap_correlation,
        global_heterogeneity_wald_chi_square=wald,
        global_heterogeneity_degrees_of_freedom=2,
        global_heterogeneity_p_value=p_value,
        simultaneous_bootstrap_critical_value=simultaneous_critical,
        confidence=confidence,
        bootstrap_draws=expected_bootstrap_draws,
    )
