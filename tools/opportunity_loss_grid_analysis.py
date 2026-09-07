# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Independent-cell synthesis for the preregistered M4 grid interaction."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any

from tools.opportunity_ledger_join import JoinedOpportunityAssignment
from tools.opportunity_loss_adjusted_inference import _bootstrap_shifts, _endpoint


CELL_NAMES = (
    "qwen3_0p6b_openmath",
    "qwen3_0p6b_gsm8k",
    "qwen3_1p7b_openmath",
    "qwen3_1p7b_gsm8k",
)
COMMON_START_VERSION = 8
COMMON_END_VERSION = 407


class OpportunityLossGridAnalysisError(ValueError):
    """Raised when cell results violate the frozen synthesis contract."""


@dataclass(frozen=True)
class GridCellInference:
    """One adjusted common-window estimate and its centered bootstrap shifts."""

    estimate: float
    hac_standard_error: float
    bootstrap_shifts: tuple[float, ...]
    primary_start_version: int = COMMON_START_VERSION
    primary_end_version: int = COMMON_END_VERSION


@dataclass(frozen=True)
class GridInteractionInference:
    """Secondary difference-in-differences result across independent cells."""

    cell_estimates: dict[str, float]
    workload_contrasts: dict[str, float]
    model_scale_contrasts: dict[str, float]
    interaction_estimate: float
    hac_standard_error: float
    hac_interval: tuple[float, float]
    bootstrap_interval: tuple[float, float]
    confidence_envelope: tuple[float, float]
    confidence: float
    bootstrap_draws: int
    conclusion: str
    role: str = "secondary_grid_synthesis_cannot_override_primary_cell"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible result."""
        return asdict(self)


def infer_common_window_cell(
    rows: Sequence[JoinedOpportunityAssignment],
    *,
    bootstrap_seed: int,
    bootstrap_draws: int = 20_000,
) -> GridCellInference:
    """Build one frozen common-window cell input from fully scored assignments."""
    versions = tuple(range(COMMON_START_VERSION, COMMON_END_VERSION + 1))
    if (
        not rows
        or len({row.assignment_id for row in rows}) != len(rows)
        or any(row.start_version not in versions for row in rows)
        or any(row.delivered is None for row in rows)
    ):
        raise OpportunityLossGridAnalysisError(
            "common-window rows must be unique, in-window, and completely scored"
        )
    opportunity_scale = math.sqrt(
        math.fsum(row.opportunity**2 for row in rows) / len(rows)
    )
    if not math.isfinite(opportunity_scale) or opportunity_scale <= 0.0:
        raise OpportunityLossGridAnalysisError("opportunity scale must be positive")
    work = _endpoint(
        rows,
        endpoint="lower",
        control_arm="control",
        treatment_arm="d5",
        versions=versions,
        folds=8,
        hac_lag=4,
        opportunity_scale=opportunity_scale,
    )
    shifts = _bootstrap_shifts(
        rows,
        work.scores,
        versions=versions,
        block_size=8,
        draws=bootstrap_draws,
        seed=bootstrap_seed,
    )
    return GridCellInference(
        estimate=work.estimate,
        hac_standard_error=work.hac_standard_error,
        bootstrap_shifts=tuple(shifts),
    )


def _probability(value: float, *, name: str) -> float:
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise OpportunityLossGridAnalysisError(f"{name} must be in (0, 1)")
    return value


def _type7(values: Sequence[float], probability: float) -> float:
    if not values:
        raise OpportunityLossGridAnalysisError("quantile requires values")
    _probability(probability, name="quantile probability")
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def infer_grid_interaction(
    cells: Mapping[str, GridCellInference],
    *,
    confidence: float = 0.95,
    expected_bootstrap_draws: int = 20_000,
) -> GridInteractionInference:
    """Combine independent common-window cell estimates.

    Each bootstrap shift must come from an independently seeded circular
    start-version block bootstrap within its cell. Draws are paired by index
    only after those independent resamples have been produced.
    """
    if set(cells) != set(CELL_NAMES):
        raise OpportunityLossGridAnalysisError("grid cell identities disagree")
    if expected_bootstrap_draws <= 0:
        raise OpportunityLossGridAnalysisError("bootstrap draws must be positive")
    confidence = _probability(confidence, name="confidence")
    for name in CELL_NAMES:
        cell = cells[name]
        if (
            not math.isfinite(cell.estimate)
            or not math.isfinite(cell.hac_standard_error)
            or cell.hac_standard_error < 0.0
            or cell.primary_start_version != COMMON_START_VERSION
            or cell.primary_end_version != COMMON_END_VERSION
            or len(cell.bootstrap_shifts) != expected_bootstrap_draws
            or any(not math.isfinite(value) for value in cell.bootstrap_shifts)
        ):
            raise OpportunityLossGridAnalysisError(f"{name} violates cell contract")

    estimates = {name: cells[name].estimate for name in CELL_NAMES}
    workload_contrasts = {
        "qwen3_0p6b_openmath_minus_gsm8k": (
            estimates["qwen3_0p6b_openmath"] - estimates["qwen3_0p6b_gsm8k"]
        ),
        "qwen3_1p7b_openmath_minus_gsm8k": (
            estimates["qwen3_1p7b_openmath"] - estimates["qwen3_1p7b_gsm8k"]
        ),
    }
    model_scale_contrasts = {
        "openmath_qwen3_1p7b_minus_0p6b": (
            estimates["qwen3_1p7b_openmath"] - estimates["qwen3_0p6b_openmath"]
        ),
        "gsm8k_qwen3_1p7b_minus_0p6b": (
            estimates["qwen3_1p7b_gsm8k"] - estimates["qwen3_0p6b_gsm8k"]
        ),
    }
    interaction = (
        workload_contrasts["qwen3_0p6b_openmath_minus_gsm8k"]
        - workload_contrasts["qwen3_1p7b_openmath_minus_gsm8k"]
    )
    standard_error = math.sqrt(
        math.fsum(cells[name].hac_standard_error ** 2 for name in CELL_NAMES)
    )
    z_value = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    hac_interval = (
        interaction - z_value * standard_error,
        interaction + z_value * standard_error,
    )
    interaction_shifts = [
        cells["qwen3_0p6b_openmath"].bootstrap_shifts[index]
        - cells["qwen3_0p6b_gsm8k"].bootstrap_shifts[index]
        - cells["qwen3_1p7b_openmath"].bootstrap_shifts[index]
        + cells["qwen3_1p7b_gsm8k"].bootstrap_shifts[index]
        for index in range(expected_bootstrap_draws)
    ]
    tail = (1.0 - confidence) / 2.0
    bootstrap_interval = (
        interaction - _type7(interaction_shifts, 1.0 - tail),
        interaction - _type7(interaction_shifts, tail),
    )
    envelope = (
        min(hac_interval[0], bootstrap_interval[0]),
        max(hac_interval[1], bootstrap_interval[1]),
    )
    conclusion = (
        "INTERACTION_DETECTED"
        if envelope[0] > 0.0 or envelope[1] < 0.0
        else "INTERACTION_INCONCLUSIVE"
    )
    return GridInteractionInference(
        cell_estimates=estimates,
        workload_contrasts=workload_contrasts,
        model_scale_contrasts=model_scale_contrasts,
        interaction_estimate=interaction,
        hac_standard_error=standard_error,
        hac_interval=hac_interval,
        bootstrap_interval=bootstrap_interval,
        confidence_envelope=envelope,
        confidence=confidence,
        bootstrap_draws=expected_bootstrap_draws,
        conclusion=conclusion,
    )
