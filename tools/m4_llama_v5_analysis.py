# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Frozen replicate-aware analysis helpers for the V5 Llama M4 study."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any

from tools.opportunity_ledger_join import JoinedOpportunityAssignment
from tools.opportunity_loss_adjusted_inference import _bootstrap_shifts, _endpoint


class M4LlamaV5AnalysisError(ValueError):
    """Raised when V5 inputs violate the prospective analysis contract."""


@dataclass(frozen=True)
class ReplicateEndpoint:
    """One independently analyzed sharp-bound endpoint."""

    estimate: float
    hac_standard_error: float
    bootstrap_shifts: tuple[float, ...]


@dataclass(frozen=True)
class ReplicateInference:
    """One 400-version acquisition replicate."""

    replicate: str
    lower: ReplicateEndpoint
    upper: ReplicateEndpoint
    control_missing_fraction: float
    treatment_missing_fraction: float
    primary_start_version: int = 8
    primary_end_version: int = 407


@dataclass(frozen=True)
class CombinedReplicateInference:
    """Equal-replicate-weight synthesis without concatenating version series."""

    replicate_estimates: dict[str, tuple[float, float]]
    identification_interval: tuple[float, float]
    confidence_envelope: tuple[float, float]
    lower_hac_interval: tuple[float, float]
    upper_hac_interval: tuple[float, float]
    lower_bootstrap_interval: tuple[float, float]
    upper_bootstrap_interval: tuple[float, float]
    material_p_value: float
    material_threshold: float
    coverage_gate_passed: bool
    conclusion: str
    confidence: float
    bootstrap_draws: int
    combination_rule: str = "equal_replicate_weight_independent_block_diagonal"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def groups_by_start_version(
    rows: Sequence[JoinedOpportunityAssignment],
) -> Counter[int]:
    """Count joined groups using the dataclass API, not mapping subscripting."""
    return Counter(row.start_version for row in rows)


def infer_replicate(
    rows: Sequence[JoinedOpportunityAssignment],
    *,
    replicate: str,
    bootstrap_seed: int,
    bootstrap_draws: int = 20_000,
) -> ReplicateInference:
    """Analyze one run independently under the accepted adjusted estimator."""
    versions = tuple(range(8, 408))
    if (
        replicate not in {"r1", "r2"}
        or not rows
        or len({row.assignment_id for row in rows}) != len(rows)
        or any(row.start_version not in versions for row in rows)
    ):
        raise M4LlamaV5AnalysisError("invalid replicate rows or identity")
    opportunity_scale = math.sqrt(
        math.fsum(row.opportunity**2 for row in rows) / len(rows)
    )
    if not math.isfinite(opportunity_scale) or opportunity_scale <= 0.0:
        raise M4LlamaV5AnalysisError("opportunity scale must be positive")
    works = {
        endpoint: _endpoint(
            rows,
            endpoint=endpoint,
            control_arm="control",
            treatment_arm="d5",
            versions=versions,
            folds=8,
            hac_lag=4,
            opportunity_scale=opportunity_scale,
        )
        for endpoint in ("lower", "upper")
    }
    endpoints = {
        name: ReplicateEndpoint(
            work.estimate,
            work.hac_standard_error,
            tuple(
                _bootstrap_shifts(
                    rows,
                    work.scores,
                    versions=versions,
                    block_size=8,
                    draws=bootstrap_draws,
                    seed=bootstrap_seed,
                )
            ),
        )
        for name, work in works.items()
    }

    def missing(arm: str) -> float:
        arm_rows = [row for row in rows if row.arm == arm]
        if not arm_rows:
            raise M4LlamaV5AnalysisError(f"replicate lacks arm {arm}")
        return sum(row.delivered is None for row in arm_rows) / len(arm_rows)

    return ReplicateInference(
        replicate=replicate,
        lower=endpoints["lower"],
        upper=endpoints["upper"],
        control_missing_fraction=missing("control"),
        treatment_missing_fraction=missing("d5"),
    )


def _type7(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 < probability < 1.0:
        raise M4LlamaV5AnalysisError("invalid quantile inputs")
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def combine_replicates(
    replicates: Mapping[str, ReplicateInference],
    *,
    confidence: float = 0.95,
    material_threshold: float = 0.2,
    material_alpha: float = 0.05,
    max_missing_fraction: float = 0.01,
    expected_bootstrap_draws: int = 20_000,
) -> CombinedReplicateInference:
    """Combine exactly two independent, identically weighted acquisition runs.

    HAC covariance is block diagonal across runs. Circular-block bootstrap draws
    are produced independently within each run before equal-weight combination.
    """
    if set(replicates) != {"r1", "r2"}:
        raise M4LlamaV5AnalysisError("exactly r1 and r2 are required")
    if not 0.0 < confidence < 1.0 or not 0.0 < material_alpha < 1.0:
        raise M4LlamaV5AnalysisError("invalid inferential probability")
    if material_threshold < 0.0 or not 0.0 <= max_missing_fraction <= 1.0:
        raise M4LlamaV5AnalysisError("invalid threshold or missingness limit")
    ordered = [replicates["r1"], replicates["r2"]]
    for value in ordered:
        if value.replicate not in {"r1", "r2"}:
            raise M4LlamaV5AnalysisError("replicate label disagrees")
        if (value.primary_start_version, value.primary_end_version) != (8, 407):
            raise M4LlamaV5AnalysisError("replicate window disagrees")
        if value.lower.estimate > value.upper.estimate + 1e-12:
            raise M4LlamaV5AnalysisError("sharp endpoints are reversed")
        for endpoint in (value.lower, value.upper):
            if (
                not math.isfinite(endpoint.estimate)
                or not math.isfinite(endpoint.hac_standard_error)
                or endpoint.hac_standard_error < 0.0
                or len(endpoint.bootstrap_shifts) != expected_bootstrap_draws
                or any(not math.isfinite(x) for x in endpoint.bootstrap_shifts)
            ):
                raise M4LlamaV5AnalysisError("invalid endpoint input")
        for fraction in (
            value.control_missing_fraction,
            value.treatment_missing_fraction,
        ):
            if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
                raise M4LlamaV5AnalysisError("invalid missingness fraction")

    def combine_endpoint(name: str) -> tuple[float, float, list[float]]:
        endpoints = [getattr(value, name) for value in ordered]
        estimate = math.fsum(value.estimate for value in endpoints) / 2.0
        standard_error = math.sqrt(
            math.fsum(value.hac_standard_error**2 for value in endpoints)
        ) / 2.0
        shifts = [
            (endpoints[0].bootstrap_shifts[index] + endpoints[1].bootstrap_shifts[index])
            / 2.0
            for index in range(expected_bootstrap_draws)
        ]
        return estimate, standard_error, shifts

    lower_estimate, lower_se, lower_shifts = combine_endpoint("lower")
    upper_estimate, upper_se, upper_shifts = combine_endpoint("upper")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    tail = (1.0 - confidence) / 2.0

    def intervals(
        estimate: float, standard_error: float, shifts: Sequence[float]
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        return (
            (estimate - z * standard_error, estimate + z * standard_error),
            (
                estimate - _type7(shifts, 1.0 - tail),
                estimate - _type7(shifts, tail),
            ),
        )

    lower_hac, lower_bootstrap = intervals(lower_estimate, lower_se, lower_shifts)
    upper_hac, upper_bootstrap = intervals(upper_estimate, upper_se, upper_shifts)
    envelope = (
        min(lower_hac[0], lower_bootstrap[0]),
        max(upper_hac[1], upper_bootstrap[1]),
    )
    if lower_se == 0.0:
        p_hac = 0.0 if lower_estimate > material_threshold else 1.0
    else:
        p_hac = 1.0 - NormalDist().cdf(
            (lower_estimate - material_threshold) / lower_se
        )
    p_bootstrap = (
        1
        + sum(shift >= lower_estimate - material_threshold for shift in lower_shifts)
    ) / (expected_bootstrap_draws + 1)
    p_value = max(p_hac, p_bootstrap)
    coverage = all(
        fraction <= max_missing_fraction
        for value in ordered
        for fraction in (
            value.control_missing_fraction,
            value.treatment_missing_fraction,
        )
    )
    if not coverage:
        conclusion = "INSUFFICIENT_TERMINAL_COVERAGE"
    elif envelope[0] > material_threshold and p_value <= material_alpha:
        conclusion = "MATERIAL"
    elif envelope[1] <= material_threshold:
        conclusion = "NOT_MATERIAL"
    else:
        conclusion = "INCONCLUSIVE"
    return CombinedReplicateInference(
        replicate_estimates={
            value.replicate: (value.lower.estimate, value.upper.estimate)
            for value in ordered
        },
        identification_interval=(lower_estimate, upper_estimate),
        confidence_envelope=envelope,
        lower_hac_interval=lower_hac,
        upper_hac_interval=upper_hac,
        lower_bootstrap_interval=lower_bootstrap,
        upper_bootstrap_interval=upper_bootstrap,
        material_p_value=p_value,
        material_threshold=material_threshold,
        coverage_gate_passed=coverage,
        conclusion=conclusion,
        confidence=confidence,
        bootstrap_draws=expected_bootstrap_draws,
    )
