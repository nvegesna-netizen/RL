# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Version-clustered inference for bounded randomized opportunity loss."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any, Literal

from tools.opportunity_ledger_join import JoinedOpportunityAssignment
from tools.opportunity_loss_analysis import bound_opportunity_loss

InferenceConclusion = Literal[
    "MATERIAL",
    "NOT_MATERIAL",
    "INCONCLUSIVE",
    "INSUFFICIENT_TERMINAL_COVERAGE",
]


class OpportunityLossInferenceError(ValueError):
    """Raised when registered clustered inference cannot be computed."""


@dataclass(frozen=True)
class EndpointInference:
    """Inference for one sharp missing-terminal endpoint."""

    estimate: float
    standard_error: float
    hac_interval: tuple[float, float]
    bootstrap_interval: tuple[float, float]
    envelope: tuple[float, float]


@dataclass(frozen=True)
class OpportunityLossInference:
    """Conservative sampling-plus-missingness result for Delta_L."""

    lower_endpoint: EndpointInference
    upper_endpoint: EndpointInference
    identification_interval: tuple[float, float]
    confidence_envelope: tuple[float, float]
    material_threshold: float
    material_p_value: float
    max_missing_fraction: float
    coverage_gate_passed: bool
    conclusion: InferenceConclusion
    bootstrap_draws: int
    bootstrap_seed: int
    hac_lag: int
    block_size: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible result."""
        return asdict(self)


def _validate_probability(value: float, *, name: str) -> float:
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise OpportunityLossInferenceError(
            f"{name} must be strictly between zero and one"
        )
    return value


def _type7(values: Sequence[float], probability: float) -> float:
    if not values:
        raise OpportunityLossInferenceError("quantile requires at least one value")
    _validate_probability(probability, name="quantile probability")
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _endpoint_outcome(
    row: JoinedOpportunityAssignment,
    *,
    endpoint: Literal["lower", "upper"],
    control_arm: str,
    treatment_arm: str,
) -> float:
    if row.delivered is True:
        d = 0.0
    elif row.delivered is False:
        d = 1.0
    elif row.arm == treatment_arm:
        d = 0.0 if endpoint == "lower" else 1.0
    elif row.arm == control_arm:
        d = 1.0 if endpoint == "lower" else 0.0
    else:
        d = 0.0
    return row.opportunity * d


def _arm_mean(
    rows: Sequence[JoinedOpportunityAssignment],
    *,
    arm: str,
    value: Mapping[str, float],
) -> float:
    selected = [value[row.assignment_id] for row in rows if row.arm == arm]
    if not selected:
        raise OpportunityLossInferenceError(f"bootstrap sample lacks arm {arm!r}")
    return math.fsum(selected) / len(selected)


def _estimate(
    rows: Sequence[JoinedOpportunityAssignment],
    *,
    endpoint: Literal["lower", "upper"],
    control_arm: str,
    treatment_arm: str,
) -> float:
    y = {
        row.assignment_id: _endpoint_outcome(
            row,
            endpoint=endpoint,
            control_arm=control_arm,
            treatment_arm=treatment_arm,
        )
        for row in rows
    }
    q = {row.assignment_id: row.opportunity for row in rows}
    control_q = _arm_mean(rows, arm=control_arm, value=q)
    if control_q <= 0.0:
        raise OpportunityLossInferenceError("mean control opportunity must be positive")
    return (
        _arm_mean(rows, arm=treatment_arm, value=y)
        - _arm_mean(rows, arm=control_arm, value=y)
    ) / control_q


def _influence_scores(
    rows: Sequence[JoinedOpportunityAssignment],
    *,
    endpoint: Literal["lower", "upper"],
    control_arm: str,
    treatment_arm: str,
    propensities: Mapping[str, float],
) -> tuple[float, list[float]]:
    for arm in (control_arm, treatment_arm):
        if arm not in propensities:
            raise OpportunityLossInferenceError(f"missing propensity for arm {arm!r}")
        _validate_probability(propensities[arm], name=f"propensity for {arm}")
    y = {
        row.assignment_id: _endpoint_outcome(
            row,
            endpoint=endpoint,
            control_arm=control_arm,
            treatment_arm=treatment_arm,
        )
        for row in rows
    }
    q = {row.assignment_id: row.opportunity for row in rows}
    mean_y_t = _arm_mean(rows, arm=treatment_arm, value=y)
    mean_y_c = _arm_mean(rows, arm=control_arm, value=y)
    mean_q_c = _arm_mean(rows, arm=control_arm, value=q)
    if mean_q_c <= 0.0:
        raise OpportunityLossInferenceError("mean control opportunity must be positive")
    numerator = mean_y_t - mean_y_c
    estimate = numerator / mean_q_c
    weight_means = {
        arm: math.fsum(
            (1.0 / propensities[arm] if row.arm == arm else 0.0) for row in rows
        )
        / len(rows)
        for arm in (control_arm, treatment_arm)
    }
    scores: list[float] = []
    for row in rows:
        psi_t_y = (
            (1.0 / propensities[treatment_arm])
            / weight_means[treatment_arm]
            * (y[row.assignment_id] - mean_y_t)
            if row.arm == treatment_arm
            else 0.0
        )
        psi_c_y = (
            (1.0 / propensities[control_arm])
            / weight_means[control_arm]
            * (y[row.assignment_id] - mean_y_c)
            if row.arm == control_arm
            else 0.0
        )
        psi_c_q = (
            (1.0 / propensities[control_arm])
            / weight_means[control_arm]
            * (q[row.assignment_id] - mean_q_c)
            if row.arm == control_arm
            else 0.0
        )
        scores.append(
            (psi_t_y - psi_c_y) / mean_q_c - numerator * psi_c_q / mean_q_c**2
        )
    return estimate, scores


def _hac_standard_error(
    rows: Sequence[JoinedOpportunityAssignment],
    scores: Sequence[float],
    *,
    versions: Sequence[int],
    lag: int,
) -> float:
    if lag < 0 or lag >= len(versions):
        raise OpportunityLossInferenceError("HAC lag must be in [0, cohort_count)")
    by_version: dict[int, float] = defaultdict(float)
    version_set = set(versions)
    for row, score in zip(rows, scores):
        if row.start_version not in version_set:
            raise OpportunityLossInferenceError("row falls outside registered cohorts")
        by_version[row.start_version] += score
    clustered = [by_version[version] for version in versions]
    long_run = math.fsum(value * value for value in clustered)
    for offset in range(1, lag + 1):
        weight = 1.0 - offset / (lag + 1.0)
        long_run += (
            2.0
            * weight
            * math.fsum(
                clustered[index] * clustered[index - offset]
                for index in range(offset, len(clustered))
            )
        )
    variance = len(versions) / (len(versions) - 1.0) * long_run / len(rows) ** 2
    if variance < -1e-15:
        raise OpportunityLossInferenceError("HAC variance is materially negative")
    return math.sqrt(max(variance, 0.0))


def _bootstrap_estimates(
    rows: Sequence[JoinedOpportunityAssignment],
    *,
    endpoint: Literal["lower", "upper"],
    control_arm: str,
    treatment_arm: str,
    versions: Sequence[int],
    block_size: int,
    draws: int,
    seed: int,
) -> list[float]:
    if block_size <= 0 or block_size > len(versions):
        raise OpportunityLossInferenceError("invalid circular bootstrap block size")
    if draws <= 0:
        raise OpportunityLossInferenceError("bootstrap draws must be positive")
    by_version: dict[int, list[JoinedOpportunityAssignment]] = defaultdict(list)
    for row in rows:
        by_version[row.start_version].append(row)
    rng = random.Random(seed)
    block_count = math.ceil(len(versions) / block_size)
    estimates: list[float] = []
    for draw in range(draws):
        sampled_versions: list[int] = []
        for _ in range(block_count):
            start = rng.randrange(len(versions))
            sampled_versions.extend(
                versions[(start + offset) % len(versions)]
                for offset in range(block_size)
            )
        sampled_versions = sampled_versions[: len(versions)]
        sampled: list[JoinedOpportunityAssignment] = []
        copy_index = 0
        for sampled_version in sampled_versions:
            for row in by_version[sampled_version]:
                sampled.append(
                    JoinedOpportunityAssignment(
                        assignment_id=f"{draw}:{copy_index}",
                        ordinal=row.ordinal,
                        start_version=sampled_version,
                        arm=row.arm,
                        opportunity=row.opportunity,
                        delivered=row.delivered,
                    )
                )
                copy_index += 1
        estimates.append(
            _estimate(
                sampled,
                endpoint=endpoint,
                control_arm=control_arm,
                treatment_arm=treatment_arm,
            )
        )
    return estimates


def infer_opportunity_loss(
    rows: Sequence[JoinedOpportunityAssignment],
    *,
    propensities: Mapping[str, float],
    primary_start_version: int,
    primary_end_version: int,
    control_arm: str = "control",
    treatment_arm: str = "d5",
    material_threshold: float = 0.2,
    max_missing_fraction: float = 0.01,
    confidence: float = 0.95,
    material_alpha: float = 0.05,
    hac_lag: int = 4,
    block_size: int = 8,
    bootstrap_draws: int = 20_000,
    bootstrap_seed: int = 20260811,
) -> OpportunityLossInference:
    """Run registered HAC/bootstrap inference on both missingness endpoints."""
    if not rows or len({row.assignment_id for row in rows}) != len(rows):
        raise OpportunityLossInferenceError("joined rows must be nonempty and unique")
    confidence = _validate_probability(confidence, name="confidence")
    material_alpha = _validate_probability(material_alpha, name="material_alpha")
    if primary_start_version < 0 or primary_end_version <= primary_start_version:
        raise OpportunityLossInferenceError("invalid primary cohort window")
    versions = tuple(range(primary_start_version, primary_end_version + 1))
    analysis_rows = [row.analysis_assignment() for row in rows]
    point = bound_opportunity_loss(
        analysis_rows,
        control_arm=control_arm,
        treatment_arm=treatment_arm,
        material_threshold=material_threshold,
        max_missing_fraction=max_missing_fraction,
    )
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    endpoints: dict[str, EndpointInference] = {}
    bootstrap_by_endpoint: dict[str, list[float]] = {}
    for endpoint in ("lower", "upper"):
        estimate, scores = _influence_scores(
            rows,
            endpoint=endpoint,
            control_arm=control_arm,
            treatment_arm=treatment_arm,
            propensities=propensities,
        )
        standard_error = _hac_standard_error(
            rows, scores, versions=versions, lag=hac_lag
        )
        hac_interval = (
            estimate - z * standard_error,
            estimate + z * standard_error,
        )
        bootstrap = _bootstrap_estimates(
            rows,
            endpoint=endpoint,
            control_arm=control_arm,
            treatment_arm=treatment_arm,
            versions=versions,
            block_size=block_size,
            draws=bootstrap_draws,
            seed=bootstrap_seed,
        )
        bootstrap_by_endpoint[endpoint] = bootstrap
        tail = (1.0 - confidence) / 2.0
        bootstrap_interval = (
            2.0 * estimate - _type7(bootstrap, 1.0 - tail),
            2.0 * estimate - _type7(bootstrap, tail),
        )
        endpoints[endpoint] = EndpointInference(
            estimate=estimate,
            standard_error=standard_error,
            hac_interval=hac_interval,
            bootstrap_interval=bootstrap_interval,
            envelope=(
                min(hac_interval[0], bootstrap_interval[0]),
                max(hac_interval[1], bootstrap_interval[1]),
            ),
        )

    lower = endpoints["lower"]
    upper = endpoints["upper"]
    if lower.estimate != point.delta_l_lower or upper.estimate != point.delta_l_upper:
        raise OpportunityLossInferenceError(
            "inference and identification endpoint estimates disagree"
        )
    if lower.standard_error == 0.0:
        p_hac = 0.0 if lower.estimate > material_threshold else 1.0
    else:
        p_hac = 1.0 - NormalDist().cdf(
            (lower.estimate - material_threshold) / lower.standard_error
        )
    bootstrap_lower = bootstrap_by_endpoint["lower"]
    p_bootstrap = (
        1
        + sum(
            value - lower.estimate >= lower.estimate - material_threshold
            for value in bootstrap_lower
        )
    ) / (len(bootstrap_lower) + 1)
    material_p = max(p_hac, p_bootstrap)
    confidence_envelope = (lower.envelope[0], upper.envelope[1])
    if not point.coverage_gate_passed:
        conclusion: InferenceConclusion = "INSUFFICIENT_TERMINAL_COVERAGE"
    elif confidence_envelope[0] > material_threshold and material_p <= material_alpha:
        conclusion = "MATERIAL"
    elif confidence_envelope[1] <= material_threshold:
        conclusion = "NOT_MATERIAL"
    else:
        conclusion = "INCONCLUSIVE"
    return OpportunityLossInference(
        lower_endpoint=lower,
        upper_endpoint=upper,
        identification_interval=(point.delta_l_lower, point.delta_l_upper),
        confidence_envelope=confidence_envelope,
        material_threshold=material_threshold,
        material_p_value=material_p,
        max_missing_fraction=max_missing_fraction,
        coverage_gate_passed=point.coverage_gate_passed,
        conclusion=conclusion,
        bootstrap_draws=bootstrap_draws,
        bootstrap_seed=bootstrap_seed,
        hac_lag=hac_lag,
        block_size=block_size,
    )
