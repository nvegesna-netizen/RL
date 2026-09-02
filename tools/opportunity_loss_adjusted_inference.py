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

"""Cross-fitted pre-treatment adjustment for opportunity-loss follow-ups."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any, Literal

from tools.opportunity_ledger_join import JoinedOpportunityAssignment
from tools.opportunity_loss_inference import _hac_standard_error

Endpoint = Literal["lower", "upper"]


class AdjustedOpportunityLossError(ValueError):
    """Raised when adjusted opportunity-loss inference is not identified."""


@dataclass(frozen=True)
class AdjustedEndpointInference:
    """One cross-fitted sharp-bound endpoint used for prospective design."""

    estimate: float
    hac_standard_error: float
    numerator: float
    pooled_mean_opportunity: float
    hac_interval: tuple[float, float]
    bootstrap_interval: tuple[float, float]
    envelope: tuple[float, float]


@dataclass(frozen=True)
class AdjustedOpportunityLossInference:
    """Cross-fitted lower and upper opportunity-loss endpoints."""

    lower_endpoint: AdjustedEndpointInference
    upper_endpoint: AdjustedEndpointInference
    folds: int
    hac_lag: int
    covariates: tuple[str, ...]
    denominator: str
    identification_interval: tuple[float, float]
    confidence_envelope: tuple[float, float]
    material_threshold: float
    material_p_value: float
    max_missing_fraction: float
    coverage_gate_passed: bool
    conclusion: str
    confidence: float
    material_alpha: float
    bootstrap_draws: int
    bootstrap_seed: int
    block_size: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible result."""
        return asdict(self)


@dataclass(frozen=True)
class _EndpointWork:
    estimate: float
    hac_standard_error: float
    numerator: float
    pooled_mean_opportunity: float
    scores: tuple[float, ...]


def _probability(value: float, *, name: str) -> float:
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise AdjustedOpportunityLossError(f"{name} must be in (0, 1)")
    return value


def _type7(values: Sequence[float], probability: float) -> float:
    if not values:
        raise AdjustedOpportunityLossError("quantile requires values")
    _probability(probability, name="quantile probability")
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _solve(matrix: Sequence[Sequence[float]], target: Sequence[float]) -> list[float]:
    size = len(target)
    if size == 0 or len(matrix) != size or any(len(row) != size for row in matrix):
        raise AdjustedOpportunityLossError("invalid regression dimensions")
    augmented = [list(row) + [target[index]] for index, row in enumerate(matrix)]
    scale = max((abs(value) for row in matrix for value in row), default=1.0)
    ridge = max(scale * 1e-10, 1e-12)
    for index in range(size):
        augmented[index][index] += ridge
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1e-15:
            raise AdjustedOpportunityLossError("singular regression design")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            multiple = augmented[row][column]
            augmented[row] = [
                left - multiple * right
                for left, right in zip(augmented[row], augmented[column], strict=True)
            ]
    return [augmented[index][-1] for index in range(size)]


def _fit(features: Sequence[Sequence[float]], outcomes: Sequence[float]) -> list[float]:
    if not features or len(features) != len(outcomes):
        raise AdjustedOpportunityLossError("empty or mismatched regression data")
    width = len(features[0])
    if width == 0 or any(len(row) != width for row in features):
        raise AdjustedOpportunityLossError("inconsistent regression features")
    gram = [
        [
            math.fsum(row[left] * row[right] for row in features)
            for right in range(width)
        ]
        for left in range(width)
    ]
    cross = [
        math.fsum(row[column] * value for row, value in zip(features, outcomes))
        for column in range(width)
    ]
    return _solve(gram, cross)


def _outcome(
    row: JoinedOpportunityAssignment,
    *,
    endpoint: Endpoint,
    control_arm: str,
    treatment_arm: str,
) -> float:
    if row.delivered is True:
        disposition = 0.0
    elif row.delivered is False:
        disposition = 1.0
    elif row.arm == treatment_arm:
        disposition = 0.0 if endpoint == "lower" else 1.0
    elif row.arm == control_arm:
        disposition = 1.0 if endpoint == "lower" else 0.0
    else:
        disposition = 0.0
    return row.opportunity * disposition


def _features(
    row: JoinedOpportunityAssignment, *, opportunity_scale: float
) -> tuple[float, ...]:
    return (
        1.0,
        row.opportunity / opportunity_scale,
        float(row.opportunity == 0.0),
    )


def _fold_by_version(versions: Sequence[int], folds: int) -> dict[int, int]:
    if folds < 2 or folds > len(versions):
        raise AdjustedOpportunityLossError("folds must be in [2, cohort_count]")
    return {
        version: min(folds - 1, index * folds // len(versions))
        for index, version in enumerate(versions)
    }


def _cross_fitted_predictions(
    rows: Sequence[JoinedOpportunityAssignment],
    *,
    outcomes: Sequence[float],
    arm: str,
    versions: Sequence[int],
    folds: int,
    opportunity_scale: float,
) -> list[float]:
    fold_by_version = _fold_by_version(versions, folds)
    predictions = [0.0] * len(rows)
    row_features = [_features(row, opportunity_scale=opportunity_scale) for row in rows]
    for fold in range(folds):
        training = [
            index
            for index, row in enumerate(rows)
            if fold_by_version[row.start_version] != fold and row.arm == arm
        ]
        if not training:
            raise AdjustedOpportunityLossError(f"fold {fold} lacks training arm {arm}")
        coefficients = _fit(
            [row_features[index] for index in training],
            [outcomes[index] for index in training],
        )
        for index, row in enumerate(rows):
            if fold_by_version[row.start_version] == fold:
                predictions[index] = math.fsum(
                    coefficient * value
                    for coefficient, value in zip(
                        coefficients, row_features[index], strict=True
                    )
                )
    return predictions


def _endpoint(
    rows: Sequence[JoinedOpportunityAssignment],
    *,
    endpoint: Endpoint,
    control_arm: str,
    treatment_arm: str,
    versions: Sequence[int],
    folds: int,
    hac_lag: int,
    opportunity_scale: float,
) -> _EndpointWork:
    outcomes = [
        _outcome(
            row,
            endpoint=endpoint,
            control_arm=control_arm,
            treatment_arm=treatment_arm,
        )
        for row in rows
    ]
    treatment_predictions = _cross_fitted_predictions(
        rows,
        outcomes=outcomes,
        arm=treatment_arm,
        versions=versions,
        folds=folds,
        opportunity_scale=opportunity_scale,
    )
    control_predictions = _cross_fitted_predictions(
        rows,
        outcomes=outcomes,
        arm=control_arm,
        versions=versions,
        folds=folds,
        opportunity_scale=opportunity_scale,
    )
    treatment_count = sum(row.arm == treatment_arm for row in rows)
    control_count = sum(row.arm == control_arm for row in rows)
    if treatment_count == 0 or control_count == 0:
        raise AdjustedOpportunityLossError("both primary arms must be observed")
    row_count = len(rows)
    pseudo_outcomes = []
    for index, row in enumerate(rows):
        value = treatment_predictions[index] - control_predictions[index]
        if row.arm == treatment_arm:
            value += (
                row_count
                / treatment_count
                * (outcomes[index] - treatment_predictions[index])
            )
        elif row.arm == control_arm:
            value -= (
                row_count
                / control_count
                * (outcomes[index] - control_predictions[index])
            )
        pseudo_outcomes.append(value)
    numerator = math.fsum(pseudo_outcomes) / row_count
    mean_opportunity = math.fsum(row.opportunity for row in rows) / row_count
    if mean_opportunity <= 0.0:
        raise AdjustedOpportunityLossError("pooled mean opportunity must be positive")
    estimate = numerator / mean_opportunity
    scores = [
        (pseudo - numerator) / mean_opportunity
        - estimate * (row.opportunity - mean_opportunity) / mean_opportunity
        for row, pseudo in zip(rows, pseudo_outcomes, strict=True)
    ]
    standard_error = _hac_standard_error(rows, scores, versions=versions, lag=hac_lag)
    return _EndpointWork(
        estimate=estimate,
        hac_standard_error=standard_error,
        numerator=numerator,
        pooled_mean_opportunity=mean_opportunity,
        scores=tuple(scores),
    )


def _bootstrap_shifts(
    rows: Sequence[JoinedOpportunityAssignment],
    scores: Sequence[float],
    *,
    versions: Sequence[int],
    block_size: int,
    draws: int,
    seed: int,
) -> list[float]:
    if block_size <= 0 or block_size > len(versions):
        raise AdjustedOpportunityLossError("invalid circular bootstrap block size")
    if draws <= 0:
        raise AdjustedOpportunityLossError("bootstrap draws must be positive")
    by_version: dict[int, list[float]] = {version: [] for version in versions}
    for row, score in zip(rows, scores, strict=True):
        by_version[row.start_version].append(score)
    rng = random.Random(seed)
    block_count = math.ceil(len(versions) / block_size)
    shifts = []
    for _draw in range(draws):
        sampled_versions = []
        for _block in range(block_count):
            start = rng.randrange(len(versions))
            sampled_versions.extend(
                versions[(start + offset) % len(versions)]
                for offset in range(block_size)
            )
        sampled_scores = [
            score
            for version in sampled_versions[: len(versions)]
            for score in by_version[version]
        ]
        if not sampled_scores:
            raise AdjustedOpportunityLossError("bootstrap sample has no assignments")
        shifts.append(math.fsum(sampled_scores) / len(sampled_scores))
    return shifts


def _finalize_endpoint(
    work: _EndpointWork,
    *,
    shifts: Sequence[float],
    confidence: float,
) -> AdjustedEndpointInference:
    tail = (1.0 - confidence) / 2.0
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    hac_interval = (
        work.estimate - z * work.hac_standard_error,
        work.estimate + z * work.hac_standard_error,
    )
    bootstrap_interval = (
        work.estimate - _type7(shifts, 1.0 - tail),
        work.estimate - _type7(shifts, tail),
    )
    return AdjustedEndpointInference(
        estimate=work.estimate,
        hac_standard_error=work.hac_standard_error,
        numerator=work.numerator,
        pooled_mean_opportunity=work.pooled_mean_opportunity,
        hac_interval=hac_interval,
        bootstrap_interval=bootstrap_interval,
        envelope=(
            min(hac_interval[0], bootstrap_interval[0]),
            max(hac_interval[1], bootstrap_interval[1]),
        ),
    )


def infer_adjusted_opportunity_loss(
    rows: Sequence[JoinedOpportunityAssignment],
    *,
    propensities: Mapping[str, float],
    primary_start_version: int,
    primary_end_version: int,
    control_arm: str = "control",
    treatment_arm: str = "d5",
    folds: int = 8,
    hac_lag: int = 4,
    confidence: float = 0.95,
    material_threshold: float = 0.2,
    material_alpha: float = 0.05,
    max_missing_fraction: float = 0.01,
    block_size: int = 8,
    bootstrap_draws: int = 20_000,
    bootstrap_seed: int = 20260903,
) -> AdjustedOpportunityLossInference:
    """Estimate the same population contrast with pre-delay adjustment.

    Nuisance regressions use only opportunity ``Q`` and ``1[Q=0]``. They are fit
    outside each contiguous start-version fold. The estimand denominator is pooled
    pre-treatment opportunity, which equals control opportunity in the randomized
    population while using information from every assignment.
    """
    if not rows or len({row.assignment_id for row in rows}) != len(rows):
        raise AdjustedOpportunityLossError("rows must be nonempty and unique")
    if not control_arm or not treatment_arm or control_arm == treatment_arm:
        raise AdjustedOpportunityLossError("primary arms must be distinct")
    for arm in (control_arm, treatment_arm):
        propensity = propensities.get(arm)
        if (
            isinstance(propensity, bool)
            or not isinstance(propensity, (int, float))
            or not math.isfinite(propensity)
            or not 0.0 < propensity < 1.0
        ):
            raise AdjustedOpportunityLossError(f"invalid propensity for {arm}")
    confidence = _probability(confidence, name="confidence")
    material_alpha = _probability(material_alpha, name="material alpha")
    if (
        not math.isfinite(material_threshold)
        or material_threshold < 0.0
        or not math.isfinite(max_missing_fraction)
        or not 0.0 <= max_missing_fraction <= 1.0
    ):
        raise AdjustedOpportunityLossError("invalid threshold or missingness limit")
    if primary_start_version < 0 or primary_end_version <= primary_start_version:
        raise AdjustedOpportunityLossError("invalid primary version window")
    versions = tuple(range(primary_start_version, primary_end_version + 1))
    version_set = set(versions)
    if any(row.start_version not in version_set for row in rows):
        raise AdjustedOpportunityLossError("row lies outside the primary window")
    if hac_lag < 0 or hac_lag >= len(versions):
        raise AdjustedOpportunityLossError("invalid HAC lag")
    opportunity_scale = math.sqrt(
        math.fsum(row.opportunity**2 for row in rows) / len(rows)
    )
    if not math.isfinite(opportunity_scale) or opportunity_scale <= 0.0:
        raise AdjustedOpportunityLossError("opportunity scale must be positive")
    lower_work = _endpoint(
        rows,
        endpoint="lower",
        control_arm=control_arm,
        treatment_arm=treatment_arm,
        versions=versions,
        folds=folds,
        hac_lag=hac_lag,
        opportunity_scale=opportunity_scale,
    )
    upper_work = _endpoint(
        rows,
        endpoint="upper",
        control_arm=control_arm,
        treatment_arm=treatment_arm,
        versions=versions,
        folds=folds,
        hac_lag=hac_lag,
        opportunity_scale=opportunity_scale,
    )
    if lower_work.estimate > upper_work.estimate + 1e-12:
        raise AdjustedOpportunityLossError("adjusted sharp endpoints are reversed")
    lower_shifts = _bootstrap_shifts(
        rows,
        lower_work.scores,
        versions=versions,
        block_size=block_size,
        draws=bootstrap_draws,
        seed=bootstrap_seed,
    )
    upper_shifts = _bootstrap_shifts(
        rows,
        upper_work.scores,
        versions=versions,
        block_size=block_size,
        draws=bootstrap_draws,
        seed=bootstrap_seed,
    )
    lower = _finalize_endpoint(lower_work, shifts=lower_shifts, confidence=confidence)
    upper = _finalize_endpoint(upper_work, shifts=upper_shifts, confidence=confidence)
    if lower.hac_standard_error == 0.0:
        p_hac = 0.0 if lower.estimate > material_threshold else 1.0
    else:
        p_hac = 1.0 - NormalDist().cdf(
            (lower.estimate - material_threshold) / lower.hac_standard_error
        )
    p_bootstrap = (
        1 + sum(shift >= lower.estimate - material_threshold for shift in lower_shifts)
    ) / (len(lower_shifts) + 1)
    material_p_value = max(p_hac, p_bootstrap)
    arm_missing = {
        arm: (
            sum(row.delivered is None for row in rows if row.arm == arm)
            / sum(row.arm == arm for row in rows)
        )
        for arm in (control_arm, treatment_arm)
    }
    coverage_gate_passed = all(
        fraction <= max_missing_fraction for fraction in arm_missing.values()
    )
    confidence_envelope = (lower.envelope[0], upper.envelope[1])
    if not coverage_gate_passed:
        conclusion = "INSUFFICIENT_TERMINAL_COVERAGE"
    elif (
        confidence_envelope[0] > material_threshold
        and material_p_value <= material_alpha
    ):
        conclusion = "MATERIAL"
    elif confidence_envelope[1] <= material_threshold:
        conclusion = "NOT_MATERIAL"
    else:
        conclusion = "INCONCLUSIVE"
    return AdjustedOpportunityLossInference(
        lower_endpoint=lower,
        upper_endpoint=upper,
        folds=folds,
        hac_lag=hac_lag,
        covariates=("opportunity_Q", "opportunity_is_zero"),
        denominator="pooled_pre_delay_mean_opportunity",
        identification_interval=(lower.estimate, upper.estimate),
        confidence_envelope=confidence_envelope,
        material_threshold=material_threshold,
        material_p_value=material_p_value,
        max_missing_fraction=max_missing_fraction,
        coverage_gate_passed=coverage_gate_passed,
        conclusion=conclusion,
        confidence=confidence,
        material_alpha=material_alpha,
        bootstrap_draws=bootstrap_draws,
        bootstrap_seed=bootstrap_seed,
        block_size=block_size,
    )
