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
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class AdjustedOpportunityLossInference:
    """Cross-fitted lower and upper opportunity-loss endpoints."""

    lower_endpoint: AdjustedEndpointInference
    upper_endpoint: AdjustedEndpointInference
    folds: int
    hac_lag: int
    covariates: tuple[str, ...]
    denominator: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible result."""
        return asdict(self)


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
) -> AdjustedEndpointInference:
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
    return AdjustedEndpointInference(
        estimate=estimate,
        hac_standard_error=standard_error,
        numerator=numerator,
        pooled_mean_opportunity=mean_opportunity,
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
    lower = _endpoint(
        rows,
        endpoint="lower",
        control_arm=control_arm,
        treatment_arm=treatment_arm,
        versions=versions,
        folds=folds,
        hac_lag=hac_lag,
        opportunity_scale=opportunity_scale,
    )
    upper = _endpoint(
        rows,
        endpoint="upper",
        control_arm=control_arm,
        treatment_arm=treatment_arm,
        versions=versions,
        folds=folds,
        hac_lag=hac_lag,
        opportunity_scale=opportunity_scale,
    )
    if lower.estimate > upper.estimate + 1e-12:
        raise AdjustedOpportunityLossError("adjusted sharp endpoints are reversed")
    return AdjustedOpportunityLossInference(
        lower_endpoint=lower,
        upper_endpoint=upper,
        folds=folds,
        hac_lag=hac_lag,
        covariates=("opportunity_Q", "opportunity_is_zero"),
        denominator="pooled_pre_delay_mean_opportunity",
    )
