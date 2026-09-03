#!/usr/bin/env python3
"""Explore prospective pre-treatment adjustment on frozen M4 assignments.

The output is design evidence only. It estimates potential variance reduction for
a future independent study and cannot replace the registered confirmatory result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path

from tools.opportunity_ledger_join import JoinedOpportunityAssignment
from tools.opportunity_loss_adjusted_inference import (
    infer_adjusted_opportunity_loss,
)
from tools.opportunity_loss_inference import (
    _hac_standard_error,
    _influence_scores,
)
from tools.opportunity_loss_pipeline import _parse_jsonl, _parse_protocol

ARTIFACT_SHA256 = "e81a29747628b6f0490323207f317755de411d0f8e7527f715368919aee12fb7"
RESULT_SHA256 = "6c3fc0cf0567d4dab63153769c075dc7d491e41cb2ceb982b569649f5b42ced3"
PREFIX = "workspace/assets/basic/m4-opportunity-loss-confirmatory-acquisition-r4/"
MEMBERS = {
    "lifecycle": PREFIX + "m4-opportunity-loss-lifecycle.jsonl",
    "opportunity": PREFIX + "m4-opportunity-loss-opportunity.jsonl",
    "result": PREFIX + "m4-opportunity-loss-result.json",
}
CONTROL_ARM = "control"
TREATMENT_ARM = "d5"
FOLD_COUNTS = (4, 8, 16)
HAC_LAG = 4


class EfficiencyAuditError(ValueError):
    """Raised when evidence or exploratory calculations fail validation."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _solve(matrix: Sequence[Sequence[float]], target: Sequence[float]) -> list[float]:
    """Solve a small regularized linear system with pivoted elimination."""
    size = len(target)
    if size == 0 or len(matrix) != size or any(len(row) != size for row in matrix):
        raise EfficiencyAuditError("invalid normal-equation dimensions")
    augmented = [list(row) + [target[index]] for index, row in enumerate(matrix)]
    scale = max((abs(value) for row in matrix for value in row), default=1.0)
    ridge = max(scale * 1e-10, 1e-12)
    for index in range(size):
        augmented[index][index] += ridge
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1e-15:
            raise EfficiencyAuditError("singular adjustment model")
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


def _fit_no_intercept(
    predictors: Sequence[Sequence[float]], target: Sequence[float]
) -> list[float]:
    if len(predictors) != len(target) or not predictors:
        raise EfficiencyAuditError("invalid adjustment training data")
    width = len(predictors[0])
    if width == 0 or any(len(row) != width for row in predictors):
        raise EfficiencyAuditError("invalid adjustment feature width")
    gram = [
        [
            math.fsum(row[left] * row[right] for row in predictors)
            for right in range(width)
        ]
        for left in range(width)
    ]
    cross = [
        math.fsum(row[column] * value for row, value in zip(predictors, target))
        for column in range(width)
    ]
    return _solve(gram, cross)


def _cluster_sums(
    rows: Sequence[JoinedOpportunityAssignment],
    values: Sequence[float],
    versions: Sequence[int],
) -> list[float]:
    by_version = {version: 0.0 for version in versions}
    for row, value in zip(rows, values, strict=True):
        by_version[row.start_version] += value
    return [by_version[version] for version in versions]


def _arm_contrast_weights(
    rows: Sequence[JoinedOpportunityAssignment], propensities: dict[str, float]
) -> list[float]:
    weight_means = {
        arm: math.fsum(
            1.0 / propensities[arm] if row.arm == arm else 0.0 for row in rows
        )
        / len(rows)
        for arm in (CONTROL_ARM, TREATMENT_ARM)
    }
    return [
        (
            (1.0 / propensities[TREATMENT_ARM]) / weight_means[TREATMENT_ARM]
            if row.arm == TREATMENT_ARM
            else 0.0
        )
        - (
            (1.0 / propensities[CONTROL_ARM]) / weight_means[CONTROL_ARM]
            if row.arm == CONTROL_ARM
            else 0.0
        )
        for row in rows
    ]


def _cross_fitted_adjustment(
    *,
    rows: Sequence[JoinedOpportunityAssignment],
    original_scores: Sequence[float],
    contrast_weights: Sequence[float],
    versions: Sequence[int],
    feature: Callable[[JoinedOpportunityAssignment], Sequence[float]],
    folds: int,
) -> tuple[list[float], list[list[float]]]:
    row_features = [tuple(feature(row)) for row in rows]
    width = len(row_features[0])
    if width == 0 or any(len(values) != width for values in row_features):
        raise EfficiencyAuditError("inconsistent adjustment feature shape")
    imbalance = [
        [weight * value for value in values]
        for weight, values in zip(contrast_weights, row_features, strict=True)
    ]
    score_clusters = _cluster_sums(rows, original_scores, versions)
    feature_clusters = [
        _cluster_sums(rows, [values[column] for values in imbalance], versions)
        for column in range(width)
    ]
    coefficients_by_fold: list[list[float]] = []
    adjusted = list(original_scores)
    version_to_index = {version: index for index, version in enumerate(versions)}
    fold_for_version = {
        version: min(folds - 1, index * folds // len(versions))
        for version, index in version_to_index.items()
    }
    for fold in range(folds):
        training_indices = [
            index
            for index, version in enumerate(versions)
            if fold_for_version[version] != fold
        ]
        predictors = [
            [feature_clusters[column][index] for column in range(width)]
            for index in training_indices
        ]
        target = [score_clusters[index] for index in training_indices]
        coefficients = _fit_no_intercept(predictors, target)
        coefficients_by_fold.append(coefficients)
        for index, row in enumerate(rows):
            if fold_for_version[row.start_version] == fold:
                adjusted[index] -= math.fsum(
                    coefficient * value
                    for coefficient, value in zip(
                        coefficients, imbalance[index], strict=True
                    )
                )
    mean_adjusted = math.fsum(adjusted) / len(adjusted)
    adjusted = [value - mean_adjusted for value in adjusted]
    return adjusted, coefficients_by_fold


def _arm_summary(
    rows: Sequence[JoinedOpportunityAssignment], arm: str
) -> dict[str, float | int]:
    selected = [row for row in rows if row.arm == arm]
    opportunities = [row.opportunity for row in selected]
    lost = [row.opportunity * (row.delivered is False) for row in selected]
    return {
        "assignments": len(selected),
        "mean_opportunity": math.fsum(opportunities) / len(selected),
        "zero_opportunity_fraction": sum(value == 0.0 for value in opportunities)
        / len(selected),
        "undelivered_fraction": sum(row.delivered is False for row in selected)
        / len(selected),
        "mean_lost_opportunity": math.fsum(lost) / len(selected),
        "missing_terminal_count": sum(row.delivered is None for row in selected),
    }


def build_audit(artifact: Path, protocol_path: Path) -> dict[str, object]:
    artifact_raw = artifact.read_bytes()
    if _sha256(artifact_raw) != ARTIFACT_SHA256:
        raise EfficiencyAuditError("artifact SHA-256 disagrees")
    protocol_raw = protocol_path.read_bytes()
    protocol_object, protocol, _options = _parse_protocol(protocol_raw)
    try:
        with zipfile.ZipFile(artifact) as archive:
            raw = {name: archive.read(member) for name, member in MEMBERS.items()}
    except (OSError, KeyError, zipfile.BadZipFile) as error:
        raise EfficiencyAuditError(f"cannot read artifact: {error}") from error
    if _sha256(raw["result"]) != RESULT_SHA256:
        raise EfficiencyAuditError("result SHA-256 disagrees")
    result = json.loads(raw["result"])
    if _canonical_json(result) != raw["result"]:
        raise EfficiencyAuditError("result is not canonical JSON")
    if result["protocol"] != {
        "sha256": _sha256(protocol_raw),
        "size": len(protocol_raw),
    }:
        raise EfficiencyAuditError("protocol does not match the frozen result")
    lifecycle = _parse_jsonl(raw["lifecycle"], compact=False, name="lifecycle")
    opportunity = _parse_jsonl(raw["opportunity"], compact=True, name="opportunity")
    from tools.opportunity_ledger_join import join_opportunity_ledgers

    rows = join_opportunity_ledgers(
        protocol=protocol,
        lifecycle_rows=lifecycle,
        opportunity_rows=opportunity,
    )
    total_mass = sum(arm.mass for arm in protocol.arms)
    propensities = {arm.label: arm.mass / total_mass for arm in protocol.arms}
    estimate, original_scores = _influence_scores(
        rows,
        endpoint="lower",
        control_arm=CONTROL_ARM,
        treatment_arm=TREATMENT_ARM,
        propensities=propensities,
    )
    versions = tuple(
        range(protocol.primary_start_version, protocol.primary_end_version + 1)
    )
    original_se = _hac_standard_error(
        rows, original_scores, versions=versions, lag=HAC_LAG
    )
    frozen_endpoint = result["causal_inference"]["lower_endpoint"]
    if (
        estimate != frozen_endpoint["estimate"]
        or original_se != frozen_endpoint["standard_error"]
    ):
        raise EfficiencyAuditError("assignment reconstruction does not match result")

    q_scale = math.sqrt(math.fsum(row.opportunity**2 for row in rows) / len(rows))
    if q_scale <= 0.0:
        raise EfficiencyAuditError("opportunity scale must be positive")
    first_version = versions[0]
    version_scale = versions[-1] - first_version
    contrast_weights = _arm_contrast_weights(rows, propensities)
    feature_sets: dict[
        str, Callable[[JoinedOpportunityAssignment], Sequence[float]]
    ] = {
        "q": lambda row: (row.opportunity / q_scale,),
        "q_and_zero_indicator": lambda row: (
            row.opportunity / q_scale,
            float(row.opportunity == 0.0),
        ),
        "q_zero_and_linear_version": lambda row: (
            row.opportunity / q_scale,
            float(row.opportunity == 0.0),
            (row.start_version - first_version) / version_scale,
        ),
    }
    models = []
    for name, feature in feature_sets.items():
        fold_sensitivity = []
        for folds in FOLD_COUNTS:
            adjusted, coefficients = _cross_fitted_adjustment(
                rows=rows,
                original_scores=original_scores,
                contrast_weights=contrast_weights,
                versions=versions,
                feature=feature,
                folds=folds,
            )
            adjusted_se = _hac_standard_error(
                rows, adjusted, versions=versions, lag=HAC_LAG
            )
            fold_sensitivity.append(
                {
                    "cross_fit_folds": folds,
                    "adjusted_hac_standard_error": adjusted_se,
                    "variance_ratio_vs_registered": (adjusted_se / original_se) ** 2,
                    "estimated_variance_reduction": 1.0
                    - (adjusted_se / original_se) ** 2,
                    "coefficients_by_fold": coefficients,
                }
            )
        models.append(
            {
                "name": name,
                "fold_sensitivity": fold_sensitivity,
                "minimum_estimated_variance_reduction": min(
                    float(item["estimated_variance_reduction"])
                    for item in fold_sensitivity
                ),
            }
        )

    primary_pair_rows = [row for row in rows if row.arm in (CONTROL_ARM, TREATMENT_ARM)]
    formal_estimator = {}
    for universe, universe_rows, universe_propensities in (
        ("all_randomized_arms", rows, propensities),
        (
            "control_and_d5_only",
            primary_pair_rows,
            {CONTROL_ARM: 0.5, TREATMENT_ARM: 0.5},
        ),
    ):
        formal_estimator[universe] = []
        for folds in FOLD_COUNTS:
            adjusted_result = infer_adjusted_opportunity_loss(
                universe_rows,
                propensities=universe_propensities,
                primary_start_version=protocol.primary_start_version,
                primary_end_version=protocol.primary_end_version,
                folds=folds,
                hac_lag=HAC_LAG,
            )
            formal_estimator[universe].append(adjusted_result.to_dict())

    return {
        "schema": "m4-opportunity-loss-pretreatment-efficiency-audit-v1",
        "status": "EXPLORATORY_DESIGN_ONLY_DOES_NOT_MODIFY_CONFIRMATORY_RESULT",
        "source": {
            "artifact_sha256": ARTIFACT_SHA256,
            "result_sha256": RESULT_SHA256,
            "protocol_sha256": _sha256(protocol_raw),
            "protocol": protocol_object["protocol"],
        },
        "reconstruction": {
            "assignment_count": len(rows),
            "cohort_count": len(versions),
            "registered_estimate": estimate,
            "registered_hac_standard_error": original_se,
            "exactly_matches_frozen_result": True,
        },
        "arm_summaries": {
            arm: _arm_summary(rows, arm) for arm in (CONTROL_ARM, TREATMENT_ARM, "d10")
        },
        "candidate_adjustments": models,
        "formal_adjusted_estimator_sensitivity": formal_estimator,
        "interpretation_contract": {
            "pretreatment_covariates_only": ["opportunity_Q", "start_weight_version"],
            "cross_fit_unit": "contiguous_start_weight_version_blocks",
            "point_estimate_reclassification_forbidden": True,
            "use": "prospective_estimator_and_sample_size_selection_only",
        },
    }


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(_canonical_json(value))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    _write_atomic(args.output, build_audit(args.artifact, args.protocol))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
