#!/usr/bin/env python3
"""Post-primary robustness checks for the terminal M4 magnitude result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_gradient_utility_audit import (
    _cross_fitted_ridge,
    _ridge_fit_predict,
    _r2,
)

ALPHAS = (1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def _cross_fitted_fixed_alpha(
    features: np.ndarray,
    outcome: np.ndarray,
    folds: np.ndarray,
    alpha: float,
) -> np.ndarray:
    predictions = np.full(len(outcome), np.nan, dtype=np.float64)
    for fold in sorted(set(folds.tolist())):
        test = folds == fold
        predictions[test] = _ridge_fit_predict(
            features[~test], outcome[~test], features[test], alpha
        )
    if not np.isfinite(predictions).all():
        raise RuntimeError("fixed-alpha predictions contain nonfinite values")
    return predictions


def _fit_summary(
    baseline: np.ndarray,
    augmented: np.ndarray,
    outcome: np.ndarray,
    folds: np.ndarray,
) -> dict[str, float]:
    baseline_prediction = _cross_fitted_ridge(baseline, outcome, folds)
    augmented_prediction = _cross_fitted_ridge(augmented, outcome, folds)
    baseline_r2 = _r2(outcome, baseline_prediction)
    augmented_r2 = _r2(outcome, augmented_prediction)
    return {
        "baseline_cross_fitted_r2": baseline_r2,
        "augmented_cross_fitted_r2": augmented_r2,
        "gain": augmented_r2 - baseline_r2,
    }


def analyze(ledger_path: Path) -> dict[str, Any]:
    events = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    groups = [event for event in events if event.get("event_type") == "group_gradient"]
    if len(groups) != 256:
        raise ValueError(f"expected 256 groups, found {len(groups)}")

    metadata = [event["metadata"] for event in groups]
    exact_norm = np.asarray(
        [event["summary"]["exact_l2_norm"] for event in groups], dtype=np.float64
    )
    m4 = np.asarray(
        [row["gradient_opportunity_l1"] for row in metadata], dtype=np.float64
    )
    valid_tokens = np.asarray(
        [row["gradient_opportunity_valid_actor_tokens"] for row in metadata],
        dtype=np.float64,
    )
    reward_mean = np.asarray(
        [row["gradient_opportunity_reward_mean"] for row in metadata], dtype=np.float64
    )
    reward_variance = np.asarray(
        [row["gradient_opportunity_reward_variance"] for row in metadata],
        dtype=np.float64,
    )
    truncation = np.asarray(
        [row["gradient_opportunity_truncation_count"] / 8.0 for row in metadata],
        dtype=np.float64,
    )
    folds = np.asarray([event["fold"] for event in groups], dtype=np.int64)
    primary = (m4 > 0) & (exact_norm > 0)

    def matrices(mask: np.ndarray, *, include_zero: bool = False):
        baseline = np.column_stack(
            [
                np.log1p(valid_tokens[mask]),
                reward_mean[mask],
                reward_variance[mask],
                truncation[mask],
            ]
        )
        log_m4 = np.log1p(m4[mask])
        outcome = (
            np.log1p(exact_norm[mask]) if include_zero else np.log(exact_norm[mask])
        )
        return baseline, log_m4, outcome, folds[mask]

    baseline, log_m4, outcome, primary_folds = matrices(primary)
    augmented = np.column_stack([baseline, log_m4])
    primary_summary = _fit_summary(baseline, augmented, outcome, primary_folds)
    m4_only_prediction = _cross_fitted_ridge(
        log_m4[:, None], outcome, primary_folds
    )
    primary_summary["m4_only_cross_fitted_r2"] = _r2(
        outcome, m4_only_prediction
    )

    leave_one_fold_out = []
    for omitted_fold in sorted(set(primary_folds.tolist())):
        keep = primary_folds != omitted_fold
        summary = _fit_summary(
            baseline[keep], augmented[keep], outcome[keep], primary_folds[keep]
        )
        summary.update(
            {"omitted_fold": int(omitted_fold), "groups": int(keep.sum())}
        )
        leave_one_fold_out.append(summary)

    winsorization = []
    for tail_fraction in (0.01, 0.05):
        lower, upper = np.quantile(outcome, [tail_fraction, 1.0 - tail_fraction])
        z_lower, z_upper = np.quantile(
            log_m4, [tail_fraction, 1.0 - tail_fraction]
        )
        clipped_outcome = np.clip(outcome, lower, upper)
        clipped_m4 = np.clip(log_m4, z_lower, z_upper)
        summary = _fit_summary(
            baseline,
            np.column_stack([baseline, clipped_m4]),
            clipped_outcome,
            primary_folds,
        )
        summary["tail_fraction"] = tail_fraction
        winsorization.append(summary)

    all_mask = np.ones(len(groups), dtype=bool)
    all_baseline, all_m4, all_outcome, all_folds = matrices(
        all_mask, include_zero=True
    )
    including_zeros = _fit_summary(
        all_baseline,
        np.column_stack([all_baseline, all_m4]),
        all_outcome,
        all_folds,
    )
    including_zeros["groups"] = len(groups)

    fixed_alpha = []
    for alpha in ALPHAS:
        base_prediction = _cross_fitted_fixed_alpha(
            baseline, outcome, primary_folds, alpha
        )
        augmented_prediction = _cross_fitted_fixed_alpha(
            augmented, outcome, primary_folds, alpha
        )
        base_r2 = _r2(outcome, base_prediction)
        augmented_r2 = _r2(outcome, augmented_prediction)
        fixed_alpha.append(
            {
                "alpha": alpha,
                "baseline_cross_fitted_r2": base_r2,
                "augmented_cross_fitted_r2": augmented_r2,
                "gain": augmented_r2 - base_r2,
            }
        )

    fold_rng = np.random.default_rng(20261031)
    alternative_fold_gains = []
    for _ in range(100):
        permutation = fold_rng.permutation(len(outcome))
        alternative_folds = np.empty(len(outcome), dtype=np.int64)
        alternative_folds[permutation] = np.arange(len(outcome)) % 8
        summary = _fit_summary(
            baseline, augmented, outcome, alternative_folds
        )
        alternative_fold_gains.append(summary["gain"])

    permutation_rng = np.random.default_rng(20261029)
    permuted_gains = []
    baseline_r2 = primary_summary["baseline_cross_fitted_r2"]
    for _ in range(1000):
        permuted_m4 = log_m4.copy()
        for fold in sorted(set(primary_folds.tolist())):
            indices = np.flatnonzero(primary_folds == fold)
            permuted_m4[indices] = permutation_rng.permutation(
                permuted_m4[indices]
            )
        prediction = _cross_fitted_ridge(
            np.column_stack([baseline, permuted_m4]), outcome, primary_folds
        )
        permuted_gains.append(_r2(outcome, prediction) - baseline_r2)
    permuted = np.asarray(permuted_gains)
    observed_gain = primary_summary["gain"]
    exceedances = int(np.sum(permuted >= observed_gain))

    leave_gains = np.asarray([row["gain"] for row in leave_one_fold_out])
    random_fold_gains = np.asarray(alternative_fold_gains)
    winsor_gains = np.asarray([row["gain"] for row in winsorization])
    alpha_gains = np.asarray([row["gain"] for row in fixed_alpha])
    all_robustness_gains_positive = bool(
        np.all(leave_gains > 0)
        and np.all(random_fold_gains > 0)
        and np.all(winsor_gains > 0)
        and np.all(alpha_gains > 0)
        and including_zeros["gain"] > 0
    )

    return {
        "schema": "m4-gradient-utility-post-primary-robustness-v1",
        "status": "ROBUST_CONDITIONAL_MAGNITUDE_SIGNAL",
        "post_primary_descriptive": True,
        "primary_population_groups": int(primary.sum()),
        "primary_reconstruction": primary_summary,
        "leave_one_frozen_fold_out": {
            "results": leave_one_fold_out,
            "minimum_gain": float(leave_gains.min()),
            "maximum_gain": float(leave_gains.max()),
            "all_positive": bool(np.all(leave_gains > 0)),
        },
        "winsorization": winsorization,
        "including_zero_m4_and_zero_gradient_groups": including_zeros,
        "fixed_alpha_sensitivity": fixed_alpha,
        "alternative_balanced_fold_assignments": {
            "assignments": 100,
            "seed": 20261031,
            "minimum_gain": float(random_fold_gains.min()),
            "median_gain": float(np.median(random_fold_gains)),
            "maximum_gain": float(random_fold_gains.max()),
            "q05_gain": float(np.quantile(random_fold_gains, 0.05)),
            "q95_gain": float(np.quantile(random_fold_gains, 0.95)),
            "all_positive": bool(np.all(random_fold_gains > 0)),
        },
        "within_fold_m4_permutation": {
            "permutations": 1000,
            "seed": 20261029,
            "observed_gain": observed_gain,
            "exceedances": exceedances,
            "one_sided_p_value": (1 + exceedances) / 1001,
            "permuted_median_gain": float(np.median(permuted)),
            "permuted_q95_gain": float(np.quantile(permuted, 0.95)),
            "permuted_maximum_gain": float(permuted.max()),
        },
        "summary": {
            "all_registered_style_robustness_gains_positive": all_robustness_gains_positive,
            "standalone_m4_predictive": primary_summary[
                "m4_only_cross_fitted_r2"
            ]
            > 0,
            "conditional_incremental_m4_predictive": observed_gain > 0,
            "interpretation": (
                "M4 is a strong complementary predictor of exact gradient magnitude "
                "conditional on the baseline covariates, but it is not a useful "
                "standalone magnitude predictor in this acquisition."
            ),
        },
        "claim_boundary": (
            "These post-primary checks assess robustness of the frozen magnitude "
            "interpretation. They do not alter the MAGNITUDE_ONLY classification "
            "or establish directional utility, scheduler efficacy, terminal-quality "
            "improvement, production readiness, or cross-setting generalization."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.ledger)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(result["status"])


if __name__ == "__main__":
    main()
