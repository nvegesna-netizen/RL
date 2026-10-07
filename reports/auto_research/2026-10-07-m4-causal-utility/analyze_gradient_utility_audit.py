#!/usr/bin/env python3
"""Analyze the frozen M4 no-update gradient-utility audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3:
        return float("nan")
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _ridge_fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    alpha: float,
) -> np.ndarray:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0] = 1.0
    x_train = (train_x - mean) / scale
    x_test = (test_x - mean) / scale
    y_mean = float(train_y.mean())
    gram = x_train.T @ x_train
    coefficients = np.linalg.solve(
        gram + alpha * np.eye(gram.shape[0]),
        x_train.T @ (train_y - y_mean),
    )
    return y_mean + x_test @ coefficients


def _cross_fitted_ridge(
    features: np.ndarray,
    outcome: np.ndarray,
    folds: np.ndarray,
) -> np.ndarray:
    alphas = (1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    predictions = np.full(len(outcome), np.nan, dtype=np.float64)
    for outer in sorted(set(folds.tolist())):
        outer_test = folds == outer
        outer_train = ~outer_test
        best_alpha = alphas[0]
        best_error = float("inf")
        for alpha in alphas:
            inner_predictions: list[np.ndarray] = []
            inner_outcomes: list[np.ndarray] = []
            for inner in sorted(set(folds[outer_train].tolist())):
                inner_test = outer_train & (folds == inner)
                inner_train = outer_train & (folds != inner)
                inner_predictions.append(
                    _ridge_fit_predict(
                        features[inner_train],
                        outcome[inner_train],
                        features[inner_test],
                        alpha,
                    )
                )
                inner_outcomes.append(outcome[inner_test])
            error = float(
                np.mean(
                    (np.concatenate(inner_predictions) - np.concatenate(inner_outcomes))
                    ** 2
                )
            )
            if error < best_error:
                best_error = error
                best_alpha = alpha
        predictions[outer_test] = _ridge_fit_predict(
            features[outer_train],
            outcome[outer_train],
            features[outer_test],
            best_alpha,
        )
    if not np.isfinite(predictions).all():
        raise RuntimeError("cross-fitted predictions contain nonfinite values")
    return predictions


def _r2(outcome: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.sum((outcome - outcome.mean()) ** 2))
    if denominator <= 0:
        return 0.0
    return 1.0 - float(np.sum((outcome - prediction) ** 2)) / denominator


def _stratified_resample(rng: np.random.Generator, folds: np.ndarray) -> np.ndarray:
    pieces = []
    for fold in sorted(set(folds.tolist())):
        indices = np.flatnonzero(folds == fold)
        pieces.append(rng.choice(indices, size=len(indices), replace=True))
    return np.concatenate(pieces)


def analyze(
    ledger_path: Path,
    protocol_path: Path,
    *,
    bootstrap_resamples: int | None = None,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not events or events[0].get("event_type") != "header":
        raise ValueError("gradient utility ledger lacks a header")
    groups = [event for event in events if event.get("event_type") == "group_gradient"]
    terminals = [event for event in events if event.get("event_type") == "terminal"]
    expected_groups = int(protocol["acquisition"]["prompt_groups"])
    if len(groups) != expected_groups or len(terminals) != 1:
        raise ValueError("gradient utility ledger is incomplete")
    terminal = terminals[0]
    group_ids = [event["group_id"] for event in groups]
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("gradient utility group IDs are not unique")
    if [event["audit_index"] for event in groups] != list(range(expected_groups)):
        raise ValueError("gradient utility audit indices are not contiguous")

    integrity = {
        "unique_groups": len(set(group_ids)) == expected_groups,
        "eight_siblings_per_group": all(
            len(event["sample_ids"]) == protocol["acquisition"]["siblings_per_group"]
            for event in groups
        ),
        "all_aborts_acknowledged": all(
            event.get("abort_acknowledged") is True for event in groups
        ),
        "finish_train_step_calls_zero": terminal.get("finish_train_step_calls") == 0,
        "optimizer_steps_zero": terminal.get("optimizer_steps") == 0,
        "scheduler_steps_zero": terminal.get("scheduler_steps") == 0,
        "learner_version_zero": terminal.get("learner_version") == 0,
        "parameter_hash_unchanged": terminal.get("parameter_hash_unchanged") is True,
    }

    exact_norm = np.asarray(
        [event["summary"]["exact_l2_norm"] for event in groups], dtype=np.float64
    )
    sketches = np.asarray(
        [event["summary"]["sketches"] for event in groups], dtype=np.float64
    )
    expected_seeds = protocol["gradient_measurement"]["sketch_seeds"]
    expected_bins = protocol["gradient_measurement"]["sketch_bins"]
    if sketches.shape != (expected_groups, len(expected_seeds), expected_bins):
        raise ValueError(f"unexpected sketch shape {sketches.shape}")

    metadata = [event["metadata"] for event in groups]
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
    finite = np.isfinite(
        np.column_stack(
            [exact_norm, m4, valid_tokens, reward_mean, reward_variance, truncation]
        )
    ).all(axis=1)
    integrity["finite_scalar_summaries"] = bool(finite.all())

    positive = finite & (m4 > 0) & (exact_norm > 0)
    if int(positive.sum()) < 96:
        integrity["minimum_primary_population"] = False
    else:
        integrity["minimum_primary_population"] = True

    sketch_norm = np.linalg.norm(sketches, axis=2)
    relative_error = np.abs(sketch_norm - exact_norm[:, None]) / np.maximum(
        exact_norm[:, None], np.finfo(np.float64).tiny
    )
    fidelity = []
    for seed_index, seed in enumerate(expected_seeds):
        errors = relative_error[exact_norm > 0, seed_index]
        fidelity.append(
            {
                "seed": seed,
                "median_norm_relative_error": float(np.median(errors)),
                "p95_norm_relative_error": float(np.quantile(errors, 0.95)),
            }
        )
    integrity["sketch_fidelity"] = all(
        row["median_norm_relative_error"]
        <= protocol["integrity_gate"]["median_sketch_norm_relative_error_max"]
        and row["p95_norm_relative_error"]
        <= protocol["integrity_gate"]["p95_sketch_norm_relative_error_max"]
        for row in fidelity
    )

    population = np.flatnonzero(positive)
    pop_folds = folds[population]
    base_features = np.column_stack(
        [
            np.log1p(valid_tokens[population]),
            reward_mean[population],
            reward_variance[population],
            truncation[population],
        ]
    )
    augmented_features = np.column_stack([base_features, np.log1p(m4[population])])
    norm_outcome = np.log(exact_norm[population])
    norm_base_prediction = _cross_fitted_ridge(base_features, norm_outcome, pop_folds)
    norm_augmented_prediction = _cross_fitted_ridge(
        augmented_features, norm_outcome, pop_folds
    )
    norm_gain = _r2(norm_outcome, norm_augmented_prediction) - _r2(
        norm_outcome, norm_base_prediction
    )

    utilities: list[np.ndarray] = []
    directional: list[dict[str, Any]] = []
    utility_predictions: list[tuple[np.ndarray, np.ndarray]] = []
    for seed_index, seed in enumerate(expected_seeds):
        utility = np.full(expected_groups, np.nan, dtype=np.float64)
        for fold in range(protocol["analysis"]["folds"]):
            reference = sketches[folds != fold, seed_index].mean(axis=0)
            reference_norm = float(np.linalg.norm(reference))
            if reference_norm == 0:
                raise ValueError("cross-fitted consensus gradient is zero")
            held_out = folds == fold
            utility[held_out] = (
                sketches[held_out, seed_index] @ reference / reference_norm
            )
        utilities.append(utility)
        pop_utility = utility[population]
        base_prediction = _cross_fitted_ridge(base_features, pop_utility, pop_folds)
        augmented_prediction = _cross_fitted_ridge(
            augmented_features, pop_utility, pop_folds
        )
        utility_predictions.append((base_prediction, augmented_prediction))
        mse_gain = float(
            np.mean((pop_utility - base_prediction) ** 2)
            - np.mean((pop_utility - augmented_prediction) ** 2)
        )
        positive_utility = np.maximum(pop_utility, 0.0)
        rank_discriminant = _spearman(m4[population], positive_utility) - _spearman(
            reward_variance[population], positive_utility
        )
        directional.append(
            {
                "seed": seed,
                "signed_utility_mse_gain": mse_gain,
                "m4_minus_reward_variance_spearman": rank_discriminant,
                "m4_spearman_positive_utility": _spearman(
                    m4[population], positive_utility
                ),
                "reward_variance_spearman_positive_utility": _spearman(
                    reward_variance[population], positive_utility
                ),
            }
        )

    resamples = int(
        bootstrap_resamples
        if bootstrap_resamples is not None
        else protocol["analysis"]["bootstrap_resamples"]
    )
    rng = np.random.default_rng(protocol["analysis"]["bootstrap_seed"])
    bootstrap = np.empty((resamples, 1 + 2 * len(expected_seeds)), dtype=np.float64)
    for replicate in range(resamples):
        selected = _stratified_resample(rng, pop_folds)
        y = norm_outcome[selected]
        bootstrap[replicate, 0] = _r2(y, norm_augmented_prediction[selected]) - _r2(
            y, norm_base_prediction[selected]
        )
        column = 1
        for seed_index in range(len(expected_seeds)):
            pop_utility = utilities[seed_index][population]
            base_prediction, augmented_prediction = utility_predictions[seed_index]
            utility_selected = pop_utility[selected]
            bootstrap[replicate, column] = float(
                np.mean((utility_selected - base_prediction[selected]) ** 2)
                - np.mean((utility_selected - augmented_prediction[selected]) ** 2)
            )
            positive_utility = np.maximum(utility_selected, 0.0)
            bootstrap[replicate, column + 1] = _spearman(
                m4[population][selected], positive_utility
            ) - _spearman(reward_variance[population][selected], positive_utility)
            column += 2

    simultaneous_alpha = 0.05 / bootstrap.shape[1]
    lower = np.quantile(bootstrap, simultaneous_alpha, axis=0)
    upper = np.quantile(bootstrap, 1.0 - simultaneous_alpha, axis=0)
    norm_interval = [float(lower[0]), float(upper[0])]
    for seed_index, row in enumerate(directional):
        row["simultaneous_interval_signed_utility_mse_gain"] = [
            float(lower[1 + 2 * seed_index]),
            float(upper[1 + 2 * seed_index]),
        ]
        row["simultaneous_interval_m4_minus_reward_variance_spearman"] = [
            float(lower[2 + 2 * seed_index]),
            float(upper[2 + 2 * seed_index]),
        ]

    integrity_pass = all(integrity.values())
    norm_pass = norm_interval[0] > 0
    directional_pass = all(
        row["simultaneous_interval_signed_utility_mse_gain"][0] > 0
        and row["simultaneous_interval_m4_minus_reward_variance_spearman"][0] > 0
        for row in directional
    )
    if not integrity_pass:
        status = "MEASUREMENT_FAILURE"
    elif norm_pass and directional_pass:
        status = "DIRECTIONAL_CONSTRUCT_PASS"
    elif norm_pass:
        status = "MAGNITUDE_ONLY"
    else:
        status = "NO_CONSTRUCT_SUPPORT"

    return {
        "schema": "m4-gradient-utility-audit-result-v1",
        "status": status,
        "groups": expected_groups,
        "primary_population_groups": int(positive.sum()),
        "zero_m4_groups": int(np.sum(m4 == 0)),
        "zero_gradient_groups": int(np.sum(exact_norm == 0)),
        "integrity": integrity,
        "sketch_fidelity": fidelity,
        "magnitude": {
            "cross_fitted_r2_gain_log_exact_gradient_norm": norm_gain,
            "simultaneous_interval": norm_interval,
            "m4_spearman_exact_gradient_norm": _spearman(
                m4[population], exact_norm[population]
            ),
            "reward_variance_spearman_exact_gradient_norm": _spearman(
                reward_variance[population], exact_norm[population]
            ),
        },
        "directional": directional,
        "bootstrap": {
            "resamples": resamples,
            "seed": protocol["analysis"]["bootstrap_seed"],
            "simultaneous_method": "bonferroni_percentile",
            "family_size": int(bootstrap.shape[1]),
        },
        "claim_boundary": (
            "This no-update construct audit does not establish terminal-quality "
            "improvement, scheduler efficacy, or production readiness."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int)
    args = parser.parse_args()
    result = analyze(
        args.ledger,
        args.protocol,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(result["status"])


if __name__ == "__main__":
    main()
