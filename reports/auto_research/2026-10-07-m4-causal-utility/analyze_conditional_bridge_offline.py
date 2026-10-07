#!/usr/bin/env python3
"""Fit and replay the frozen token-conditioned M4 bridge policy."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import zipfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analyze_gradient_utility_audit import _r2, _ridge_fit_predict, _spearman


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> bytes:
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
        raise RuntimeError("cross-fitted predictions are nonfinite")
    return predictions


def fit_deployment_model(
    groups: Sequence[dict[str, Any]], protocol: dict[str, Any]
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    metadata = [group["metadata"] for group in groups]
    exact_norm = np.asarray(
        [group["summary"]["exact_l2_norm"] for group in groups], dtype=np.float64
    )
    m4 = np.asarray(
        [row["gradient_opportunity_l1"] for row in metadata], dtype=np.float64
    )
    valid_tokens = np.asarray(
        [row["gradient_opportunity_valid_actor_tokens"] for row in metadata],
        dtype=np.float64,
    )
    reward_mean = np.asarray(
        [row["gradient_opportunity_reward_mean"] for row in metadata],
        dtype=np.float64,
    )
    reward_variance = np.asarray(
        [row["gradient_opportunity_reward_variance"] for row in metadata],
        dtype=np.float64,
    )
    folds = np.asarray([group["fold"] for group in groups], dtype=np.int64)
    finite = np.isfinite(
        np.column_stack([exact_norm, m4, valid_tokens, reward_mean, reward_variance])
    ).all(axis=1)
    primary = finite & (m4 > 0) & (exact_norm > 0)
    features = np.column_stack(
        [
            np.log1p(valid_tokens[primary]),
            reward_mean[primary],
            reward_variance[primary],
            np.log1p(m4[primary]),
        ]
    )
    baseline = features[:, :3]
    outcome = np.log(exact_norm[primary])
    primary_folds = folds[primary]
    alphas = tuple(float(value) for value in protocol["deployment_model"]["alpha_grid"])

    def choose_alpha(
        matrix: np.ndarray,
    ) -> tuple[float, np.ndarray, list[dict[str, float]]]:
        rows: list[dict[str, float]] = []
        predictions_by_alpha: list[np.ndarray] = []
        for alpha in alphas:
            prediction = _cross_fitted_fixed_alpha(
                matrix, outcome, primary_folds, alpha
            )
            error = float(np.mean((prediction - outcome) ** 2))
            rows.append({"alpha": alpha, "out_of_fold_mse": error})
            predictions_by_alpha.append(prediction)
        best_index = min(
            range(len(rows)),
            key=lambda index: (
                rows[index]["out_of_fold_mse"],
                rows[index]["alpha"],
            ),
        )
        return rows[best_index]["alpha"], predictions_by_alpha[best_index], rows

    baseline_alpha, baseline_prediction, baseline_grid = choose_alpha(baseline)
    alpha, prediction, grid = choose_alpha(features)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0] = 1.0
    standardized = (features - mean) / scale
    standardized_coefficients = np.linalg.solve(
        standardized.T @ standardized + alpha * np.eye(features.shape[1]),
        standardized.T @ (outcome - outcome.mean()),
    )
    coefficients = standardized_coefficients / scale
    intercept = float(outcome.mean() - mean @ coefficients)
    baseline_r2 = _r2(outcome, baseline_prediction)
    augmented_r2 = _r2(outcome, prediction)
    model = {
        "feature_order": protocol["deployment_model"]["features_in_order"],
        "primary_population_groups": int(primary.sum()),
        "zero_m4_groups": int((m4 == 0).sum()),
        "alpha": alpha,
        "alpha_grid_results": grid,
        "baseline_alpha": baseline_alpha,
        "baseline_alpha_grid_results": baseline_grid,
        "feature_means": mean.tolist(),
        "feature_scales": scale.tolist(),
        "standardized_coefficients": standardized_coefficients.tolist(),
        "raw_coefficients": coefficients.tolist(),
        "raw_intercept": intercept,
        "baseline_cross_fitted_r2": baseline_r2,
        "augmented_cross_fitted_r2": augmented_r2,
        "incremental_cross_fitted_r2": augmented_r2 - baseline_r2,
        "augmented_prediction_spearman": _spearman(prediction, outcome),
        "zero_m4_prediction": 0.0,
    }
    return model, coefficients, np.asarray([intercept], dtype=np.float64)


def predict_gradient_norm(
    candidate: dict[str, Any], coefficients: np.ndarray, intercept: float
) -> float:
    m4 = float(candidate["l1"])
    if m4 <= 0:
        return 0.0
    features = np.asarray(
        [
            math.log1p(float(candidate["valid_actor_tokens"])),
            float(candidate["reward_mean"]),
            float(candidate["reward_variance"]),
            math.log1p(m4),
        ],
        dtype=np.float64,
    )
    prediction = math.exp(intercept + float(features @ coefficients))
    if not math.isfinite(prediction) or prediction < 0:
        raise ValueError("predicted gradient norm is invalid")
    return prediction


def select_conditional_batch(
    decision: dict[str, Any],
    coefficients: np.ndarray,
    intercept: float,
    *,
    tolerance: float,
) -> dict[str, Any]:
    candidates = sorted(decision["candidates"], key=lambda row: row["group_id"])
    by_id = {row["group_id"]: row for row in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("candidate group IDs are not unique")
    base_proposal = decision["proposals"]["reward_variance_risk"]
    base_ids = tuple(base_proposal["proposed_group_ids"])
    base = tuple(by_id[group_id] for group_id in base_ids)
    base_variance = math.fsum(float(row["reward_variance"]) for row in base)
    base_tokens = sum(int(row["valid_actor_tokens"]) for row in base)
    base_score = math.fsum(
        predict_gradient_norm(row, coefficients, intercept) for row in base
    )
    minimum_tokens = int(base_proposal["minimum_tokens"])
    maximum_tokens = int(base_proposal["maximum_tokens"])
    cardinality = len(base_ids)
    best: (
        tuple[float, float, int, tuple[str, ...], tuple[dict[str, Any], ...]] | None
    ) = None
    feasible_count = 0
    for combination in itertools.combinations(candidates, cardinality):
        tokens = sum(int(row["valid_actor_tokens"]) for row in combination)
        if tokens < minimum_tokens or tokens > maximum_tokens:
            continue
        variance = math.fsum(float(row["reward_variance"]) for row in combination)
        if variance + tolerance < base_variance:
            continue
        feasible_count += 1
        score = math.fsum(
            predict_gradient_norm(row, coefficients, intercept) for row in combination
        )
        ids = tuple(row["group_id"] for row in combination)
        # Python's max would prefer larger IDs. Negate only the scientific and
        # service fields here; handle identity explicitly below.
        candidate_key = (score, variance, -tokens)
        if best is None:
            best = (*candidate_key, ids, combination)
            continue
        best_key = best[:3]
        if candidate_key > best_key or (candidate_key == best_key and ids < best[3]):
            best = (*candidate_key, ids, combination)
    if best is None or feasible_count < 1:
        raise RuntimeError("base reward-variance proposal is not feasible")
    proposed_score, proposed_variance, negated_tokens, proposed_ids, _ = best
    strict_gain = proposed_score > base_score + tolerance
    selected_ids = proposed_ids if strict_gain else base_ids
    selected = tuple(by_id[group_id] for group_id in selected_ids)
    selected_tokens = sum(int(row["valid_actor_tokens"]) for row in selected)
    selected_variance = math.fsum(float(row["reward_variance"]) for row in selected)
    return {
        "base_group_ids": list(base_ids),
        "selected_group_ids": list(selected_ids),
        "strict_intervention": set(selected_ids) != set(base_ids),
        "base_predicted_norm_sum": base_score,
        "selected_predicted_norm_sum": (proposed_score if strict_gain else base_score),
        "relative_predicted_norm_gain": (
            (proposed_score / base_score - 1.0)
            if strict_gain and base_score > 0
            else 0.0
        ),
        "base_reward_variance": base_variance,
        "selected_reward_variance": selected_variance,
        "reward_variance_retention": (
            selected_variance / base_variance if base_variance > 0 else 1.0
        ),
        "base_tokens": base_tokens,
        "selected_tokens": selected_tokens,
        "token_ratio": selected_tokens / base_tokens,
        "base_overlap_count": len(set(selected_ids) & set(base_ids)),
        "feasible_combination_count": feasible_count,
        "minimum_tokens": minimum_tokens,
        "maximum_tokens": maximum_tokens,
        # Retained in the canonical replay so an independent reviewer can
        # authenticate the complete deterministic tie-break tuple.
        "unused_best_negated_tokens": negated_tokens,
        "unused_best_variance": proposed_variance,
    }


def analyze(
    gradient_ledger_path: Path,
    frontier_archive_path: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    gradient_bytes = gradient_ledger_path.read_bytes()
    archive_bytes = frontier_archive_path.read_bytes()
    expected_inputs = protocol["development_inputs"]
    if _sha256(gradient_bytes) != expected_inputs["gradient_utility_ledger"]["sha256"]:
        raise ValueError("gradient utility ledger hash differs from protocol")
    if (
        _sha256(archive_bytes)
        != expected_inputs["controlled_frontier_archive"]["sha256"]
    ):
        raise ValueError("controlled-frontier archive hash differs from protocol")

    gradient_events = [
        json.loads(line) for line in gradient_bytes.splitlines() if line.strip()
    ]
    groups = [
        event
        for event in gradient_events
        if event.get("event_type") == "group_gradient"
    ]
    if len(groups) != expected_inputs["gradient_utility_ledger"]["expected_groups"]:
        raise ValueError("gradient utility group count differs")
    model, coefficients, intercept_array = fit_deployment_model(groups, protocol)
    intercept = float(intercept_array[0])

    with zipfile.ZipFile(frontier_archive_path) as archive:
        suffix = expected_inputs["controlled_frontier_archive"]["ledger_member_suffix"]
        members = [name for name in archive.namelist() if name.endswith(suffix)]
        if len(members) != 1:
            raise ValueError("controlled-frontier ledger member is ambiguous")
        frontier_bytes = archive.read(members[0])
    if (
        _sha256(frontier_bytes)
        != expected_inputs["controlled_frontier_archive"]["ledger_sha256"]
    ):
        raise ValueError("controlled-frontier ledger hash differs")
    frontier_events = [
        json.loads(line) for line in frontier_bytes.splitlines() if line.strip()
    ]
    if not frontier_events or frontier_events[0].get("event_type") != "header":
        raise ValueError("controlled-frontier ledger lacks a header")
    decisions = [
        event for event in frontier_events if event.get("event_type") == "decision"
    ]
    if (
        len(decisions)
        != expected_inputs["controlled_frontier_archive"]["expected_decisions"]
    ):
        raise ValueError("controlled-frontier decision count differs")
    if not all(
        len(row["candidates"])
        == expected_inputs["controlled_frontier_archive"][
            "expected_candidates_per_decision"
        ]
        for row in decisions
    ):
        raise ValueError("controlled-frontier candidate count differs")

    tolerance = float(protocol["frontier_policy"]["numeric_tolerance"])
    replay = [
        select_conditional_batch(decision, coefficients, intercept, tolerance=tolerance)
        for decision in decisions
    ]
    interventions = [row for row in replay if row["strict_intervention"]]
    relative_gains = np.asarray(
        [row["relative_predicted_norm_gain"] for row in interventions],
        dtype=np.float64,
    )
    token_ratios = np.asarray([row["token_ratio"] for row in replay], dtype=np.float64)
    variance_retentions = np.asarray(
        [row["reward_variance_retention"] for row in replay], dtype=np.float64
    )
    gate = protocol["offline_gate"]
    checks = {
        "primary_population": model["primary_population_groups"]
        >= gate["primary_population_groups_minimum"],
        "augmented_cross_fitted_r2": model["augmented_cross_fitted_r2"]
        >= gate["augmented_cross_fitted_r2_minimum"],
        "incremental_cross_fitted_r2": model["incremental_cross_fitted_r2"]
        >= gate["incremental_cross_fitted_r2_minimum"],
        "strict_interventions": len(interventions)
        >= gate["strict_interventions_minimum"],
        "strict_intervention_rate": len(interventions) / len(replay)
        >= gate["strict_intervention_rate_minimum"],
        "median_relative_predicted_norm_gain": len(interventions) > 0
        and float(np.median(relative_gains))
        >= gate["median_relative_predicted_norm_gain_minimum"],
        "reward_variance_preserved": bool(
            np.all(
                variance_retentions + tolerance
                >= gate["minimum_reward_variance_retention"]
            )
        ),
        "token_band_preserved": bool(
            np.all(token_ratios + tolerance >= gate["minimum_token_ratio"])
            and np.all(token_ratios - tolerance <= gate["maximum_token_ratio"])
        ),
    }
    result = {
        "schema": "m4-conditional-one-update-bridge-offline-result-v1",
        "status": (
            "PASS_AUTHORIZE_NO_UPDATE_CAPSULE_QUALIFICATION"
            if all(checks.values())
            else "FAIL_DO_NOT_BUILD_CAUSAL_BRIDGE"
        ),
        "protocol_sha256": _sha256(protocol_bytes),
        "authenticated_inputs": {
            "gradient_utility_ledger_sha256": _sha256(gradient_bytes),
            "controlled_frontier_archive_sha256": _sha256(archive_bytes),
            "controlled_frontier_ledger_sha256": _sha256(frontier_bytes),
        },
        "deployment_model": model,
        "frontier_replay": {
            "decisions": len(replay),
            "strict_interventions": len(interventions),
            "strict_intervention_rate": len(interventions) / len(replay),
            "mean_relative_predicted_norm_gain_on_interventions": (
                float(np.mean(relative_gains)) if len(interventions) else 0.0
            ),
            "median_relative_predicted_norm_gain_on_interventions": (
                float(np.median(relative_gains)) if len(interventions) else 0.0
            ),
            "minimum_relative_predicted_norm_gain_on_interventions": (
                float(np.min(relative_gains)) if len(interventions) else 0.0
            ),
            "interventions_with_at_least_ten_percent_predicted_gain": int(
                np.sum(relative_gains >= 0.1)
            ),
            "mean_base_overlap_on_interventions": (
                float(np.mean([row["base_overlap_count"] for row in interventions]))
                if interventions
                else 4.0
            ),
            "minimum_reward_variance_retention": float(np.min(variance_retentions)),
            "mean_reward_variance_retention": float(np.mean(variance_retentions)),
            "minimum_token_ratio": float(np.min(token_ratios)),
            "maximum_token_ratio": float(np.max(token_ratios)),
            "mean_token_ratio": float(np.mean(token_ratios)),
            "first_strict_intervention_decision_index": (
                next(
                    index
                    for index, row in enumerate(replay)
                    if row["strict_intervention"]
                )
                if interventions
                else None
            ),
        },
        "checks": checks,
        "decision": protocol["decision_ladder"][
            "pass" if all(checks.values()) else "fail"
        ],
        "claim_boundary": protocol["claim_boundary"],
    }
    replay_hash = _sha256(_canonical_json(replay))
    result["frontier_replay"]["canonical_decision_replay_sha256"] = replay_hash
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gradient-ledger", type=Path, required=True)
    parser.add_argument("--frontier-archive", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.gradient_ledger, args.frontier_archive, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_json(result))
    print(json.dumps({"output": str(args.output), "status": result["status"]}))


if __name__ == "__main__":
    main()
