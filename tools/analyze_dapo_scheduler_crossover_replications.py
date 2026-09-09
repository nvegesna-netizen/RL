# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Aggregate all four frozen DAPO scheduler-crossover replications."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nemo_rl.algorithms.async_utils.structured_scheduler_crossover import (
    DapoSchedulerCrossoverPlan,
    load_dapo_scheduler_crossover_plan,
)


class DapoSchedulerCrossoverAggregateError(ValueError):
    """The complete replication set or a frozen aggregate gate is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DapoSchedulerCrossoverAggregateError(message)


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        rank = (start + 1 + end) / 2
        for index, _ in ordered[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    _require(len(left) == len(right) and len(left) > 1, "invalid Spearman inputs")
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = statistics.fmean(left_ranks)
    right_mean = statistics.fmean(right_ranks)
    left_centered = [value - left_mean for value in left_ranks]
    right_centered = [value - right_mean for value in right_ranks]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator == 0:
        return 0.0
    return sum(
        left_value * right_value
        for left_value, right_value in zip(
            left_centered, right_centered, strict=True
        )
    ) / denominator


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _pooled_within_replication_spearman(
    replications: Sequence[Mapping[str, Any]],
) -> float:
    centered_load: list[float] = []
    centered_latency: list[float] = []
    for replication in replications:
        observations = replication["runs"]["in_order"]["observations"]
        load_ranks = _average_ranks(
            [float(item["mean_generated_tokens"]) for item in observations]
        )
        latency_ranks = _average_ranks(
            [float(item["ready_latency_ns"]) for item in observations]
        )
        load_mean = statistics.fmean(load_ranks)
        latency_mean = statistics.fmean(latency_ranks)
        centered_load.extend(value - load_mean for value in load_ranks)
        centered_latency.extend(value - latency_mean for value in latency_ranks)
    denominator = math.sqrt(
        sum(value * value for value in centered_load)
        * sum(value * value for value in centered_latency)
    )
    if denominator == 0:
        return 0.0
    return sum(
        left * right
        for left, right in zip(centered_load, centered_latency, strict=True)
    ) / denominator


def _secondary(replication: Mapping[str, Any]) -> dict[str, Any]:
    in_order = replication["runs"]["in_order"]
    lower = set(replication["lower_load_source_prompt_ids"])
    output: dict[str, Any] = {}
    for arm_id in ("ready_first", "in_order"):
        run = replication["runs"][arm_id]
        order = run["selection_order_source_prompt_ids"]
        observations = {
            item["source_prompt_id"]: item for item in run["observations"]
        }
        output[arm_id] = {
            "lower_load_share_by_horizon": {
                str(horizon): sum(item in lower for item in order[:horizon]) / horizon
                for horizon in (4, 8, 12)
            },
            "selected_reward_mean_by_horizon": {
                str(horizon): statistics.fmean(
                    float(observations[item]["reward_mean"])
                    for item in order[:horizon]
                )
                for horizon in (4, 8, 12)
            },
            "reference_load_selection_rank_spearman": _spearman(
                [
                    float(item["mean_generated_tokens"])
                    for item in in_order["observations"]
                ],
                [float(order.index(item["source_prompt_id"]) + 1) for item in in_order["observations"]],
            ),
        }
    return output


def analyze(
    plan: DapoSchedulerCrossoverPlan,
    replications: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _require(len(replications) == 4, "all four replications are required")
    _require(
        [item.get("order_seed") for item in replications] == list(plan.replication_order),
        "replication order mismatch",
    )
    _require(
        all(
            item.get("plan_id") == plan.plan_id
            and item.get("analysis_status") == plan.analysis_status
            and item.get("calibration_only") is True
            and item.get("confirmatory_eligible") is False
            for item in replications
        ),
        "replication identity or evidence labels mismatch",
    )
    primary = [
        float(item["lower_load_normalized_selection_rank_advantage"])
        for item in replications
    ]
    in_order_observations = [
        observation
        for replication in replications
        for observation in replication["runs"]["in_order"]["observations"]
    ]
    latencies = [float(item["ready_latency_ns"]) for item in in_order_observations]
    p10 = _percentile(latencies, 0.1)
    p90 = _percentile(latencies, 0.9)
    latency_ratio = p90 / p10 if p10 > 0 else math.inf
    load_latency = _pooled_within_replication_spearman(replications)
    reward_variance = sum(
        float(item["reward_min"]) < float(item["reward_max"])
        for item in in_order_observations
    ) / len(in_order_observations)
    thresholds = plan.thresholds
    validity = {
        "all_four_replications_valid": all(
            item.get("all_replication_validity_gates_passed") is True
            for item in replications
        ),
        "pooled_in_order_load_latency_spearman_ge_0_5": (
            load_latency
            >= thresholds.pooled_in_order_within_pool_spearman_group_mean_tokens_ready_latency_min
        ),
        "pooled_in_order_ready_latency_p90_p10_ratio_ge_1_5": (
            latency_ratio
            >= thresholds.ready_latency_p90_p10_ratio_min_pooled_in_order
        ),
        "pooled_in_order_reward_variance_fraction_ge_0_25": (
            reward_variance
            >= thresholds.pooled_in_order_reward_variance_group_fraction_min
        ),
    }
    effect = {
        "median_advantage_ge_1_over_15": (
            statistics.median(primary)
            >= thresholds.median_lower_load_normalized_selection_rank_advantage_min
        ),
        "at_least_three_replications_ge_1_over_15": sum(
            value
            >= thresholds.median_lower_load_normalized_selection_rank_advantage_min
            for value in primary
        )
        >= thresholds.replications_at_or_above_minimum_required,
        "zero_replications_below_zero": sum(value < 0 for value in primary)
        <= thresholds.replications_below_zero_max,
    }
    passed = all(validity.values()) and all(effect.values())
    return {
        "schema_version": 1,
        "analysis_status": plan.analysis_status,
        "calibration_only": True,
        "confirmatory_eligible": False,
        "population_claim_authorized": False,
        "counterfactual_replay_authorized": False,
        "training_authorized": False,
        "plan_id": plan.plan_id,
        "replication_order": list(plan.replication_order),
        "primary": {
            "per_replication_lower_load_normalized_selection_rank_advantage": primary,
            "median_lower_load_normalized_selection_rank_advantage": statistics.median(primary),
        },
        "secondary_by_replication": [
            {
                "order_seed": item["order_seed"],
                **_secondary(item),
            }
            for item in replications
        ],
        "pooled_in_order": {
            "within_replication_spearman_group_mean_tokens_ready_latency": load_latency,
            "ready_latency_p90_p10_ratio": latency_ratio,
            "reward_variance_group_fraction": reward_variance,
        },
        "validity_checks": validity,
        "effect_checks": effect,
        "decision": (
            "controlled_dapo_scheduler_selection_calibration_passed"
            if passed
            else "stop_no_dapo_replay_or_training_consequence_study"
        ),
        "interpretation": (
            "controlled exact-setup scheduler calibration only; not a population, "
            "counterfactual replay, or learner-training result"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--replication", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = load_dapo_scheduler_crossover_plan(args.plan)
    records = [json.loads(path.read_text()) for path in args.replication]
    result = analyze(plan, records)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
