# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Aggregate the three frozen scheduler pressure-response replications."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from nemo_rl.algorithms.async_utils.structured_scheduler_crossover import (
    SchedulerPressureResponsePlan,
    load_scheduler_pressure_response_plan,
)


class SchedulerPressureResponseReplicationError(ValueError):
    """A replication result violates the frozen aggregate contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SchedulerPressureResponseReplicationError(message)


def analyze(
    *, plan: SchedulerPressureResponsePlan, result_paths: dict[int, Path]
) -> dict[str, Any]:
    _require(
        tuple(result_paths) == plan.replication_order, "results are not in frozen order"
    )
    results = []
    for order_seed, path in result_paths.items():
        value = json.loads(path.read_text())
        _require(
            isinstance(value, dict)
            and value.get("schema_version") == 1
            and value.get("analysis_status") == plan.analysis_status
            and value.get("calibration_only") is True
            and value.get("confirmatory_eligible") is False
            and value.get("population_claim_authorized") is False
            and value.get("counterfactual_replay_authorized") is False
            and value.get("training_authorized") is False
            and value.get("plan_id") == plan.plan_id
            and value.get("order_seed") == order_seed,
            f"replication {order_seed} identity mismatch",
        )
        results.append(value)

    def in_order_pressure(result: dict[str, Any], level: str) -> float:
        arm = result["arms"][f"natural_{level}_in_order"]
        return float(
            arm["selection_pressure"]["decisions_with_ready_greater_than_eligible"]
        ) / float(arm["selection_steps"])

    def promotion(result: dict[str, Any], level: str) -> float:
        return float(
            result["comparisons"][f"natural_{level}"]["fixed_stratum_reference"][
                "lower_reference_mean_normalized_rank_promotion"
            ]
        )

    pressure_l0 = [in_order_pressure(result, "l0") for result in results]
    pressure_l1 = [in_order_pressure(result, "l1") for result in results]
    pressure_l3 = [in_order_pressure(result, "l3") for result in results]
    promotion_l0 = [promotion(result, "l0") for result in results]
    promotion_l1 = [promotion(result, "l1") for result in results]
    promotion_l3 = [promotion(result, "l3") for result in results]
    threshold = plan.thresholds
    checks = {
        "all_three_replications_valid": all(
            all(result["replication_checks"].values()) for result in results
        ),
        "median_l3_pressure_ge_0_5": statistics.median(pressure_l3)
        >= threshold.pressure_activation_median_min,
        "at_least_two_l3_pressure_values_exceed_l0": sum(
            l3 > l0 for l3, l0 in zip(pressure_l3, pressure_l0, strict=True)
        )
        >= threshold.pressure_activation_replications_exceeding_l0_min,
        "median_l3_promotion_ge_1_over_31": statistics.median(promotion_l3)
        >= threshold.composition_median_normalized_rank_promotion_min,
        "at_least_two_l3_promotions_ge_1_over_31": sum(
            value >= threshold.composition_median_normalized_rank_promotion_min
            for value in promotion_l3
        )
        >= threshold.composition_replications_at_or_above_minimum,
        "at_most_one_negative_l3_promotion": sum(value < 0 for value in promotion_l3)
        <= threshold.composition_negative_replications_max,
        "median_l3_pressure_exceeds_l0": statistics.median(pressure_l3)
        > statistics.median(pressure_l0),
        "median_l3_promotion_exceeds_l0": statistics.median(promotion_l3)
        > statistics.median(promotion_l0),
    }
    passed = all(checks.values())
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
        "pressure_fraction_by_level": {
            "l0": pressure_l0,
            "l1": pressure_l1,
            "l3": pressure_l3,
        },
        "fixed_stratum_normalized_rank_promotion_by_level": {
            "l0": promotion_l0,
            "l1": promotion_l1,
            "l3": promotion_l3,
        },
        "aggregate_checks": checks,
        "decision": (
            "draft_separate_fresh_operational_mixture_zero_update_protocol"
            if passed
            else "stop_ready_first_consequence_sequence_no_replay_or_training"
        ),
        "interpretation": "calibration-only pressure-response result; no population, replay, or training claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--result", action="append", required=True, metavar="ORDER_SEED=PATH"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = load_scheduler_pressure_response_plan(args.plan)
    paths = {}
    for assignment in args.result:
        raw_seed, separator, raw_path = assignment.partition("=")
        _require(bool(separator and raw_seed and raw_path), "invalid --result")
        seed = int(raw_seed)
        _require(seed not in paths, f"duplicate order seed {seed}")
        paths[seed] = Path(raw_path)
    result = analyze(plan=plan, result_paths=paths)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
