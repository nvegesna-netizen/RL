# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Apply the frozen aggregate gates to three operational-mixture replications."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from nemo_rl.algorithms.async_utils.dapo_operational_mixture import (
    DapoOperationalMixturePlan,
    load_dapo_operational_mixture_plan,
)


class DapoOperationalMixtureAggregateError(ValueError):
    """A replication result or aggregate invariant is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DapoOperationalMixtureAggregateError(message)


def analyze(plan: DapoOperationalMixturePlan, paths: list[Path]) -> dict[str, object]:
    _require(len(paths) == 3, "exactly three replication results are required")
    results = []
    for seed, path in zip(plan.replication_order, paths, strict=True):
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise DapoOperationalMixtureAggregateError(
                f"cannot load replication {seed}"
            ) from error
        _require(
            isinstance(value, dict)
            and value.get("plan_id") == plan.plan_id
            and value.get("order_seed") == seed
            and value.get("decision") == "pass_to_next_replication",
            f"replication {seed} did not pass its frozen validity gates",
        )
        results.append(value)

    def pressure(result: dict[str, object], level: str) -> float:
        arms = result["arms"]
        arm = arms[f"{level}_in_order"]  # type: ignore[index]
        summary = arm["selection_pressure"]
        return float(summary["decisions_with_ready_greater_than_eligible"]) / 8

    def promotion(result: dict[str, object], level: str) -> float:
        comparisons = result["comparisons"]
        return float(
            comparisons[level][  # type: ignore[index]
                "fixed_harder_mean_normalized_selection_step_promotion"
            ]
        )

    l0_pressure = [pressure(result, "l0") for result in results]
    l1_pressure = [pressure(result, "l1") for result in results]
    l1_promotions = [promotion(result, "l1") for result in results]
    l3_promotions = [promotion(result, "l3") for result in results]
    thresholds = plan.thresholds
    checks = {
        "all_three_replications_valid": True,
        "l0_fixed_harder_prompt_promotion_equals_zero_each_replication": all(
            promotion(result, "l0") == 0 for result in results
        ),
        "l1_pressure_median_ge_0_25": statistics.median(l1_pressure)
        >= thresholds.l1_pressure_median_min,
        "l1_pressure_exceeds_l0_in_at_least_two_replications": sum(
            l1 > l0 for l0, l1 in zip(l0_pressure, l1_pressure, strict=True)
        )
        >= thresholds.l1_pressure_replications_exceeding_l0_min,
        "l1_promotion_median_ge_1_over_28": statistics.median(l1_promotions)
        >= thresholds.l1_median_normalized_selection_step_promotion_min,
        "l1_promotion_at_or_above_1_over_28_in_at_least_two_replications": sum(
            value >= thresholds.l1_median_normalized_selection_step_promotion_min
            for value in l1_promotions
        )
        >= thresholds.l1_replications_at_or_above_minimum,
        "l1_negative_replications_at_most_one": sum(
            value < 0 for value in l1_promotions
        )
        <= thresholds.l1_negative_replications_max,
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "analysis_status": "aggregate_dapo_operational_mixture_zero_update_shadow",
        "calibration_only": True,
        "confirmatory_eligible": False,
        "population_claim_authorized": False,
        "counterfactual_replay_authorized": False,
        "training_authorized": False,
        "plan_id": plan.plan_id,
        "replication_order": list(plan.replication_order),
        "l0_pressure_fraction": l0_pressure,
        "l1_pressure_fraction": l1_pressure,
        "l1_fixed_harder_prompt_normalized_selection_step_promotion": l1_promotions,
        "l3_fixed_harder_prompt_normalized_selection_step_promotion": l3_promotions,
        "locked_checks": checks,
        "decision": (
            "draft_separate_counterfactual_replay_protocol"
            if passed
            else "stop_no_counterfactual_replay_or_training"
        ),
        "l3_cannot_rescue_l1": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--replication", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = analyze(load_dapo_operational_mixture_plan(args.plan), args.replication)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
