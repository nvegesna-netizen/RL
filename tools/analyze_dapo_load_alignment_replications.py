# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Apply frozen gates to three DAPO load-alignment replications."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from nemo_rl.algorithms.async_utils.dapo_load_alignment import (
    DapoLoadAlignmentPlan,
    load_dapo_load_alignment_plan,
)


class DapoLoadAlignmentAggregateError(ValueError):
    """A replication result or aggregate invariant is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DapoLoadAlignmentAggregateError(message)


def analyze(plan: DapoLoadAlignmentPlan, paths: list[Path]) -> dict[str, object]:
    _require(len(paths) == 3, "exactly three replication results are required")
    results: list[dict[str, Any]] = []
    for seed, path in zip(plan.replication_order, paths, strict=True):
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise DapoLoadAlignmentAggregateError(
                f"cannot load replication {seed}"
            ) from error
        _require(
            isinstance(value, dict)
            and value.get("plan_id") == plan.plan_id
            and value.get("order_seed") == seed
            and value.get("decision") == "pass_to_next_replication",
            f"replication {seed} did not pass frozen validity gates",
        )
        results.append(value)

    def pressure(result: dict[str, Any], arm_id: str) -> float:
        summary = result["arms"][arm_id]["selection_pressure"]
        return float(summary["decisions_with_ready_greater_than_eligible"]) / 8

    def promotion(result: dict[str, Any], condition: str, target: str) -> float:
        return float(
            result["comparisons"][condition][
                f"fixed_{target}_mean_normalized_selection_step_promotion"
            ]
        )

    l0_pressure = [pressure(result, "l0_natural_in_order") for result in results]
    high_pressure = [
        pressure(result, "l3_high_load_delayed_in_order") for result in results
    ]
    low_pressure = [
        pressure(result, "l3_low_load_delayed_in_order") for result in results
    ]
    high_promotions = [
        promotion(result, "l3_high_load_delayed", "lower_load") for result in results
    ]
    low_promotions = [
        promotion(result, "l3_low_load_delayed", "lower_load") for result in results
    ]
    natural_load = [promotion(result, "l3_natural", "lower_load") for result in results]
    natural_easier = [promotion(result, "l3_natural", "easier") for result in results]
    separations = [
        high - low for high, low in zip(high_promotions, low_promotions, strict=True)
    ]
    thresholds = plan.thresholds
    validity = {
        "all_three_replications_valid": True,
        "negative_control_exact_each_replication": all(
            promotion(result, "l0_natural", "lower_load") == 0
            and promotion(result, "l0_natural", "easier") == 0
            for result in results
        ),
    }
    signed_pressure = {
        "high_load_delayed_pressure_median_ge_0_5": statistics.median(high_pressure)
        >= thresholds.signed_pressure_median_min,
        "low_load_delayed_pressure_median_ge_0_5": statistics.median(low_pressure)
        >= thresholds.signed_pressure_median_min,
        "high_load_delayed_pressure_exceeds_l0_at_least_twice": sum(
            high > l0 for high, l0 in zip(high_pressure, l0_pressure, strict=True)
        )
        >= thresholds.signed_pressure_replications_exceeding_l0_min,
        "low_load_delayed_pressure_exceeds_l0_at_least_twice": sum(
            low > l0 for low, l0 in zip(low_pressure, l0_pressure, strict=True)
        )
        >= thresholds.signed_pressure_replications_exceeding_l0_min,
    }
    signed_response = {
        "high_load_delayed_promotion_median_ge_1_over_28": statistics.median(
            high_promotions
        )
        >= thresholds.high_load_delayed_promotion_median_min,
        "high_load_delayed_promotion_ge_1_over_28_at_least_twice": sum(
            value >= thresholds.high_load_delayed_promotion_median_min
            for value in high_promotions
        )
        >= thresholds.high_load_delayed_replications_at_or_above_min,
        "low_load_delayed_promotion_median_le_minus_1_over_28": statistics.median(
            low_promotions
        )
        <= thresholds.low_load_delayed_promotion_median_max,
        "low_load_delayed_promotion_le_minus_1_over_28_at_least_twice": sum(
            value <= thresholds.low_load_delayed_promotion_median_max
            for value in low_promotions
        )
        >= thresholds.low_load_delayed_replications_at_or_below_max,
        "signed_separation_median_ge_1_over_14": statistics.median(separations)
        >= thresholds.signed_separation_median_min,
    }
    natural_process = {
        "natural_lower_load_promotion_median_ge_1_over_28": statistics.median(
            natural_load
        )
        >= thresholds.natural_lower_load_promotion_median_min,
        "natural_lower_load_promotion_ge_1_over_28_at_least_twice": sum(
            value >= thresholds.natural_lower_load_promotion_median_min
            for value in natural_load
        )
        >= thresholds.natural_lower_load_replications_at_or_above_min,
        "natural_lower_load_negative_at_most_once": sum(
            value < 0 for value in natural_load
        )
        <= thresholds.natural_lower_load_negative_replications_max,
    }
    learner_relevance = {
        "natural_easier_promotion_median_ge_1_over_28": statistics.median(
            natural_easier
        )
        >= thresholds.natural_easier_promotion_median_min,
        "natural_easier_promotion_ge_1_over_28_at_least_twice": sum(
            value >= thresholds.natural_easier_promotion_median_min
            for value in natural_easier
        )
        >= thresholds.natural_easier_replications_at_or_above_min,
        "natural_easier_negative_at_most_once": sum(
            value < 0 for value in natural_easier
        )
        <= thresholds.natural_easier_negative_replications_max,
    }
    validity_passed = all(validity.values())
    signed_passed = all(signed_pressure.values()) and all(signed_response.values())
    natural_passed = all(natural_process.values())
    relevance_passed = all(learner_relevance.values())
    if not validity_passed:
        decision = "stop_and_diagnose_without_posthoc_substitution"
    elif not signed_passed:
        decision = "stop_mechanism_not_identifiable_at_fixed_pressure_and_delay"
    elif not natural_passed:
        decision = "close_controllable_but_not_material_under_natural_dapo_conditions"
    elif not relevance_passed:
        decision = "close_load_composition_effect_without_difficulty_exposure_shift"
    else:
        decision = "draft_separate_fresh_counterfactual_replay_candidate"
    return {
        "schema_version": 1,
        "analysis_status": "aggregate_dapo_load_alignment_signed_control",
        "calibration_only": True,
        "confirmatory_eligible": False,
        "population_claim_authorized": False,
        "counterfactual_replay_authorized": False,
        "training_authorized": False,
        "plan_id": plan.plan_id,
        "replication_order": list(plan.replication_order),
        "l0_pressure_fraction": l0_pressure,
        "high_load_delayed_pressure_fraction": high_pressure,
        "low_load_delayed_pressure_fraction": low_pressure,
        "high_load_delayed_lower_load_promotion": high_promotions,
        "low_load_delayed_lower_load_promotion": low_promotions,
        "signed_separation": separations,
        "natural_lower_load_promotion": natural_load,
        "natural_easier_promotion": natural_easier,
        "locked_checks": {
            "validity": validity,
            "signed_pressure": signed_pressure,
            "signed_response": signed_response,
            "natural_process": natural_process,
            "learner_relevance": learner_relevance,
        },
        "decision": decision,
        "automatic_replay": False,
        "automatic_training": False,
        "automatic_population_claim": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--replication", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = analyze(load_dapo_load_alignment_plan(args.plan), args.replication)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
