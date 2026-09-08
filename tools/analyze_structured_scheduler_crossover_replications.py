# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Aggregate the four frozen structured scheduler-crossover replications."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from nemo_rl.algorithms.async_utils.structured_scheduler_crossover import (
    StructuredSchedulerCrossoverPlan,
    load_structured_scheduler_crossover_plan,
)


class StructuredSchedulerReplicationAnalysisError(ValueError):
    """A block result does not match the final frozen protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StructuredSchedulerReplicationAnalysisError(message)


def analyze(
    *, plan: StructuredSchedulerCrossoverPlan, result_paths: dict[int, Path]
) -> dict[str, Any]:
    _require(
        tuple(result_paths) == plan.replication_order,
        "results must be supplied in frozen replication order",
    )
    results = []
    for order_seed, path in result_paths.items():
        value = json.loads(path.read_text())
        _require(isinstance(value, dict), "block result must be an object")
        _require(
            value.get("schema_version") == 1
            and value.get("analysis_status") == plan.analysis_status
            and value.get("calibration_only") is True
            and value.get("confirmatory_eligible") is False
            and value.get("natural_benchmark_claim_authorized") is False
            and value.get("counterfactual_replay_authorized") is False
            and value.get("training_authorized") is False
            and value.get("plan_id") == plan.plan_id
            and value.get("order_seed") == order_seed,
            f"replication {order_seed} identity or evidence labels mismatch",
        )
        results.append(value)
    contrasts = [
        float(value["ready_first_minus_in_order_short_share"]) for value in results
    ]
    all_valid = all(
        value["progression_checks"]["both_arms_valid"] is True for value in results
    )
    thresholds = plan.thresholds
    checks = {
        "all_four_replications_valid": all_valid,
        "no_negative_contrast": (
            sum(value < 0 for value in contrasts)
            <= thresholds.maximum_negative_contrasts
        ),
        "at_least_three_contrasts_ge_0_25": (
            sum(value >= thresholds.primary_contrast_min for value in contrasts)
            >= thresholds.contrasts_at_or_above_minimum_required
        ),
        "median_contrast_ge_0_25": (
            statistics.median(contrasts) >= thresholds.median_contrast_min
        ),
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "analysis_status": plan.analysis_status,
        "calibration_only": True,
        "confirmatory_eligible": False,
        "natural_benchmark_claim_authorized": False,
        "counterfactual_replay_authorized": False,
        "training_authorized": False,
        "plan_id": plan.plan_id,
        "replication_order": list(plan.replication_order),
        "contrasts": contrasts,
        "median_contrast": statistics.median(contrasts),
        "final_acceptance_checks": checks,
        "decision": (
            "controlled_scheduler_crossover_calibration_passed"
            if passed
            else "stop_no_training_or_broader_claim"
        ),
        "interpretation": (
            "controlled synthetic-workload scheduler calibration only; not a "
            "natural benchmark, counterfactual replay, or training result"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--result", action="append", required=True, metavar="ORDER_SEED=PATH"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = load_structured_scheduler_crossover_plan(args.plan)
    paths = {}
    for assignment in args.result:
        raw_seed, separator, raw_path = assignment.partition("=")
        _require(bool(separator and raw_seed and raw_path), "invalid --result")
        order_seed = int(raw_seed)
        _require(order_seed not in paths, f"duplicate order seed {order_seed}")
        paths[order_seed] = Path(raw_path)
    result = analyze(plan=plan, result_paths=paths)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
