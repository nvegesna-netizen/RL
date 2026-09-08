# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for four-block structured scheduler-crossover aggregation."""

import json

from nemo_rl.algorithms.async_utils.structured_scheduler_crossover import (
    StructuredSchedulerCrossoverPlan,
)
from tests.unit.single_controller.test_structured_scheduler_crossover import (
    _plan_record,
)
from tools.analyze_structured_scheduler_crossover_replications import analyze


def test_final_rule_permits_one_weak_nonnegative_replication(tmp_path) -> None:
    plan = StructuredSchedulerCrossoverPlan.model_validate(_plan_record())
    paths = {}
    for order_seed, contrast in zip(
        plan.replication_order, (0.5, 0.25, 0.125, 0.375), strict=True
    ):
        path = tmp_path / f"{order_seed}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "analysis_status": plan.analysis_status,
                    "calibration_only": True,
                    "confirmatory_eligible": False,
                    "natural_benchmark_claim_authorized": False,
                    "counterfactual_replay_authorized": False,
                    "training_authorized": False,
                    "plan_id": plan.plan_id,
                    "order_seed": order_seed,
                    "ready_first_minus_in_order_short_share": contrast,
                    "progression_checks": {"both_arms_valid": True},
                }
            )
        )
        paths[order_seed] = path

    result = analyze(plan=plan, result_paths=paths)

    assert result["decision"] == "controlled_scheduler_crossover_calibration_passed"
    assert all(result["final_acceptance_checks"].values())
