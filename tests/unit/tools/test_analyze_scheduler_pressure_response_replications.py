# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from pathlib import Path

from nemo_rl.algorithms.async_utils.structured_scheduler_crossover import (
    SchedulerPressureResponsePlan,
)
from tests.unit.single_controller.test_scheduler_pressure_response import (
    pressure_plan_record,
)
from tools.analyze_scheduler_pressure_response_replications import analyze


def _result(plan: SchedulerPressureResponsePlan, seed: int) -> dict[str, object]:
    arms = {}
    for level, positive in (("l0", 0), ("l1", 3), ("l3", 6)):
        arms[f"natural_{level}_in_order"] = {
            "selection_steps": 8,
            "selection_pressure": {
                "decisions_with_ready_greater_than_eligible": positive
            },
        }
    comparisons = {
        f"natural_{level}": {
            "fixed_stratum_reference": {
                "lower_reference_mean_normalized_rank_promotion": promotion
            }
        }
        for level, promotion in (("l0", 0.0), ("l1", 1 / 62), ("l3", 2 / 31))
    }
    return {
        "schema_version": 1,
        "analysis_status": plan.analysis_status,
        "calibration_only": True,
        "confirmatory_eligible": False,
        "population_claim_authorized": False,
        "counterfactual_replay_authorized": False,
        "training_authorized": False,
        "plan_id": plan.plan_id,
        "order_seed": seed,
        "arms": arms,
        "comparisons": comparisons,
        "replication_checks": {"all_ten_arms_valid": True},
    }


def test_aggregate_applies_pressure_and_fixed_stratum_progression(
    tmp_path: Path,
) -> None:
    plan = SchedulerPressureResponsePlan.model_validate(pressure_plan_record())
    paths = {}
    for seed in plan.replication_order:
        path = tmp_path / f"{seed}.json"
        path.write_text(json.dumps(_result(plan, seed)))
        paths[seed] = path
    result = analyze(plan=plan, result_paths=paths)
    assert all(result["aggregate_checks"].values())
    assert result["decision"] == (
        "draft_separate_fresh_operational_mixture_zero_update_protocol"
    )
