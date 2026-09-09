import json
from pathlib import Path

import pytest

from nemo_rl.algorithms.async_utils.structured_scheduler_crossover import (
    DapoSchedulerCrossoverPlan,
    compute_dapo_scheduler_crossover_plan_id,
    load_scheduler_protocol,
)


def _plan() -> DapoSchedulerCrossoverPlan:
    value = {
        "schema_version": 1,
        "analysis_status": "controlled_dapo_natural_latency_scheduler_crossover",
        "calibration_only": True,
        "confirmatory_eligible": False,
        "population_claim_authorized": False,
        "counterfactual_replay_authorized": False,
        "training_authorized": False,
        "confirmed_candidate_sha256": "c49f9604225db847eded1d298c592938299fb3c3d508cf60d2b598f1e8b604c7",
        "confirmation_record_sha256": "a73a4b56744cc31fe5a50f4cccb2b1dd985357ca1879b8377df19c6b1622e91e",
        "analysis_code_commit": "1" * 40,
        "expected_base_commit": "2" * 40,
        "expected_image_sha256": "3" * 64,
        "source_design_id": "dapo_math_scheduler_crossover_v1",
        "model_revision": "b101308fe89651ea5ce025f25317fea6fc07e96e",
        "model_weights_sha256": "70a914a3466bf064ca88ae875cb33c366856147e475a2b8da54e7a3d94d8074a",
        "pools": [
            {
                "replication_id": f"replication_{seed}",
                "order_seed": seed,
                "dispatch_order_seed": seed + 10000,
                "generation_study_seed": seed + 20000,
                "arm_execution_order": list(order),
                "pool_id": f"{index + 4:064x}",
                "manifest_sha256": f"{index + 8:064x}",
            }
            for index, (seed, order) in enumerate(
                (
                    (48001, ("ready_first", "in_order")),
                    (48002, ("in_order", "ready_first")),
                    (48003, ("in_order", "ready_first")),
                    (48004, ("ready_first", "in_order")),
                )
            )
        ],
        "arms": [
            {"arm_id": "ready_first", "sampler": "ready_first"},
            {"arm_id": "in_order", "sampler": "in_order"},
        ],
        "prompt_groups": 16,
        "dispatch_cohorts": 4,
        "groups_per_cohort": 4,
        "completions_per_group": 16,
        "max_total_sequence_length": 6144,
        "data_max_input_seq_length": 2048,
        "hf_config_override_max_position_embeddings": 6144,
        "max_new_tokens": 4096,
        "temperature": 1.0,
        "top_p": 0.7,
        "max_inflight_prompts": 4,
        "max_buffered_rollouts": 8,
        "sampler_lookahead_versions": 1,
        "release_delay_seconds": 0.0,
        "replication_order": [48001, 48002, 48003, 48004],
        "thresholds": {
            "backend_length_termination_rate_max_each_arm": 0.2,
            "distinct_in_order_group_reference_loads_min_each_replication": 12,
            "pooled_in_order_within_pool_spearman_group_mean_tokens_ready_latency_min": 0.5,
            "ready_latency_p90_p10_ratio_min_pooled_in_order": 1.5,
            "median_lower_load_normalized_selection_rank_advantage_min": 1 / 15,
            "replications_at_or_above_minimum_required": 3,
            "replications_below_zero_max": 0,
            "pooled_in_order_reward_variance_group_fraction_min": 0.25,
        },
        "plan_id": "0" * 64,
    }
    draft = DapoSchedulerCrossoverPlan.model_validate(value)
    value["plan_id"] = compute_dapo_scheduler_crossover_plan_id(draft)
    return DapoSchedulerCrossoverPlan.model_validate(value)


def test_dapo_plan_round_trips_through_protocol_loader(tmp_path: Path) -> None:
    plan = _plan()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan.model_dump(mode="json")))

    assert load_scheduler_protocol(path) == plan
    assert plan.pool(48002).generation_study_seed == 68002
    assert plan.arm("ready_first").sampler == "ready_first"


def test_dapo_plan_rejects_changed_threshold() -> None:
    value = _plan().model_dump(mode="json")
    value["thresholds"]["median_lower_load_normalized_selection_rank_advantage_min"] = 0.1

    with pytest.raises(ValueError, match="thresholds mismatch"):
        DapoSchedulerCrossoverPlan.model_validate(value)
