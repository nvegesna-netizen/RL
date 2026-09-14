import json
from pathlib import Path

import pytest

from nemo_rl.algorithms.async_utils.dapo_operational_mixture import (
    DapoOperationalMixturePlan,
    compute_dapo_operational_mixture_plan_id,
)
from nemo_rl.algorithms.async_utils.structured_scheduler_crossover import (
    load_scheduler_protocol,
)
from tools.materialize_dapo_operational_mixture import ARM_EXECUTION_ORDERS


def _plan() -> DapoOperationalMixturePlan:
    arms = [
        {
            "arm_id": f"{level}_{sampler}",
            "sampler": sampler,
            "pressure_level": level,
            "sampler_lookahead_versions": {"l0": 0, "l1": 1, "l3": 3}[level],
            "max_buffered_rollouts": {"l0": 4, "l1": 8, "l3": 16}[level],
        }
        for level in ("l0", "l1", "l3")
        for sampler in ("in_order", "ready_first")
    ]
    value = {
        "schema_version": 1,
        "analysis_status": "controlled_dapo_operational_mixture_zero_update_shadow",
        "calibration_only": True,
        "confirmatory_eligible": False,
        "population_claim_authorized": False,
        "counterfactual_replay_authorized": False,
        "training_authorized": False,
        "confirmed_protocol_sha256": "e9942c4afbec0b3b622079d1208eaba6a7c83a7759fc307cfd47293e797a9f63",
        "implementation_confirmation_sha256": "1" * 64,
        "reference_execution_confirmation_sha256": "2" * 64,
        "reference_result_sha256": "3" * 64,
        "final_plan_confirmation_sha256": "4" * 64,
        "analysis_code_commit": "5" * 40,
        "expected_base_commit": "6" * 40,
        "expected_image_sha256": "7" * 64,
        "source_design_id": "dapo_math_operational_mixture_v1",
        "model_revision": "b101308fe89651ea5ce025f25317fea6fc07e96e",
        "model_weights_sha256": "70a914a3466bf064ca88ae875cb33c366856147e475a2b8da54e7a3d94d8074a",
        "pools": [
            {
                "replication_id": f"replication_{seed}",
                "order_seed": seed,
                "scheduler_selection_seed": 2026091301 + index,
                "scheduler_generation_seed": 71001 + index,
                "arm_execution_order": list(ARM_EXECUTION_ORDERS[seed]),
                "pool_id": f"{8 + index:064x}",
                "manifest_sha256": f"{11 + index:064x}",
                "private_reference_manifest_sha256": f"{14 + index:064x}",
            }
            for index, seed in enumerate((50001, 50002, 50003))
        ],
        "arms": arms,
        "prompt_groups": 32,
        "completions_per_group": 16,
        "selection_groups_per_step": 4,
        "selection_steps": 8,
        "max_inflight_prompts": 4,
        "max_total_sequence_length": 6144,
        "data_max_input_seq_length": 2048,
        "hf_config_override_max_position_embeddings": 6144,
        "max_new_tokens": 4096,
        "temperature": 1.0,
        "top_p": 0.7,
        "release_delay_seconds": 0.0,
        "replication_order": [50001, 50002, 50003],
        "thresholds": {
            "backend_length_termination_rate_max_each_arm": 0.2,
            "maximum_active_generation_groups_required_each_arm": 4,
            "l0_maximum_unreleased_groups_required_each_arm": 4,
            "l1_pressure_median_min": 0.25,
            "l1_pressure_replications_exceeding_l0_min": 2,
            "l1_median_normalized_selection_step_promotion_min": 1 / 28,
            "l1_replications_at_or_above_minimum": 2,
            "l1_negative_replications_max": 1,
        },
        "plan_id": "0" * 64,
    }
    draft = DapoOperationalMixturePlan.model_validate(value)
    value["plan_id"] = compute_dapo_operational_mixture_plan_id(draft)
    return DapoOperationalMixturePlan.model_validate(value)


def test_plan_round_trips_through_shared_loader(tmp_path: Path) -> None:
    plan = _plan()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan.model_dump(mode="json")))

    assert load_scheduler_protocol(path) == plan
    assert plan.arm("l3_ready_first").max_buffered_rollouts == 16


def test_plan_rejects_changed_primary_threshold() -> None:
    value = _plan().model_dump(mode="json")
    value["thresholds"]["l1_median_normalized_selection_step_promotion_min"] = 0.1
    with pytest.raises(ValueError, match="thresholds mismatch"):
        DapoOperationalMixturePlan.model_validate(value)
