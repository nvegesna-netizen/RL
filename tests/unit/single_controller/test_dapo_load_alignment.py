import json
from pathlib import Path

import pytest

from nemo_rl.algorithms.async_utils.dapo_load_alignment import (
    CONFIRMED_PROTOCOL_SHA256,
    IMPLEMENTATION_CONFIRMATION_SHA256,
    DapoLoadAlignmentPlan,
    compute_dapo_load_alignment_plan_id,
)
from nemo_rl.algorithms.async_utils.structured_scheduler_crossover import (
    load_scheduler_protocol,
)
from tools.materialize_dapo_load_alignment import ARM_EXECUTION_ORDERS


def _plan() -> DapoLoadAlignmentPlan:
    definitions = (
        ("l0", "natural", "none"),
        ("l3", "natural", "none"),
        ("l3", "high_load_delayed", "fixed_higher_load_half"),
        ("l3", "low_load_delayed", "fixed_lower_load_half"),
    )
    arms = [
        {
            "arm_id": f"{level}_{condition}_{sampler}",
            "sampler": sampler,
            "pressure_level": level,
            "sampler_lookahead_versions": {"l0": 0, "l3": 3}[level],
            "max_buffered_rollouts": {"l0": 4, "l3": 16}[level],
            "delay_target": target,
            "release_delay_seconds": 0.0 if target == "none" else 16.0,
        }
        for level, condition, target in definitions
        for sampler in ("in_order", "ready_first")
    ]
    value = {
        "schema_version": 1,
        "analysis_status": "controlled_dapo_load_alignment_signed_control",
        "calibration_only": True,
        "confirmatory_eligible": False,
        "population_claim_authorized": False,
        "counterfactual_replay_authorized": False,
        "training_authorized": False,
        "confirmed_protocol_sha256": CONFIRMED_PROTOCOL_SHA256,
        "implementation_confirmation_sha256": IMPLEMENTATION_CONFIRMATION_SHA256,
        "materialization_confirmation_sha256": "1" * 64,
        "reference_result_sha256": "2" * 64,
        "final_plan_confirmation_sha256": "3" * 64,
        "analysis_code_commit": "4" * 40,
        "expected_base_commit": "5" * 40,
        "expected_image_sha256": "6" * 64,
        "source_design_id": "dapo_math_load_alignment_v1",
        "model_revision": "b101308fe89651ea5ce025f25317fea6fc07e96e",
        "model_weights_sha256": "70a914a3466bf064ca88ae875cb33c366856147e475a2b8da54e7a3d94d8074a",
        "pools": [
            {
                "replication_id": f"replication_{seed}",
                "order_seed": seed,
                "scheduler_selection_seed": 2026092001 + index,
                "reference_generation_seeds": [72001 + index, 72011 + index],
                "scheduler_generation_seed": 73001 + index,
                "arm_execution_order": list(ARM_EXECUTION_ORDERS[seed]),
                "pool_id": f"{10 + index:064x}",
                "manifest_sha256": f"{20 + index:064x}",
                "private_reference_manifest_sha256": f"{30 + index:064x}",
                "fixed_lower_load_prompt_ids": [
                    f"{1000 + index * 100 + n:064x}" for n in range(16)
                ],
                "fixed_higher_load_prompt_ids": [
                    f"{1050 + index * 100 + n:064x}" for n in range(16)
                ],
            }
            for index, seed in enumerate((51001, 51002, 51003))
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
        "replication_order": [51001, 51002, 51003],
        "thresholds": {
            "backend_length_termination_rate_max_each_arm": 0.2,
            "maximum_active_generation_groups_required_each_arm": 4,
            "signed_pressure_median_min": 0.5,
            "signed_pressure_replications_exceeding_l0_min": 2,
            "high_load_delayed_promotion_median_min": 1 / 28,
            "high_load_delayed_replications_at_or_above_min": 2,
            "low_load_delayed_promotion_median_max": -1 / 28,
            "low_load_delayed_replications_at_or_below_max": 2,
            "signed_separation_median_min": 1 / 14,
            "natural_lower_load_promotion_median_min": 1 / 28,
            "natural_lower_load_replications_at_or_above_min": 2,
            "natural_lower_load_negative_replications_max": 1,
            "natural_easier_promotion_median_min": 1 / 28,
            "natural_easier_replications_at_or_above_min": 2,
            "natural_easier_negative_replications_max": 1,
        },
        "plan_id": "0" * 64,
    }
    draft = DapoLoadAlignmentPlan.model_validate(value)
    value["plan_id"] = compute_dapo_load_alignment_plan_id(draft)
    return DapoLoadAlignmentPlan.model_validate(value)


def test_plan_round_trips_through_shared_loader(tmp_path: Path) -> None:
    plan = _plan()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan.model_dump(mode="json")))

    assert load_scheduler_protocol(path) == plan
    assert plan.delayed_prompt_ids(
        "l3_low_load_delayed_ready_first", 51001
    ) == frozenset(plan.pools[0].fixed_lower_load_prompt_ids)
    assert plan.delayed_prompt_ids("l3_natural_ready_first", 51001) == frozenset()


def test_plan_rejects_changed_signed_threshold() -> None:
    value = _plan().model_dump(mode="json")
    value["thresholds"]["signed_separation_median_min"] = 0.01
    with pytest.raises(ValueError, match="thresholds mismatch"):
        DapoLoadAlignmentPlan.model_validate(value)


@pytest.mark.parametrize(("field", "value"), [("temperature", 0.9), ("top_p", 0.8)])
def test_plan_rejects_changed_sampling_parameter(field: str, value: float) -> None:
    plan = _plan().model_dump(mode="json")
    plan[field] = value
    with pytest.raises(ValueError, match="sampling parameters mismatch"):
        DapoLoadAlignmentPlan.model_validate(plan)
