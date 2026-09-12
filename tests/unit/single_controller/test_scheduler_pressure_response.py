# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from pathlib import Path

import pytest

from nemo_rl.algorithms.async_utils.structured_scheduler_crossover import (
    PRESSURE_RESPONSE_CONCURRENCY_AMENDMENT_CONFIRMATION_SHA256,
    PRESSURE_RESPONSE_CONCURRENCY_AMENDMENT_SHA256,
    PRESSURE_RESPONSE_CONFIRMED_CANDIDATE_SHA256,
    PRESSURE_RESPONSE_POOL_49005_MATERIALIZATION_RESULT_SHA256,
    PRESSURE_RESPONSE_REPLACED_PLAN_ID,
    PRESSURE_RESPONSE_REPLACEMENT_AMENDMENT_SHA256,
    PRESSURE_RESPONSE_REPLACEMENT_CONFIRMATION_SHA256,
    PRESSURE_RESPONSE_SUPERSEDED_PLAN_ID,
    PRESSURE_RESPONSE_TQ_RUNTIME_VALIDATION_RESULT_SHA256,
    SchedulerPressureResponsePlan,
    compute_scheduler_pressure_response_plan_id,
    load_scheduler_pressure_response_plan,
    load_scheduler_protocol,
)


def pressure_plan_record() -> dict[str, object]:
    arms = []
    for level, lookahead, buffer in (("l0", 0, 4), ("l1", 1, 8), ("l3", 3, 16)):
        for sampler in ("ready_first", "in_order"):
            arms.append(
                {
                    "arm_id": f"natural_{level}_{sampler}",
                    "sampler": sampler,
                    "pressure_level": level,
                    "sampler_lookahead_versions": lookahead,
                    "max_buffered_rollouts": buffer,
                    "latency_condition": "natural",
                    "delay_mapping": "none",
                    "delayed_task": "none",
                    "release_delay_seconds": 0.0,
                }
            )
    for mapping, task in (
        ("short_delayed", "structured_short"),
        ("long_delayed", "structured_long"),
    ):
        for sampler in ("ready_first", "in_order"):
            arms.append(
                {
                    "arm_id": f"controlled_positive_control_l3_{mapping}_{sampler}",
                    "sampler": sampler,
                    "pressure_level": "l3",
                    "sampler_lookahead_versions": 3,
                    "max_buffered_rollouts": 16,
                    "latency_condition": "controlled_positive_control",
                    "delay_mapping": mapping,
                    "delayed_task": task,
                    "release_delay_seconds": 30.0,
                }
            )
    arm_ids = [arm["arm_id"] for arm in arms]
    record: dict[str, object] = {
        "schema_version": 1,
        "analysis_status": "controlled_zero_update_scheduler_pressure_response_surface",
        "calibration_only": True,
        "confirmatory_eligible": False,
        "population_claim_authorized": False,
        "counterfactual_replay_authorized": False,
        "training_authorized": False,
        "confirmed_candidate_sha256": PRESSURE_RESPONSE_CONFIRMED_CANDIDATE_SHA256,
        "confirmation_record_sha256": "1" * 64,
        "analysis_code_commit": "2" * 40,
        "expected_base_commit": "3" * 40,
        "expected_image_sha256": "4" * 64,
        "source_design_id": "structured_scheduler_pressure_response_v1",
        "model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "model_weights_sha256": "456f5eff514d78f7b0ef52a057118046acf15d86606655767db068cabf5f49f7",
        "pools": [
            {
                "replication_id": f"replication_{seed}",
                "order_seed": seed,
                "selection_seed": selection,
                "generation_study_seed": generation,
                "arm_execution_order": arm_ids,
                "pool_id": str(index) * 64,
                "manifest_sha256": str(index + 3) * 64,
            }
            for index, (seed, selection, generation) in enumerate(
                (
                    (49001, 2026091001, 69001),
                    (49002, 2026091002, 69002),
                    (49003, 2026091003, 69003),
                ),
                1,
            )
        ],
        "arms": arms,
        "prompt_groups": 32,
        "dispatch_cohorts": 8,
        "groups_per_cohort": 4,
        "groups_per_stratum_per_cohort": 2,
        "completions_per_group": 2,
        "selection_groups_per_step": 4,
        "max_inflight_prompts": 4,
        "max_total_sequence_length": 768,
        "max_new_tokens": 768,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "replication_order": [49001, 49002, 49003],
        "thresholds": {
            "backend_length_termination_rate_max_each_stratum_each_arm": 0.125,
            "reward_mean_min_each_stratum_each_arm": 0.75,
            "maximum_concurrent_groups_required_each_arm": 4,
            "natural_long_short_generated_token_median_ratio_min_each_arm": 2.0,
            "natural_long_short_ready_latency_median_ratio_min_each_arm": 1.5,
            "positive_control_undelayed_share_ready_first_min": 0.75,
            "positive_control_undelayed_share_in_order_required": 0.5,
            "positive_control_contrast_min_each_mapping": 0.25,
            "pressure_activation_median_min": 0.5,
            "pressure_activation_replications_exceeding_l0_min": 2,
            "composition_median_normalized_rank_promotion_min": 1 / 31,
            "composition_replications_at_or_above_minimum": 2,
            "composition_negative_replications_max": 1,
        },
        "plan_id": "0" * 64,
    }
    draft = SchedulerPressureResponsePlan.model_validate(record)
    record["plan_id"] = compute_scheduler_pressure_response_plan_id(draft)
    return record


def amended_pressure_plan_record() -> dict[str, object]:
    record = pressure_plan_record()
    pools = record["pools"]
    assert isinstance(pools, list)
    arm_order = pools[0]["arm_execution_order"]
    record.update(
        schema_version=2,
        confirmed_concurrency_amendment_sha256=(
            PRESSURE_RESPONSE_CONCURRENCY_AMENDMENT_SHA256
        ),
        concurrency_amendment_confirmation_sha256=(
            PRESSURE_RESPONSE_CONCURRENCY_AMENDMENT_CONFIRMATION_SHA256
        ),
        supersedes_plan_id=PRESSURE_RESPONSE_SUPERSEDED_PLAN_ID,
        excluded_calibration_order_seed=49001,
        pools=[
            *pools[1:],
            {
                "replication_id": "replication_49004",
                "order_seed": 49004,
                "selection_seed": 2026091004,
                "generation_study_seed": 69004,
                "arm_execution_order": arm_order,
                "pool_id": "4" * 64,
                "manifest_sha256": "7" * 64,
            },
        ],
        replication_order=[49002, 49003, 49004],
    )
    thresholds = record["thresholds"]
    assert isinstance(thresholds, dict)
    thresholds.pop("maximum_concurrent_groups_required_each_arm")
    thresholds["maximum_active_generation_groups_required_each_arm"] = 4
    thresholds["natural_maximum_unreleased_groups_required_each_arm"] = 4
    record["plan_id"] = "0" * 64
    draft = SchedulerPressureResponsePlan.model_validate(record)
    record["plan_id"] = compute_scheduler_pressure_response_plan_id(draft)
    return record


def replacement_pressure_plan_record() -> dict[str, object]:
    record = amended_pressure_plan_record()
    pools = record["pools"]
    assert isinstance(pools, list)
    replaced_pool = pools[-1]
    assert isinstance(replaced_pool, dict)
    record.update(
        schema_version=3,
        confirmed_replacement_amendment_sha256=(
            PRESSURE_RESPONSE_REPLACEMENT_AMENDMENT_SHA256
        ),
        replacement_amendment_confirmation_sha256=(
            PRESSURE_RESPONSE_REPLACEMENT_CONFIRMATION_SHA256
        ),
        tq_runtime_validation_result_sha256=(
            PRESSURE_RESPONSE_TQ_RUNTIME_VALIDATION_RESULT_SHA256
        ),
        replacement_pool_materialization_result_sha256=(
            PRESSURE_RESPONSE_POOL_49005_MATERIALIZATION_RESULT_SHA256
        ),
        excluded_infrastructure_order_seed=49004,
        data_plane_actor_runtime_env_mode="inherit_baked_single_node",
        required_live_ray_nodes=1,
        inherited_completed_replications=[
            {
                "order_seed": 49002,
                "source_plan_id": PRESSURE_RESPONSE_REPLACED_PLAN_ID,
                "scientific_result_sha256": (
                    "b6e467b3100d287f0928d2dd26b08af77f70de446364015f58cd94fcd61c4bbf"
                ),
                "audit_sha256": (
                    "085801de8a031d00eb1748dba455af13831a84c7e9d5eedf9dfcddcd4faa41b4"
                ),
            },
            {
                "order_seed": 49003,
                "source_plan_id": PRESSURE_RESPONSE_REPLACED_PLAN_ID,
                "scientific_result_sha256": (
                    "09ace64da91e2cb063bbbfc9d52c5656f19fbb8ec54819d3e23bc3fabdca271c"
                ),
                "audit_sha256": (
                    "224a22aac661b322a5c23020c1d0794c0f27d44cfbe6d14a00ddecd4b288c3e6"
                ),
            },
        ],
        supersedes_plan_id=PRESSURE_RESPONSE_REPLACED_PLAN_ID,
        pools=[
            *pools[:-1],
            {
                **replaced_pool,
                "replication_id": "replication_49005",
                "order_seed": 49005,
                "selection_seed": 2026091005,
                "generation_study_seed": 69005,
                "pool_id": "8" * 64,
                "manifest_sha256": "9" * 64,
            },
        ],
        replication_order=[49002, 49003, 49005],
    )
    record["plan_id"] = "0" * 64
    draft = SchedulerPressureResponsePlan.model_validate(record)
    record["plan_id"] = compute_scheduler_pressure_response_plan_id(draft)
    return record


def test_pressure_plan_round_trip_and_generic_dispatch(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(pressure_plan_record()))
    plan = load_scheduler_pressure_response_plan(path)
    assert plan.prompt_groups == 32
    assert len(plan.arms) == 10
    assert load_scheduler_protocol(path) == plan


def test_amended_pressure_plan_round_trip_and_replication_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "plan.v2.json"
    path.write_text(json.dumps(amended_pressure_plan_record()))
    plan = load_scheduler_pressure_response_plan(path)
    assert plan.schema_version == 2
    assert plan.replication_order == (49002, 49003, 49004)
    assert plan.excluded_calibration_order_seed == 49001
    assert plan.thresholds.maximum_active_generation_groups_required_each_arm == 4
    assert plan.thresholds.natural_maximum_unreleased_groups_required_each_arm == 4


def test_replacement_pressure_plan_round_trip_and_runtime_binding(
    tmp_path: Path,
) -> None:
    path = tmp_path / "plan.v3.json"
    path.write_text(json.dumps(replacement_pressure_plan_record()))
    plan = load_scheduler_pressure_response_plan(path)
    assert plan.schema_version == 3
    assert plan.replication_order == (49002, 49003, 49005)
    assert plan.excluded_infrastructure_order_seed == 49004
    assert plan.data_plane_actor_runtime_env_mode == "inherit_baked_single_node"
    assert plan.required_live_ray_nodes == 1
    assert tuple(
        binding.order_seed for binding in plan.inherited_completed_replications or ()
    ) == (49002, 49003)


def test_replacement_pressure_plan_rejects_missing_validation_binding() -> None:
    record = replacement_pressure_plan_record()
    record["tq_runtime_validation_result_sha256"] = None
    with pytest.raises(ValueError, match="replacement binding mismatch"):
        SchedulerPressureResponsePlan.model_validate(record)


def test_amended_pressure_plan_rejects_old_concurrency_gate() -> None:
    record = amended_pressure_plan_record()
    thresholds = record["thresholds"]
    assert isinstance(thresholds, dict)
    thresholds["maximum_concurrent_groups_required_each_arm"] = 4
    with pytest.raises(ValueError, match="thresholds mismatch"):
        SchedulerPressureResponsePlan.model_validate(record)


def test_pressure_plan_rejects_mutated_pressure_geometry() -> None:
    record = pressure_plan_record()
    record["arms"][0]["max_buffered_rollouts"] = 8  # type: ignore[index]
    with pytest.raises(ValueError, match="pressure level"):
        SchedulerPressureResponsePlan.model_validate(record)
