# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the structured natural-latency scheduler crossover protocol."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nemo_rl.algorithms.async_utils.structured_scheduler_crossover import (
    StructuredSchedulerCrossoverPlan,
    StructuredSchedulerCrossoverPlanError,
    compute_structured_scheduler_crossover_plan_id,
    load_scheduler_protocol,
    load_structured_scheduler_crossover_plan,
)


def _plan_record() -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": 1,
        "analysis_status": "controlled_natural_latency_scheduler_crossover",
        "calibration_only": True,
        "confirmatory_eligible": False,
        "natural_benchmark_claim_authorized": False,
        "counterfactual_replay_authorized": False,
        "training_authorized": False,
        "confirmed_candidate_sha256": "ddd7536aa3d67351974629c2c56fbe41456b9e3b48b0d26e47ef7f7e2728f7ef",
        "confirmation_record_sha256": "a" * 64,
        "analysis_code_commit": "b" * 40,
        "expected_base_commit": "c" * 40,
        "expected_image_sha256": "d" * 64,
        "source_design_id": "structured_generation_scheduler_crossover_v1",
        "model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "model_weights_sha256": "456f5eff514d78f7b0ef52a057118046acf15d86606655767db068cabf5f49f7",
        "pools": [
            {
                "replication_id": f"replication_{order_seed}",
                "order_seed": order_seed,
                "selection_seed": selection_seed,
                "generation_study_seed": generation_seed,
                "arm_execution_order": arm_order,
                "pool_id": f"{index + 1:064x}",
                "manifest_sha256": f"{index + 11:064x}",
            }
            for index, (
                order_seed,
                selection_seed,
                generation_seed,
                arm_order,
            ) in enumerate(
                (
                    (46001, 2026090801, 65001, ["ready_first", "in_order"]),
                    (46002, 2026090802, 65002, ["in_order", "ready_first"]),
                    (46003, 2026090803, 65003, ["in_order", "ready_first"]),
                    (46004, 2026090804, 65004, ["ready_first", "in_order"]),
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
        "groups_per_task_per_cohort": 2,
        "completions_per_group": 2,
        "max_total_sequence_length": 768,
        "max_new_tokens": 768,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "max_inflight_prompts": 4,
        "max_buffered_rollouts": 8,
        "sampler_lookahead_versions": 1,
        "release_delay_seconds": 0.0,
        "primary_horizon": 8,
        "replication_order": [46001, 46002, 46003, 46004],
        "thresholds": {
            "backend_length_termination_rate_max_each_stratum_each_arm": 0.125,
            "in_order_short_share_at_primary_horizon": 0.5,
            "long_short_generated_token_median_ratio_min_each_arm": 2.0,
            "long_short_ready_latency_median_ratio_min_each_arm": 1.5,
            "maximum_concurrent_groups_min_each_arm": 2,
            "primary_contrast_min": 0.25,
            "ready_first_short_share_min": 0.75,
            "reward_mean_min_each_stratum_each_arm": 0.75,
            "contrasts_at_or_above_minimum_required": 3,
            "maximum_negative_contrasts": 0,
            "median_contrast_min": 0.25,
        },
        "plan_id": "0" * 64,
    }
    draft = StructuredSchedulerCrossoverPlan.model_construct(**record)
    record["plan_id"] = compute_structured_scheduler_crossover_plan_id(draft)
    return record


def test_load_crossover_plan_verifies_identity_and_abba(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(_plan_record()))

    plan = load_structured_scheduler_crossover_plan(path)

    assert plan.plan_id == compute_structured_scheduler_crossover_plan_id(plan)
    assert plan.pool(46001).arm_execution_order == ("ready_first", "in_order")
    assert plan.pool(46003).arm_execution_order == ("in_order", "ready_first")
    assert plan.arm("ready_first").sampler == "ready_first"
    assert load_scheduler_protocol(path) == plan


def test_crossover_plan_rejects_changed_execution_order(tmp_path: Path) -> None:
    record = _plan_record()
    record["pools"][0]["arm_execution_order"] = ["in_order", "ready_first"]  # type: ignore[index]
    draft = StructuredSchedulerCrossoverPlan.model_construct(**record)
    record["plan_id"] = compute_structured_scheduler_crossover_plan_id(draft)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(record))

    with pytest.raises(
        StructuredSchedulerCrossoverPlanError,
        match="replication bindings mismatch",
    ):
        load_structured_scheduler_crossover_plan(path)
