# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import json

import pytest

from nemo_rl.algorithms.async_utils.scheduler_replay import ReplayGroup
from nemo_rl.algorithms.async_utils.scheduler_replay_plan import (
    ExploratoryClosurePlan,
    ReplayPlanError,
    build_deadline_ticks,
    compute_plan_id,
    load_exploratory_closure_plan,
)


def _plan_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "analysis_status": "exploratory_post_observation",
        "calibration_only": True,
        "confirmatory_eligible": False,
        "semantics": "fixed_tick_target_cohort_deadline_ablation_v1",
        "analysis_code_commit": "a" * 40,
        "expected_completions_per_group": 2,
        "observed_before_plan": [
            {"run_id": "run", "median_latency_ratio": 1.1, "rank_biserial": 0.2}
        ],
        "source_runs": [
            {
                "run_id": "run",
                "pool_id": "b" * 64,
                "manifest_sha256": "c" * 64,
                "trace_sha256": "d" * 64,
                "validation_sha256": "e" * 64,
                "order_seed": 1,
                "generation_seed": 2,
            }
        ],
        "deadlines_ns": [
            250_000_000,
            500_000_000,
            1_000_000_000,
            2_000_000_000,
            4_000_000_000,
        ],
        "tick_plan": {
            "anchor": "max_dispatch_in_cohort",
            "min_prompt_groups": 4,
            "max_prompt_groups": 4,
        },
        "policies": [
            {"label": "target_cohort_deadline", "kernel": "in_order"},
            {
                "label": "ready_first_fixed_admission",
                "kernel": "ready_first",
            },
            {
                "label": "windowed_fifo_w1_fixed_admission",
                "kernel": "windowed",
                "max_staleness_versions": 1,
            },
        ],
        "release_intervention": {
            "method": "sha256_sorted_unrestricted_block_permutation_v1",
            "seed_start": 0,
            "seed_count": 1000,
            "block": "dispatch_cohort_and_decorrelation_block",
            "identity_assignments_allowed": True,
        },
        "horizons": [4, 12, 24, 48],
        "stop_rules": {
            "primary_horizon": 12,
            "adjacent_deadline_scenarios_required": 2,
            "minimum_absolute_tv_interaction": 0.02,
            "maximum_tv_interaction_mcse": 0.002,
            "require_same_direction_all_runs": True,
        },
        "plan_id": "0" * 64,
    }


def _valid_plan() -> ExploratoryClosurePlan:
    draft = ExploratoryClosurePlan.model_validate(_plan_payload())
    return ExploratoryClosurePlan.model_validate(
        {**draft.model_dump(mode="json"), "plan_id": compute_plan_id(draft)}
    )


def _group(group_id: str, *, cohort: int, slot: int, dispatch_ns: int) -> ReplayGroup:
    return ReplayGroup(
        logical_group_id=group_id,
        prompt_uid=f"prompt-{group_id}",
        source_prompt_id=f"source-{group_id}",
        task_stratum="task",
        repeated_prompt_cluster_uid=f"cluster-{group_id}",
        dispatch_cohort=str(cohort),
        decorrelation_block=f"block-{cohort}",
        slot_order=slot,
        dispatch_ns=dispatch_ns,
        ready_ns=dispatch_ns + 1,
        nominal_start_version=cohort,
        target_step=cohort,
    )


def test_plan_round_trip_and_hash_validation(tmp_path):
    plan = _valid_plan()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan.model_dump(mode="json")), encoding="utf-8")
    assert load_exploratory_closure_plan(path) == plan

    corrupted = plan.model_dump(mode="json")
    corrupted["observed_before_plan"][0]["median_latency_ratio"] = 9.0
    path.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(ReplayPlanError, match="ID mismatch"):
        load_exploratory_closure_plan(path)


def test_deadline_ticks_use_last_dispatch_and_fail_on_nonmonotone_anchors():
    plan = _valid_plan()
    groups = (
        _group("c0a", cohort=0, slot=0, dispatch_ns=10),
        _group("c0b", cohort=0, slot=1, dispatch_ns=20),
        _group("c1a", cohort=1, slot=2, dispatch_ns=30),
    )
    ticks = build_deadline_ticks(groups, deadline_ns=100, tick_plan=plan.tick_plan)
    assert [item.monotonic_ns for item in ticks] == [120, 130]
    assert [item.nominal_trainer_version for item in ticks] == [0, 1]

    nonmonotone = (
        _group("c0", cohort=0, slot=0, dispatch_ns=30),
        _group("c1", cohort=1, slot=1, dispatch_ns=20),
    )
    with pytest.raises(ReplayPlanError, match="strictly increasing"):
        build_deadline_ticks(nonmonotone, deadline_ns=100, tick_plan=plan.tick_plan)
