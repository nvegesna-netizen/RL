# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the outcome-excluded OARS-v2 actuation gate."""

from __future__ import annotations

import copy

from tools.m4_oars_v2_actuation_qualification import (
    OARSV2_SCORERS,
    assess_v2_actuation,
)


def _artifacts(scorer: str) -> tuple[list[dict], list[dict], dict]:
    header = {
        "event_type": "header",
        "schema_version": 2,
        "mode": "act",
        "policy": "multi_scorer_oars_v2_actuator",
        "actuation_scorer": scorer,
        "candidate_window_policy": "controlled_frontier",
        "selection_candidate_watermark": 8,
        "scorers": list(OARSV2_SCORERS),
        "minimum_service_multiplier": 0.98,
        "maximum_service_multiplier": 1.02,
        "max_candidate_groups": 64,
        "exact_search_max_candidates": 16,
        "decision_time_budget_ns": 5_000_000,
        "decision_time_budget_scope": "per_scorer",
        "candidate_mutation": "configured_scorer_exact_removal",
        "stale_replenishment_policy": "one_batch_drop_newest_excess_v1",
    }
    decisions = []
    lifecycle = []
    for index in range(64):
        ids = [f"g{index}-{candidate}" for candidate in range(8)]
        proposals = {
            name: {
                "status": "proposed",
                "proposed_group_ids": ids[-4:],
                "proposed_valid_actor_tokens": 400,
                "minimum_tokens": 392,
                "maximum_tokens": 408,
                "combination_count": 70,
                "search_candidate_count": 8,
                "search_strategy": "exact",
            }
            for name in OARSV2_SCORERS
        }
        total = index + 1
        decisions.append(
            {
                "event_type": "decision",
                "mode": "act",
                "actuation_scorer": scorer,
                "candidate_group_count": 8,
                "eligible_candidate_count": 9,
                "candidate_excess_count": 1,
                "candidates": [{"group_id": group_id} for group_id in ids],
                "actual_selected_group_count": 4,
                "actual_selected_group_ids": ids[-4:],
                "proposal_matches_actual": True,
                "skip_reason": None,
                "decision_latency_ns": 100_000,
                "proposals": proposals,
                "liveness_accounting": {
                    "candidate_excess_pending_groups": 1,
                    "candidate_excess_removed_groups_total": total,
                    "stale_evicted_groups_total": total,
                    "replenishment_batches_earned_total": total,
                    "replenishment_batches_consumed_total": total,
                    "replenishment_credits_outstanding": 0,
                },
            }
        )
        lifecycle.extend(
            [
                {
                    "stage": "learner_version_advanced",
                    "learner_weight_version": total,
                },
                {"stage": "removed", "removal_reason": "oars_candidate_excess"},
                {"stage": "removed", "removal_reason": "stale_evicted"},
            ]
        )
    duty = {"active_window_ns": 6_400_000_000, "corrected_observer_duty": 0.001}
    return [header, *decisions], lifecycle, duty


def _assess(scorer: str = "absolute_m4_risk") -> dict:
    oars, lifecycle, duty = _artifacts(scorer)
    return assess_v2_actuation(
        scorer=scorer,
        oars_rows=oars,
        lifecycle_rows=lifecycle,
        observer_duty=duty,
        source_commit="a" * 40,
        run_start_ns=0,
        run_end_ns=6_500_000_000,
    )


def test_both_preregistered_scorers_pass_complete_gate() -> None:
    for scorer in ("reward_variance_risk", "absolute_m4_risk"):
        result = _assess(scorer)
        assert result["status"] == "PASS_OARS_V2_ACTUATION_QUALIFIED"
        assert all(result["checks"].values())
        assert result["training_quality_analyzed"] is False


def test_wrong_actual_selection_fails_identity_gate() -> None:
    oars, lifecycle, duty = _artifacts("absolute_m4_risk")
    broken = copy.deepcopy(oars)
    broken[1]["actual_selected_group_ids"] = broken[1]["actual_selected_group_ids"][
        ::-1
    ]
    result = assess_v2_actuation(
        scorer="absolute_m4_risk",
        oars_rows=broken,
        lifecycle_rows=lifecycle,
        observer_duty=duty,
        source_commit="b" * 40,
        run_start_ns=0,
        run_end_ns=6_500_000_000,
    )
    assert result["status"] == "FAIL"
    assert result["checks"]["configured_proposal_identity"] is False


def test_unreconciled_liveness_count_fails_gate() -> None:
    oars, lifecycle, duty = _artifacts("reward_variance_risk")
    broken = copy.deepcopy(oars)
    broken[-1]["liveness_accounting"]["stale_evicted_groups_total"] -= 1
    result = assess_v2_actuation(
        scorer="reward_variance_risk",
        oars_rows=broken,
        lifecycle_rows=lifecycle,
        observer_duty=duty,
        source_commit="c" * 40,
        run_start_ns=0,
        run_end_ns=6_500_000_000,
    )
    assert result["status"] == "FAIL"
    assert result["checks"]["complete_liveness_accounting"] is False
