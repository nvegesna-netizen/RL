"""Tests for the outcome-excluded M4-Shield qualification gate."""

from __future__ import annotations

import copy

from tools.m4_shield_qualification import EXPECTED_SCORERS, assess_m4_shield


def artifacts() -> tuple[list[dict], list[dict], dict]:
    header = {
        "event_type": "header",
        "schema_version": 4,
        "mode": "act",
        "policy": "multi_scorer_oars_v2_actuator",
        "actuation_scorer": "m4_shield",
        "candidate_window_policy": "controlled_frontier",
        "selection_candidate_watermark": 8,
        "scorers": list(EXPECTED_SCORERS),
        "minimum_service_multiplier": 0.98,
        "maximum_service_multiplier": 1.02,
        "max_candidate_groups": 64,
        "exact_search_max_candidates": 16,
        "candidate_mutation": "configured_scorer_exact_removal",
        "stale_replenishment_policy": "one_batch_drop_newest_excess_v1",
        "m4_shield_base_proposer": "reward_variance_risk",
        "m4_shield_minimum_base_overlap_groups": 2,
        "m4_shield_numeric_tolerance": 1e-12,
        "m4_shield_utility_contract": "exact_imminent_and_total_reward_variance",
        "m4_shield_actuation_rule": "strict_imminent_l1_gain_else_base_proposal",
    }
    decisions: list[dict] = []
    lifecycle: list[dict] = []
    for index in range(64):
        ids = [f"g{index}-{candidate}" for candidate in range(8)]
        base_ids = ids[:4]
        shield_ids = [ids[2], ids[3], ids[4], ids[5]]
        proposals = {
            name: {
                "status": "proposed",
                "proposed_group_ids": base_ids,
                "proposed_valid_actor_tokens": 400,
                "proposed_imminent_l1": 4.0,
                "proposed_reward_variance_sum": 4.0,
                "proposed_imminent_reward_variance_sum": 4.0,
                "minimum_tokens": 392,
                "maximum_tokens": 408,
                "combination_count": 70,
                "search_candidate_count": 8,
                "search_strategy": "exact",
            }
            for name in EXPECTED_SCORERS
        }
        proposals["m4_shield"].update(
            {
                "proposed_group_ids": shield_ids,
                "proposed_imminent_l1": 8.0,
            }
        )
        decisions.append(
            {
                "event_type": "decision",
                "mode": "act",
                "actuation_scorer": "m4_shield",
                "baseline_group_ids": base_ids,
                "candidate_group_count": 8,
                "eligible_candidate_count": 8,
                "candidate_excess_count": 0,
                "candidates": [{"group_id": group_id} for group_id in ids],
                "actual_selected_group_count": 4,
                "actual_selected_group_ids": shield_ids,
                "proposal_matches_actual": True,
                "skip_reason": None,
                "decision_latency_ns": 100_000,
                "proposals": proposals,
                "liveness_accounting": {
                    "candidate_excess_pending_groups": 0,
                    "candidate_excess_removed_groups_total": 0,
                    "stale_evicted_groups_total": 0,
                    "replenishment_batches_earned_total": 0,
                    "replenishment_batches_consumed_total": 0,
                    "replenishment_credits_outstanding": 0,
                },
            }
        )
        lifecycle.append(
            {
                "stage": "learner_version_advanced",
                "learner_weight_version": index + 1,
            }
        )
    duty = {"active_window_ns": 6_400_000_000, "corrected_observer_duty": 0.001}
    return [header, *decisions], lifecycle, duty


def assess(oars: list[dict], lifecycle: list[dict], duty: dict) -> dict:
    return assess_m4_shield(
        oars_rows=oars,
        lifecycle_rows=lifecycle,
        observer_duty=duty,
        source_commit="a" * 40,
        protocol_sha256="b" * 64,
        run_start_ns=0,
        run_end_ns=6_500_000_000,
    )


def test_complete_exact_shield_qualification_passes_without_removals() -> None:
    result = assess(*artifacts())

    assert result["status"] == "PASS_M4_SHIELD_QUALIFIED"
    assert all(result["checks"].values())
    assert result["intervention_count"] == 64
    assert result["training_quality_analyzed"] is False
    assert result["scientific_outcome_acquisition"] is False


def test_zero_shield_activity_fails_reachability_gate() -> None:
    oars, lifecycle, duty = artifacts()
    broken = copy.deepcopy(oars)
    for row in broken[1:]:
        base = row["proposals"]["reward_variance_risk"]
        shield = row["proposals"]["m4_shield"]
        shield["proposed_group_ids"] = base["proposed_group_ids"]
        shield["proposed_imminent_l1"] = base["proposed_imminent_l1"]
        row["actual_selected_group_ids"] = shield["proposed_group_ids"]

    result = assess(broken, lifecycle, duty)

    assert result["status"] == "FAIL"
    assert result["checks"]["nonzero_shield_activity"] is False


def test_reward_variance_utility_change_fails_semantic_gate() -> None:
    oars, lifecycle, duty = artifacts()
    broken = copy.deepcopy(oars)
    broken[1]["proposals"]["m4_shield"]["proposed_reward_variance_sum"] = 3.0

    result = assess(broken, lifecycle, duty)

    assert result["status"] == "FAIL"
    assert result["checks"]["m4_shield_semantics"] is False


def test_intervention_without_strict_imminent_gain_fails_semantic_gate() -> None:
    oars, lifecycle, duty = artifacts()
    broken = copy.deepcopy(oars)
    broken[1]["proposals"]["m4_shield"]["proposed_imminent_l1"] = 4.0

    result = assess(broken, lifecycle, duty)

    assert result["status"] == "FAIL"
    assert result["checks"]["m4_shield_semantics"] is False


def test_liveness_mismatch_fails_even_without_positive_event_requirement() -> None:
    oars, lifecycle, duty = artifacts()
    broken = copy.deepcopy(oars)
    broken[-1]["liveness_accounting"]["stale_evicted_groups_total"] = 1

    result = assess(broken, lifecycle, duty)

    assert result["status"] == "FAIL"
    assert result["checks"]["complete_liveness_accounting"] is False
