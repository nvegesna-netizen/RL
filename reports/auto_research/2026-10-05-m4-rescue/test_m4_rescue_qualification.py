"""Tests for the outcome-excluded M4-Rescue qualification gate."""

from __future__ import annotations

import copy

from tools.m4_rescue_qualification import EXPECTED_SCORERS, assess_m4_rescue


def artifacts() -> tuple[list[dict], list[dict], dict]:
    header = {
        "event_type": "header",
        "schema_version": 3,
        "mode": "act",
        "policy": "multi_scorer_oars_v2_actuator",
        "actuation_scorer": "m4_rescue",
        "candidate_window_policy": "controlled_frontier",
        "selection_candidate_watermark": 8,
        "scorers": list(EXPECTED_SCORERS),
        "minimum_service_multiplier": 0.98,
        "maximum_service_multiplier": 1.02,
        "max_candidate_groups": 64,
        "exact_search_max_candidates": 16,
        "candidate_mutation": "configured_scorer_exact_removal",
        "stale_replenishment_policy": "one_batch_drop_newest_excess_v1",
        "m4_rescue_minimum_fifo_overlap_groups": 3,
        "m4_rescue_reward_variance_floor_multiplier": 1.0,
        "m4_rescue_numeric_tolerance": 1e-12,
        "m4_rescue_actuation_rule": "strict_imminent_l1_gain_else_fifo",
    }
    decisions: list[dict] = []
    lifecycle: list[dict] = []
    for index in range(64):
        ids = [f"g{index}-{candidate}" for candidate in range(8)]
        selected = [*ids[:3], ids[4]]
        proposals = {
            name: {
                "status": "proposed",
                "proposed_group_ids": selected,
                "proposed_valid_actor_tokens": 400,
                "minimum_tokens": 392,
                "maximum_tokens": 408,
                "combination_count": 70,
                "search_candidate_count": 8,
                "search_strategy": "exact",
            }
            for name in EXPECTED_SCORERS
        }
        proposals["m4_rescue"].update(
            {
                "fifo_overlap_count": 3,
                "proposed_reward_variance_sum": 5.0,
                "proposed_imminent_l1": 5.0,
            }
        )
        total = index + 1
        decisions.append(
            {
                "event_type": "decision",
                "mode": "act",
                "actuation_scorer": "m4_rescue",
                "baseline_group_ids": ids[:4],
                "baseline_imminent_l1": 4.0,
                "baseline_reward_variance_sum": 4.0,
                "candidate_group_count": 8,
                "eligible_candidate_count": 9,
                "candidate_excess_count": 1,
                "candidates": [{"group_id": group_id} for group_id in ids],
                "actual_selected_group_count": 4,
                "actual_selected_group_ids": selected,
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


def assess(oars: list[dict], lifecycle: list[dict], duty: dict) -> dict:
    return assess_m4_rescue(
        oars_rows=oars,
        lifecycle_rows=lifecycle,
        observer_duty=duty,
        source_commit="a" * 40,
        protocol_sha256="b" * 64,
        run_start_ns=0,
        run_end_ns=6_500_000_000,
    )


def test_complete_exact_rescue_qualification_passes() -> None:
    result = assess(*artifacts())

    assert result["status"] == "PASS_M4_RESCUE_QUALIFIED"
    assert all(result["checks"].values())
    assert result["intervention_count"] == 64
    assert result["training_quality_analyzed"] is False
    assert result["scientific_outcome_acquisition"] is False


def test_reward_variance_regression_fails_semantic_gate() -> None:
    oars, lifecycle, duty = artifacts()
    broken = copy.deepcopy(oars)
    broken[1]["proposals"]["m4_rescue"]["proposed_reward_variance_sum"] = 3.0

    result = assess(broken, lifecycle, duty)

    assert result["status"] == "FAIL"
    assert result["checks"]["m4_rescue_semantics"] is False


def test_second_fifo_replacement_fails_semantic_gate() -> None:
    oars, lifecycle, duty = artifacts()
    broken = copy.deepcopy(oars)
    baseline = broken[1]["baseline_group_ids"]
    replacement = broken[1]["candidates"][5]["group_id"]
    proposal = broken[1]["proposals"]["m4_rescue"]
    proposal["proposed_group_ids"] = [baseline[0], baseline[1], replacement, "extra"]
    proposal["fifo_overlap_count"] = 2
    broken[1]["actual_selected_group_ids"] = proposal["proposed_group_ids"]

    result = assess(broken, lifecycle, duty)

    assert result["status"] == "FAIL"
    assert result["checks"]["m4_rescue_semantics"] is False


def test_intervention_without_strict_imminent_gain_fails_semantic_gate() -> None:
    oars, lifecycle, duty = artifacts()
    broken = copy.deepcopy(oars)
    broken[1]["proposals"]["m4_rescue"]["proposed_imminent_l1"] = 4.0

    result = assess(broken, lifecycle, duty)

    assert result["status"] == "FAIL"
    assert result["checks"]["m4_rescue_semantics"] is False
