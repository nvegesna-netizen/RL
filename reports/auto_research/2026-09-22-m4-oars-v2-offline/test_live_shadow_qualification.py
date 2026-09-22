# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the frozen OARS-v2 live-shadow qualification gate."""

from __future__ import annotations

import copy

from tools.m4_oars_v2_live_shadow_qualification import (
    OARSV2_SCORERS,
    assess_live_shadow,
)


def _artifacts(*, contended: int = 8) -> tuple[list[dict], list[dict], dict]:
    header = {
        "event_type": "header",
        "schema_version": 1,
        "mode": "observe",
        "policy": "multi_scorer_oars_v2_shadow",
        "candidate_window_policy": "all_naturally_ready_in_window_v2",
        "scorers": list(OARSV2_SCORERS),
        "minimum_service_multiplier": 0.98,
        "maximum_service_multiplier": 1.02,
        "max_candidate_groups": 64,
        "exact_search_max_candidates": 16,
        "decision_time_budget_ns": 5_000_000,
        "decision_time_budget_scope": "per_scorer",
        "candidate_mutation": "none",
    }
    decisions = []
    for index in range(64):
        count = 8 if index < contended else 4
        ids = [f"g{index}-{candidate}" for candidate in range(count)]
        baseline = ids[:4]
        candidates = [
            {
                "group_id": group_id,
                "valid_actor_tokens": 100,
                "l1": float(candidate + 1),
                "l2": float(candidate + 1),
                "reward_mean": 0.0,
                "reward_variance": 0.25,
            }
            for candidate, group_id in enumerate(ids)
        ]
        proposals = {}
        for scorer in OARSV2_SCORERS:
            proposed = ids[-4:] if count > 4 else baseline
            overlap = len(set(proposed) & set(baseline))
            proposals[scorer] = {
                "status": "proposed",
                "proposed_group_ids": proposed,
                "proposed_valid_actor_tokens": 400,
                "fifo_overlap_count": overlap,
                "decision_latency_ns": 100_000,
            }
        decisions.append(
            {
                "event_type": "decision",
                "candidate_group_count": count,
                "candidates": candidates,
                "baseline_group_ids": baseline,
                "baseline_valid_actor_tokens": 400,
                "actual_selected_group_count": 4,
                "actual_selected_group_ids": baseline,
                "baseline_matches_actual": True,
                "skip_reason": None,
                "decision_latency_ns": 400_000,
                "proposals": proposals,
            }
        )
    lifecycle = [
        {
            "stage": "learner_version_advanced",
            "learner_weight_version": version,
            "timestamp_ns": version * 100_000_000,
        }
        for version in range(1, 65)
    ]
    duty = {"active_window_ns": 6_400_000_000, "corrected_observer_duty": 0.001}
    return [header, *decisions], lifecycle, duty


def _assess(*, contended: int = 8) -> dict:
    oars, lifecycle, duty = _artifacts(contended=contended)
    return assess_live_shadow(
        oars_rows=oars,
        lifecycle_rows=lifecycle,
        observer_duty=duty,
        source_commit="a" * 40,
        run_start_ns=0,
        run_end_ns=6_500_000_000,
    )


def test_passes_safety_and_natural_support() -> None:
    result = _assess(contended=8)
    assert result["status"] == "PASS_SHADOW_SYSTEMS_READY"
    assert result["safety_pass"] is True
    assert result["natural_contention_support_pass"] is True
    assert result["candidate_group_count"]["complete_contended"] == 8


def test_safe_sparse_ready_sets_are_not_called_scheduler_ready() -> None:
    result = _assess(contended=7)
    assert result["status"] == "PASS_SAFE_INSUFFICIENT_NATURAL_CONTENTION"
    assert result["safety_pass"] is True
    assert result["natural_contention_support_pass"] is False


def test_fifo_identity_mismatch_fails_safety() -> None:
    oars, lifecycle, duty = _artifacts()
    broken = copy.deepcopy(oars)
    broken[1]["actual_selected_group_ids"] = list(
        reversed(broken[1]["baseline_group_ids"])
    )
    broken[1]["baseline_matches_actual"] = False
    result = assess_live_shadow(
        oars_rows=broken,
        lifecycle_rows=lifecycle,
        observer_duty=duty,
        source_commit="b" * 40,
        run_start_ns=0,
        run_end_ns=6_500_000_000,
    )
    assert result["status"] == "FAIL_SHADOW_SYSTEMS_GATE"
    assert result["checks"]["actual_selection_exactly_matches_fifo"] is False
