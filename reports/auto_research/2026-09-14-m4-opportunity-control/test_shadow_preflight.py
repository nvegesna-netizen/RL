#!/usr/bin/env python3
"""Synthetic tests for the outcome-neutral OARS shadow systems gates."""

from __future__ import annotations

import copy

from tools.m4_oars_shadow_preflight import assess_oars_shadow_preflight


def _inputs() -> tuple[
    list[dict[str, object]], list[dict[str, object]], dict[str, object]
]:
    oars: list[dict[str, object]] = [
        {
            "event_type": "header",
            "max_candidate_groups": 25,
            "policy": "baseline_budgeted_oars_v1",
            "schema_version": 1,
            "service_budget_multiplier": 1.02,
        }
    ]
    for index in range(64):
        baseline = [f"g{index:02d}-{offset}" for offset in range(4)]
        proposed = [f"p{index:02d}-{offset}" for offset in range(4)]
        oars.append(
            {
                "event_type": "decision",
                "candidate_group_count": 8 if index < 8 else 4,
                "baseline_group_ids": baseline,
                "actual_selected_group_ids": baseline,
                "actual_selected_group_count": 4,
                "baseline_matches_actual": True,
                "proposed_group_ids": proposed,
                "baseline_valid_actor_tokens": 400,
                "proposed_valid_actor_tokens": 404,
                "token_budget": 408,
                "decision_latency_ns": 1_000_000,
                "skip_reason": None,
            }
        )
    lifecycle = [
        {
            "stage": "learner_version_advanced",
            "learner_weight_version": version,
            "timestamp_ns": version * 1_000_000_000,
        }
        for version in range(1, 65)
    ]
    duty = {
        "active_window_ns": 64_000_000_000,
        "corrected_observer_duty": 0.001,
    }
    return oars, lifecycle, duty


def test_complete_fifo_controlled_shadow_passes() -> None:
    oars, lifecycle, duty = _inputs()

    result = assess_oars_shadow_preflight(
        oars_rows=oars,
        lifecycle_rows=lifecycle,
        observer_duty=duty,
        source_commit="abc123",
        run_start_ns=0,
        run_end_ns=64_000_000_000,
    )

    assert result["status"] == "PASS"
    assert all(result["checks"].values())
    assert result["oars_actuated"] is False
    assert result["training_quality_analyzed"] is False


def test_fifo_identity_mismatch_fails_closed() -> None:
    oars, lifecycle, duty = _inputs()
    mismatched = copy.deepcopy(oars)
    mismatched[1]["baseline_matches_actual"] = False

    result = assess_oars_shadow_preflight(
        oars_rows=mismatched,
        lifecycle_rows=lifecycle,
        observer_duty=duty,
        source_commit="abc123",
        run_start_ns=0,
        run_end_ns=64_000_000_000,
    )

    assert result["status"] == "FAIL"
    assert result["checks"]["actual_selection_exactly_matches_fifo"] is False
