#!/usr/bin/env python3
"""Synthetic tests for the repaired OARS contention systems gate."""

from __future__ import annotations

import copy
from typing import Any

from tools.m4_oars_shadow_contention_preflight import (
    assess_oars_shadow_contention_preflight,
)


def _inputs() -> tuple[
    list[dict[str, object]], list[dict[str, object]], dict[str, object]
]:
    oars: list[dict[str, object]] = [
        {
            "event_type": "header",
            "max_candidate_groups": 25,
            "policy": "baseline_budgeted_oars_v1",
            "schema_version": 1,
            "selection_candidate_watermark": 8,
            "service_budget_multiplier": 1.02,
        }
    ]
    for index in range(64):
        baseline = [f"g{index:02d}-{offset}" for offset in range(4)]
        proposed = [f"p{index:02d}-{offset}" for offset in range(4)]
        oars.append(
            {
                "event_type": "decision",
                "candidate_group_count": 8,
                "baseline_group_ids": baseline,
                "actual_selected_group_ids": baseline,
                "actual_selected_group_count": 4,
                "baseline_matches_actual": True,
                "proposed_group_ids": proposed,
                "baseline_valid_actor_tokens": 400,
                "proposed_valid_actor_tokens": 404,
                "token_budget": 408,
                "combination_count": 70,
                "feasible_combination_count": 70,
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


def _assess(
    oars: list[dict[str, object]],
    lifecycle: list[dict[str, object]],
    duty: dict[str, object],
) -> dict[str, Any]:
    return assess_oars_shadow_contention_preflight(
        oars_rows=oars,
        lifecycle_rows=lifecycle,
        observer_duty=duty,
        source_commit="abc123",
        run_start_ns=0,
        run_end_ns=64_000_000_000,
    )


def test_complete_contention_qualification_passes() -> None:
    result = _assess(*_inputs())

    assert result["status"] == "PASS"
    assert all(result["checks"].values())
    assert result["natural_eager_fifo_representativeness_claim"] is False


def test_uncontended_decision_fails_closed() -> None:
    oars, lifecycle, duty = _inputs()
    uncontended = copy.deepcopy(oars)
    uncontended[1]["candidate_group_count"] = 4
    uncontended[1]["combination_count"] = 1

    result = _assess(uncontended, lifecycle, duty)

    assert result["status"] == "FAIL"
    assert result["checks"]["candidate_watermark_reached_every_decision"] is False
    assert result["checks"]["nontrivial_choice_set_every_decision"] is False


def test_wrong_declared_watermark_fails_closed() -> None:
    oars, lifecycle, duty = _inputs()
    wrong_header = copy.deepcopy(oars)
    wrong_header[0]["selection_candidate_watermark"] = 4

    result = _assess(wrong_header, lifecycle, duty)

    assert result["status"] == "FAIL"
    assert result["checks"]["declared_selection_candidate_watermark"] is False
