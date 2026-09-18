# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Synthetic tests for the OARS actuation qualification gate."""

from __future__ import annotations

import copy
from typing import Any

from tools.m4_oars_actuation_qualification import (
    assess_oars_actuation_qualification,
)


def _inputs(
    mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "candidate_window_policy": "oldest_ready_exact_watermark_v1",
            "event_type": "header",
            "max_candidate_groups": 25,
            "mode": mode,
            "policy": "baseline_budgeted_oars_v1",
            "schema_version": 3,
            "selection_candidate_watermark": 8,
            "service_budget_multiplier": 1.02,
            "stale_replenishment_policy": (
                "one_batch_drop_newest_excess_v1" if mode == "act" else "none"
            ),
        }
    ]
    for index in range(64):
        baseline = [f"b{index}-{offset}" for offset in range(4)]
        proposal = [f"p{index}-{offset}" for offset in range(4)]
        actual = baseline if mode == "observe" else proposal
        rows.append(
            {
                "event_type": "decision",
                "candidate_excess_count": 0,
                "candidate_group_count": 8,
                "combination_count": 70,
                "eligible_candidate_count": 8,
                "baseline_group_ids": baseline,
                "proposed_group_ids": proposal,
                "actual_selected_group_ids": actual,
                "actual_selected_group_count": 4,
                "baseline_matches_actual": mode == "observe",
                "proposal_matches_actual": mode == "act",
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
        }
        for version in range(1, 65)
    ]
    duty = {"active_window_ns": 64_000_000_000, "corrected_observer_duty": 0.001}
    return rows, lifecycle, duty


def _assess(mode: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    _, lifecycle, duty = _inputs(mode)
    return assess_oars_actuation_qualification(
        mode=mode,
        oars_rows=rows,
        lifecycle_rows=lifecycle,
        observer_duty=duty,
        source_commit="a" * 40,
        run_start_ns=0,
        run_end_ns=64_000_000_000,
    )


def test_both_frozen_modes_pass() -> None:
    for mode in ("observe", "act"):
        rows, _, _ = _inputs(mode)
        result = _assess(mode, rows)
        assert result["status"] == "PASS"
        assert result["oars_actuated"] is (mode == "act")


def test_act_mode_rejects_fifo_fallback() -> None:
    rows, _, _ = _inputs("act")
    failed = copy.deepcopy(rows)
    failed[1]["proposal_matches_actual"] = False
    failed[1]["baseline_matches_actual"] = True

    result = _assess("act", failed)

    assert result["status"] == "FAIL"
    assert result["checks"]["exact_policy_identity"] is False


def test_gate_rejects_unsafe_candidate_or_budget() -> None:
    rows, _, _ = _inputs("observe")
    failed = copy.deepcopy(rows)
    failed[1]["candidate_group_count"] = 9
    failed[2]["proposed_valid_actor_tokens"] = 409

    result = _assess("observe", failed)

    assert result["status"] == "FAIL"
    assert result["checks"]["exact_candidate_set"] is False
    assert result["checks"]["service_budget"] is False


def test_gate_rejects_unbounded_replenishment() -> None:
    rows, _, _ = _inputs("act")
    failed = copy.deepcopy(rows)
    failed[1]["eligible_candidate_count"] = 12
    failed[1]["candidate_excess_count"] = 4

    result = _assess("act", failed)

    assert result["status"] == "FAIL"
    assert result["checks"]["bounded_replenishment"] is False
