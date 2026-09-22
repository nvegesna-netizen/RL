# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Frozen systems gates for the behavior-neutral OARS-v2 live shadow."""

from __future__ import annotations

import argparse
import collections
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

EXPECTED_STEPS = 64
EXPECTED_BATCH_GROUPS = 4
OARSV2_SCORERS = (
    "age",
    "reward_variance_risk",
    "token_normalized_m4_risk",
    "absolute_m4_risk",
)
MINIMUM_CONTENDED_DECISIONS = 8
MAXIMUM_FALLBACK_FRACTION = 0.05
MAXIMUM_COMBINED_OBSERVER_DUTY = 0.01
MAXIMUM_P95_LATENCY_FRACTION = 0.01


class LiveShadowQualificationError(ValueError):
    """Raised when a qualification artifact violates its frozen schema."""


def _finite_nonnegative(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _nearest_rank(values: Sequence[int], probability: float) -> int:
    if not values:
        raise LiveShadowQualificationError("percentile requires observations")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def _median(values: Sequence[int]) -> float:
    if not values:
        raise LiveShadowQualificationError("median requires observations")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def assess_live_shadow(
    *,
    oars_rows: Sequence[Mapping[str, Any]],
    lifecycle_rows: Sequence[Mapping[str, Any]],
    observer_duty: Mapping[str, Any],
    source_commit: str,
    run_start_ns: int,
    run_end_ns: int,
) -> dict[str, Any]:
    """Evaluate safety, equivalence, overhead, and natural support only."""
    if len(oars_rows) < 2 or run_start_ns < 0 or run_end_ns <= run_start_ns:
        raise LiveShadowQualificationError("invalid qualification inputs")
    header = oars_rows[0]
    decisions = tuple(oars_rows[1:])
    expected_header = {
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
    if dict(header) != expected_header:
        raise LiveShadowQualificationError("OARS-v2 header differs from protocol")
    if any(row.get("event_type") != "decision" for row in decisions):
        raise LiveShadowQualificationError("unknown OARS-v2 event type")

    advances = sorted(
        (
            row
            for row in lifecycle_rows
            if row.get("stage") == "learner_version_advanced"
        ),
        key=lambda row: int(row["learner_weight_version"]),
    )
    advance_versions = [int(row["learner_weight_version"]) for row in advances]
    advance_times = [int(row["timestamp_ns"]) for row in advances]
    step_intervals = [
        right - left for left, right in zip(advance_times, advance_times[1:])
    ]
    if any(interval <= 0 for interval in step_intervals):
        raise LiveShadowQualificationError("learner timestamps are not increasing")

    skip_counts = collections.Counter(
        str(row["skip_reason"])
        for row in decisions
        if row.get("skip_reason") is not None
    )
    fallback_counts: collections.Counter[str] = collections.Counter()
    proposed_count = 0
    proposal_count = 0
    proposal_contracts: list[bool] = []
    scorer_fifo_overlap: dict[str, list[int]] = {
        scorer: [] for scorer in OARSV2_SCORERS
    }
    scorer_disagreement: dict[str, int] = {scorer: 0 for scorer in OARSV2_SCORERS}
    complete_contended = 0
    candidate_counts: list[int] = []
    identity_checks: list[bool] = []
    decision_latencies: list[int] = []

    for row in decisions:
        candidate_count = int(row.get("candidate_group_count", -1))
        candidate_counts.append(candidate_count)
        latency = row.get("decision_latency_ns")
        if not isinstance(latency, int) or isinstance(latency, bool) or latency < 0:
            raise LiveShadowQualificationError("invalid decision latency")
        decision_latencies.append(latency)
        baseline = row.get("baseline_group_ids")
        actual = row.get("actual_selected_group_ids")
        identity_checks.append(
            row.get("baseline_matches_actual") is True
            and isinstance(baseline, list)
            and isinstance(actual, list)
            and len(baseline) == EXPECTED_BATCH_GROUPS
            and actual == baseline
            and row.get("actual_selected_group_count") == EXPECTED_BATCH_GROUPS
        )
        candidates = row.get("candidates")
        candidate_ids = (
            {
                candidate.get("group_id")
                for candidate in candidates
                if isinstance(candidate, dict)
            }
            if isinstance(candidates, list)
            else set()
        )
        proposals = row.get("proposals")
        row_complete = row.get("skip_reason") is None and isinstance(proposals, dict)
        for scorer in OARSV2_SCORERS:
            proposal_count += 1
            proposal = proposals.get(scorer) if isinstance(proposals, dict) else None
            if not isinstance(proposal, dict):
                proposal_contracts.append(False)
                row_complete = False
                continue
            status = proposal.get("status")
            ids = proposal.get("proposed_group_ids")
            valid_ids = (
                isinstance(ids, list)
                and len(ids) == EXPECTED_BATCH_GROUPS
                and len(set(ids)) == EXPECTED_BATCH_GROUPS
                and set(ids).issubset(candidate_ids)
            )
            tokens = proposal.get("proposed_valid_actor_tokens")
            baseline_tokens = row.get("baseline_valid_actor_tokens")
            token_band = (
                _finite_nonnegative(tokens)
                and _finite_nonnegative(baseline_tokens)
                and math.ceil(int(baseline_tokens) * 0.98 - 1e-12)
                <= int(tokens)
                <= math.floor(int(baseline_tokens) * 1.02 + 1e-12)
            )
            valid_status = status in {"proposed", "fallback"}
            proposal_contracts.append(valid_ids and token_band and valid_status)
            if status == "proposed":
                proposed_count += 1
            elif status == "fallback":
                fallback_counts[str(proposal.get("fallback_reason"))] += 1
                row_complete = False
            else:
                row_complete = False
            overlap = proposal.get("fifo_overlap_count")
            if isinstance(overlap, int) and not isinstance(overlap, bool):
                scorer_fifo_overlap[scorer].append(overlap)
                scorer_disagreement[scorer] += int(overlap < EXPECTED_BATCH_GROUPS)
        if candidate_count > EXPECTED_BATCH_GROUPS and row_complete:
            complete_contended += 1

    active_window_ns = observer_duty.get("active_window_ns")
    gradient_duty = observer_duty.get("corrected_observer_duty")
    if not _finite_nonnegative(active_window_ns) or int(active_window_ns) <= 0:
        raise LiveShadowQualificationError("invalid observer active window")
    if not _finite_nonnegative(gradient_duty):
        raise LiveShadowQualificationError("invalid gradient observer duty")
    oars_observer_ns = sum(decision_latencies)
    oars_duty = oars_observer_ns / int(active_window_ns)
    combined_duty = float(gradient_duty) + oars_duty
    median_step_interval_ns = _median(step_intervals)
    p95_latency_ns = _nearest_rank(decision_latencies, 0.95)
    latency_fraction = p95_latency_ns / median_step_interval_ns
    fallback_fraction = (proposal_count - proposed_count) / proposal_count

    safety_checks = {
        "complete_learner_steps": advance_versions
        == list(range(1, EXPECTED_STEPS + 1)),
        "one_shadow_decision_per_step": len(decisions) == EXPECTED_STEPS,
        "eager_fifo_candidate_bounds": all(
            EXPECTED_BATCH_GROUPS <= count <= 64 for count in candidate_counts
        ),
        "actual_selection_exactly_matches_fifo": all(identity_checks),
        "no_shadow_skip": not skip_counts,
        "proposal_contracts": all(proposal_contracts)
        and len(proposal_contracts) == EXPECTED_STEPS * len(OARSV2_SCORERS),
        "fallback_fraction": fallback_fraction <= MAXIMUM_FALLBACK_FRACTION,
        "combined_observer_duty": combined_duty <= MAXIMUM_COMBINED_OBSERVER_DUTY,
        "p95_shadow_latency": latency_fraction <= MAXIMUM_P95_LATENCY_FRACTION,
        "runtime": (run_end_ns - run_start_ns) / 1e9 <= 14_400.0,
    }
    support_checks = {
        "minimum_natural_contended_decisions": complete_contended
        >= MINIMUM_CONTENDED_DECISIONS
    }
    safety_pass = all(safety_checks.values())
    support_pass = all(support_checks.values())
    if safety_pass and support_pass:
        status = "PASS_SHADOW_SYSTEMS_READY"
    elif safety_pass:
        status = "PASS_SAFE_INSUFFICIENT_NATURAL_CONTENTION"
    else:
        status = "FAIL_SHADOW_SYSTEMS_GATE"
    return {
        "schema": "m4-oars-v2-live-shadow-qualification-result-v1",
        "status": status,
        "source_commit": source_commit,
        "checks": {**safety_checks, **support_checks},
        "safety_pass": safety_pass,
        "natural_contention_support_pass": support_pass,
        "decision_count": len(decisions),
        "candidate_group_count": {
            "minimum": min(candidate_counts),
            "maximum": max(candidate_counts),
            "contended": sum(
                count > EXPECTED_BATCH_GROUPS for count in candidate_counts
            ),
            "complete_contended": complete_contended,
        },
        "skip_reason_counts": dict(sorted(skip_counts.items())),
        "fallback_reason_counts": dict(sorted(fallback_counts.items())),
        "fallback_fraction": fallback_fraction,
        "proposal_disagreement_with_fifo": scorer_disagreement,
        "mean_fifo_overlap": {
            scorer: math.fsum(values) / len(values) if values else None
            for scorer, values in scorer_fifo_overlap.items()
        },
        "gradient_observer_duty": gradient_duty,
        "oars_observer_duty": oars_duty,
        "combined_observer_duty": combined_duty,
        "decision_latency_ns": {
            "p95_nearest_rank": p95_latency_ns,
            "maximum": max(decision_latencies),
        },
        "median_learner_step_interval_ns": median_step_interval_ns,
        "p95_latency_fraction_of_median_step_interval": latency_fraction,
        "runtime_seconds": (run_end_ns - run_start_ns) / 1e9,
        "acting_policy": "eager_weight_fifo",
        "candidate_watermark": None,
        "oars_v2_actuated": False,
        "training_quality_analyzed": False,
        "scientific_outcome_acquisition": False,
    }


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LiveShadowQualificationError(f"{path.name} must be an object")
    return value


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise LiveShadowQualificationError(f"{path.name} must be newline JSONL")
    rows = [json.loads(line) for line in raw.splitlines()]
    if any(not isinstance(row, dict) for row in rows):
        raise LiveShadowQualificationError(f"{path.name} rows must be objects")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oars", type=Path, required=True)
    parser.add_argument("--lifecycle", type=Path, required=True)
    parser.add_argument("--observer-duty", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-start-ns", type=int, required=True)
    parser.add_argument("--run-end-ns", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess_live_shadow(
        oars_rows=_read_jsonl(args.oars),
        lifecycle_rows=_read_jsonl(args.lifecycle),
        observer_duty=_read_json(args.observer_duty),
        source_commit=args.source_commit,
        run_start_ns=args.run_start_ns,
        run_end_ns=args.run_end_ns,
    )
    args.output.write_text(json.dumps(result, allow_nan=False, sort_keys=True) + "\n")
    if not result["safety_pass"]:
        raise SystemExit(f"OARS-v2 live-shadow safety gate failed: {result['checks']}")
    print(result["status"])


if __name__ == "__main__":
    main()
