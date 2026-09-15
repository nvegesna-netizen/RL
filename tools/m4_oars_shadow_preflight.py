# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Outcome-neutral gates for the FIFO-controlled OARS shadow preflight."""

from __future__ import annotations

import argparse
import collections
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeGuard


class OARSShadowPreflightError(ValueError):
    """Raised when a preflight artifact violates its frozen schema."""


def _nearest_rank(values: Sequence[int], probability: float) -> int:
    if not values or not 0.0 < probability <= 1.0:
        raise OARSShadowPreflightError("invalid percentile inputs")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def _median(values: Sequence[int]) -> float:
    if not values:
        raise OARSShadowPreflightError("median requires observations")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _finite_nonnegative(value: object) -> TypeGuard[int | float]:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and value >= 0
    )


def assess_oars_shadow_preflight(
    *,
    oars_rows: Sequence[Mapping[str, Any]],
    lifecycle_rows: Sequence[Mapping[str, Any]],
    observer_duty: Mapping[str, Any],
    source_commit: str,
    run_start_ns: int,
    run_end_ns: int,
    expected_steps: int = 64,
    expected_batch_groups: int = 4,
    minimum_contended_decisions: int = 8,
    minimum_metadata_coverage: float = 0.99,
    maximum_observer_duty: float = 0.01,
    maximum_latency_fraction_of_step_interval: float = 0.10,
    maximum_runtime_seconds: float = 14_400.0,
) -> dict[str, Any]:
    """Evaluate the frozen systems gates without inspecting quality outcomes."""
    if (
        len(oars_rows) < 2
        or expected_steps < 1
        or expected_batch_groups < 1
        or minimum_contended_decisions < 1
        or not 0.0 < minimum_metadata_coverage <= 1.0
        or not 0.0 < maximum_observer_duty <= 1.0
        or not 0.0 < maximum_latency_fraction_of_step_interval <= 1.0
        or maximum_runtime_seconds <= 0.0
        or run_start_ns < 0
        or run_end_ns <= run_start_ns
    ):
        raise OARSShadowPreflightError("invalid frozen preflight controls")

    header = oars_rows[0]
    decisions = tuple(
        row for row in oars_rows[1:] if row.get("event_type") == "decision"
    )
    if len(decisions) != len(oars_rows) - 1:
        raise OARSShadowPreflightError("OARS ledger contains unknown event types")
    if (
        header.get("event_type") != "header"
        or header.get("schema_version") != 1
        or header.get("policy") != "baseline_budgeted_oars_v1"
        or header.get("service_budget_multiplier") != 1.02
        or header.get("max_candidate_groups") != 25
    ):
        raise OARSShadowPreflightError("OARS header differs from the frozen policy")

    advances = sorted(
        (
            row
            for row in lifecycle_rows
            if row.get("stage") == "learner_version_advanced"
        ),
        key=lambda row: int(row["learner_weight_version"]),
    )
    advance_versions = [int(row["learner_weight_version"]) for row in advances]
    advance_timestamps = [int(row["timestamp_ns"]) for row in advances]
    intervals_ns = [
        right - left for left, right in zip(advance_timestamps, advance_timestamps[1:])
    ]
    if any(value <= 0 for value in intervals_ns):
        raise OARSShadowPreflightError("learner timestamps are not strictly increasing")

    contended = tuple(
        row
        for row in decisions
        if int(row.get("candidate_group_count", 0)) > expected_batch_groups
    )
    complete_contended = tuple(
        row
        for row in contended
        if row.get("skip_reason") is None
        and row.get("baseline_matches_actual") is True
        and len(row.get("baseline_group_ids", ())) == expected_batch_groups
        and len(row.get("proposed_group_ids", ())) == expected_batch_groups
    )
    metadata_coverage = len(complete_contended) / len(contended) if contended else 0.0
    skip_reason_counts = collections.Counter(
        str(row["skip_reason"])
        for row in decisions
        if row.get("skip_reason") is not None
    )

    decision_latencies = [int(row["decision_latency_ns"]) for row in decisions]
    if any(value < 0 for value in decision_latencies):
        raise OARSShadowPreflightError("negative OARS decision latency")
    latency_p95_ns = _nearest_rank(decision_latencies, 0.95)
    median_step_interval_ns = _median(intervals_ns)
    latency_fraction = latency_p95_ns / median_step_interval_ns

    active_window_ns = int(observer_duty.get("active_window_ns", 0))
    oars_observer_ns = sum(decision_latencies)
    oars_observer_duty = (
        oars_observer_ns / active_window_ns if active_window_ns > 0 else math.inf
    )
    gradient_observer_duty = observer_duty.get("corrected_observer_duty")
    runtime_seconds = (run_end_ns - run_start_ns) / 1e9

    identity_complete = all(
        row.get("baseline_matches_actual") is True
        and row.get("actual_selected_group_count") == expected_batch_groups
        and len(row.get("actual_selected_group_ids") or ()) == expected_batch_groups
        for row in decisions
    )
    budget_complete = all(
        row.get("skip_reason") is None
        and _finite_nonnegative(row.get("baseline_valid_actor_tokens"))
        and _finite_nonnegative(row.get("proposed_valid_actor_tokens"))
        and _finite_nonnegative(row.get("token_budget"))
        and int(row["proposed_valid_actor_tokens"]) <= int(row["token_budget"])
        and int(row["token_budget"])
        == math.floor(int(row["baseline_valid_actor_tokens"]) * 1.02 + 1e-12)
        for row in decisions
    )
    proposal_complete = all(
        len(row.get("proposed_group_ids", ())) == expected_batch_groups
        and len(set(row["proposed_group_ids"])) == expected_batch_groups
        for row in decisions
    )
    checks = {
        "complete_learner_steps": advance_versions
        == list(range(1, expected_steps + 1)),
        "one_shadow_decision_per_step": len(decisions) == expected_steps,
        "minimum_contended_support": len(contended) >= minimum_contended_decisions,
        "metadata_coverage": metadata_coverage >= minimum_metadata_coverage,
        "actual_selection_exactly_matches_fifo": identity_complete,
        "proposal_cardinality": proposal_complete,
        "service_budget": budget_complete,
        "no_shadow_skip": not skip_reason_counts,
        "gradient_observer_duty": (
            _finite_nonnegative(gradient_observer_duty)
            and float(gradient_observer_duty) <= maximum_observer_duty
        ),
        "oars_observer_duty": (
            math.isfinite(oars_observer_duty)
            and oars_observer_duty <= maximum_observer_duty
        ),
        "decision_latency": latency_fraction
        <= maximum_latency_fraction_of_step_interval,
        "runtime": runtime_seconds <= maximum_runtime_seconds,
    }
    return {
        "schema": "m4-oars-shadow-systems-preflight-result-v1",
        "source_commit": source_commit,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "expected_steps": expected_steps,
        "decision_count": len(decisions),
        "contended_decision_count": len(contended),
        "complete_contended_decision_count": len(complete_contended),
        "metadata_coverage": metadata_coverage,
        "skip_reason_counts": dict(sorted(skip_reason_counts.items())),
        "decision_latency_ns": {
            "p95_nearest_rank": latency_p95_ns,
            "maximum": max(decision_latencies),
        },
        "median_learner_step_interval_ns": median_step_interval_ns,
        "p95_latency_fraction_of_median_step_interval": latency_fraction,
        "gradient_observer_duty": gradient_observer_duty,
        "oars_observer_ns": oars_observer_ns,
        "oars_observer_duty": oars_observer_duty,
        "runtime_seconds": runtime_seconds,
        "oars_actuated": False,
        "training_quality_analyzed": False,
        "scientific_outcome_acquisition": False,
    }


def _reject_constant(value: str) -> None:
    raise OARSShadowPreflightError(f"non-finite JSON constant {value}")


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=_reject_constant
    )
    if not isinstance(value, dict):
        raise OARSShadowPreflightError(f"{path.name} must contain one object")
    return value


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise OARSShadowPreflightError(f"{path.name} must be nonempty newline JSONL")
    rows = [
        json.loads(line, parse_constant=_reject_constant) for line in raw.splitlines()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise OARSShadowPreflightError(f"{path.name} rows must be objects")
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
    result = assess_oars_shadow_preflight(
        oars_rows=_read_jsonl(args.oars),
        lifecycle_rows=_read_jsonl(args.lifecycle),
        observer_duty=_read_json(args.observer_duty),
        source_commit=args.source_commit,
        run_start_ns=args.run_start_ns,
        run_end_ns=args.run_end_ns,
    )
    args.output.write_text(
        json.dumps(result, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    if result["status"] != "PASS":
        raise SystemExit(f"OARS shadow preflight failed: {result['checks']}")
    print("M4_OARS_FIFO_CONTROLLED_SHADOW_PREFLIGHT_PASS")


if __name__ == "__main__":
    main()
