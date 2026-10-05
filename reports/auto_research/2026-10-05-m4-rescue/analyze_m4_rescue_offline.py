#!/usr/bin/env python3
"""Authenticate historical OARS choices and qualify the frozen M4-Rescue policy."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
PARENT_PATH = (
    HERE.parent / "2026-09-22-m4-oars-v2-offline" / "analyze_offline_autopsy.py"
)
SPEC = importlib.util.spec_from_file_location("m4_rescue_parent", PARENT_PATH)
assert SPEC is not None and SPEC.loader is not None
PARENT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PARENT
SPEC.loader.exec_module(PARENT)

EXPECTED_ARMS = 20
EXPECTED_DECISIONS = 1280
EXPECTED_CANDIDATES = 8
SELECTED_GROUPS = 4
MAX_STALENESS_VERSIONS = 1


def _sum(candidates: Sequence[Any], field: str) -> float:
    return math.fsum(float(getattr(candidate, field)) for candidate in candidates)


def _imminent_l1(decision: Any, candidates: Sequence[Any]) -> float:
    return math.fsum(
        candidate.l1
        for candidate in candidates
        if candidate.start_weight_version + MAX_STALENESS_VERSIONS
        <= decision.current_learner_version
    )


def select_m4_rescue(decision: Any, protocol: dict[str, Any]) -> tuple[Any, ...]:
    """Return the frozen one-swap rescue selection for one reconstructed decision."""
    policy = protocol["policy"]
    tolerance = float(policy["numeric_comparison_tolerance"])
    minimum_overlap = int(policy["minimum_fifo_overlap_groups"])
    minimum_multiplier = float(policy["minimum_service_multiplier"])
    maximum_multiplier = float(policy["maximum_service_multiplier"])
    reward_floor_multiplier = float(policy["reward_variance_floor_multiplier"])
    by_id = {candidate.group_id: candidate for candidate in decision.candidates}
    PARENT.require(
        len(by_id) == len(decision.candidates),
        f"{decision.identity} d{decision.index}: duplicate candidate ID",
    )
    baseline = tuple(by_id[group_id] for group_id in decision.baseline_group_ids)
    baseline_ids = set(decision.baseline_group_ids)
    baseline_imminent_l1 = _imminent_l1(decision, baseline)
    baseline_reward_variance = _sum(baseline, "reward_variance")
    minimum_tokens = math.ceil(
        decision.baseline_tokens * minimum_multiplier - tolerance
    )
    maximum_tokens = math.floor(
        decision.baseline_tokens * maximum_multiplier + tolerance
    )
    best = baseline
    best_ids = tuple(sorted(candidate.group_id for candidate in baseline))
    best_score = (
        baseline_imminent_l1,
        _sum(baseline, "l1"),
        baseline_reward_variance,
        float(-decision.baseline_tokens),
    )
    feasible = 0
    for combination in itertools.combinations(
        sorted(decision.candidates, key=lambda candidate: candidate.group_id),
        SELECTED_GROUPS,
    ):
        ids = tuple(candidate.group_id for candidate in combination)
        if len(set(ids) & baseline_ids) < minimum_overlap:
            continue
        tokens = sum(candidate.valid_actor_tokens for candidate in combination)
        if tokens < minimum_tokens or tokens > maximum_tokens:
            continue
        reward_variance = _sum(combination, "reward_variance")
        if (
            reward_variance + tolerance
            < baseline_reward_variance * reward_floor_multiplier
        ):
            continue
        feasible += 1
        score = (
            _imminent_l1(decision, combination),
            _sum(combination, "l1"),
            reward_variance,
            float(-tokens),
        )
        if score > best_score or (score == best_score and ids < best_ids):
            best = combination
            best_ids = ids
            best_score = score
    PARENT.require(
        feasible >= 1,
        f"{decision.identity} d{decision.index}: FIFO unexpectedly infeasible",
    )
    if _imminent_l1(decision, best) <= baseline_imminent_l1 + tolerance:
        return baseline
    return tuple(best)


def _selection_record(
    decision: Any, selection: Sequence[Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    tolerance = float(protocol["policy"]["numeric_comparison_tolerance"])
    baseline_ids = set(decision.baseline_group_ids)
    selected_ids = tuple(candidate.group_id for candidate in selection)
    by_id = {candidate.group_id: candidate for candidate in decision.candidates}
    baseline = tuple(by_id[group_id] for group_id in decision.baseline_group_ids)
    tokens = sum(candidate.valid_actor_tokens for candidate in selection)
    baseline_reward_variance = _sum(baseline, "reward_variance")
    selected_reward_variance = _sum(selection, "reward_variance")
    overlap = len(set(selected_ids) & baseline_ids)
    minimum_tokens = math.ceil(
        decision.baseline_tokens
        * float(protocol["policy"]["minimum_service_multiplier"])
        - tolerance
    )
    maximum_tokens = math.floor(
        decision.baseline_tokens
        * float(protocol["policy"]["maximum_service_multiplier"])
        + tolerance
    )
    return {
        "identity": decision.identity,
        "decision_index": decision.index,
        "baseline_group_ids": list(decision.baseline_group_ids),
        "selected_group_ids": list(selected_ids),
        "intervened": set(selected_ids) != baseline_ids,
        "fifo_overlap_count": overlap,
        "baseline_l1": _sum(baseline, "l1"),
        "selected_l1": _sum(selection, "l1"),
        "baseline_imminent_l1": _imminent_l1(decision, baseline),
        "selected_imminent_l1": _imminent_l1(decision, selection),
        "baseline_valid_actor_tokens": decision.baseline_tokens,
        "selected_valid_actor_tokens": tokens,
        "minimum_tokens": minimum_tokens,
        "maximum_tokens": maximum_tokens,
        "baseline_reward_variance_sum": baseline_reward_variance,
        "selected_reward_variance_sum": selected_reward_variance,
        "constraint_checks": {
            "cardinality": len(selection) == SELECTED_GROUPS
            and len(set(selected_ids)) == SELECTED_GROUPS,
            "fifo_overlap": overlap
            >= int(protocol["policy"]["minimum_fifo_overlap_groups"]),
            "token_band": minimum_tokens <= tokens <= maximum_tokens,
            "reward_variance_floor": selected_reward_variance + tolerance
            >= baseline_reward_variance
            * float(protocol["policy"]["reward_variance_floor_multiplier"]),
            "strict_imminent_gain_or_fifo": (
                set(selected_ids) == baseline_ids
                or _imminent_l1(decision, selection)
                > _imminent_l1(decision, baseline) + tolerance
            ),
        },
    }


def _canonical(records: Sequence[dict[str, Any]]) -> bytes:
    return (
        json.dumps(
            records,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _percent_gain(numerator: float, denominator: float) -> float:
    PARENT.require(denominator > 0.0, "percent-gain denominator must be positive")
    return 100.0 * (numerator / denominator - 1.0)


def analyze(
    repository: Path,
    archive_root: Path,
    authentication_report_dir: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    """Run the provenance-bound canonical development qualification."""
    protocol = PARENT.load_object(protocol_path)
    PARENT.require(
        protocol.get("schema") == "m4-rescue-protocol-v1",
        "M4-Rescue protocol schema differs",
    )
    PARENT.require(
        protocol.get("status") == "FROZEN_BEFORE_CANONICAL_OFFLINE_REPLAY",
        "M4-Rescue protocol was not frozen before replay",
    )
    all_decisions: list[Any] = []
    provenance: dict[str, Any] = {}
    for pair in range(1, 11):
        for mode in ("fifo", "oars"):
            identity = f"p{pair:02d}-{mode}"
            arm_provenance, ledgers = PARENT.archive_ledgers(
                repository, archive_root, authentication_report_dir, identity
            )
            opportunities = PARENT.opportunity_map(
                ledgers["opportunity.jsonl"], identity=identity
            )
            decisions, _ = PARENT.reconstruct_decisions(
                identity,
                ledgers["lifecycle.jsonl"],
                ledgers["oars.jsonl"],
                opportunities,
            )
            all_decisions.extend(decisions)
            provenance[identity] = arm_provenance

    PARENT.require(len(provenance) == EXPECTED_ARMS, "authenticated arm count differs")
    PARENT.require(
        len(all_decisions) == EXPECTED_DECISIONS,
        "reconstructed decision count differs",
    )
    PARENT.require(
        all(
            len(decision.candidates) == EXPECTED_CANDIDATES
            for decision in all_decisions
        ),
        "candidate count differs",
    )

    first_records = [
        _selection_record(decision, select_m4_rescue(decision, protocol), protocol)
        for decision in all_decisions
    ]
    second_records = [
        _selection_record(decision, select_m4_rescue(decision, protocol), protocol)
        for decision in all_decisions
    ]
    deterministic = _canonical(first_records) == _canonical(second_records)
    intervention_count = sum(bool(row["intervened"]) for row in first_records)
    selected_count = len(first_records) * SELECTED_GROUPS
    baseline_l1 = math.fsum(float(row["baseline_l1"]) for row in first_records)
    selected_l1 = math.fsum(float(row["selected_l1"]) for row in first_records)
    baseline_imminent_l1 = math.fsum(
        float(row["baseline_imminent_l1"]) for row in first_records
    )
    selected_imminent_l1 = math.fsum(
        float(row["selected_imminent_l1"]) for row in first_records
    )
    baseline_tokens = sum(
        int(row["baseline_valid_actor_tokens"]) for row in first_records
    )
    selected_tokens = sum(
        int(row["selected_valid_actor_tokens"]) for row in first_records
    )
    baseline_reward_variance = math.fsum(
        float(row["baseline_reward_variance_sum"]) for row in first_records
    )
    selected_reward_variance = math.fsum(
        float(row["selected_reward_variance_sum"]) for row in first_records
    )
    overlap_groups = sum(int(row["fifo_overlap_count"]) for row in first_records)
    all_constraints = all(
        all(bool(value) for value in row["constraint_checks"].values())
        for row in first_records
    )
    intervention_rate = intervention_count / len(first_records)
    overlap_fraction = overlap_groups / selected_count
    token_ratio = selected_tokens / baseline_tokens
    imminent_gain = _percent_gain(selected_imminent_l1, baseline_imminent_l1)
    requirements = protocol["offline_qualification"]["requirements"]
    checks = {
        "provenance": len(provenance) == EXPECTED_ARMS
        and len(first_records) == EXPECTED_DECISIONS,
        "determinism": deterministic,
        "intervention_rate": float(requirements["intervention_rate"]["minimum"])
        <= intervention_rate
        <= float(requirements["intervention_rate"]["maximum"]),
        "fifo_overlap": overlap_fraction
        >= float(requirements["mean_fifo_overlap_fraction_minimum"]),
        "imminent_l1_gain": imminent_gain
        >= float(requirements["aggregate_imminent_l1_gain_over_fifo_percent_minimum"]),
        "aggregate_token_ratio": float(
            requirements["aggregate_token_ratio_to_fifo"]["minimum"]
        )
        <= token_ratio
        <= float(requirements["aggregate_token_ratio_to_fifo"]["maximum"]),
        "aggregate_reward_variance": selected_reward_variance
        >= baseline_reward_variance - 1e-12,
        "every_decision_constraints": all_constraints,
    }
    status = (
        protocol["offline_qualification"]["pass_status"]
        if all(checks.values())
        else protocol["offline_qualification"]["stop_status"]
    )
    return {
        "schema": "m4-rescue-offline-qualification-result-v1",
        "status": status,
        "protocol": {
            "path": str(protocol_path.relative_to(repository)),
            "sha256": PARENT.sha256_file(protocol_path),
        },
        "scope": {
            "authenticated_arms": len(provenance),
            "reconstructed_decisions": len(first_records),
            "candidate_rows": len(first_records) * EXPECTED_CANDIDATES,
        },
        "checks": checks,
        "metrics": {
            "intervention_count": intervention_count,
            "intervention_rate": intervention_rate,
            "mean_fifo_overlap_fraction": overlap_fraction,
            "selected_l1_gain_over_fifo_percent": _percent_gain(
                selected_l1, baseline_l1
            ),
            "selected_imminent_l1_gain_over_fifo_percent": imminent_gain,
            "aggregate_token_ratio_to_fifo": token_ratio,
            "selected_reward_variance_gain_over_fifo_percent": _percent_gain(
                selected_reward_variance, baseline_reward_variance
            ),
            "minimum_decision_token_ratio_to_fifo": min(
                int(row["selected_valid_actor_tokens"])
                / int(row["baseline_valid_actor_tokens"])
                for row in first_records
            ),
            "maximum_decision_token_ratio_to_fifo": max(
                int(row["selected_valid_actor_tokens"])
                / int(row["baseline_valid_actor_tokens"])
                for row in first_records
            ),
        },
        "selection_records_sha256": PARENT.sha256_bytes(_canonical(first_records)),
        "provenance": provenance,
        "interpretation_boundary": protocol["interpretation_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--authentication-report-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.repository.resolve(),
        args.archive_root.resolve(),
        args.authentication_report_dir.resolve(),
        args.protocol.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: result[key] for key in ("status", "scope", "checks", "metrics")},
            indent=2,
            sort_keys=True,
        )
    )
    if result["status"] != "PASS_M4_RESCUE_OFFLINE_QUALIFICATION":
        raise SystemExit("M4-Rescue offline qualification stopped")


if __name__ == "__main__":
    main()
