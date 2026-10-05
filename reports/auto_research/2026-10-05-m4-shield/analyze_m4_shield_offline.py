#!/usr/bin/env python3
"""Authenticate and qualify the frozen M4-Shield development policy."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import sys
import zipfile
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
PARENT_PATH = (
    HERE.parent / "2026-09-22-m4-oars-v2-offline" / "analyze_offline_autopsy.py"
)
SPEC = importlib.util.spec_from_file_location("m4_shield_parent", PARENT_PATH)
assert SPEC is not None and SPEC.loader is not None
PARENT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PARENT
SPEC.loader.exec_module(PARENT)

EXPECTED_HISTORICAL_ARMS = 20
EXPECTED_HISTORICAL_DECISIONS = 1280
EXPECTED_LIVE_DECISIONS = 64
EXPECTED_CANDIDATES = 8
SELECTED_GROUPS = 4
MAX_STALENESS_VERSIONS = 1


def _sum(candidates: Sequence[Any], field: str) -> float:
    return math.fsum(float(getattr(candidate, field)) for candidate in candidates)


def _imminent_sum(decision: Any, candidates: Sequence[Any], field: str) -> float:
    return math.fsum(
        float(getattr(candidate, field))
        for candidate in candidates
        if candidate.start_weight_version + MAX_STALENESS_VERSIONS
        <= decision.current_learner_version
    )


def _token_bounds(decision: Any, protocol: dict[str, Any]) -> tuple[int, int]:
    policy = protocol["policy"]
    tolerance = float(policy["numeric_comparison_tolerance"])
    minimum = math.ceil(
        decision.baseline_tokens * float(policy["minimum_service_multiplier"])
        - tolerance
    )
    maximum = math.floor(
        decision.baseline_tokens * float(policy["maximum_service_multiplier"])
        + tolerance
    )
    return minimum, maximum


def select_reward_variance_base(
    decision: Any, protocol: dict[str, Any]
) -> tuple[Any, ...]:
    """Reproduce the exact OARS-v2 reward-variance proposal."""
    minimum_tokens, maximum_tokens = _token_bounds(decision, protocol)
    best: tuple[Any, ...] | None = None
    best_score: tuple[float, ...] | None = None
    for combination in itertools.combinations(
        sorted(decision.candidates, key=lambda candidate: candidate.group_id),
        SELECTED_GROUPS,
    ):
        tokens = sum(candidate.valid_actor_tokens for candidate in combination)
        if tokens < minimum_tokens or tokens > maximum_tokens:
            continue
        score = (
            _imminent_sum(decision, combination, "reward_variance"),
            _sum(combination, "reward_variance"),
            float(-tokens),
        )
        # Combinations are lexicographically ordered, so retaining the first exact
        # score tie reproduces the runtime's deterministic group-ID tie break.
        if best_score is None or score > best_score:
            best = combination
            best_score = score
    PARENT.require(
        best is not None,
        f"{decision.identity} d{decision.index}: no feasible base proposal",
    )
    return best


def select_m4_shield(
    decision: Any, base: Sequence[Any], protocol: dict[str, Any]
) -> tuple[Any, ...]:
    """Refine a base proposal only within its exact reward-utility level set."""
    policy = protocol["policy"]
    tolerance = float(policy["numeric_comparison_tolerance"])
    minimum_overlap = int(policy["minimum_base_proposal_overlap_groups"])
    minimum_tokens, maximum_tokens = _token_bounds(decision, protocol)
    base_ids = {candidate.group_id for candidate in base}
    base_imminent_variance = _imminent_sum(decision, base, "reward_variance")
    base_total_variance = _sum(base, "reward_variance")
    base_imminent_l1 = _imminent_sum(decision, base, "l1")
    best = tuple(base)
    best_ids = tuple(sorted(base_ids))
    best_score = (
        base_imminent_l1,
        _sum(base, "l1"),
        float(-sum(candidate.valid_actor_tokens for candidate in base)),
    )
    feasible = 0
    for combination in itertools.combinations(
        sorted(decision.candidates, key=lambda candidate: candidate.group_id),
        SELECTED_GROUPS,
    ):
        ids = tuple(candidate.group_id for candidate in combination)
        if len(set(ids) & base_ids) < minimum_overlap:
            continue
        tokens = sum(candidate.valid_actor_tokens for candidate in combination)
        if tokens < minimum_tokens or tokens > maximum_tokens:
            continue
        if not math.isclose(
            _imminent_sum(decision, combination, "reward_variance"),
            base_imminent_variance,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            continue
        if not math.isclose(
            _sum(combination, "reward_variance"),
            base_total_variance,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            continue
        feasible += 1
        score = (
            _imminent_sum(decision, combination, "l1"),
            _sum(combination, "l1"),
            float(-tokens),
        )
        if score > best_score or (score == best_score and ids < best_ids):
            best = combination
            best_ids = ids
            best_score = score
    PARENT.require(
        feasible >= 1,
        f"{decision.identity} d{decision.index}: base proposal is infeasible",
    )
    if _imminent_sum(decision, best, "l1") <= base_imminent_l1 + tolerance:
        return tuple(base)
    return tuple(best)


def _selection_record(decision: Any, protocol: dict[str, Any]) -> dict[str, Any]:
    base = select_reward_variance_base(decision, protocol)
    shield = select_m4_shield(decision, base, protocol)
    by_id = {candidate.group_id: candidate for candidate in decision.candidates}
    fifo = tuple(by_id[group_id] for group_id in decision.baseline_group_ids)
    base_ids = tuple(candidate.group_id for candidate in base)
    shield_ids = tuple(candidate.group_id for candidate in shield)
    minimum_tokens, maximum_tokens = _token_bounds(decision, protocol)
    tolerance = float(protocol["policy"]["numeric_comparison_tolerance"])
    base_imminent_l1 = _imminent_sum(decision, base, "l1")
    shield_imminent_l1 = _imminent_sum(decision, shield, "l1")
    fifo_imminent_l1 = _imminent_sum(decision, fifo, "l1")
    base_imminent_variance = _imminent_sum(decision, base, "reward_variance")
    shield_imminent_variance = _imminent_sum(decision, shield, "reward_variance")
    base_total_variance = _sum(base, "reward_variance")
    shield_total_variance = _sum(shield, "reward_variance")
    shield_tokens = sum(candidate.valid_actor_tokens for candidate in shield)
    overlap = len(set(base_ids) & set(shield_ids))
    intervened = set(base_ids) != set(shield_ids)
    return {
        "identity": decision.identity,
        "decision_index": decision.index,
        "fifo_group_ids": list(decision.baseline_group_ids),
        "base_group_ids": list(base_ids),
        "shield_group_ids": list(shield_ids),
        "base_differs_from_fifo": set(base_ids) != set(decision.baseline_group_ids),
        "shield_intervened": intervened,
        "base_proposal_overlap_count": overlap,
        "fifo_imminent_l1": fifo_imminent_l1,
        "base_imminent_l1": base_imminent_l1,
        "shield_imminent_l1": shield_imminent_l1,
        "base_total_l1": _sum(base, "l1"),
        "shield_total_l1": _sum(shield, "l1"),
        "base_imminent_reward_variance": base_imminent_variance,
        "shield_imminent_reward_variance": shield_imminent_variance,
        "base_total_reward_variance": base_total_variance,
        "shield_total_reward_variance": shield_total_variance,
        "base_valid_actor_tokens": sum(
            candidate.valid_actor_tokens for candidate in base
        ),
        "shield_valid_actor_tokens": shield_tokens,
        "minimum_tokens": minimum_tokens,
        "maximum_tokens": maximum_tokens,
        "base_harm_vs_fifo": max(0.0, fifo_imminent_l1 - base_imminent_l1),
        "shield_harm_vs_fifo": max(0.0, fifo_imminent_l1 - shield_imminent_l1),
        "constraint_checks": {
            "cardinality": len(shield) == SELECTED_GROUPS
            and len(set(shield_ids)) == SELECTED_GROUPS,
            "base_overlap": overlap
            >= int(protocol["policy"]["minimum_base_proposal_overlap_groups"]),
            "token_band": minimum_tokens <= shield_tokens <= maximum_tokens,
            "imminent_reward_variance_identity": math.isclose(
                shield_imminent_variance,
                base_imminent_variance,
                rel_tol=0.0,
                abs_tol=tolerance,
            ),
            "total_reward_variance_identity": math.isclose(
                shield_total_variance,
                base_total_variance,
                rel_tol=0.0,
                abs_tol=tolerance,
            ),
            "strict_imminent_gain_or_base": (
                not intervened or shield_imminent_l1 > base_imminent_l1 + tolerance
            ),
        },
    }


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _load_historical(
    repository: Path, archive_root: Path, report_dir: Path
) -> tuple[list[Any], dict[str, Any]]:
    decisions: list[Any] = []
    provenance: dict[str, Any] = {}
    for pair in range(1, 11):
        for mode in ("fifo", "oars"):
            identity = f"p{pair:02d}-{mode}"
            arm_provenance, ledgers = PARENT.archive_ledgers(
                repository, archive_root, report_dir, identity
            )
            opportunities = PARENT.opportunity_map(
                ledgers["opportunity.jsonl"], identity=identity
            )
            arm_decisions, _ = PARENT.reconstruct_decisions(
                identity,
                ledgers["lifecycle.jsonl"],
                ledgers["oars.jsonl"],
                opportunities,
            )
            decisions.extend(arm_decisions)
            provenance[identity] = arm_provenance
    PARENT.require(
        len(provenance) == EXPECTED_HISTORICAL_ARMS,
        "historical authenticated arm count differs",
    )
    PARENT.require(
        len(decisions) == EXPECTED_HISTORICAL_DECISIONS,
        "historical decision count differs",
    )
    return decisions, provenance


def _live_member(archive: zipfile.ZipFile, suffix: str) -> tuple[str, bytes]:
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    PARENT.require(
        len(matches) == 1,
        f"expected one live member ending in {suffix}, found {len(matches)}",
    )
    member = matches[0]
    return member, archive.read(member)


def _load_live(
    repository: Path, archive_path: Path, receipt_path: Path
) -> tuple[list[Any], list[tuple[str, ...]], dict[str, Any]]:
    receipt = PARENT.load_object(receipt_path)
    PARENT.require(
        receipt.get("status") == "AUTHENTICATED_QUALIFICATION_FAIL",
        "live authentication status differs",
    )
    PARENT.require(archive_path.is_file(), "live archive missing")
    archive_sha = PARENT.sha256_file(archive_path)
    PARENT.require(
        archive_sha == receipt["workload_artifact_archive"]["sha256"],
        "live archive SHA-256 differs",
    )
    declared = receipt["declared_artifacts"]
    verified: dict[str, str] = {}
    with zipfile.ZipFile(archive_path) as archive:
        PARENT.validate_zip_names(archive)
        suffixes = {
            "manifest_sha256": "artifacts.sha256",
            "lifecycle.jsonl": "lifecycle.jsonl",
            "oars-v2.jsonl": "oars-v2.jsonl",
            "observer-duty.json": "observer-duty.json",
            "opportunity.jsonl": "opportunity.jsonl",
            "result.json": "result.json",
            "run.log": "run.log",
        }
        payloads: dict[str, bytes] = {}
        for label, suffix in suffixes.items():
            _, payload = _live_member(archive, suffix)
            digest = PARENT.sha256_bytes(payload)
            PARENT.require(digest == declared[label], f"live {label} SHA-256 differs")
            verified[label] = digest
            payloads[label] = payload
    rows = PARENT.parse_jsonl(payloads["oars-v2.jsonl"], label="live:oars-v2")
    headers = [row for row in rows if row.get("event_type") == "header"]
    events = [row for row in rows if row.get("event_type") == "decision"]
    PARENT.require(len(headers) == 1, "live OARS header count differs")
    PARENT.require(headers[0].get("schema_version") == 3, "live schema differs")
    PARENT.require(
        headers[0].get("selection_candidate_watermark") == EXPECTED_CANDIDATES,
        "live candidate watermark differs",
    )
    PARENT.require(
        len(events) == EXPECTED_LIVE_DECISIONS,
        "live decision count differs",
    )
    decisions: list[Any] = []
    recorded_reward_variance: list[tuple[str, ...]] = []
    for index, event in enumerate(events):
        candidates = tuple(
            PARENT.Candidate(
                group_id=str(row["group_id"]),
                start_weight_version=int(row["start_weight_version"]),
                ready_timestamp_ns=int(row["ready_timestamp_ns"]),
                l1=float(row["l1"]),
                l2=float(row["l2"]),
                valid_actor_tokens=int(row["valid_actor_tokens"]),
                mean_reward=float(row["reward_mean"]),
                reward_variance=float(row["reward_variance"]),
            )
            for row in event["candidates"]
        )
        PARENT.require(
            len(candidates) == EXPECTED_CANDIDATES,
            f"live d{index}: candidate count differs",
        )
        baseline_ids = tuple(event["baseline_group_ids"])
        decisions.append(
            PARENT.Decision(
                identity="m4-rescue-live-negative-control",
                index=index,
                current_learner_version=int(event["current_learner_version"]),
                timestamp_ns=0,
                candidates=candidates,
                baseline_group_ids=baseline_ids,
                proposal_group_ids=tuple(
                    event["proposals"]["reward_variance_risk"]["proposed_group_ids"]
                ),
                actual_group_ids=tuple(event["actual_selected_group_ids"]),
                baseline_tokens=int(event["baseline_valid_actor_tokens"]),
            )
        )
        recorded_reward_variance.append(
            tuple(event["proposals"]["reward_variance_risk"]["proposed_group_ids"])
        )
    try:
        archive_label = str(archive_path.relative_to(repository))
    except ValueError:
        archive_label = str(archive_path)
    return (
        decisions,
        recorded_reward_variance,
        {
            "authentication_receipt": str(receipt_path.relative_to(repository)),
            "authentication_receipt_sha256": PARENT.sha256_file(receipt_path),
            "archive": archive_label,
            "archive_sha256": archive_sha,
            "verified_artifact_sha256": verified,
        },
    )


def _summarize_arms(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    identities = sorted({str(row["identity"]) for row in records})
    result: list[dict[str, Any]] = []
    for identity in identities:
        rows = [row for row in records if row["identity"] == identity]
        result.append(
            {
                "identity": identity,
                "decision_count": len(rows),
                "base_differs_from_fifo_count": sum(
                    bool(row["base_differs_from_fifo"]) for row in rows
                ),
                "shield_intervention_count": sum(
                    bool(row["shield_intervened"]) for row in rows
                ),
                "imminent_l1_gain_over_base": math.fsum(
                    float(row["shield_imminent_l1"]) - float(row["base_imminent_l1"])
                    for row in rows
                ),
                "base_harm_vs_fifo": math.fsum(
                    float(row["base_harm_vs_fifo"]) for row in rows
                ),
                "shield_harm_vs_fifo": math.fsum(
                    float(row["shield_harm_vs_fifo"]) for row in rows
                ),
            }
        )
    return result


def analyze(
    repository: Path,
    archive_root: Path,
    authentication_report_dir: Path,
    live_archive: Path,
    live_receipt: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    """Run the provenance-bound canonical development qualification."""
    protocol = PARENT.load_object(protocol_path)
    PARENT.require(
        protocol.get("schema") == "m4-shield-development-protocol-v1",
        "M4-Shield protocol schema differs",
    )
    PARENT.require(
        protocol.get("status") == "FROZEN_BEFORE_CANONICAL_OFFLINE_REPLAY",
        "M4-Shield protocol was not frozen before replay",
    )
    historical, historical_provenance = _load_historical(
        repository, archive_root, authentication_report_dir
    )
    live, recorded_live_base, live_provenance = _load_live(
        repository, live_archive, live_receipt
    )
    first_historical = [
        _selection_record(decision, protocol) for decision in historical
    ]
    second_historical = [
        _selection_record(decision, protocol) for decision in historical
    ]
    first_live = [_selection_record(decision, protocol) for decision in live]
    second_live = [_selection_record(decision, protocol) for decision in live]
    deterministic = _canonical(
        {"historical": first_historical, "live": first_live}
    ) == _canonical({"historical": second_historical, "live": second_live})
    live_base_identity_count = sum(
        tuple(row["base_group_ids"]) == recorded
        for row, recorded in zip(first_live, recorded_live_base)
    )
    historical_interventions = sum(
        bool(row["shield_intervened"]) for row in first_historical
    )
    fifo_interventions = sum(
        bool(row["shield_intervened"])
        for row in first_historical
        if str(row["identity"]).endswith("-fifo")
    )
    live_interventions = sum(bool(row["shield_intervened"]) for row in first_live)
    oars_active_arms = len(
        {
            str(row["identity"])
            for row in first_historical
            if str(row["identity"]).endswith("-oars") and bool(row["shield_intervened"])
        }
    )
    gain = math.fsum(
        float(row["shield_imminent_l1"]) - float(row["base_imminent_l1"])
        for row in first_historical
    )
    base_harm = math.fsum(float(row["base_harm_vs_fifo"]) for row in first_historical)
    shield_harm = math.fsum(
        float(row["shield_harm_vs_fifo"]) for row in first_historical
    )
    PARENT.require(base_harm > 0.0, "historical base harm is unexpectedly zero")
    harm_reduction = (base_harm - shield_harm) / base_harm
    harmful_improved = sum(
        float(row["base_harm_vs_fifo"]) > 0.0
        and float(row["shield_harm_vs_fifo"]) < float(row["base_harm_vs_fifo"])
        for row in first_historical
    )
    all_records = first_historical + first_live
    constraints = all(
        all(bool(value) for value in row["constraint_checks"].values())
        for row in all_records
    )
    utility_identity = all(
        bool(row["constraint_checks"]["imminent_reward_variance_identity"])
        and bool(row["constraint_checks"]["total_reward_variance_identity"])
        for row in all_records
    )
    requirements = protocol["offline_qualification"]["requirements"]
    checks = {
        "all_sources_authenticate": len(historical_provenance)
        == EXPECTED_HISTORICAL_ARMS
        and bool(live_provenance),
        "deterministic": deterministic,
        "live_base_proposal_identity": live_base_identity_count
        == int(requirements["live_reward_variance_base_identity_count"]),
        "every_decision_constraints": constraints,
        "reward_variance_utility_identity": utility_identity,
        "historical_intervention_count": historical_interventions
        >= int(requirements["historical_intervention_count_minimum"]),
        "oars_trajectory_arm_activity": oars_active_arms
        >= int(requirements["oars_trajectory_arms_with_intervention_minimum"]),
        "fifo_negative_control": fifo_interventions
        <= int(requirements["fifo_trajectory_intervention_count_maximum"]),
        "live_negative_control": live_interventions
        <= int(requirements["live_identity_intervention_count_maximum"]),
        "aggregate_imminent_l1_gain": gain
        >= float(requirements["aggregate_historical_imminent_l1_gain_minimum"]),
        "base_harm_reduction": harm_reduction
        >= float(requirements["base_harm_reduction_fraction_minimum"]),
        "harmful_base_decisions_improved": harmful_improved
        >= int(requirements["harmful_base_decisions_improved_minimum"]),
    }
    status = (
        protocol["offline_qualification"]["pass_status"]
        if all(checks.values())
        else protocol["offline_qualification"]["stop_status"]
    )
    base_action_count = sum(
        bool(row["base_differs_from_fifo"]) for row in first_historical
    )
    return {
        "schema": "m4-shield-development-result-v1",
        "status": status,
        "protocol": {
            "path": str(protocol_path.relative_to(repository)),
            "sha256": PARENT.sha256_file(protocol_path),
        },
        "scope": {
            "authenticated_historical_arms": len(historical_provenance),
            "historical_decisions": len(first_historical),
            "live_identity_decisions": len(first_live),
            "candidate_rows": len(all_records) * EXPECTED_CANDIDATES,
        },
        "checks": checks,
        "metrics": {
            "historical_base_action_count": base_action_count,
            "historical_shield_intervention_count": historical_interventions,
            "oars_trajectory_arms_with_intervention": oars_active_arms,
            "fifo_trajectory_intervention_count": fifo_interventions,
            "live_base_proposal_identity_count": live_base_identity_count,
            "live_shield_intervention_count": live_interventions,
            "aggregate_imminent_l1_gain_over_base": gain,
            "base_harm_vs_fifo": base_harm,
            "shield_harm_vs_fifo": shield_harm,
            "base_harm_reduction_fraction": harm_reduction,
            "harmful_base_decisions": sum(
                float(row["base_harm_vs_fifo"]) > 0.0 for row in first_historical
            ),
            "harmful_base_decisions_improved": harmful_improved,
            "maximum_absolute_imminent_reward_variance_difference": max(
                abs(
                    float(row["shield_imminent_reward_variance"])
                    - float(row["base_imminent_reward_variance"])
                )
                for row in all_records
            ),
            "maximum_absolute_total_reward_variance_difference": max(
                abs(
                    float(row["shield_total_reward_variance"])
                    - float(row["base_total_reward_variance"])
                )
                for row in all_records
            ),
        },
        "per_historical_arm": _summarize_arms(first_historical),
        "selection_records_sha256": PARENT.sha256_bytes(
            _canonical({"historical": first_historical, "live": first_live})
        ),
        "provenance": {
            "historical": historical_provenance,
            "live_identity": live_provenance,
        },
        "interpretation_boundary": protocol["interpretation_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--authentication-report-dir", type=Path, required=True)
    parser.add_argument("--live-archive", type=Path, required=True)
    parser.add_argument("--live-receipt", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.repository.resolve(),
        args.archive_root.resolve(),
        args.authentication_report_dir.resolve(),
        args.live_archive.resolve(),
        args.live_receipt.resolve(),
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
    if result["status"] != "PASS_M4_SHIELD_DEVELOPMENT_QUALIFICATION":
        raise SystemExit("M4-Shield development qualification stopped")


if __name__ == "__main__":
    main()
