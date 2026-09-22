#!/usr/bin/env python3
"""Authenticate and replay the 20-arm OARS confirmatory decision ledgers."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import statistics
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Sequence


EXPECTED_ARMS = 20
EXPECTED_DECISIONS = 64
EXPECTED_CANDIDATES = 8
SELECTED_GROUPS = 4
LOWER_TOKEN_MULTIPLIER = 0.98
UPPER_TOKEN_MULTIPLIER = 1.02
AUTHENTICATION_STATUS = "PASS_AUTHENTICATED_OUTCOME_EMBARGOED"


class AutopsyError(ValueError):
    """Raised when authenticated evidence or a replay invariant differs."""


def require(condition: bool, message: str) -> None:
    """Raise a stable analysis error when a required condition is false."""
    if not condition:
        raise AutopsyError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    require(isinstance(value, dict), f"{path}: expected JSON object")
    return value


def parse_jsonl(payload: bytes, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line.strip():
            continue
        value = json.loads(raw_line)
        require(isinstance(value, dict), f"{label}:{line_number}: expected object")
        rows.append(value)
    require(bool(rows), f"{label}: empty JSONL")
    return rows


def validate_zip_names(archive: zipfile.ZipFile) -> None:
    names = archive.namelist()
    require(len(names) == len(set(names)), "duplicate ZIP member")
    for name in names:
        pure = PurePosixPath(name)
        require(
            not pure.is_absolute() and ".." not in pure.parts,
            f"unsafe ZIP member: {name}",
        )


def one_member(archive: zipfile.ZipFile, basename: str) -> str:
    matches = [name for name in archive.namelist() if name.endswith(f"/{basename}")]
    require(len(matches) == 1, f"expected one {basename}, found {len(matches)}")
    return matches[0]


def close(left: float, right: float, *, label: str) -> None:
    require(
        math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-7),
        f"{label}: {left} != {right}",
    )


@dataclass(frozen=True)
class Candidate:
    group_id: str
    start_weight_version: int
    ready_timestamp_ns: int
    l1: float
    l2: float
    valid_actor_tokens: int
    mean_reward: float
    reward_variance: float


@dataclass(frozen=True)
class Decision:
    identity: str
    index: int
    current_learner_version: int
    timestamp_ns: int
    candidates: tuple[Candidate, ...]
    baseline_group_ids: tuple[str, ...]
    proposal_group_ids: tuple[str, ...]
    actual_group_ids: tuple[str, ...]
    baseline_tokens: int


def candidate_metric(candidate: Candidate, field: str) -> float:
    return float(getattr(candidate, field))


def select_band(
    decision: Decision,
    score: Callable[[tuple[Candidate, ...]], tuple[float, ...]],
) -> tuple[Candidate, ...]:
    lower = math.ceil(decision.baseline_tokens * LOWER_TOKEN_MULTIPLIER - 1e-12)
    upper = math.floor(decision.baseline_tokens * UPPER_TOKEN_MULTIPLIER + 1e-12)
    ordered = sorted(decision.candidates, key=lambda candidate: candidate.group_id)
    best: tuple[Candidate, ...] | None = None
    best_score: tuple[float, ...] | None = None
    for combination in itertools.combinations(ordered, SELECTED_GROUPS):
        tokens = sum(candidate.valid_actor_tokens for candidate in combination)
        if tokens < lower or tokens > upper:
            continue
        combination_score = score(combination)
        if best_score is None or combination_score > best_score:
            best = combination
            best_score = combination_score
    require(best is not None, f"{decision.identity} d{decision.index}: no feasible batch")
    return best


def policy_selection(
    decision: Decision,
    policy: str,
    *,
    shuffled_l1: dict[str, float] | None = None,
) -> tuple[Candidate, ...]:
    by_id = {candidate.group_id: candidate for candidate in decision.candidates}
    if policy == "recorded_fifo":
        return tuple(by_id[group_id] for group_id in decision.baseline_group_ids)
    if policy == "recorded_oars_v1":
        return tuple(by_id[group_id] for group_id in decision.proposal_group_ids)
    if policy == "age_only_band":
        return select_band(
            decision,
            lambda combination: (
                float(
                    sum(
                        decision.current_learner_version
                        - candidate.start_weight_version
                        for candidate in combination
                    )
                ),
                float(
                    sum(
                        decision.timestamp_ns - candidate.ready_timestamp_ns
                        for candidate in combination
                    )
                ),
                float(-sum(candidate.valid_actor_tokens for candidate in combination)),
            ),
        )
    if policy == "m4_total_band":
        return select_band(
            decision,
            lambda combination: (
                math.fsum(candidate.l1 for candidate in combination),
                float(-sum(candidate.valid_actor_tokens for candidate in combination)),
            ),
        )
    if policy in {"m4_risk_band", "shuffled_m4_risk_band"}:
        values = shuffled_l1 or {candidate.group_id: candidate.l1 for candidate in decision.candidates}
        return select_band(
            decision,
            lambda combination: (
                math.fsum(
                    values[candidate.group_id]
                    for candidate in combination
                    if candidate.start_weight_version + 1
                    <= decision.current_learner_version
                ),
                math.fsum(values[candidate.group_id] for candidate in combination),
                float(-sum(candidate.valid_actor_tokens for candidate in combination)),
            ),
        )
    raise AutopsyError(f"unknown policy: {policy}")


def summarize_selection(decisions: Sequence[Decision], policy: str) -> dict[str, Any]:
    totals = {
        "l1": 0.0,
        "l2": 0.0,
        "tokens": 0,
        "mean_reward_sum": 0.0,
        "reward_variance_sum": 0.0,
        "version_lag_sum": 0,
        "ready_age_seconds_sum": 0.0,
        "imminent_l1": 0.0,
        "fifo_overlap": 0,
    }
    minimum_ratio = math.inf
    maximum_ratio = -math.inf
    for decision in decisions:
        selection = policy_selection(decision, policy)
        tokens = sum(candidate.valid_actor_tokens for candidate in selection)
        ratio = tokens / decision.baseline_tokens
        minimum_ratio = min(minimum_ratio, ratio)
        maximum_ratio = max(maximum_ratio, ratio)
        totals["l1"] += math.fsum(candidate.l1 for candidate in selection)
        totals["l2"] += math.fsum(candidate.l2 for candidate in selection)
        totals["tokens"] += tokens
        totals["mean_reward_sum"] += math.fsum(candidate.mean_reward for candidate in selection)
        totals["reward_variance_sum"] += math.fsum(candidate.reward_variance for candidate in selection)
        totals["version_lag_sum"] += sum(
            decision.current_learner_version - candidate.start_weight_version
            for candidate in selection
        )
        totals["ready_age_seconds_sum"] += math.fsum(
            (decision.timestamp_ns - candidate.ready_timestamp_ns) / 1e9
            for candidate in selection
        )
        totals["imminent_l1"] += math.fsum(
            candidate.l1
            for candidate in selection
            if candidate.start_weight_version + 1 <= decision.current_learner_version
        )
        totals["fifo_overlap"] += len(
            set(candidate.group_id for candidate in selection)
            & set(decision.baseline_group_ids)
        )
    selected_count = len(decisions) * SELECTED_GROUPS
    baseline_tokens = sum(decision.baseline_tokens for decision in decisions)
    return {
        "selected_l1": totals["l1"],
        "selected_l2": totals["l2"],
        "selected_imminent_l1": totals["imminent_l1"],
        "selected_valid_actor_tokens": totals["tokens"],
        "aggregate_token_ratio_to_fifo": totals["tokens"] / baseline_tokens,
        "minimum_decision_token_ratio_to_fifo": minimum_ratio,
        "maximum_decision_token_ratio_to_fifo": maximum_ratio,
        "mean_group_reward": totals["mean_reward_sum"] / selected_count,
        "mean_group_reward_variance": totals["reward_variance_sum"] / selected_count,
        "mean_version_lag": totals["version_lag_sum"] / selected_count,
        "mean_ready_age_seconds": totals["ready_age_seconds_sum"] / selected_count,
        "mean_fifo_overlap_fraction": totals["fifo_overlap"] / selected_count,
    }


def quantile(values: Sequence[float], probability: float) -> float:
    require(bool(values), "quantile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def shuffled_replay(
    decisions: Sequence[Decision],
    *,
    identity: str,
    replicates: int,
    seed_label: str,
) -> dict[str, float]:
    replicate_l1: list[float] = []
    replicate_l2: list[float] = []
    replicate_tokens: list[int] = []
    for replicate in range(replicates):
        l1_total = 0.0
        l2_total = 0.0
        token_total = 0
        for decision in decisions:
            digest = hashlib.sha256(
                f"{seed_label}:{identity}:{decision.index}:{replicate}".encode()
            ).digest()
            rng = random.Random(int.from_bytes(digest[:8], "big"))
            observed = [candidate.l1 for candidate in decision.candidates]
            rng.shuffle(observed)
            shuffled = {
                candidate.group_id: value
                for candidate, value in zip(decision.candidates, observed)
            }
            selection = policy_selection(
                decision,
                "shuffled_m4_risk_band",
                shuffled_l1=shuffled,
            )
            l1_total += math.fsum(candidate.l1 for candidate in selection)
            l2_total += math.fsum(candidate.l2 for candidate in selection)
            token_total += sum(candidate.valid_actor_tokens for candidate in selection)
        replicate_l1.append(l1_total)
        replicate_l2.append(l2_total)
        replicate_tokens.append(token_total)
    return {
        "selected_true_l1_mean": statistics.fmean(replicate_l1),
        "selected_true_l1_q025": quantile(replicate_l1, 0.025),
        "selected_true_l1_q975": quantile(replicate_l1, 0.975),
        "selected_l2_mean": statistics.fmean(replicate_l2),
        "selected_tokens_mean": statistics.fmean(replicate_tokens),
    }


def archive_ledgers(
    repository: Path,
    archive_root: Path,
    report_dir: Path,
    identity: str,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    authentication_path = report_dir / f"confirmatory_{identity.replace('-', '_')}_terminal_authentication.json"
    authentication = load_object(authentication_path)
    require(authentication.get("identity") == identity, f"{identity}: identity differs")
    require(authentication.get("status") == AUTHENTICATION_STATUS, f"{identity}: status differs")
    main_job = authentication["jobs"]["main"]
    archive_path = archive_root / f"confirmatory-{identity}-terminal-archives" / f"main-{main_job}.zip"
    require(archive_path.is_file(), f"{identity}: archive missing")
    archive_digest = sha256_file(archive_path)
    require(
        archive_digest == authentication["archives"]["main"]["sha256"],
        f"{identity}: archive SHA-256 differs",
    )
    ledgers: dict[str, list[dict[str, Any]]] = {}
    member_hashes: dict[str, str] = {}
    with zipfile.ZipFile(archive_path) as archive:
        validate_zip_names(archive)
        for basename in ("lifecycle.jsonl", "opportunity.jsonl", "oars.jsonl"):
            member = one_member(archive, basename)
            payload = archive.read(member)
            digest = sha256_bytes(payload)
            require(
                digest == authentication["artifact_sha256"][basename],
                f"{identity}: {basename} SHA-256 differs",
            )
            ledgers[basename] = parse_jsonl(payload, label=f"{identity}:{basename}")
            member_hashes[basename] = digest
    try:
        relative_archive = str(archive_path.resolve().relative_to(repository.resolve()))
    except ValueError:
        relative_archive = str(archive_path.resolve())
    provenance = {
        "authentication_receipt": str(authentication_path.relative_to(repository)),
        "authentication_receipt_sha256": sha256_file(authentication_path),
        "archive": relative_archive,
        "archive_sha256": archive_digest,
        "ledger_sha256": member_hashes,
    }
    return provenance, ledgers


def opportunity_map(rows: Sequence[dict[str, Any]], *, identity: str) -> dict[str, Candidate]:
    groups: dict[str, Candidate] = {}
    headers = [row for row in rows if row.get("event_type") == "header"]
    require(len(headers) == 1, f"{identity}: opportunity header count differs")
    for row in rows:
        if row.get("event_type") != "group":
            continue
        group_id = row.get("group_id")
        siblings = row.get("siblings")
        require(isinstance(group_id, str) and group_id, f"{identity}: invalid opportunity group")
        require(isinstance(siblings, list) and siblings, f"{identity}: missing siblings")
        rewards = [float(sibling["reward"]) for sibling in siblings]
        groups[group_id] = Candidate(
            group_id=group_id,
            start_weight_version=int(row["start_weight_version"]),
            ready_timestamp_ns=-1,
            l1=float(row["opportunity"]),
            l2=float(row["l2_coefficient_mass"]),
            valid_actor_tokens=int(row["valid_actor_tokens"]),
            mean_reward=statistics.fmean(rewards),
            reward_variance=statistics.pvariance(rewards),
        )
    require(bool(groups), f"{identity}: no opportunity groups")
    return groups


def reconstruct_decisions(
    identity: str,
    lifecycle_rows: Sequence[dict[str, Any]],
    oars_rows: Sequence[dict[str, Any]],
    opportunities: dict[str, Candidate],
) -> tuple[list[Decision], dict[str, Any]]:
    headers = [row for row in oars_rows if row.get("event_type") == "header"]
    events = [row for row in oars_rows if row.get("event_type") == "decision"]
    require(len(headers) == 1, f"{identity}: OARS header count differs")
    header = headers[0]
    require(header.get("schema_version") == 3, f"{identity}: OARS schema differs")
    require(
        header.get("selection_candidate_watermark") == EXPECTED_CANDIDATES,
        f"{identity}: candidate watermark differs",
    )
    require(len(events) == EXPECTED_DECISIONS, f"{identity}: decision count differs")
    lifecycle = sorted(lifecycle_rows, key=lambda row: int(row["controller_sequence"]))
    sequences = [int(row["controller_sequence"]) for row in lifecycle]
    require(len(sequences) == len(set(sequences)), f"{identity}: duplicate lifecycle sequence")
    live: list[str] = []
    state: dict[str, dict[str, Any]] = {}
    decisions: list[Decision] = []
    selected_block: list[str] = []
    captured: Decision | None = None
    deferred_first: dict[str, dict[str, Any]] = {}
    removal_reason: dict[str, str] = {}

    for row in lifecycle:
        stage = row.get("stage")
        group_id = row.get("group_id")
        if stage == "reserved":
            require(isinstance(group_id, str) and group_id not in state, f"{identity}: invalid reservation")
            live.append(group_id)
            state[group_id] = {
                "ready": False,
                "start": int(row["start_weight_version"]),
                "ready_timestamp_ns": None,
            }
        elif stage == "group_ready":
            require(group_id in state and not state[group_id]["ready"], f"{identity}: invalid ready event")
            state[group_id]["ready"] = True
            state[group_id]["ready_timestamp_ns"] = int(row["timestamp_ns"])
        elif stage == "removed":
            require(group_id in state and group_id in live, f"{identity}: unknown removal")
            reason = row.get("removal_reason")
            require(isinstance(reason, str), f"{identity}: missing removal reason")
            if reason == "selected" and not selected_block:
                decision_index = len(decisions)
                require(decision_index < len(events), f"{identity}: excess selection block")
                event = events[decision_index]
                current = int(event["current_learner_version"])
                minimum = max(0, current - 1)
                eligible = [
                    candidate_id
                    for candidate_id in live
                    if state[candidate_id]["ready"]
                    and minimum <= state[candidate_id]["start"] <= current
                ]
                eligible.sort(key=lambda candidate_id: (state[candidate_id]["start"], live.index(candidate_id)))
                candidate_ids = eligible[:EXPECTED_CANDIDATES]
                require(
                    len(candidate_ids) == EXPECTED_CANDIDATES,
                    f"{identity} d{decision_index}: reconstructed candidate count differs",
                )
                require(
                    int(event["candidate_group_count"]) == EXPECTED_CANDIDATES
                    and int(event["eligible_candidate_count"]) >= EXPECTED_CANDIDATES,
                    f"{identity} d{decision_index}: recorded candidate counts differ",
                )
                candidates: list[Candidate] = []
                for candidate_id in candidate_ids:
                    require(candidate_id in opportunities, f"{identity}: candidate opportunity missing")
                    source = opportunities[candidate_id]
                    require(
                        source.start_weight_version == state[candidate_id]["start"],
                        f"{identity}: candidate start version differs",
                    )
                    ready_timestamp = state[candidate_id]["ready_timestamp_ns"]
                    require(isinstance(ready_timestamp, int), f"{identity}: ready timestamp missing")
                    candidates.append(
                        Candidate(
                            group_id=source.group_id,
                            start_weight_version=source.start_weight_version,
                            ready_timestamp_ns=ready_timestamp,
                            l1=source.l1,
                            l2=source.l2,
                            valid_actor_tokens=source.valid_actor_tokens,
                            mean_reward=source.mean_reward,
                            reward_variance=source.reward_variance,
                        )
                    )
                baseline_ids = tuple(event["baseline_group_ids"])
                proposal_ids = tuple(event["proposed_group_ids"])
                actual_ids = tuple(event["actual_selected_group_ids"])
                candidate_id_set = set(candidate_ids)
                for label, ids in (
                    ("baseline", baseline_ids),
                    ("proposal", proposal_ids),
                    ("actual", actual_ids),
                ):
                    require(
                        len(ids) == SELECTED_GROUPS and set(ids) <= candidate_id_set,
                        f"{identity} d{decision_index}: {label} identities differ",
                    )
                by_id = {candidate.group_id: candidate for candidate in candidates}
                baseline_l1 = math.fsum(by_id[group].l1 for group in baseline_ids)
                proposal_l1 = math.fsum(by_id[group].l1 for group in proposal_ids)
                baseline_tokens = sum(by_id[group].valid_actor_tokens for group in baseline_ids)
                proposal_tokens = sum(by_id[group].valid_actor_tokens for group in proposal_ids)
                close(baseline_l1, float(event["baseline_l1"]), label=f"{identity} baseline L1")
                close(proposal_l1, float(event["proposed_l1"]), label=f"{identity} proposal L1")
                require(baseline_tokens == int(event["baseline_valid_actor_tokens"]), f"{identity}: baseline tokens differ")
                require(proposal_tokens == int(event["proposed_valid_actor_tokens"]), f"{identity}: proposal tokens differ")
                captured = Decision(
                    identity=identity,
                    index=decision_index,
                    current_learner_version=current,
                    timestamp_ns=int(row["timestamp_ns"]),
                    candidates=tuple(candidates),
                    baseline_group_ids=baseline_ids,
                    proposal_group_ids=proposal_ids,
                    actual_group_ids=actual_ids,
                    baseline_tokens=baseline_tokens,
                )
                for candidate_id in candidate_ids:
                    if candidate_id not in actual_ids and candidate_id not in deferred_first:
                        deferred_first[candidate_id] = {
                            "l1": by_id[candidate_id].l1,
                            "l2": by_id[candidate_id].l2,
                        }
            if reason == "selected":
                require(captured is not None, f"{identity}: selected block was not captured")
                selected_block.append(group_id)
            removal_reason[group_id] = reason
            live.remove(group_id)
            del state[group_id]
            if len(selected_block) == SELECTED_GROUPS:
                require(captured is not None, f"{identity}: captured decision missing")
                require(
                    set(selected_block) == set(captured.actual_group_ids),
                    f"{identity} d{captured.index}: selected lifecycle identities differ",
                )
                expected_actual = (
                    captured.proposal_group_ids
                    if header.get("mode") == "act"
                    else captured.baseline_group_ids
                )
                require(
                    set(captured.actual_group_ids) == set(expected_actual),
                    f"{identity} d{captured.index}: enacted policy differs",
                )
                decisions.append(captured)
                selected_block = []
                captured = None
    require(not selected_block and captured is None, f"{identity}: incomplete selected block")
    require(len(decisions) == EXPECTED_DECISIONS, f"{identity}: reconstructed decision count differs")

    deferred_summary: dict[str, dict[str, float | int]] = {}
    for candidate_id, metrics in deferred_first.items():
        reason = removal_reason.get(candidate_id, "not_removed")
        row = deferred_summary.setdefault(reason, {"groups": 0, "l1": 0.0, "l2": 0.0})
        row["groups"] = int(row["groups"]) + 1
        row["l1"] = float(row["l1"]) + float(metrics["l1"])
        row["l2"] = float(row["l2"]) + float(metrics["l2"])
    return decisions, {
        "mode": header.get("mode"),
        "first_deferred_group_fates": deferred_summary,
    }


def percent_gain(numerator: float, denominator: float) -> float:
    require(denominator > 0, "percent-gain denominator must be positive")
    return 100.0 * (numerator / denominator - 1.0)


def analyze(
    repository: Path,
    archive_root: Path,
    report_dir: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    protocol = load_object(protocol_path)
    require(
        protocol.get("schema") == "m4-oars-v2-offline-autopsy-protocol-v1",
        "protocol schema differs",
    )
    replicates = int(protocol["shuffled_replays_per_arm"])
    seed_label = str(protocol["shuffle_seed"])
    identities = [f"p{pair:02d}-{mode}" for pair in range(1, 11) for mode in ("fifo", "oars")]
    require(len(identities) == EXPECTED_ARMS, "identity count differs")
    arm_rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    all_decisions: list[Decision] = []
    for identity in identities:
        arm_provenance, ledgers = archive_ledgers(
            repository, archive_root, report_dir, identity
        )
        opportunities = opportunity_map(ledgers["opportunity.jsonl"], identity=identity)
        decisions, lifecycle_summary = reconstruct_decisions(
            identity,
            ledgers["lifecycle.jsonl"],
            ledgers["oars.jsonl"],
            opportunities,
        )
        summaries = {
            policy: summarize_selection(decisions, policy)
            for policy in (
                "recorded_fifo",
                "recorded_oars_v1",
                "age_only_band",
                "m4_total_band",
                "m4_risk_band",
            )
        }
        shuffled = shuffled_replay(
            decisions,
            identity=identity,
            replicates=replicates,
            seed_label=seed_label,
        )
        m4 = summaries["m4_risk_band"]
        age = summaries["age_only_band"]
        arm_rows.append(
            {
                "identity": identity,
                "mode": lifecycle_summary["mode"],
                "decision_count": len(decisions),
                "policies": summaries,
                "shuffled_m4_risk_band": shuffled,
                "m4_l1_gain_over_age_percent": percent_gain(m4["selected_l1"], age["selected_l1"]),
                "m4_l2_gain_over_age_percent": percent_gain(m4["selected_l2"], age["selected_l2"]),
                "m4_l1_gain_over_shuffled_mean_percent": percent_gain(
                    m4["selected_l1"], shuffled["selected_true_l1_mean"]
                ),
                "first_deferred_group_fates": lifecycle_summary["first_deferred_group_fates"],
            }
        )
        provenance[identity] = arm_provenance
        all_decisions.extend(decisions)

    def sum_policy(policy: str, metric: str) -> float:
        return math.fsum(float(row["policies"][policy][metric]) for row in arm_rows)

    aggregate: dict[str, Any] = {}
    for policy in (
        "recorded_fifo",
        "recorded_oars_v1",
        "age_only_band",
        "m4_total_band",
        "m4_risk_band",
    ):
        l1 = sum_policy(policy, "selected_l1")
        l2 = sum_policy(policy, "selected_l2")
        tokens = sum_policy(policy, "selected_valid_actor_tokens")
        fifo_tokens = sum_policy("recorded_fifo", "selected_valid_actor_tokens")
        aggregate[policy] = {
            "selected_l1": l1,
            "selected_l2": l2,
            "selected_valid_actor_tokens": int(tokens),
            "aggregate_token_ratio_to_fifo": tokens / fifo_tokens,
        }
    shuffled_l1 = math.fsum(
        float(row["shuffled_m4_risk_band"]["selected_true_l1_mean"])
        for row in arm_rows
    )
    shuffled_l2 = math.fsum(
        float(row["shuffled_m4_risk_band"]["selected_l2_mean"])
        for row in arm_rows
    )
    aggregate["shuffled_m4_risk_band"] = {
        "selected_true_l1_mean": shuffled_l1,
        "selected_l2_mean": shuffled_l2,
    }
    m4_l1 = aggregate["m4_risk_band"]["selected_l1"]
    m4_l2 = aggregate["m4_risk_band"]["selected_l2"]
    age_l1 = aggregate["age_only_band"]["selected_l1"]
    age_l2 = aggregate["age_only_band"]["selected_l2"]
    band_minimum = min(
        float(row["policies"]["m4_risk_band"]["minimum_decision_token_ratio_to_fifo"])
        for row in arm_rows
    )
    band_maximum = max(
        float(row["policies"]["m4_risk_band"]["maximum_decision_token_ratio_to_fifo"])
        for row in arm_rows
    )
    positive_l1_age = sum(float(row["m4_l1_gain_over_age_percent"]) > 0 for row in arm_rows)
    positive_l2_age = sum(float(row["m4_l2_gain_over_age_percent"]) > 0 for row in arm_rows)
    positive_l1_shuffled = sum(
        float(row["m4_l1_gain_over_shuffled_mean_percent"]) > 0 for row in arm_rows
    )
    gate_checks = {
        "provenance": len(arm_rows) == EXPECTED_ARMS
        and len(all_decisions) == EXPECTED_ARMS * EXPECTED_DECISIONS,
        "service": band_minimum >= LOWER_TOKEN_MULTIPLIER - 1e-12
        and band_maximum <= UPPER_TOKEN_MULTIPLIER + 1e-12,
        "signal_ablation": positive_l1_shuffled == EXPECTED_ARMS
        and percent_gain(float(m4_l1), shuffled_l1) >= 10.0,
        "staleness_comparator": positive_l1_age >= 16
        and percent_gain(float(m4_l1), float(age_l1)) >= 5.0,
        "metric_robustness": positive_l2_age >= 16,
    }
    return {
        "schema": "m4-oars-v2-offline-autopsy-result-v1",
        "status": "PASS_OARS_V2_SHADOW_IMPLEMENTATION_GATE"
        if all(gate_checks.values())
        else "STOP_OARS_V2_SHADOW_IMPLEMENTATION_GATE",
        "protocol": {
            "path": str(protocol_path.relative_to(repository)),
            "sha256": sha256_file(protocol_path),
        },
        "scope": {
            "authenticated_arms": len(arm_rows),
            "reconstructed_decisions": len(all_decisions),
            "candidate_rows": len(all_decisions) * EXPECTED_CANDIDATES,
            "shuffled_replays_per_arm": replicates,
        },
        "gate_checks": gate_checks,
        "gate_diagnostics": {
            "m4_l1_positive_vs_shuffled_arms": positive_l1_shuffled,
            "m4_l1_gain_over_shuffled_mean_percent": percent_gain(float(m4_l1), shuffled_l1),
            "m4_l1_positive_vs_age_arms": positive_l1_age,
            "m4_l1_gain_over_age_percent": percent_gain(float(m4_l1), float(age_l1)),
            "m4_l2_positive_vs_age_arms": positive_l2_age,
            "m4_l2_gain_over_age_percent": percent_gain(float(m4_l2), float(age_l2)),
            "m4_band_minimum_decision_token_ratio": band_minimum,
            "m4_band_maximum_decision_token_ratio": band_maximum,
        },
        "aggregate": aggregate,
        "arms": arm_rows,
        "provenance": provenance,
        "interpretation_boundary": protocol["interpretation_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.repository.resolve(),
        args.archive_root.resolve(),
        args.report_dir.resolve(),
        args.protocol.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "scope", "gate_checks", "gate_diagnostics")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
