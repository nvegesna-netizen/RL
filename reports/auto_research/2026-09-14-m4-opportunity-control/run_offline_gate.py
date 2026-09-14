#!/usr/bin/env python3
"""Run the authenticated retrospective M4 opportunity-control design gate."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


G_HERE = Path(__file__).resolve().parent
G_ROOT = G_HERE.parents[2]
G_PAPER = G_ROOT / "reports/auto_research/2026-09-08-m4-opportunity-loss-paper"
sys.path.insert(0, str(G_ROOT))
sys.path.insert(0, str(G_PAPER))

from run_publication_synthesis import numina_join_protocol  # noqa: E402
from tools.m4_llama3b_terminal_analysis import THREE_B_CELLS  # noqa: E402
from tools.m4_llama_v5_terminal_analysis import CELLS as ONE_B_CELLS  # noqa: E402
from tools.opportunity_ledger_join import (  # noqa: E402
    JoinedOpportunityAssignment,
    LedgerJoinProtocol,
    ReleaseArm,
    join_opportunity_ledgers,
)
from tools.opportunity_loss_adjusted_inference import (  # noqa: E402
    infer_adjusted_opportunity_loss,
)
from tools.opportunity_loss_analysis import bound_opportunity_loss  # noqa: E402
from tools.opportunity_loss_pipeline import _parse_protocol  # noqa: E402


G_METRIC_NAMES = (
    "registered_l1",
    "l2",
    "token_normalized_l1",
    "nonzero_token_support",
)
G_POLICY_NAMES = (
    "actual_selection",
    "ready_fifo",
    "freshest_first",
    "weight_fifo",
    "l1_only",
    "urgency_only",
    "deterministic_oars",
)


class OfflineGateError(ValueError):
    """Raised when authenticated evidence violates the frozen gate contract."""


@dataclass(frozen=True)
class MetricVector:
    """Nonnegative coefficient summaries available before release."""

    registered_l1: float
    l2: float
    token_normalized_l1: float
    nonzero_token_support: float
    valid_actor_tokens: int

    def value(self, name: str) -> float:
        """Return one registered metric-family member."""
        values = {
            "registered_l1": self.registered_l1,
            "l2": self.l2,
            "token_normalized_l1": self.token_normalized_l1,
            "nonzero_token_support": self.nonzero_token_support,
        }
        try:
            return values[name]
        except KeyError as error:
            raise OfflineGateError(f"unsupported metric {name!r}") from error


@dataclass(frozen=True)
class GroupRecord:
    """One primary-window group with decision-time lifecycle state."""

    group_id: str
    ordinal: int
    start_version: int
    arm: str
    delivered: bool
    metrics: MetricVector
    reserved_timestamp_ns: int
    ready_timestamp_ns: int
    ready_learner_version: int
    removed_timestamp_ns: int
    removed_learner_version: int
    removal_reason: str

    @property
    def l1_per_token(self) -> float:
        """Return L1 opportunity per valid actor token."""
        if self.metrics.valid_actor_tokens == 0:
            return 0.0
        return self.metrics.registered_l1 / self.metrics.valid_actor_tokens


@dataclass(frozen=True)
class CellData:
    """Authenticated raw and joined evidence for one acquisition."""

    label: str
    family: str
    workload: str
    replicate: str
    protocol: LedgerJoinProtocol
    lifecycle_rows: tuple[Mapping[str, Any], ...]
    opportunity_rows: tuple[Mapping[str, Any], ...]
    joined: tuple[JoinedOpportunityAssignment, ...]
    groups: Mapping[str, GroupRecord]
    lifecycle_sha256: str
    opportunity_sha256: str


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Mapping[str, Any]:
    """Load one JSON object."""
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise OfflineGateError(f"{path} must contain a JSON object")
    return value


def load_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    """Load a newline-terminated JSONL ledger."""
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise OfflineGateError(f"{path} must be nonempty and newline terminated")
    rows = tuple(json.loads(line) for line in raw.splitlines())
    if any(not isinstance(row, dict) for row in rows):
        raise OfflineGateError(f"{path} contains a non-object row")
    return rows


def require_hash(path: Path, expected: str) -> str:
    """Require a path to match its authenticated digest."""
    actual = sha256(path)
    if actual != expected:
        raise OfflineGateError(f"authenticated hash mismatch for {path}")
    return actual


def type7(values: Sequence[float], probability: float) -> float | None:
    """Return the R type-7 quantile, or None for an empty sequence."""
    if not values:
        return None
    if not 0.0 <= probability <= 1.0:
        raise OfflineGateError("quantile probability must be in [0, 1]")
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def metric_vector(row: Mapping[str, Any]) -> MetricVector:
    """Parse the pre-release metric family from a validated opportunity row."""
    tokens = int(row["valid_actor_tokens"])
    l1 = float(row["opportunity"])
    l2 = float(row["l2_coefficient_mass"])
    support = float(row["nonzero_advantage_tokens"])
    values = (l1, l2, support)
    if tokens < 0 or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise OfflineGateError("opportunity metric family must be finite and nonnegative")
    token_normalized = l1 / tokens if tokens > 0 else 0.0
    return MetricVector(
        registered_l1=l1,
        l2=l2,
        token_normalized_l1=token_normalized,
        nonzero_token_support=support,
        valid_actor_tokens=tokens,
    )


def one_stage(
    stages: Mapping[str, list[Mapping[str, Any]]],
    name: str,
    *,
    group_id: str,
) -> Mapping[str, Any]:
    """Return exactly one lifecycle stage for a group."""
    values = stages.get(name, [])
    if len(values) != 1:
        raise OfflineGateError(
            f"{group_id}: expected one {name!r} stage, observed {len(values)}"
        )
    return values[0]


def build_groups(
    *,
    joined: Sequence[JoinedOpportunityAssignment],
    lifecycle_rows: Sequence[Mapping[str, Any]],
    opportunity_rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, GroupRecord]:
    """Attach validated metric and lifecycle records to primary assignments."""
    opportunity_by_group = {
        str(row["group_id"]): row
        for row in opportunity_rows
        if row.get("event_type") == "group"
    }
    stages: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in lifecycle_rows:
        group_id = row.get("group_id")
        stage = row.get("stage")
        if isinstance(group_id, str) and isinstance(stage, str):
            stages[group_id][stage].append(row)

    result: dict[str, GroupRecord] = {}
    for assignment in joined:
        group_id = assignment.assignment_id
        raw_opportunity = opportunity_by_group.get(group_id)
        if raw_opportunity is None:
            raise OfflineGateError(f"{group_id}: missing opportunity group")
        metrics = metric_vector(raw_opportunity)
        if not math.isclose(
            metrics.registered_l1,
            assignment.opportunity,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise OfflineGateError(f"{group_id}: joined L1 opportunity differs")
        if assignment.delivered is None:
            raise OfflineGateError(f"{group_id}: terminal disposition is missing")
        group_stages = stages[group_id]
        reserved = one_stage(group_stages, "reserved", group_id=group_id)
        ready = one_stage(group_stages, "group_ready", group_id=group_id)
        removed = one_stage(group_stages, "removed", group_id=group_id)
        result[group_id] = GroupRecord(
            group_id=group_id,
            ordinal=assignment.ordinal,
            start_version=assignment.start_version,
            arm=assignment.arm,
            delivered=assignment.delivered,
            metrics=metrics,
            reserved_timestamp_ns=int(reserved["timestamp_ns"]),
            ready_timestamp_ns=int(ready["timestamp_ns"]),
            ready_learner_version=int(ready["end_weight_version"]),
            removed_timestamp_ns=int(removed["timestamp_ns"]),
            removed_learner_version=int(removed["end_weight_version"]),
            removal_reason=str(removed["removal_reason"]),
        )
    return result


def make_cell(
    *,
    label: str,
    family: str,
    workload: str,
    replicate: str,
    protocol: LedgerJoinProtocol,
    lifecycle_path: Path,
    opportunity_path: Path,
    expected_lifecycle_sha256: str,
    expected_opportunity_sha256: str,
) -> CellData:
    """Authenticate, strictly join, and enrich one acquisition."""
    lifecycle_digest = require_hash(lifecycle_path, expected_lifecycle_sha256)
    opportunity_digest = require_hash(opportunity_path, expected_opportunity_sha256)
    lifecycle_rows = load_jsonl(lifecycle_path)
    opportunity_rows = load_jsonl(opportunity_path)
    joined = join_opportunity_ledgers(
        protocol=protocol,
        lifecycle_rows=lifecycle_rows,
        opportunity_rows=opportunity_rows,
    )
    groups = build_groups(
        joined=joined,
        lifecycle_rows=lifecycle_rows,
        opportunity_rows=opportunity_rows,
    )
    if len(groups) != len(joined):
        raise OfflineGateError(f"{label}: group enrichment coverage differs")
    return CellData(
        label=label,
        family=family,
        workload=workload,
        replicate=replicate,
        protocol=protocol,
        lifecycle_rows=lifecycle_rows,
        opportunity_rows=opportunity_rows,
        joined=joined,
        groups=groups,
        lifecycle_sha256=lifecycle_digest,
        opportunity_sha256=opportunity_digest,
    )


def load_qwen_cells(evidence_root: Path) -> list[CellData]:
    """Load the six authenticated Qwen acquisitions."""
    synthesis = load_json(G_PAPER / "six_cell_synthesis.json")
    numina_path = (
        evidence_root
        / "reports/auto_research/2026-09-07-m4-opportunity-loss-numinamath-generalization/protocol_config.json"
    )
    numina = load_json(numina_path)
    cells = []
    evidence = synthesis["evidence"]
    for name in sorted(synthesis["cell_results"]):
        record = evidence[name]
        lifecycle_path = evidence_root / record["lifecycle_path"]
        opportunity_path = evidence_root / record["opportunity_path"]
        protocol_path = evidence_root / record["protocol_path"]
        require_hash(protocol_path, record["protocol_sha256"])
        if name.endswith("numinamath"):
            protocol = numina_join_protocol(numina, name)
        else:
            _, protocol, _ = _parse_protocol(protocol_path.read_bytes())
        scale = "0p6b" if "0p6b" in name else "1p7b"
        workload = name.rsplit("_", maxsplit=1)[-1]
        cells.append(
            make_cell(
                label=f"qwen-{name}",
                family="qwen3",
                workload=workload,
                replicate=scale,
                protocol=protocol,
                lifecycle_path=lifecycle_path,
                opportunity_path=opportunity_path,
                expected_lifecycle_sha256=record["lifecycle_sha256"],
                expected_opportunity_sha256=record["opportunity_sha256"],
            )
        )
    return cells


def llama_protocol(domain: str, seed: int) -> LedgerJoinProtocol:
    """Return the shared registered Llama acquisition contract."""
    return LedgerJoinProtocol(
        assignment_domain=domain,
        assignment_seed=seed,
        arms=(ReleaseArm("control", 0.0, 1), ReleaseArm("d5", 5.0, 1)),
        primary_start_version=8,
        primary_end_version=407,
        siblings_per_group=8,
        train_batch_size=32,
    )


def load_llama_cells(evidence_root: Path) -> list[CellData]:
    """Load the eight authenticated Llama 1B and 3B acquisitions."""
    session = evidence_root / "session/20260909_m4_llama_lifecycle_derived_transport"
    one_root = session / "v5-terminal-artifacts/authenticated"
    three_root = session / "llama3b-terminal-artifacts/authenticated"
    one_auth = load_json(session / "v5-terminal-artifacts/authentication.json")
    three_auth = load_json(
        evidence_root
        / "reports/auto_research/2026-09-10-m4-llama3p2-3b-size-extension/terminal_authentication.json"
    )
    if three_auth["status"] != "AUTHENTICATED_READY_FOR_FROZEN_JOINT_ANALYSIS":
        raise OfflineGateError("Llama 3B authentication gate is not open")

    cells = []
    for prefix, family, specs, artifact_root in (
        ("llama1b", "llama3p2_1b", ONE_B_CELLS, one_root),
        ("llama3b", "llama3p2_3b", THREE_B_CELLS, three_root),
    ):
        for name, (workload, replicate, domain, seed, _bootstrap_seed) in specs.items():
            source = artifact_root / name
            if prefix == "llama1b":
                hashes = one_auth[name]["selected_sha256"]
                lifecycle_digest = hashes["lifecycle.jsonl"]
                opportunity_digest = hashes["opportunity.jsonl"]
            else:
                hashes = three_auth["cells"][name.replace("-", "_")]
                lifecycle_digest = hashes["lifecycle_sha256"]
                opportunity_digest = hashes["opportunity_sha256"]
            cells.append(
                make_cell(
                    label=f"{prefix}-{name}",
                    family=family,
                    workload=workload,
                    replicate=replicate,
                    protocol=llama_protocol(domain, seed),
                    lifecycle_path=source / "lifecycle.jsonl",
                    opportunity_path=source / "opportunity.jsonl",
                    expected_lifecycle_sha256=lifecycle_digest,
                    expected_opportunity_sha256=opportunity_digest,
                )
            )
    return cells


def metric_rows(
    cell: CellData,
    metric_name: str,
) -> tuple[JoinedOpportunityAssignment, ...]:
    """Project one metric-family member into the registered inference rows."""
    return tuple(
        dataclasses.replace(
            row,
            opportunity=cell.groups[row.assignment_id].metrics.value(metric_name),
        )
        for row in cell.joined
    )


def metric_inference(
    cell: CellData,
    *,
    metric_name: str,
    bootstrap_seed: int,
    bootstrap_draws: int,
) -> Mapping[str, Any]:
    """Run unadjusted and cross-fitted inference for one metric."""
    rows = metric_rows(cell, metric_name)
    unadjusted = bound_opportunity_loss(row.analysis_assignment() for row in rows)
    adjusted = infer_adjusted_opportunity_loss(
        rows,
        propensities={"control": 0.5, "d5": 0.5},
        primary_start_version=cell.protocol.primary_start_version,
        primary_end_version=cell.protocol.primary_end_version,
        folds=8,
        hac_lag=4,
        confidence=0.95,
        material_threshold=0.2,
        max_missing_fraction=0.01,
        block_size=8,
        bootstrap_draws=bootstrap_draws,
        bootstrap_seed=bootstrap_seed,
    )
    if adjusted.identification_interval[0] != adjusted.identification_interval[1]:
        raise OfflineGateError(f"{cell.label}: adjusted endpoints differ despite completeness")
    endpoint = adjusted.lower_endpoint
    return {
        "assignment_count": len(rows),
        "unadjusted_estimate": unadjusted.complete_data_delta_l,
        "adjusted_estimate": endpoint.estimate,
        "hac_standard_error": endpoint.hac_standard_error,
        "hac_interval": endpoint.hac_interval,
        "bootstrap_interval": endpoint.bootstrap_interval,
        "confidence_envelope": endpoint.envelope,
        "pooled_mean_metric": endpoint.pooled_mean_opportunity,
        "control_mean_metric": unadjusted.control_mean_opportunity,
    }


def natural_expiry_summary(cell: CellData, *, max_staleness: int) -> Mapping[str, Any]:
    """Describe no-added-delay expiry and lifecycle timing."""
    control = [group for group in cell.groups.values() if group.arm == "control"]
    positive = [group for group in control if group.metrics.registered_l1 > 0.0]
    stale_positive = [
        group for group in positive if group.removal_reason == "stale_evicted"
    ]
    total_positive_l1 = math.fsum(group.metrics.registered_l1 for group in positive)
    stale_positive_l1 = math.fsum(
        group.metrics.registered_l1 for group in stale_positive
    )
    latencies: dict[str, list[float]] = defaultdict(list)
    slack = []
    for group in control:
        latency = (group.removed_timestamp_ns - group.ready_timestamp_ns) / 1e9
        if latency < 0.0:
            raise OfflineGateError(f"{cell.label}: removal precedes readiness")
        latencies[group.removal_reason].append(latency)
        slack.append(group.start_version + max_staleness - group.ready_learner_version)
    reason_counts = Counter(group.removal_reason for group in control)
    return {
        "control_assignment_count": len(control),
        "control_terminal_reason_counts": dict(sorted(reason_counts.items())),
        "positive_l1_count": len(positive),
        "positive_l1_stale_eviction_count": len(stale_positive),
        "positive_l1_stale_eviction_count_fraction": (
            len(stale_positive) / len(positive) if positive else 0.0
        ),
        "positive_l1_stale_eviction_mass_fraction": (
            stale_positive_l1 / total_positive_l1 if total_positive_l1 > 0.0 else 0.0
        ),
        "ready_version_slack": {
            "minimum": min(slack),
            "median": type7(slack, 0.5),
            "p90": type7(slack, 0.9),
            "fraction_imminent_or_expired": sum(value <= 0 for value in slack)
            / len(slack),
        },
        "ready_to_removal_seconds": {
            reason: {
                "count": len(values),
                "median": type7(values, 0.5),
                "p90": type7(values, 0.9),
                "p99": type7(values, 0.99),
            }
            for reason, values in sorted(latencies.items())
        },
    }


def selected_group_ids(
    step: Mapping[str, Any],
    sample_owner: Mapping[str, str],
) -> tuple[str, ...]:
    """Recover ordered prompt groups from a completed-step sample list."""
    result = []
    seen = set()
    for sample_id in step["sample_ids"]:
        group_id = sample_owner[str(sample_id)]
        if group_id not in seen:
            result.append(group_id)
            seen.add(group_id)
    return tuple(result)


def policy_order(
    groups: Sequence[GroupRecord],
    *,
    policy: str,
    current_version: int,
) -> list[GroupRecord]:
    """Order one reconstructed decision set under a frozen shadow policy."""
    if policy == "ready_fifo":
        key = lambda group: (group.ready_timestamp_ns, group.group_id)
    elif policy == "freshest_first":
        key = lambda group: (
            -group.start_version,
            group.reserved_timestamp_ns,
            group.group_id,
        )
    elif policy == "weight_fifo":
        key = lambda group: (
            group.start_version,
            group.reserved_timestamp_ns,
            group.group_id,
        )
    elif policy == "l1_only":
        key = lambda group: (
            -group.metrics.registered_l1,
            group.ready_timestamp_ns,
            group.group_id,
        )
    elif policy == "urgency_only":
        key = lambda group: (
            group.start_version + 1 - current_version,
            group.ready_timestamp_ns,
            group.group_id,
        )
    elif policy == "deterministic_oars":
        def key(group: GroupRecord) -> tuple[float | int | str, ...]:
            slack = group.start_version + 1 - current_version
            if slack <= 0:
                return (0, -group.l1_per_token, group.ready_timestamp_ns, group.group_id)
            return (
                1,
                group.start_version,
                -group.l1_per_token,
                group.ready_timestamp_ns,
                group.group_id,
            )
    else:
        raise OfflineGateError(f"unsupported shadow policy {policy!r}")
    return sorted(groups, key=key)


def empty_policy_total() -> dict[str, float]:
    """Create an empty mutable policy accumulator."""
    return {
        "decision_count": 0.0,
        "selected_group_count": 0.0,
        "selected_l1_mass": 0.0,
        "selected_l2_mass": 0.0,
        "selected_valid_actor_tokens": 0.0,
        "overlap_fraction_sum": 0.0,
    }


def add_policy_choice(
    total: dict[str, float],
    selected: Sequence[GroupRecord],
    *,
    actual_ids: set[str],
) -> None:
    """Accumulate one shadow-policy decision."""
    total["decision_count"] += 1.0
    total["selected_group_count"] += len(selected)
    total["selected_l1_mass"] += math.fsum(
        group.metrics.registered_l1 for group in selected
    )
    total["selected_l2_mass"] += math.fsum(group.metrics.l2 for group in selected)
    total["selected_valid_actor_tokens"] += math.fsum(
        group.metrics.valid_actor_tokens for group in selected
    )
    total["overlap_fraction_sum"] += (
        sum(group.group_id in actual_ids for group in selected) / len(selected)
    )


def shadow_choice_summary(cell: CellData, *, max_staleness: int) -> Mapping[str, Any]:
    """Measure one-step policy headroom on reconstructed actual choice sets."""
    sample_owner: dict[str, str] = {}
    for row in cell.opportunity_rows:
        if row.get("event_type") != "group":
            continue
        group_id = str(row["group_id"])
        for sample_id in row["sample_ids"]:
            sample = str(sample_id)
            if sample in sample_owner:
                raise OfflineGateError(f"{cell.label}: duplicate sample owner")
            sample_owner[sample] = group_id
    steps = sorted(
        (
            row
            for row in cell.opportunity_rows
            if row.get("event_type") == "train_step_completed"
        ),
        key=lambda row: int(row["learner_version"]),
    )
    totals = {name: empty_policy_total() for name in G_POLICY_NAMES}
    primary_complete_steps = 0
    contended_steps = 0
    candidate_counts = []
    for step in steps:
        actual_ids = selected_group_ids(step, sample_owner)
        if len(actual_ids) != 4 or any(group_id not in cell.groups for group_id in actual_ids):
            continue
        primary_complete_steps += 1
        actual = [cell.groups[group_id] for group_id in actual_ids]
        if any(group.removal_reason != "selected" for group in actual):
            raise OfflineGateError(f"{cell.label}: completed step includes unselected group")
        decision_timestamp = min(group.removed_timestamp_ns for group in actual)
        current_version = int(step["previous_learner_version"])
        minimum_version = max(0, current_version - max_staleness)
        candidates = [
            group
            for group in cell.groups.values()
            if group.ready_timestamp_ns <= decision_timestamp
            and group.removed_timestamp_ns >= decision_timestamp
            and minimum_version <= group.start_version <= current_version
        ]
        candidate_ids = {group.group_id for group in candidates}
        if not set(actual_ids).issubset(candidate_ids):
            raise OfflineGateError(f"{cell.label}: actual selection absent from choice set")
        if len(candidates) <= len(actual):
            continue
        contended_steps += 1
        candidate_counts.append(len(candidates))
        actual_set = set(actual_ids)
        add_policy_choice(totals["actual_selection"], actual, actual_ids=actual_set)
        for policy in G_POLICY_NAMES[1:]:
            selected = policy_order(
                candidates,
                policy=policy,
                current_version=current_version,
            )[:4]
            add_policy_choice(totals[policy], selected, actual_ids=actual_set)

    actual_l1 = totals["actual_selection"]["selected_l1_mass"]
    actual_tokens = totals["actual_selection"]["selected_valid_actor_tokens"]
    policies = {}
    for name, total in totals.items():
        decisions = int(total.pop("decision_count"))
        selected_count = int(total.pop("selected_group_count"))
        l1 = total["selected_l1_mass"]
        tokens = total["selected_valid_actor_tokens"]
        policies[name] = {
            "decision_count": decisions,
            "selected_group_count": selected_count,
            "selected_l1_mass": l1,
            "selected_l2_mass": total["selected_l2_mass"],
            "selected_valid_actor_tokens": int(tokens),
            "l1_per_selected_token": l1 / tokens if tokens > 0.0 else None,
            "mean_overlap_with_actual": (
                total["overlap_fraction_sum"] / decisions if decisions else None
            ),
            "l1_gain_over_actual_fraction": (
                l1 / actual_l1 - 1.0 if name != "actual_selection" and actual_l1 > 0.0 else 0.0
            ),
            "selected_token_ratio_to_actual": (
                tokens / actual_tokens if name != "actual_selection" and actual_tokens > 0.0 else 1.0
            ),
        }
    return {
        "completed_steps": len(steps),
        "primary_complete_steps": primary_complete_steps,
        "contended_primary_steps": contended_steps,
        "candidate_group_count": {
            "minimum": min(candidate_counts) if candidate_counts else None,
            "median": type7(candidate_counts, 0.5),
            "p90": type7(candidate_counts, 0.9),
            "maximum": max(candidate_counts) if candidate_counts else None,
        },
        "policies": policies,
    }


def infer_cell(
    cell: CellData,
    *,
    cell_index: int,
    bootstrap_draws: int,
) -> Mapping[str, Any]:
    """Produce all frozen offline analyses for one acquisition."""
    metric_results = {}
    for metric_index, name in enumerate(G_METRIC_NAMES):
        metric_results[name] = metric_inference(
            cell,
            metric_name=name,
            bootstrap_seed=2026091400 + cell_index * 10 + metric_index,
            bootstrap_draws=bootstrap_draws,
        )
    return {
        "family": cell.family,
        "workload": cell.workload,
        "replicate": cell.replicate,
        "primary_window": [
            cell.protocol.primary_start_version,
            cell.protocol.primary_end_version,
        ],
        "assignment_count": len(cell.joined),
        "lifecycle_sha256": cell.lifecycle_sha256,
        "opportunity_sha256": cell.opportunity_sha256,
        "metrics": metric_results,
        "natural_expiry": natural_expiry_summary(cell, max_staleness=1),
        "shadow_choice": shadow_choice_summary(cell, max_staleness=1),
    }


def evaluate_gate(cells: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    """Evaluate the frozen all-required implementation gate."""
    l2_positive = sum(
        cell["metrics"]["l2"]["adjusted_estimate"] > 0.0 for cell in cells.values()
    )
    token_positive = sum(
        cell["metrics"]["token_normalized_l1"]["adjusted_estimate"] > 0.0
        for cell in cells.values()
    )
    natural = sum(
        cell["natural_expiry"]["positive_l1_stale_eviction_mass_fraction"] >= 0.10
        for cell in cells.values()
    )
    shadow = sum(
        cell["shadow_choice"]["policies"]["deterministic_oars"][
            "l1_gain_over_actual_fraction"
        ]
        >= 0.10
        for cell in cells.values()
    )
    cells_with_shadow = [
        cell
        for cell in cells.values()
        if cell["shadow_choice"]["contended_primary_steps"] > 0
    ]
    service = all(
        cell["shadow_choice"]["policies"]["deterministic_oars"][
            "selected_token_ratio_to_actual"
        ]
        <= 1.02
        for cell in cells_with_shadow
    )
    conditions = {
        "authentication_and_coverage": {
            "passed": len(cells) == 14
            and sum(cell["assignment_count"] for cell in cells.values()) == 106653,
            "observed_acquisitions": len(cells),
            "observed_assignments": sum(
                cell["assignment_count"] for cell in cells.values()
            ),
        },
        "metric_robustness": {
            "passed": l2_positive >= 12 and token_positive >= 12,
            "l2_positive_acquisitions": l2_positive,
            "token_normalized_l1_positive_acquisitions": token_positive,
            "required_each": 12,
        },
        "natural_relevance": {
            "passed": natural >= 10,
            "acquisitions_at_or_above_0p10": natural,
            "required": 10,
        },
        "shadow_headroom": {
            "passed": shadow >= 8,
            "acquisitions_at_or_above_0p10": shadow,
            "required": 8,
        },
        "service_cost": {
            "passed": service and len(cells_with_shadow) > 0,
            "acquisitions_with_eligible_choice_sets": len(cells_with_shadow),
            "maximum_allowed_token_ratio": 1.02,
        },
    }
    return {
        "all_required": True,
        "passed": all(value["passed"] for value in conditions.values()),
        "conditions": conditions,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    return parser.parse_args()


def main() -> None:
    """Authenticate evidence, run analyses, evaluate the frozen gate, and write JSON."""
    args = parse_args()
    protocol_path = G_HERE / "offline_gate_protocol.json"
    protocol = load_json(protocol_path)
    if protocol["schema"] != "m4-opportunity-control-offline-gate-v1":
        raise OfflineGateError("offline gate protocol schema differs")
    if args.bootstrap_draws != protocol["inference"]["bootstrap_draws"]:
        raise OfflineGateError("bootstrap draw count differs from frozen protocol")
    cells = load_qwen_cells(args.evidence_root)
    cells.extend(load_llama_cells(args.evidence_root))
    if len({cell.label for cell in cells}) != len(cells):
        raise OfflineGateError("acquisition labels are not unique")
    inferred = {
        cell.label: infer_cell(
            cell,
            cell_index=index,
            bootstrap_draws=args.bootstrap_draws,
        )
        for index, cell in enumerate(sorted(cells, key=lambda value: value.label))
    }
    result = {
        "schema": "m4-opportunity-control-offline-gate-result-v1",
        "status": "COMPLETE_RETROSPECTIVE_POLICY_DEVELOPMENT",
        "analysis_role": protocol["analysis_role"],
        "source_commit": protocol["source_commit"],
        "protocol_sha256": sha256(protocol_path),
        "analysis_script_sha256": sha256(Path(__file__)),
        "bootstrap_draws": args.bootstrap_draws,
        "cells": inferred,
        "gate": evaluate_gate(inferred),
        "claim_boundary": protocol["forbidden_claims"],
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "acquisitions": len(inferred),
                "assignments": sum(
                    cell["assignment_count"] for cell in inferred.values()
                ),
                "gate": result["gate"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
