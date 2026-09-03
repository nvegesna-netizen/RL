# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Assess the registered M4 delay-mechanism support condition."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from tools.opportunity_ledger_join import JoinedOpportunityAssignment


class OpportunityLossMechanismError(ValueError):
    """Raised when mechanism inputs or the registered contract disagree."""


@dataclass(frozen=True)
class MechanismReplicationResult:
    """Value-only mechanism-support result."""

    conclusion: str
    assignment_count: int
    unscored_assignment_count: int
    arm_summaries: dict[str, dict[str, int | float]]
    contrasts: dict[str, float]
    checks: dict[str, bool]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OpportunityLossMechanismError(f"{name} must be an integer")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpportunityLossMechanismError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise OpportunityLossMechanismError(f"{name} must be finite")
    return result


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise OpportunityLossMechanismError(f"{name} must be an object")
    return value


def _one(
    stages: Mapping[str, list[Mapping[str, Any]]], stage: str
) -> Mapping[str, Any] | None:
    rows = stages.get(stage, [])
    if len(rows) > 1:
        raise OpportunityLossMechanismError(f"duplicate {stage} lifecycle stage")
    return rows[0] if rows else None


def _expected_contract(value: object) -> dict[str, float]:
    contract = _mapping(value, name="mechanism_replication")
    expected: dict[str, float] = {
        "maximum_control_direct_chain_rate": 0.02,
        "minimum_d5_direct_chain_rate": 0.10,
        "minimum_d10_direct_chain_rate": 0.40,
        "minimum_d5_control_direct_chain_contrast": 0.10,
        "minimum_d10_d5_direct_chain_contrast": 0.20,
        "minimum_d5_control_version_advance_contrast": 0.20,
        "minimum_d10_d5_version_advance_contrast": 0.25,
        "minimum_nonzero_delay_compliance": 0.99,
        "maximum_delay_overshoot_p99_seconds": 2.0,
    }
    if set(contract) != {
        "accepted_predecessor_commit",
        "role",
        "score_window",
        "thresholds",
    }:
        raise OpportunityLossMechanismError("mechanism contract keyset disagrees")
    if (
        contract.get("accepted_predecessor_commit")
        != ("a6be6971b79ad0d6b48c0dc85bb7f09682c2f7dd")
        or contract.get("role") != "support_condition_not_primary_endpoint"
    ):
        raise OpportunityLossMechanismError("mechanism provenance disagrees")
    if contract.get("score_window") != "primary_start_versions":
        raise OpportunityLossMechanismError("mechanism score window disagrees")
    thresholds = _mapping(contract.get("thresholds"), name="mechanism thresholds")
    if set(thresholds) != set(expected):
        raise OpportunityLossMechanismError("mechanism threshold keyset disagrees")
    for name, registered in expected.items():
        if _number(thresholds.get(name), name=name) != registered:
            raise OpportunityLossMechanismError(f"mechanism threshold {name} disagrees")
    return expected


def _quantile_99(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * 0.99
    lower = math.floor(position)
    fraction = position - lower
    return ordered[lower] + fraction * (
        ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]
    )


def assess_mechanism_replication(
    *,
    lifecycle_rows: Sequence[Mapping[str, Any]],
    assignments: Sequence[JoinedOpportunityAssignment],
    contract: object,
    max_staleness_versions: int,
) -> MechanismReplicationResult:
    """Score version advance and strict stale-boundary chains in the primary cohort."""
    thresholds = _expected_contract(contract)
    if max_staleness_versions != 1:
        raise OpportunityLossMechanismError("mechanism requires max staleness one")
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in lifecycle_rows:
        if row.get("stage") != "learner_version_advanced":
            group_id = row.get("group_id")
            if isinstance(group_id, str) and group_id:
                by_group[group_id].append(row)

    scored: dict[str, list[tuple[float, bool, float | None]]] = defaultdict(list)
    unscored = 0
    for assignment in assignments:
        stages: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in by_group.get(assignment.assignment_id, []):
            stage = row.get("stage")
            if isinstance(stage, str):
                stages[stage].append(row)
        started = _one(stages, "release_delay_started")
        completed = _one(stages, "release_delay_completed")
        group_completed = _one(stages, "group_completed")
        ready = _one(stages, "group_ready")
        removed = _one(stages, "removed")
        if started is None or completed is None:
            unscored += 1
            continue
        start_version = started.get("learner_weight_version")
        end_version = completed.get("learner_weight_version")
        if (
            isinstance(start_version, bool)
            or not isinstance(start_version, int)
            or isinstance(end_version, bool)
            or not isinstance(end_version, int)
            or end_version < start_version
        ):
            unscored += 1
            continue
        version_advance = float(end_version - start_version)
        direct_chain = False
        if group_completed is not None and ready is not None and removed is not None:
            sequences = [
                completed.get("controller_sequence"),
                group_completed.get("controller_sequence"),
                ready.get("controller_sequence"),
                removed.get("controller_sequence"),
            ]
            if all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in sequences
            ):
                boundary = assignment.start_version + max_staleness_versions
                direct_chain = bool(
                    start_version <= boundary < end_version
                    and sequences == sorted(sequences)
                    and len(set(sequences)) == 4
                    and removed.get("removal_reason") == "stale_evicted"
                )
        elapsed_seconds = (
            _integer(completed.get("timestamp_ns"), name="delay completed timestamp")
            - _integer(started.get("timestamp_ns"), name="delay started timestamp")
        ) / 1e9
        requested = _number(started.get("release_delay_seconds"), name="release delay")
        scored[assignment.arm].append(
            (
                version_advance,
                direct_chain,
                elapsed_seconds - requested if requested > 0 else None,
            )
        )

    arm_summaries: dict[str, dict[str, int | float]] = {}
    for arm in ("control", "d5", "d10"):
        rows = scored.get(arm, [])
        if not rows:
            continue
        arm_summaries[arm] = {
            "assignment_count": len(rows),
            "direct_chain_count": sum(item[1] for item in rows),
            "direct_chain_rate": statistics.mean(item[1] for item in rows),
            "mean_version_advance": statistics.mean(item[0] for item in rows),
        }
    contrasts: dict[str, float] = {}
    if set(arm_summaries) == {"control", "d5", "d10"}:
        contrasts = {
            "d5_minus_control_direct_chain": float(
                arm_summaries["d5"]["direct_chain_rate"]
            )
            - float(arm_summaries["control"]["direct_chain_rate"]),
            "d10_minus_d5_direct_chain": float(
                arm_summaries["d10"]["direct_chain_rate"]
            )
            - float(arm_summaries["d5"]["direct_chain_rate"]),
            "d5_minus_control_version_advance": float(
                arm_summaries["d5"]["mean_version_advance"]
            )
            - float(arm_summaries["control"]["mean_version_advance"]),
            "d10_minus_d5_version_advance": float(
                arm_summaries["d10"]["mean_version_advance"]
            )
            - float(arm_summaries["d5"]["mean_version_advance"]),
        }
    overshoots = [
        item[2] for rows in scored.values() for item in rows if item[2] is not None
    ]
    compliance = (
        statistics.mean(value >= 0 for value in overshoots) if overshoots else 0.0
    )
    overshoot_p99 = _quantile_99(overshoots) if overshoots else math.inf
    complete = (
        unscored == 0
        and len(assignments) > 0
        and set(arm_summaries) == {"control", "d5", "d10"}
    )
    checks = {
        "complete_primary_scoring": complete,
        "control_direct_chain_rate": complete
        and float(arm_summaries["control"]["direct_chain_rate"])
        <= thresholds["maximum_control_direct_chain_rate"],
        "d5_direct_chain_rate": complete
        and float(arm_summaries["d5"]["direct_chain_rate"])
        >= thresholds["minimum_d5_direct_chain_rate"],
        "d10_direct_chain_rate": complete
        and float(arm_summaries["d10"]["direct_chain_rate"])
        >= thresholds["minimum_d10_direct_chain_rate"],
        "d5_control_direct_chain_ordering": complete
        and contrasts.get("d5_minus_control_direct_chain", -math.inf)
        >= thresholds["minimum_d5_control_direct_chain_contrast"],
        "d10_d5_direct_chain_ordering": complete
        and contrasts.get("d10_minus_d5_direct_chain", -math.inf)
        >= thresholds["minimum_d10_d5_direct_chain_contrast"],
        "d5_control_version_advance_ordering": complete
        and contrasts.get("d5_minus_control_version_advance", -math.inf)
        >= thresholds["minimum_d5_control_version_advance_contrast"],
        "d10_d5_version_advance_ordering": complete
        and contrasts.get("d10_minus_d5_version_advance", -math.inf)
        >= thresholds["minimum_d10_d5_version_advance_contrast"],
        "nonzero_delay_compliance": complete
        and compliance >= thresholds["minimum_nonzero_delay_compliance"],
        "delay_overshoot_p99": complete
        and overshoot_p99 <= thresholds["maximum_delay_overshoot_p99_seconds"],
    }
    conclusion = (
        "INSUFFICIENT_MECHANISM_EVIDENCE"
        if not complete
        else "REPLICATED"
        if all(checks.values())
        else "NOT_REPLICATED"
    )
    return MechanismReplicationResult(
        conclusion=conclusion,
        assignment_count=len(assignments),
        unscored_assignment_count=unscored,
        arm_summaries=arm_summaries,
        contrasts=contrasts,
        checks=checks,
    )


def assess_followup_mechanism(
    *,
    lifecycle_rows: Sequence[Mapping[str, Any]],
    assignments: Sequence[JoinedOpportunityAssignment],
    contract: object,
    max_staleness_versions: int,
) -> MechanismReplicationResult:
    """Score the prospectively registered control:d5 mechanism support condition."""
    value = _mapping(contract, name="mechanism_followup")
    expected_thresholds = {
        "maximum_control_direct_chain_rate": 0.02,
        "maximum_delay_overshoot_p99_seconds": 2.0,
        "minimum_d5_control_direct_chain_contrast": 0.10,
        "minimum_d5_control_version_advance_contrast": 0.20,
        "minimum_d5_direct_chain_rate": 0.10,
        "minimum_nonzero_delay_compliance": 0.99,
    }
    if set(value) != {
        "historical_dose_order_result_sha256",
        "role",
        "score_window",
        "thresholds",
    }:
        raise OpportunityLossMechanismError("follow-up mechanism keyset disagrees")
    if (
        value.get("historical_dose_order_result_sha256")
        != "6c3fc0cf0567d4dab63153769c075dc7d491e41cb2ceb982b569649f5b42ced3"
        or value.get("role") != "d5_control_support_condition_not_primary_endpoint"
        or value.get("score_window") != "primary_start_versions"
    ):
        raise OpportunityLossMechanismError("follow-up mechanism provenance disagrees")
    raw_thresholds = _mapping(value.get("thresholds"), name="follow-up thresholds")
    if set(raw_thresholds) != set(expected_thresholds):
        raise OpportunityLossMechanismError("follow-up threshold keyset disagrees")
    thresholds = {
        name: _number(raw_thresholds.get(name), name=name)
        for name in expected_thresholds
    }
    if thresholds != expected_thresholds or max_staleness_versions != 1:
        raise OpportunityLossMechanismError("follow-up mechanism contract disagrees")

    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in lifecycle_rows:
        if row.get("stage") != "learner_version_advanced":
            group_id = row.get("group_id")
            if isinstance(group_id, str) and group_id:
                by_group[group_id].append(row)
    scored: dict[str, list[tuple[float, bool, float | None]]] = defaultdict(list)
    unscored = 0
    for assignment in assignments:
        stages: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in by_group.get(assignment.assignment_id, []):
            stage = row.get("stage")
            if isinstance(stage, str):
                stages[stage].append(row)
        started = _one(stages, "release_delay_started")
        completed = _one(stages, "release_delay_completed")
        group_completed = _one(stages, "group_completed")
        ready = _one(stages, "group_ready")
        removed = _one(stages, "removed")
        if started is None or completed is None:
            unscored += 1
            continue
        start_version = started.get("learner_weight_version")
        end_version = completed.get("learner_weight_version")
        if (
            isinstance(start_version, bool)
            or not isinstance(start_version, int)
            or isinstance(end_version, bool)
            or not isinstance(end_version, int)
            or end_version < start_version
        ):
            unscored += 1
            continue
        direct_chain = False
        if group_completed is not None and ready is not None and removed is not None:
            sequences = [
                completed.get("controller_sequence"),
                group_completed.get("controller_sequence"),
                ready.get("controller_sequence"),
                removed.get("controller_sequence"),
            ]
            if all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in sequences
            ):
                boundary = assignment.start_version + max_staleness_versions
                direct_chain = bool(
                    start_version <= boundary < end_version
                    and sequences == sorted(sequences)
                    and len(set(sequences)) == 4
                    and removed.get("removal_reason") == "stale_evicted"
                )
        elapsed = (
            _integer(completed.get("timestamp_ns"), name="delay completed timestamp")
            - _integer(started.get("timestamp_ns"), name="delay started timestamp")
        ) / 1e9
        requested = _number(started.get("release_delay_seconds"), name="release delay")
        scored[assignment.arm].append(
            (
                float(end_version - start_version),
                direct_chain,
                elapsed - requested if requested > 0 else None,
            )
        )

    summaries: dict[str, dict[str, int | float]] = {}
    for arm in ("control", "d5"):
        arm_rows = scored.get(arm, [])
        if arm_rows:
            summaries[arm] = {
                "assignment_count": len(arm_rows),
                "direct_chain_count": sum(item[1] for item in arm_rows),
                "direct_chain_rate": statistics.mean(item[1] for item in arm_rows),
                "mean_version_advance": statistics.mean(item[0] for item in arm_rows),
            }
    complete = (
        unscored == 0 and bool(assignments) and set(summaries) == {"control", "d5"}
    )
    contrasts = (
        {
            "d5_minus_control_direct_chain": float(summaries["d5"]["direct_chain_rate"])
            - float(summaries["control"]["direct_chain_rate"]),
            "d5_minus_control_version_advance": float(
                summaries["d5"]["mean_version_advance"]
            )
            - float(summaries["control"]["mean_version_advance"]),
        }
        if complete
        else {}
    )
    overshoots = [
        item[2] for rows in scored.values() for item in rows if item[2] is not None
    ]
    compliance = (
        statistics.mean(value >= 0 for value in overshoots) if overshoots else 0.0
    )
    overshoot_p99 = _quantile_99(overshoots) if overshoots else math.inf
    checks = {
        "complete_primary_scoring": complete,
        "control_direct_chain_rate": complete
        and float(summaries["control"]["direct_chain_rate"])
        <= thresholds["maximum_control_direct_chain_rate"],
        "d5_direct_chain_rate": complete
        and float(summaries["d5"]["direct_chain_rate"])
        >= thresholds["minimum_d5_direct_chain_rate"],
        "d5_control_direct_chain_ordering": complete
        and contrasts.get("d5_minus_control_direct_chain", -math.inf)
        >= thresholds["minimum_d5_control_direct_chain_contrast"],
        "d5_control_version_advance_ordering": complete
        and contrasts.get("d5_minus_control_version_advance", -math.inf)
        >= thresholds["minimum_d5_control_version_advance_contrast"],
        "nonzero_delay_compliance": complete
        and compliance >= thresholds["minimum_nonzero_delay_compliance"],
        "delay_overshoot_p99": complete
        and overshoot_p99 <= thresholds["maximum_delay_overshoot_p99_seconds"],
    }
    conclusion = (
        "INSUFFICIENT_MECHANISM_EVIDENCE"
        if not complete
        else "REPLICATED"
        if all(checks.values())
        else "NOT_REPLICATED"
    )
    return MechanismReplicationResult(
        conclusion=conclusion,
        assignment_count=len(assignments),
        unscored_assignment_count=unscored,
        arm_summaries=summaries,
        contrasts=contrasts,
        checks=checks,
    )
