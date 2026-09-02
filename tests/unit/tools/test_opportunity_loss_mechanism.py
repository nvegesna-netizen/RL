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

from __future__ import annotations

import copy

import pytest

from tools.opportunity_ledger_join import JoinedOpportunityAssignment
from tools.opportunity_loss_mechanism import (
    OpportunityLossMechanismError,
    assess_mechanism_replication,
)


def _contract() -> dict[str, object]:
    return {
        "accepted_predecessor_commit": "a6be6971b79ad0d6b48c0dc85bb7f09682c2f7dd",
        "role": "support_condition_not_primary_endpoint",
        "score_window": "primary_start_versions",
        "thresholds": {
            "maximum_control_direct_chain_rate": 0.02,
            "maximum_delay_overshoot_p99_seconds": 2.0,
            "minimum_d10_d5_direct_chain_contrast": 0.2,
            "minimum_d10_d5_version_advance_contrast": 0.25,
            "minimum_d10_direct_chain_rate": 0.4,
            "minimum_d5_control_direct_chain_contrast": 0.1,
            "minimum_d5_control_version_advance_contrast": 0.2,
            "minimum_d5_direct_chain_rate": 0.1,
            "minimum_nonzero_delay_compliance": 0.99,
        },
    }


def _fixture() -> tuple[list[dict[str, object]], list[JoinedOpportunityAssignment]]:
    lifecycle: list[dict[str, object]] = []
    assignments: list[JoinedOpportunityAssignment] = []
    sequence = 0
    ordinal = 0
    for arm, delay, direct_count in (("control", 0, 0), ("d5", 5, 20), ("d10", 10, 50)):
        for index in range(100):
            group_id = f"{arm}-{index}"
            direct = index < direct_count
            start_version = 0
            end_version = 2 if direct else (0 if arm == "control" else 1)
            start_ns = ordinal * 20_000_000_000
            stages = [
                {
                    "group_id": group_id,
                    "stage": "release_delay_started",
                    "controller_sequence": sequence,
                    "timestamp_ns": start_ns,
                    "learner_weight_version": start_version,
                    "release_delay_seconds": delay,
                },
                {
                    "group_id": group_id,
                    "stage": "release_delay_completed",
                    "controller_sequence": sequence + 1,
                    "timestamp_ns": start_ns + delay * 1_000_000_000,
                    "learner_weight_version": end_version,
                },
                {
                    "group_id": group_id,
                    "stage": "group_completed",
                    "controller_sequence": sequence + 2,
                },
                {
                    "group_id": group_id,
                    "stage": "group_ready",
                    "controller_sequence": sequence + 3,
                },
                {
                    "group_id": group_id,
                    "stage": "removed",
                    "controller_sequence": sequence + 4,
                    "removal_reason": "stale_evicted" if direct else "selected",
                },
            ]
            lifecycle.extend(stages)
            assignments.append(
                JoinedOpportunityAssignment(group_id, ordinal, 0, arm, 1.0, not direct)
            )
            sequence += len(stages)
            ordinal += 1
    return lifecycle, assignments


def test_registered_m4_mechanism_replication_is_green() -> None:
    lifecycle, assignments = _fixture()

    result = assess_mechanism_replication(
        lifecycle_rows=lifecycle,
        assignments=assignments,
        contract=_contract(),
        max_staleness_versions=1,
    )

    assert result.conclusion == "REPLICATED"
    assert result.arm_summaries["control"]["direct_chain_rate"] == 0.0
    assert result.arm_summaries["d5"]["direct_chain_rate"] == 0.2
    assert result.arm_summaries["d10"]["direct_chain_rate"] == 0.5
    assert all(result.checks.values())


def test_broken_dose_order_is_not_replicated() -> None:
    lifecycle, assignments = _fixture()
    for row in lifecycle:
        if row["group_id"].startswith("d10-") and row["stage"] == "removed":
            row["removal_reason"] = "selected"

    result = assess_mechanism_replication(
        lifecycle_rows=lifecycle,
        assignments=assignments,
        contract=_contract(),
        max_staleness_versions=1,
    )

    assert result.conclusion == "NOT_REPLICATED"
    assert result.checks["d10_direct_chain_rate"] is False


def test_missing_primary_delay_epoch_is_insufficient() -> None:
    lifecycle, assignments = _fixture()
    lifecycle = [
        row
        for row in lifecycle
        if not (row["group_id"] == "d5-0" and row["stage"] == "release_delay_started")
    ]

    result = assess_mechanism_replication(
        lifecycle_rows=lifecycle,
        assignments=assignments,
        contract=_contract(),
        max_staleness_versions=1,
    )

    assert result.conclusion == "INSUFFICIENT_MECHANISM_EVIDENCE"
    assert result.unscored_assignment_count == 1


def test_contract_and_duplicate_stage_mutations_fail_closed() -> None:
    lifecycle, assignments = _fixture()
    contract = copy.deepcopy(_contract())
    contract["thresholds"]["minimum_d5_direct_chain_rate"] = 0.09
    with pytest.raises(OpportunityLossMechanismError, match="threshold"):
        assess_mechanism_replication(
            lifecycle_rows=lifecycle,
            assignments=assignments,
            contract=contract,
            max_staleness_versions=1,
        )

    lifecycle.append(copy.deepcopy(lifecycle[0]))
    with pytest.raises(OpportunityLossMechanismError, match="duplicate"):
        assess_mechanism_replication(
            lifecycle_rows=lifecycle,
            assignments=assignments,
            contract=_contract(),
            max_staleness_versions=1,
        )
