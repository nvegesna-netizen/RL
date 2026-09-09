# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Fail-closed lifecycle checks for neutral qualification runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def assess_neutral_qualification_lifecycle(
    lifecycle: Sequence[Mapping[str, object]],
    *,
    opportunity_group_ids: set[str],
) -> dict[str, object]:
    """Validate zero-dose lifecycle records with bounded terminal censoring.

    An opportunity is recorded synchronously before its release hold begins. At
    a bounded step-count shutdown, that neutral zero-second hold may therefore
    have started without publishing its completion event. Such a terminal group
    is accepted only when its final event explicitly records
    ``bounded_shutdown`` after the start. Unexplained missing events and every
    non-neutral assignment remain failures.
    """
    release_stages = (
        "release_delay_assigned",
        "release_delay_started",
        "release_delay_completed",
    )
    by_stage: dict[str, list[Mapping[str, object]]] = {
        stage: [row for row in lifecycle if row.get("stage") == stage]
        for stage in release_stages
    }

    def group_ids(stage: str) -> set[str]:
        values = [row.get("group_id") for row in by_stage[stage]]
        return {value for value in values if isinstance(value, str) and value}

    def is_neutral(row: Mapping[str, object]) -> bool:
        return (
            row.get("release_arm") == "neutral"
            and row.get("release_delay_seconds") == 0.0
            and row.get("release_arm_mass") == 1
            and row.get("release_total_mass") == 1
        )

    assigned = group_ids("release_delay_assigned")
    started = group_ids("release_delay_started")
    completed = group_ids("release_delay_completed")
    unique_stage_ids = all(
        len(by_stage[stage]) == len(group_ids(stage)) for stage in release_stages
    )
    all_release_fields_neutral = bool(assigned) and all(
        is_neutral(row) for stage in release_stages for row in by_stage[stage]
    )

    removals: dict[str, list[Mapping[str, object]]] = {}
    for row in lifecycle:
        if row.get("stage") != "removed":
            continue
        group_id = row.get("group_id")
        if isinstance(group_id, str) and group_id:
            removals.setdefault(group_id, []).append(row)

    def terminally_censored(group_id: str, prior_stage: str) -> bool:
        rows = removals.get(group_id, [])
        prior = [
            row for row in by_stage[prior_stage] if row.get("group_id") == group_id
        ]
        if len(rows) != 1 or len(prior) != 1:
            return False
        removal_sequence = rows[0].get("controller_sequence")
        prior_sequence = prior[0].get("controller_sequence")
        return (
            rows[0].get("removal_reason") == "bounded_shutdown"
            and isinstance(removal_sequence, int)
            and not isinstance(removal_sequence, bool)
            and isinstance(prior_sequence, int)
            and not isinstance(prior_sequence, bool)
            and removal_sequence > prior_sequence
        )

    assigned_not_started = assigned - started
    started_not_completed = started - completed
    bounded_before_start = all(
        terminally_censored(group_id, "release_delay_assigned")
        for group_id in assigned_not_started
    )
    bounded_during_release = all(
        terminally_censored(group_id, "release_delay_started")
        for group_id in started_not_completed
    )
    topology = (
        opportunity_group_ids == started
        and completed <= started <= assigned
        and bounded_before_start
        and bounded_during_release
    )
    supported = unique_stage_ids and all_release_fields_neutral and topology
    return {
        "supported": supported,
        "all_release_fields_neutral": all_release_fields_neutral,
        "unique_stage_group_ids": unique_stage_ids,
        "opportunity_groups_equal_started_groups": opportunity_group_ids == started,
        "completed_groups_subset_started_groups": completed <= started,
        "started_groups_subset_assigned_groups": started <= assigned,
        "assigned_group_count": len(assigned),
        "started_group_count": len(started),
        "completed_group_count": len(completed),
        "bounded_shutdown_before_start_count": len(assigned_not_started),
        "bounded_shutdown_during_release_count": len(started_not_completed),
        "bounded_shutdown_before_start_valid": bounded_before_start,
        "bounded_shutdown_during_release_valid": bounded_during_release,
    }
