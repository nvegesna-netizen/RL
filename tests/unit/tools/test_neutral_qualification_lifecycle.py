"""Tests for neutral qualification terminal-censoring validation."""

from __future__ import annotations

import pytest

from tools.neutral_qualification_lifecycle import (
    assess_neutral_qualification_lifecycle,
)


def _release_event(
    group_id: str,
    stage: str,
    sequence: int,
    *,
    removal_reason: str | None = None,
) -> dict[str, object]:
    event: dict[str, object] = {
        "group_id": group_id,
        "stage": stage,
        "controller_sequence": sequence,
        "release_arm": None,
        "release_delay_seconds": None,
        "release_arm_mass": None,
        "release_total_mass": None,
        "removal_reason": removal_reason,
    }
    if stage.startswith("release_delay_"):
        event.update(
            {
                "release_arm": "neutral",
                "release_delay_seconds": 0.0,
                "release_arm_mass": 1,
                "release_total_mass": 1,
            }
        )
    return event


def test_accepts_only_explicit_bounded_terminal_censoring() -> None:
    lifecycle = [
        _release_event("complete", "release_delay_assigned", 1),
        _release_event("complete", "release_delay_started", 2),
        _release_event("complete", "release_delay_completed", 3),
        _release_event("terminal", "release_delay_assigned", 4),
        _release_event("terminal", "release_delay_started", 5),
        _release_event("terminal", "removed", 6, removal_reason="bounded_shutdown"),
        _release_event("never-started", "release_delay_assigned", 7),
        _release_event(
            "never-started", "removed", 8, removal_reason="bounded_shutdown"
        ),
    ]

    result = assess_neutral_qualification_lifecycle(
        lifecycle,
        opportunity_group_ids={"complete", "terminal"},
    )

    assert result["supported"] is True
    assert result["bounded_shutdown_before_start_count"] == 1
    assert result["bounded_shutdown_during_release_count"] == 1


@pytest.mark.parametrize(
    ("mutation", "opportunity_group_ids"),
    [
        ("nonzero_delay", {"group"}),
        ("unexplained_missing_completion", {"group"}),
        ("wrong_terminal_reason", {"group"}),
        ("opportunity_without_start", {"group", "unknown"}),
    ],
)
def test_rejects_non_neutral_or_unexplained_censoring(
    mutation: str, opportunity_group_ids: set[str]
) -> None:
    lifecycle = [
        _release_event("group", "release_delay_assigned", 1),
        _release_event("group", "release_delay_started", 2),
        _release_event("group", "removed", 3, removal_reason="bounded_shutdown"),
    ]
    if mutation == "nonzero_delay":
        lifecycle[1]["release_delay_seconds"] = 5.0
    elif mutation == "unexplained_missing_completion":
        lifecycle.pop()
    elif mutation == "wrong_terminal_reason":
        lifecycle[-1]["removal_reason"] = "failed"

    result = assess_neutral_qualification_lifecycle(
        lifecycle,
        opportunity_group_ids=opportunity_group_ids,
    )

    assert result["supported"] is False
