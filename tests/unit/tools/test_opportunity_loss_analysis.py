"""Tests for randomized opportunity-loss terminal-missingness bounds."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.opportunity_loss_analysis import (
    OpportunityAssignment,
    OpportunityLossAnalysisError,
    bound_opportunity_loss,
    load_assignments,
)


def _row(
    assignment_id: str,
    arm: str,
    opportunity: float,
    delivered: bool | None,
) -> OpportunityAssignment:
    return OpportunityAssignment(assignment_id, arm, opportunity, delivered)


def test_complete_data_matches_registered_estimand() -> None:
    result = bound_opportunity_loss(
        [
            _row("c0", "control", 2.0, True),
            _row("c1", "control", 4.0, False),
            _row("t0", "d5", 3.0, False),
            _row("t1", "d5", 5.0, False),
        ]
    )

    # (mean_d5(QD) - mean_control(QD)) / mean_control(Q)
    assert result.complete_data_delta_l == pytest.approx((4.0 - 2.0) / 3.0)
    assert result.delta_l_lower == result.complete_data_delta_l
    assert result.delta_l_upper == result.complete_data_delta_l
    assert result.conclusion == "MATERIAL"


def test_missing_terminal_is_bounded_by_its_observed_opportunity() -> None:
    result = bound_opportunity_loss(
        [
            *(_row(f"c{i}", "control", 2.0, True) for i in range(999)),
            _row("c_missing", "control", 2.0, None),
            *(_row(f"t{i}", "d5", 2.0, False) for i in range(999)),
            _row("t_missing", "d5", 6.0, None),
        ],
        max_missing_fraction=0.001,
    )

    assert result.control.missing_fraction == 0.001
    assert result.treatment.missing_fraction == 0.001
    assert result.complete_data_delta_l is None
    assert result.delta_l_lower == pytest.approx((1.998 - 0.002) / 2.0)
    assert result.delta_l_upper == pytest.approx((2.004 - 0.0) / 2.0)
    assert result.coverage_gate_passed
    assert result.conclusion == "MATERIAL"


def test_bounds_can_make_material_conclusion_inconclusive() -> None:
    result = bound_opportunity_loss(
        [
            _row("c0", "control", 1.0, True),
            _row("c1", "control", 1.0, None),
            _row("t0", "d5", 1.0, False),
            _row("t1", "d5", 1.0, None),
        ],
        material_threshold=0.2,
        max_missing_fraction=0.5,
    )

    assert result.delta_l_lower == 0.0
    assert result.delta_l_upper == 1.0
    assert result.conclusion == "INCONCLUSIVE"


def test_terminal_coverage_gate_is_separate_from_identification_bounds() -> None:
    result = bound_opportunity_loss(
        [
            _row("c0", "control", 1.0, True),
            _row("c1", "control", 1.0, True),
            _row("t0", "d5", 100.0, False),
            _row("t1", "d5", 0.0, None),
        ],
        max_missing_fraction=0.49,
    )

    assert result.delta_l_lower > result.material_threshold
    assert not result.coverage_gate_passed
    assert result.conclusion == "INSUFFICIENT_TERMINAL_COVERAGE"


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        ([_row("x", "control", float("nan"), True)], "finite"),
        ([_row("x", "control", -1.0, True)], "nonnegative"),
        (
            [_row("x", "control", 1.0, True), _row("x", "d5", 1.0, False)],
            "duplicate",
        ),
        (
            [
                _row("c", "control", 0.0, True),
                _row("t", "d5", 1.0, False),
            ],
            "strictly positive",
        ),
    ],
)
def test_invalid_measurement_inputs_fail_closed(
    rows: list[OpportunityAssignment], match: str
) -> None:
    with pytest.raises(OpportunityLossAnalysisError, match=match):
        bound_opportunity_loss(rows)


def test_jsonl_loader_requires_exact_shape_and_preserves_null(tmp_path: Path) -> None:
    path = tmp_path / "assignments.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row, sort_keys=True)
            for row in (
                {
                    "assignment_id": "c",
                    "arm": "control",
                    "opportunity": 1.0,
                    "delivered": True,
                },
                {
                    "assignment_id": "t",
                    "arm": "d5",
                    "opportunity": 2.0,
                    "delivered": None,
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_assignments(path)
    assert loaded[1].delivered is None

    path.write_text(
        json.dumps(
            {
                "assignment_id": "c",
                "arm": "control",
                "opportunity": 1.0,
                "delivered": True,
                "unexpected": 1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(OpportunityLossAnalysisError, match="exactly"):
        load_assignments(path)
