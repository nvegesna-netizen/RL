"""Tests for version-clustered bounded opportunity-loss inference."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tools.opportunity_ledger_join import JoinedOpportunityAssignment
from tools.opportunity_loss_inference import (
    OpportunityLossInferenceError,
    infer_opportunity_loss,
)

_PROPENSITIES = {"control": 5 / 12, "d5": 5 / 12, "d10": 2 / 12}


def _rows(
    *,
    control_delivered: bool = True,
    treatment_delivered: bool = False,
) -> list[JoinedOpportunityAssignment]:
    rows: list[JoinedOpportunityAssignment] = []
    ordinal = 0
    for version in range(16):
        for replicate in range(2):
            opportunity = float(1 + (version + replicate) % 4)
            rows.append(
                JoinedOpportunityAssignment(
                    assignment_id=f"c-{version}-{replicate}",
                    ordinal=ordinal,
                    start_version=version,
                    arm="control",
                    opportunity=opportunity,
                    delivered=control_delivered,
                )
            )
            ordinal += 1
            rows.append(
                JoinedOpportunityAssignment(
                    assignment_id=f"t-{version}-{replicate}",
                    ordinal=ordinal,
                    start_version=version,
                    arm="d5",
                    opportunity=opportunity,
                    delivered=treatment_delivered,
                )
            )
            ordinal += 1
    return rows


def _infer(rows: list[JoinedOpportunityAssignment], **kwargs: object):
    return infer_opportunity_loss(
        rows,
        propensities=_PROPENSITIES,
        primary_start_version=0,
        primary_end_version=15,
        hac_lag=2,
        block_size=4,
        bootstrap_draws=199,
        bootstrap_seed=17,
        **kwargs,
    )


def test_complete_material_effect_has_identical_endpoints() -> None:
    result = _infer(_rows())

    assert result.identification_interval == pytest.approx((1.0, 1.0))
    assert result.lower_endpoint == result.upper_endpoint
    assert result.confidence_envelope == pytest.approx((1.0, 1.0))
    assert result.material_p_value == pytest.approx(1 / 200)
    assert result.conclusion == "MATERIAL"


def test_complete_zero_effect_is_not_material() -> None:
    result = _infer(_rows(treatment_delivered=True))

    assert result.identification_interval == (0.0, 0.0)
    assert result.confidence_envelope == (0.0, 0.0)
    assert result.conclusion == "NOT_MATERIAL"


def test_missing_terminal_widens_point_and_confidence_bounds() -> None:
    rows = _rows()
    treatment_index = next(index for index, row in enumerate(rows) if row.arm == "d5")
    rows[treatment_index] = replace(rows[treatment_index], delivered=None)

    result = _infer(rows, max_missing_fraction=0.04)

    assert result.identification_interval[0] < result.identification_interval[1]
    assert result.lower_endpoint.estimate == result.identification_interval[0]
    assert result.upper_endpoint.estimate == result.identification_interval[1]
    assert result.confidence_envelope[0] <= result.identification_interval[0]
    assert result.confidence_envelope[1] >= result.identification_interval[1]


def test_terminal_count_gate_overrides_material_inference() -> None:
    rows = _rows()
    treatment_index = next(index for index, row in enumerate(rows) if row.arm == "d5")
    rows[treatment_index] = replace(rows[treatment_index], delivered=None)

    result = _infer(rows, max_missing_fraction=0.01)

    assert not result.coverage_gate_passed
    assert result.conclusion == "INSUFFICIENT_TERMINAL_COVERAGE"


def test_bootstrap_and_hac_are_deterministic_and_order_invariant() -> None:
    rows = _rows()

    first = _infer(rows)
    second = _infer(list(reversed(rows)))

    assert first == second


def test_row_outside_registered_cohorts_fails_closed() -> None:
    rows = _rows()
    rows[0] = replace(rows[0], start_version=99)

    with pytest.raises(OpportunityLossInferenceError, match="outside"):
        _infer(rows)


def test_invalid_propensity_and_duplicate_identity_fail_closed() -> None:
    rows = _rows()
    duplicate = [*rows, rows[0]]
    with pytest.raises(OpportunityLossInferenceError, match="unique"):
        _infer(duplicate)

    with pytest.raises(OpportunityLossInferenceError, match="propensity"):
        infer_opportunity_loss(
            rows,
            propensities={"control": 0.0, "d5": 1.0},
            primary_start_version=0,
            primary_end_version=15,
            bootstrap_draws=10,
        )
