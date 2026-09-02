"""Tests for prospective cross-fitted opportunity-loss adjustment."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from tools.opportunity_ledger_join import JoinedOpportunityAssignment
from tools.opportunity_loss_adjusted_inference import (
    AdjustedOpportunityLossError,
    infer_adjusted_opportunity_loss,
)

_PROPENSITIES = {"control": 0.5, "d5": 0.5}


def _rows() -> list[JoinedOpportunityAssignment]:
    rows: list[JoinedOpportunityAssignment] = []
    ordinal = 0
    for version in range(16):
        for replicate in range(2):
            opportunity = float((version + replicate) % 5)
            delivered = (version + replicate) % 3 == 0
            for arm in ("control", "d5"):
                rows.append(
                    JoinedOpportunityAssignment(
                        assignment_id=f"{arm}-{version}-{replicate}",
                        ordinal=ordinal,
                        start_version=version,
                        arm=arm,
                        opportunity=opportunity,
                        delivered=delivered,
                    )
                )
                ordinal += 1
    return rows


def _infer(rows: list[JoinedOpportunityAssignment]):
    return infer_adjusted_opportunity_loss(
        rows,
        propensities=_PROPENSITIES,
        primary_start_version=0,
        primary_end_version=15,
        folds=4,
        hac_lag=2,
    )


def test_matched_arm_outcomes_have_zero_adjusted_effect() -> None:
    result = _infer(_rows())

    assert result.lower_endpoint == result.upper_endpoint
    assert result.lower_endpoint.estimate == pytest.approx(0.0, abs=1e-12)
    assert result.lower_endpoint.hac_standard_error == pytest.approx(0.0, abs=1e-12)
    assert result.covariates == ("opportunity_Q", "opportunity_is_zero")
    assert result.denominator == "pooled_pre_delay_mean_opportunity"


def test_complete_treatment_loss_is_positive() -> None:
    rows = [
        replace(row, delivered=False if row.arm == "d5" else True) for row in _rows()
    ]

    result = _infer(rows)

    assert result.lower_endpoint == result.upper_endpoint
    assert result.lower_endpoint.estimate == pytest.approx(1.0)


def test_missing_terminal_preserves_sharp_endpoint_order() -> None:
    rows = _rows()
    control = next(
        index
        for index, row in enumerate(rows)
        if row.arm == "control" and row.opportunity > 0.0
    )
    treatment = next(
        index
        for index, row in enumerate(rows)
        if row.arm == "d5" and row.opportunity > 0.0
    )
    rows[control] = replace(rows[control], delivered=None)
    rows[treatment] = replace(rows[treatment], delivered=None)

    result = _infer(rows)

    assert result.lower_endpoint.estimate < result.upper_endpoint.estimate


def test_row_order_does_not_change_result() -> None:
    forward = _infer(_rows())
    reverse = _infer(list(reversed(_rows())))

    assert reverse.lower_endpoint.estimate == pytest.approx(
        forward.lower_endpoint.estimate, abs=1e-12
    )
    assert reverse.lower_endpoint.hac_standard_error == pytest.approx(
        forward.lower_endpoint.hac_standard_error, abs=1e-12
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows + [rows[0]],
        lambda rows: [replace(rows[0], start_version=99), *rows[1:]],
    ],
)
def test_invalid_rows_fail_closed(mutation) -> None:
    with pytest.raises(AdjustedOpportunityLossError):
        _infer(mutation(_rows()))


def test_invalid_propensity_fails_closed() -> None:
    with pytest.raises(AdjustedOpportunityLossError, match="propensity"):
        infer_adjusted_opportunity_loss(
            _rows(),
            propensities={"control": 0.5, "d5": 0.0},
            primary_start_version=0,
            primary_end_version=15,
            folds=4,
            hac_lag=2,
        )


def test_sharp_null_randomization_has_bounded_false_positive_rate() -> None:
    rng = random.Random(20260902)
    rejections = 0
    draws = 200
    for draw in range(draws):
        rows = []
        ordinal = 0
        for version in range(32):
            treated = set(rng.sample(range(8), 4))
            for unit in range(8):
                opportunity = float((version * 3 + unit * 5) % 11)
                delivered = (version + unit * 2) % 5 == 0
                rows.append(
                    JoinedOpportunityAssignment(
                        assignment_id=f"{draw}-{version}-{unit}",
                        ordinal=ordinal,
                        start_version=version,
                        arm="d5" if unit in treated else "control",
                        opportunity=opportunity,
                        delivered=delivered,
                    )
                )
                ordinal += 1
        result = infer_adjusted_opportunity_loss(
            rows,
            propensities=_PROPENSITIES,
            primary_start_version=0,
            primary_end_version=31,
            folds=4,
            hac_lag=2,
        )
        endpoint = result.lower_endpoint
        if endpoint.hac_standard_error > 0.0:
            rejections += endpoint.estimate / endpoint.hac_standard_error > 1.644854

    assert rejections / draws <= 0.10
