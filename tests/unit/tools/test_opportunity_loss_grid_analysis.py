# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import math

import pytest

from tools.opportunity_ledger_join import JoinedOpportunityAssignment
from tools.opportunity_loss_grid_analysis import (
    CELL_NAMES,
    GridCellInference,
    OpportunityLossGridAnalysisError,
    infer_common_window_cell,
    infer_grid_interaction,
)


def _cells(*, shifts: int = 5) -> dict[str, GridCellInference]:
    estimates = {
        "qwen3_0p6b_openmath": 0.30,
        "qwen3_0p6b_gsm8k": 0.10,
        "qwen3_1p7b_openmath": 0.30,
        "qwen3_1p7b_gsm8k": 0.15,
    }
    return {
        name: GridCellInference(
            estimate=estimates[name],
            hac_standard_error=0.01,
            bootstrap_shifts=tuple(0.0 for _ in range(shifts)),
        )
        for name in CELL_NAMES
    }


def _common_window_rows() -> list[JoinedOpportunityAssignment]:
    rows = []
    ordinal = 0
    for version in range(8, 408):
        for arm, delivered in (("control", True), ("d5", version % 3 == 0)):
            opportunity = 0.0 if version % 5 == 0 else float(1 + version % 7)
            rows.append(
                JoinedOpportunityAssignment(
                    assignment_id=f"{version}-{arm}",
                    ordinal=ordinal,
                    start_version=version,
                    arm=arm,
                    opportunity=opportunity,
                    delivered=delivered,
                )
            )
            ordinal += 1
    return rows


def test_common_window_cell_is_fixed_and_reproducible() -> None:
    rows = _common_window_rows()
    first = infer_common_window_cell(rows, bootstrap_seed=91, bootstrap_draws=7)
    second = infer_common_window_cell(rows, bootstrap_seed=91, bootstrap_draws=7)
    assert first == second
    assert first.primary_start_version == 8
    assert first.primary_end_version == 407
    assert len(first.bootstrap_shifts) == 7


def test_common_window_cell_requires_complete_scoring() -> None:
    rows = _common_window_rows()
    row = rows[0]
    rows[0] = JoinedOpportunityAssignment(
        assignment_id=row.assignment_id,
        ordinal=row.ordinal,
        start_version=row.start_version,
        arm=row.arm,
        opportunity=row.opportunity,
        delivered=None,
    )
    with pytest.raises(OpportunityLossGridAnalysisError, match="completely scored"):
        infer_common_window_cell(rows, bootstrap_seed=91, bootstrap_draws=7)


def test_grid_interaction_combines_independent_cells() -> None:
    result = infer_grid_interaction(_cells(), expected_bootstrap_draws=5)
    assert result.workload_contrasts == {
        "qwen3_0p6b_openmath_minus_gsm8k": pytest.approx(0.20),
        "qwen3_1p7b_openmath_minus_gsm8k": pytest.approx(0.15),
    }
    assert result.model_scale_contrasts == {
        "openmath_qwen3_1p7b_minus_0p6b": pytest.approx(0.0),
        "gsm8k_qwen3_1p7b_minus_0p6b": pytest.approx(0.05),
    }
    assert result.interaction_estimate == pytest.approx(0.05)
    assert result.hac_standard_error == pytest.approx(math.sqrt(4 * 0.01**2))
    assert result.bootstrap_interval == pytest.approx((0.05, 0.05))
    assert result.conclusion == "INTERACTION_DETECTED"


def test_grid_interaction_is_secondary_and_can_be_inconclusive() -> None:
    cells = _cells()
    cells["qwen3_0p6b_gsm8k"] = GridCellInference(
        estimate=0.15,
        hac_standard_error=0.03,
        bootstrap_shifts=(-0.08, -0.04, 0.0, 0.04, 0.08),
    )
    result = infer_grid_interaction(cells, expected_bootstrap_draws=5)
    assert result.confidence_envelope[0] < 0.0 < result.confidence_envelope[1]
    assert result.conclusion == "INTERACTION_INCONCLUSIVE"
    assert result.role == "secondary_grid_synthesis_cannot_override_primary_cell"


@pytest.mark.parametrize(
    "mutation",
    ["missing_cell", "wrong_window", "wrong_draw_count", "negative_standard_error"],
)
def test_grid_interaction_rejects_contract_mutations(mutation: str) -> None:
    cells = _cells()
    if mutation == "missing_cell":
        del cells["qwen3_0p6b_gsm8k"]
    elif mutation == "wrong_window":
        cells["qwen3_0p6b_gsm8k"] = GridCellInference(
            estimate=0.1,
            hac_standard_error=0.01,
            bootstrap_shifts=(0.0,) * 5,
            primary_end_version=507,
        )
    elif mutation == "wrong_draw_count":
        cells["qwen3_0p6b_gsm8k"] = GridCellInference(
            estimate=0.1,
            hac_standard_error=0.01,
            bootstrap_shifts=(0.0,) * 4,
        )
    else:
        cells["qwen3_0p6b_gsm8k"] = GridCellInference(
            estimate=0.1,
            hac_standard_error=-0.01,
            bootstrap_shifts=(0.0,) * 5,
        )
    with pytest.raises(OpportunityLossGridAnalysisError):
        infer_grid_interaction(cells, expected_bootstrap_draws=5)
