# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the frozen two-cell M4 family-transport contrast."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tools.opportunity_loss_family_transport_analysis import (
    FamilyTransportAnalysisError,
    FamilyTransportCellInference,
    infer_family_transport,
)


def _cells(*, draws: int = 20_000):
    return {
        "llama3p2_1b_openmath": FamilyTransportCellInference(
            estimate=0.30,
            hac_standard_error=0.01,
            bootstrap_shifts=tuple(0.0 for _ in range(draws)),
            confidence_envelope=(0.28, 0.32),
            materiality_conclusion="MATERIAL",
        ),
        "llama3p2_1b_gsm8k": FamilyTransportCellInference(
            estimate=0.18,
            hac_standard_error=0.01,
            bootstrap_shifts=tuple(0.0 for _ in range(draws)),
            confidence_envelope=(0.16, 0.20),
            materiality_conclusion="NOT_MATERIAL",
        ),
    }


def test_same_direction_does_not_require_gsm8k_materiality() -> None:
    result = infer_family_transport(_cells())
    assert result.openmath_family_transport == "MATERIAL"
    assert result.gsm8k_effect_direction == "POSITIVE"
    assert result.workload_contrast_estimate == pytest.approx(0.12)
    assert result.workload_pattern_conclusion == "SAME_DIRECTION"
    assert result.cell_materiality["llama3p2_1b_gsm8k"] == "NOT_MATERIAL"


def test_inconclusive_when_outer_interval_crosses_zero() -> None:
    cells = _cells()
    cells["llama3p2_1b_openmath"] = replace(
        cells["llama3p2_1b_openmath"], estimate=0.185
    )
    result = infer_family_transport(cells)
    assert result.workload_pattern_conclusion == "INCONCLUSIVE"


def test_incomplete_or_wrong_window_cell_is_rejected() -> None:
    cells = _cells()
    cells["llama3p2_1b_gsm8k"] = replace(
        cells["llama3p2_1b_gsm8k"], terminal_scoring_complete=False
    )
    with pytest.raises(FamilyTransportAnalysisError, match="cell contract"):
        infer_family_transport(cells)


def test_wrong_identity_or_draw_count_is_rejected() -> None:
    cells = _cells(draws=10)
    with pytest.raises(FamilyTransportAnalysisError, match="draws"):
        infer_family_transport(cells)
    cells = _cells()
    cells["other"] = cells.pop("llama3p2_1b_gsm8k")
    with pytest.raises(FamilyTransportAnalysisError, match="identities"):
        infer_family_transport(cells)


def test_materiality_label_must_match_registered_envelope_rule() -> None:
    cells = _cells()
    cells["llama3p2_1b_openmath"] = replace(
        cells["llama3p2_1b_openmath"], materiality_conclusion="NOT_MATERIAL"
    )
    with pytest.raises(FamilyTransportAnalysisError, match="materiality label"):
        infer_family_transport(cells)
