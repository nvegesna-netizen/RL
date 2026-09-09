# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import math
import unittest

from tools.opportunity_loss_grid_analysis import GridCellInference
from tools.opportunity_loss_six_cell_synthesis import (
    SixCellSynthesisError,
    infer_six_cell_synthesis,
)


def _cell(estimate: float, se: float, shifts: tuple[float, ...]) -> GridCellInference:
    return GridCellInference(
        estimate=estimate,
        hac_standard_error=se,
        bootstrap_shifts=shifts,
    )


def _cells() -> dict[str, GridCellInference]:
    shifts = {
        "qwen3_0p6b_openmath": (-0.02, -0.01, 0.01, 0.02),
        "qwen3_1p7b_openmath": (-0.01, 0.01, -0.02, 0.02),
        "qwen3_0p6b_gsm8k": (-0.03, -0.01, 0.01, 0.03),
        "qwen3_1p7b_gsm8k": (-0.02, 0.0, 0.01, 0.01),
        "qwen3_0p6b_numinamath": (-0.01, -0.02, 0.01, 0.02),
        "qwen3_1p7b_numinamath": (-0.02, -0.01, 0.0, 0.03),
    }
    estimates = {
        "qwen3_0p6b_openmath": 0.30,
        "qwen3_1p7b_openmath": 0.29,
        "qwen3_0p6b_gsm8k": 0.23,
        "qwen3_1p7b_gsm8k": 0.14,
        "qwen3_0p6b_numinamath": 0.31,
        "qwen3_1p7b_numinamath": 0.30,
    }
    return {name: _cell(estimates[name], 0.01, shifts[name]) for name in estimates}


class TestSixCellSynthesis(unittest.TestCase):
    def test_shared_gsm_cells_induce_interaction_covariance(self) -> None:
        result = infer_six_cell_synthesis(_cells(), expected_bootstrap_draws=4)
        expected = {"openmath": -0.01, "gsm8k": -0.09, "numinamath": -0.01}
        for name, value in expected.items():
            self.assertAlmostEqual(result.model_scale_effects[name], value)
        self.assertAlmostEqual(
            result.reference_interactions["gsm8k_minus_openmath"].estimate, -0.08
        )
        self.assertAlmostEqual(
            result.reference_interactions["gsm8k_minus_numinamath"].estimate, -0.08
        )
        self.assertAlmostEqual(result.hac_covariance[0][1], 2 * 0.01**2)
        self.assertAlmostEqual(result.hac_correlation, 0.5)
        self.assertNotEqual(result.bootstrap_covariance[0][1], 0.0)
        self.assertTrue(math.isfinite(result.global_heterogeneity_p_value))

    def test_requires_exactly_six_cells(self) -> None:
        cells = _cells()
        cells.pop("qwen3_1p7b_numinamath")
        with self.assertRaisesRegex(SixCellSynthesisError, "identities"):
            infer_six_cell_synthesis(cells, expected_bootstrap_draws=4)

    def test_rejects_mismatched_draw_count(self) -> None:
        cells = _cells()
        cells["qwen3_1p7b_numinamath"] = _cell(0.3, 0.01, (0.0,))
        with self.assertRaisesRegex(SixCellSynthesisError, "violates"):
            infer_six_cell_synthesis(cells, expected_bootstrap_draws=4)


if __name__ == "__main__":
    unittest.main()
