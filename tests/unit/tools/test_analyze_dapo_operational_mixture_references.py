import pytest

from nemo_rl.algorithms.async_utils.fixed_pool import FIXED_POOL_RUN_MODE
from tools import analyze_dapo_operational_mixture_references as analyzer


def test_pearson_and_harder_half_use_locked_ordinal_ties() -> None:
    values = [float(index) for index in range(32)]

    assert analyzer._pearson(values, values) == pytest.approx(1.0)
    assert analyzer._pearson(values, list(reversed(values))) == pytest.approx(-1.0)
    assert analyzer._harder_ordinals([0.0] * 32) == tuple(range(16))


def test_reference_analyzer_uses_the_fixed_pool_trace_run_mode() -> None:
    assert analyzer.FIXED_POOL_RUN_MODE == FIXED_POOL_RUN_MODE == "fixed_pool"
