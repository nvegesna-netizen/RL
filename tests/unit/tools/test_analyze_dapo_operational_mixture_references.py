import pytest

from tools import analyze_dapo_operational_mixture_references as analyzer


def test_pearson_and_harder_half_use_locked_ordinal_ties() -> None:
    values = [float(index) for index in range(32)]

    assert analyzer._pearson(values, values) == pytest.approx(1.0)
    assert analyzer._pearson(values, list(reversed(values))) == pytest.approx(-1.0)
    assert analyzer._harder_ordinals([0.0] * 32) == tuple(range(16))
