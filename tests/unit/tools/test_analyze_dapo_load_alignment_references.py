import pytest

from tools import analyze_dapo_load_alignment_references as analyzer


def test_reference_statistics_and_locked_ordinal_ties() -> None:
    values = [float(index) for index in range(32)]

    assert analyzer._pearson(values, values) == pytest.approx(1.0)
    assert analyzer._spearman(values, list(reversed(values))) == pytest.approx(-1.0)
    assert analyzer._lower_ordinals([0.0] * 32) == tuple(range(16))
