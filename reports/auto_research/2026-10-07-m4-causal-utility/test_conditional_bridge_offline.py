from __future__ import annotations

import numpy as np

from analyze_conditional_bridge_offline import (
    predict_gradient_norm,
    select_conditional_batch,
)


COEFFICIENTS = np.asarray([-1.0, 0.0, 0.0, 1.0], dtype=np.float64)


def candidate(group_id: str, *, m4: float, tokens: int, variance: float) -> dict:
    return {
        "group_id": group_id,
        "l1": m4,
        "l2": m4,
        "valid_actor_tokens": tokens,
        "reward_mean": 0.5,
        "reward_variance": variance,
        "start_weight_version": 0,
    }


def decision(candidates: list[dict], base_ids: list[str]) -> dict:
    base_tokens = sum(
        row["valid_actor_tokens"] for row in candidates if row["group_id"] in base_ids
    )
    return {
        "candidates": candidates,
        "proposals": {
            "reward_variance_risk": {
                "proposed_group_ids": base_ids,
                "minimum_tokens": int(base_tokens * 0.98),
                "maximum_tokens": int(base_tokens * 1.02),
            }
        },
    }


def test_zero_m4_has_zero_predicted_gradient() -> None:
    row = candidate("zero", m4=0.0, tokens=100, variance=0.25)
    assert predict_gradient_norm(row, COEFFICIENTS, 0.0) == 0.0


def test_conditional_policy_requires_strict_gain_and_preserves_variance() -> None:
    rows = [
        candidate("a", m4=10.0, tokens=100, variance=0.25),
        candidate("b", m4=10.0, tokens=100, variance=0.25),
        candidate("c", m4=10.0, tokens=100, variance=0.25),
        candidate("d", m4=10.0, tokens=100, variance=0.25),
        candidate("e", m4=30.0, tokens=100, variance=0.25),
        candidate("f", m4=30.0, tokens=100, variance=0.25),
        candidate("g", m4=100.0, tokens=100, variance=0.0),
        candidate("h", m4=0.0, tokens=100, variance=0.5),
    ]
    result = select_conditional_batch(
        decision(rows, ["a", "b", "c", "d"]),
        COEFFICIENTS,
        0.0,
        tolerance=1e-12,
    )
    assert result["strict_intervention"] is True
    assert {"e", "f"}.issubset(result["selected_group_ids"])
    assert result["reward_variance_retention"] >= 1.0
    assert result["token_ratio"] == 1.0


def test_no_strict_gain_falls_back_to_base_identity() -> None:
    rows = [
        candidate(letter, m4=10.0, tokens=100, variance=0.25) for letter in "abcdefgh"
    ]
    base = ["a", "b", "c", "d"]
    result = select_conditional_batch(
        decision(rows, base), COEFFICIENTS, 0.0, tolerance=1e-12
    )
    assert result["strict_intervention"] is False
    assert result["selected_group_ids"] == base
    assert result["relative_predicted_norm_gain"] == 0.0
