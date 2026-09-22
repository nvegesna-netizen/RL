#!/usr/bin/env python3
"""Dependency-light unit tests for the OARS-v2 offline autopsy."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("analyze_offline_autopsy.py")
SPEC = importlib.util.spec_from_file_location("analyze_offline_autopsy", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def candidate(group_id: str, start: int, ready: int, l1: float, l2: float, tokens: int):
    return MODULE.Candidate(group_id, start, ready, l1, l2, tokens, 0.5, 0.25)


def synthetic_decision() -> object:
    candidates = tuple(
        candidate(
            f"g{index}",
            0 if index < 4 else 1,
            100 + index,
            float(index + 1),
            float((index + 1) * 2),
            100,
        )
        for index in range(8)
    )
    return MODULE.Decision(
        identity="synthetic",
        index=0,
        current_learner_version=1,
        timestamp_ns=1000,
        candidates=candidates,
        baseline_group_ids=("g0", "g1", "g2", "g3"),
        proposal_group_ids=("g4", "g5", "g6", "g7"),
        actual_group_ids=("g0", "g1", "g2", "g3"),
        baseline_tokens=400,
    )


def test_two_sided_band_and_risk_priority() -> None:
    decision = synthetic_decision()
    selection = MODULE.policy_selection(decision, "m4_risk_band")
    assert {row.group_id for row in selection} == {"g0", "g1", "g2", "g3"}
    assert sum(row.valid_actor_tokens for row in selection) == 400


def test_total_policy_differs_from_expiry_policy() -> None:
    decision = synthetic_decision()
    selection = MODULE.policy_selection(decision, "m4_total_band")
    assert {row.group_id for row in selection} == {"g4", "g5", "g6", "g7"}


def test_shuffled_score_is_evaluated_with_true_values() -> None:
    decision = synthetic_decision()
    shuffled = {f"g{index}": float(8 - index) for index in range(8)}
    selection = MODULE.policy_selection(
        decision,
        "shuffled_m4_risk_band",
        shuffled_l1=shuffled,
    )
    assert {row.group_id for row in selection} == {"g0", "g1", "g2", "g3"}
    assert math.fsum(row.l1 for row in selection) == 10.0


def test_quantile_interpolates() -> None:
    assert MODULE.quantile([0.0, 10.0], 0.25) == 2.5


def main() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
    print(f"PASS {len(tests)} tests")


if __name__ == "__main__":
    main()
