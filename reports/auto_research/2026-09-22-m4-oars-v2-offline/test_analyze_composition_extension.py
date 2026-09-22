#!/usr/bin/env python3
"""Dependency-light tests for OARS-v2 composition diagnostics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("analyze_composition_extension.py")
SPEC = importlib.util.spec_from_file_location("analyze_composition_extension", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def synthetic_decision() -> object:
    candidates = []
    for index in range(8):
        candidates.append(
            MODULE.PARENT.Candidate(
                group_id=f"g{index}",
                start_weight_version=0 if index < 4 else 1,
                ready_timestamp_ns=index,
                l1=float(index + 1),
                l2=float(index + 1),
                valid_actor_tokens=100,
                mean_reward=0.5,
                reward_variance=float(8 - index),
            )
        )
    return MODULE.PARENT.Decision(
        identity="synthetic",
        index=0,
        current_learner_version=1,
        timestamp_ns=1000,
        candidates=tuple(candidates),
        baseline_group_ids=("g0", "g1", "g2", "g3"),
        proposal_group_ids=("g0", "g1", "g2", "g3"),
        actual_group_ids=("g0", "g1", "g2", "g3"),
        baseline_tokens=400,
    )


def test_reward_variance_policy_uses_imminent_risk_first() -> None:
    selected = MODULE.selection(synthetic_decision(), "reward_variance_risk_band")
    assert {candidate.group_id for candidate in selected} == {"g0", "g1", "g2", "g3"}


def test_policy_comparison_reports_exact_overlap() -> None:
    result = MODULE.compare_selections(
        [synthetic_decision()], "m4_risk_band", "reward_variance_risk_band"
    )
    assert result == {"mean_overlap_fraction": 1.0, "identical_selection_fraction": 1.0}


def main() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
    print(f"PASS {len(tests)} tests")


if __name__ == "__main__":
    main()
