#!/usr/bin/env python3
"""Dependency-light tests for the baseline-budgeted OARS repair."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


G_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(G_HERE))
import run_offline_gate as G_BASE  # noqa: E402

G_SPEC = importlib.util.spec_from_file_location(
    "m4_opportunity_control_service_budget_repair",
    G_HERE / "run_service_budget_repair.py",
)
if G_SPEC is None or G_SPEC.loader is None:
    raise RuntimeError("cannot load service-budget repair module")
G_MODULE = importlib.util.module_from_spec(G_SPEC)
sys.modules[G_SPEC.name] = G_MODULE
G_SPEC.loader.exec_module(G_MODULE)


def group(
    group_id: str,
    *,
    start_version: int,
    l1: float,
    tokens: int,
    reserved_timestamp_ns: int,
) -> object:
    """Build one synthetic decision-time group."""
    return G_MODULE.GroupRecord(
        group_id=group_id,
        ordinal=0,
        start_version=start_version,
        arm="control",
        delivered=True,
        metrics=G_BASE.MetricVector(
            registered_l1=l1,
            l2=l1**0.5,
            token_normalized_l1=l1 / tokens,
            nonzero_token_support=float(tokens),
            valid_actor_tokens=tokens,
        ),
        reserved_timestamp_ns=reserved_timestamp_ns,
        ready_timestamp_ns=reserved_timestamp_ns + 1,
        ready_learner_version=start_version,
        removed_timestamp_ns=reserved_timestamp_ns + 100,
        removed_learner_version=start_version,
        removal_reason="selected",
    )


class ServiceBudgetRepairTest(unittest.TestCase):
    """Verify exact feasibility, opportunity ordering, and determinism."""

    def test_repaired_choice_obeys_baseline_budget(self) -> None:
        candidates = [
            group("a", start_version=9, l1=1.0, tokens=100, reserved_timestamp_ns=1),
            group("b", start_version=9, l1=1.0, tokens=100, reserved_timestamp_ns=2),
            group("c", start_version=9, l1=1.0, tokens=100, reserved_timestamp_ns=3),
            group("d", start_version=9, l1=1.0, tokens=100, reserved_timestamp_ns=4),
            group("e", start_version=9, l1=100.0, tokens=500, reserved_timestamp_ns=5),
        ]
        selected, audit = G_MODULE.baseline_budgeted_oars(
            candidates,
            current_version=10,
        )
        self.assertEqual(len(selected), 4)
        self.assertLessEqual(audit["selected_tokens"], audit["token_budget"])
        self.assertNotIn("e", {value.group_id for value in selected})

    def test_repaired_choice_improves_opportunity_when_feasible(self) -> None:
        candidates = [
            group("a", start_version=9, l1=1.0, tokens=100, reserved_timestamp_ns=1),
            group("b", start_version=9, l1=1.0, tokens=100, reserved_timestamp_ns=2),
            group("c", start_version=9, l1=1.0, tokens=100, reserved_timestamp_ns=3),
            group("d", start_version=9, l1=1.0, tokens=100, reserved_timestamp_ns=4),
            group("e", start_version=9, l1=20.0, tokens=100, reserved_timestamp_ns=5),
        ]
        selected, audit = G_MODULE.baseline_budgeted_oars(
            candidates,
            current_version=10,
        )
        self.assertIn("e", {value.group_id for value in selected})
        self.assertEqual(audit["baseline_tokens"], audit["selected_tokens"])

    def test_repaired_choice_is_deterministic_on_exact_tie(self) -> None:
        candidates = [
            group(name, start_version=9, l1=1.0, tokens=10, reserved_timestamp_ns=index)
            for index, name in enumerate(("f", "e", "d", "c", "b", "a"))
        ]
        left, _ = G_MODULE.baseline_budgeted_oars(candidates, current_version=10)
        right, _ = G_MODULE.baseline_budgeted_oars(
            list(reversed(candidates)),
            current_version=10,
        )
        self.assertEqual(
            [value.group_id for value in left],
            [value.group_id for value in right],
        )
        self.assertEqual([value.group_id for value in left], ["a", "b", "c", "d"])

    def test_invalid_budget_multiplier_fails_closed(self) -> None:
        candidates = [
            group(str(index), start_version=1, l1=1.0, tokens=1, reserved_timestamp_ns=index)
            for index in range(4)
        ]
        with self.assertRaises(G_MODULE.OfflineGateError):
            G_MODULE.baseline_budgeted_oars(
                candidates,
                current_version=1,
                budget_multiplier=0.99,
            )


if __name__ == "__main__":
    unittest.main()
