#!/usr/bin/env python3
"""Dependency-light tests for the M4 opportunity-control offline gate."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


G_HERE = Path(__file__).resolve().parent
G_SPEC = importlib.util.spec_from_file_location(
    "m4_opportunity_control_offline_gate",
    G_HERE / "run_offline_gate.py",
)
if G_SPEC is None or G_SPEC.loader is None:
    raise RuntimeError("cannot load offline gate module")
G_MODULE = importlib.util.module_from_spec(G_SPEC)
sys.modules[G_SPEC.name] = G_MODULE
G_SPEC.loader.exec_module(G_MODULE)


def group(
    group_id: str,
    *,
    start_version: int,
    l1: float,
    tokens: int,
    ready_timestamp_ns: int,
) -> object:
    """Build one synthetic decision-time group."""
    return G_MODULE.GroupRecord(
        group_id=group_id,
        ordinal=0,
        start_version=start_version,
        arm="control",
        delivered=True,
        metrics=G_MODULE.MetricVector(
            registered_l1=l1,
            l2=l1**0.5,
            token_normalized_l1=l1 / tokens,
            nonzero_token_support=float(tokens),
            valid_actor_tokens=tokens,
        ),
        reserved_timestamp_ns=ready_timestamp_ns - 1,
        ready_timestamp_ns=ready_timestamp_ns,
        ready_learner_version=start_version,
        removed_timestamp_ns=ready_timestamp_ns + 10,
        removed_learner_version=start_version,
        removal_reason="selected",
    )


class OfflineGateTest(unittest.TestCase):
    """Verify metric parsing, deterministic ordering, and gate logic."""

    def test_metric_vector_derives_token_normalized_l1(self) -> None:
        parsed = G_MODULE.metric_vector(
            {
                "opportunity": 12.0,
                "l2_coefficient_mass": 3.0,
                "nonzero_advantage_tokens": 4,
                "valid_actor_tokens": 6,
            }
        )
        self.assertEqual(parsed.registered_l1, 12.0)
        self.assertEqual(parsed.token_normalized_l1, 2.0)
        self.assertEqual(parsed.value("nonzero_token_support"), 4.0)

    def test_deterministic_oars_places_imminent_groups_first(self) -> None:
        groups = [
            group("urgent-low", start_version=9, l1=10.0, tokens=10, ready_timestamp_ns=2),
            group("urgent-high", start_version=9, l1=20.0, tokens=10, ready_timestamp_ns=3),
            group("fresh-high", start_version=10, l1=1000.0, tokens=10, ready_timestamp_ns=1),
        ]
        ordered = G_MODULE.policy_order(
            groups,
            policy="deterministic_oars",
            current_version=10,
        )
        self.assertEqual(
            [value.group_id for value in ordered],
            ["urgent-high", "urgent-low", "fresh-high"],
        )

    def test_ready_fifo_uses_ready_timestamp_then_identity(self) -> None:
        groups = [
            group("b", start_version=2, l1=1.0, tokens=1, ready_timestamp_ns=2),
            group("c", start_version=2, l1=1.0, tokens=1, ready_timestamp_ns=1),
            group("a", start_version=2, l1=1.0, tokens=1, ready_timestamp_ns=2),
        ]
        ordered = G_MODULE.policy_order(
            groups,
            policy="ready_fifo",
            current_version=2,
        )
        self.assertEqual([value.group_id for value in ordered], ["c", "a", "b"])

    def test_gate_requires_every_frozen_condition(self) -> None:
        cells = {}
        for index in range(14):
            cells[str(index)] = {
                "assignment_count": 7618 + (1 if index < 1 else 0),
                "metrics": {
                    "l2": {"adjusted_estimate": 0.1},
                    "token_normalized_l1": {"adjusted_estimate": 0.1},
                },
                "natural_expiry": {
                    "positive_l1_stale_eviction_mass_fraction": 0.2,
                },
                "shadow_choice": {
                    "contended_primary_steps": 1,
                    "policies": {
                        "deterministic_oars": {
                            "l1_gain_over_actual_fraction": 0.2,
                            "selected_token_ratio_to_actual": 1.0,
                        }
                    },
                },
            }
        # Make the synthetic total equal the frozen assignment count.
        cells["13"]["assignment_count"] += 106653 - sum(
            value["assignment_count"] for value in cells.values()
        )
        self.assertTrue(G_MODULE.evaluate_gate(cells)["passed"])
        cells["0"]["shadow_choice"]["policies"]["deterministic_oars"][
            "selected_token_ratio_to_actual"
        ] = 1.03
        self.assertFalse(G_MODULE.evaluate_gate(cells)["passed"])

    def test_type7_empty_and_interpolated(self) -> None:
        self.assertIsNone(G_MODULE.type7([], 0.5))
        self.assertEqual(G_MODULE.type7([0.0, 10.0], 0.25), 2.5)


if __name__ == "__main__":
    unittest.main()
