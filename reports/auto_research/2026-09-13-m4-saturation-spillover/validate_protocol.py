#!/usr/bin/env python3
"""Fail-closed semantic validation for the local saturation protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> None:
    path = HERE / "prospective_protocol.json"
    raw = path.read_bytes()
    value = json.loads(raw)
    assert value["status"] == "FROZEN_LOCAL_DESIGN_NO_LAUNCH_AUTHORITY"
    design = value["design"]
    assert design["matched_training_seed_blocks"] == 16
    assert design["runs_per_block"] == 2
    assert design["total_scientific_acquisitions"] == 32
    assert len(design["training_seeds"]) == len(set(design["training_seeds"])) == 16
    assert len(design["low_assignment_seeds"]) == len(set(design["low_assignment_seeds"])) == 16
    assert len(design["high_assignment_seeds"]) == len(set(design["high_assignment_seeds"])) == 16
    assert set(design["low_assignment_seeds"]).isdisjoint(design["high_assignment_seeds"])
    policies = design["saturation_policies"]
    assert policies["low"]["delayed_fraction"] == 0.25
    assert policies["high"]["delayed_fraction"] == 0.75
    for name, delayed_mass in (("low", 1), ("high", 3)):
        arms = policies[name]["arms"]
        assert [arm["label"] for arm in arms] == ["control", "d5"]
        assert [arm["delay_seconds"] for arm in arms] == [0.0, 5.0]
        assert sum(arm["mass"] for arm in arms) == 4
        assert arms[1]["mass"] == delayed_mass
    assert value["setting"]["primary_start_versions"] == [8, 407]
    assert value["setting"]["guard_start_versions"] == [408, 447]
    assert value["setting"]["terminal_quality_evaluation"] is False
    assert set(value["primary_estimands"]) >= {"control_spillover", "total_policy", "causal_unit"}
    assert value["inference"]["stopping_rule"].startswith("analyze exactly 16")
    precision = value["precision_and_capacity"]
    assert precision["projected_student_half_width"] <= precision["precision_target_half_width"]
    assert precision["aggregate_h100_gpu_hours_cap"] == 128.0
    assert precision["planning_reference_h100_gpu_hours"] < precision["aggregate_h100_gpu_hours_cap"]
    authority = value["authority"]
    assert authority and not any(authority.values())
    print(json.dumps({
        "status": "PASS_LOCAL_DESIGN_NO_LAUNCH_AUTHORITY",
        "protocol_sha256": hashlib.sha256(raw).hexdigest(),
        "blocks": 16,
        "candidate_acquisitions": 32,
        "primary_estimands": ["control_spillover", "total_policy"],
        "launch_authorized": False
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
