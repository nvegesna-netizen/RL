#!/usr/bin/env python3
"""Fail-closed validation for the frozen downstream-quality design."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "trained_paired_acquisition_protocol.json"


def main() -> None:
    raw = PROTOCOL.read_bytes()
    protocol = json.loads(raw)
    assert protocol["schema"] == "m4-downstream-quality-trained-paired-acquisition-protocol-v1"
    assert protocol["status"].startswith("FROZEN_DESIGN_")
    assert protocol["runtime_source_commit"] is None
    assert protocol["causal_unit"] == "matched_training_seed_block"
    assert protocol["evidence_anchors"]["accepted_m4"]["protocol_sha256"] == (
        "6631a4ca8cfc6010150d1c135a1f5821c717f0dea189a3309c5f0f58f9b0709c"
    )
    assert protocol["evidence_anchors"]["v8_no_training_preflight"][
        "terminal_result_sha256"
    ] == "2c638192f249fc2e3a2644b8a3879b5cfb833cf99d0a42c458dfe1d629b1c88e"
    assert protocol["anchor"]["learner_updates"] == 448
    assert protocol["anchor"]["terminal_learner_version"] == 448
    assert protocol["evaluation"]["prompt_count"] == 1024
    assert protocol["evaluation"]["prompt_manifest_sha256"] == (
        "469a34a176febe9d15c5c6d967ff21f135a43cb20a7cb7d5ae1a53066e6b3a0a"
    )
    assert protocol["evaluation"]["v8_result_enters_primary_estimator"] is False
    assert protocol["inference"]["matched_blocks"] == 16
    assert protocol["inference"]["runs"] == 32
    assert protocol["inference"]["practical_absolute_accuracy_margin"] == 0.02
    assert protocol["inference"]["prompt_count_is_degrees_of_freedom"] is False
    assert protocol["inference"]["optional_extension"] is False
    assert protocol["inference"]["sequential_analysis"] is False

    blocks = protocol["blocks"]
    assert len(blocks) == 16
    assert len({b["id"] for b in blocks}) == 16
    assert len({b["training_seed"] for b in blocks}) == 16
    assert len({b["mixed_assignment_seed"] for b in blocks}) == 16
    for block in blocks:
        expected_bit = int.from_bytes(
            hashlib.sha256(
                f"m4-downstream-quality-paired-v1|{block['id']}".encode("ascii")
            ).digest(),
            "big",
        ) % 2
        expected = (
            ["immediate", "mixed_d5"]
            if expected_bit == 0
            else ["mixed_d5", "immediate"]
        )
        assert block["submission_order"] == expected

    assert protocol["regimes"]["immediate"]["arms"] == [
        {"label": "control", "delay_seconds": 0.0, "mass": 1}
    ]
    assert protocol["regimes"]["mixed_d5"]["arms"] == [
        {"label": "control", "delay_seconds": 0.0, "mass": 1},
        {"label": "d5", "delay_seconds": 5.0, "mass": 1},
    ]
    assert protocol["artifact_contract"]["all_1024_prompt_scores_required"] is True
    assert protocol["artifact_contract"]["outcome_embargo_until_completion_gate"] is True
    assert all(value is False for value in protocol["fail_closed"].values())
    assert protocol["authority"] == {
        "local_design_work": True,
        "config_implementation": False,
        "training": False,
        "eos_submission": False,
        "qualification": False,
        "scientific_acquisition": False,
    }
    print("M4_DOWNSTREAM_QUALITY_TRAINED_PAIRED_DESIGN_PASS")
    print(f"protocol_sha256={hashlib.sha256(raw).hexdigest()}")


if __name__ == "__main__":
    main()
