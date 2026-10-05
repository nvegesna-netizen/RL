"""Synthetic tests for the frozen OARS-v2 quality-primary analysis."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "quality_primary_analysis", ROOT / "analyze_quality_primary_blocks.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture_rows() -> tuple[list[dict], dict]:
    """Return 54 synthetic rows conforming to the frozen manifest."""
    manifest = json.loads((ROOT / "quality_primary_run_manifest.json").read_bytes())
    rows = []
    for run in manifest["runs"]:
        arm_offset = {"fifo": 0, "reward_variance": 33, "absolute_m4": 66}[run["arm"]]
        correct = 600 + run["block"] + arm_offset
        is_fifo = run["arm"] == "fifo"
        rows.append(
            {
                "schema": "m4-oars-v2-quality-primary-arm-result-v1",
                "status": "PASS",
                "protocol_sha256": MODULE.PROTOCOL_SHA256,
                "run_manifest_sha256": MODULE.RUN_MANIFEST_SHA256,
                "analysis_plan_sha256": MODULE.ANALYSIS_PLAN_SHA256,
                "source_commit": MODULE.SOURCE_COMMIT,
                "identity": run["identity"],
                "global_sequence": run["global_sequence"],
                "block": run["block"],
                "arm": run["arm"],
                "mode": run["mode"],
                "actuation_scorer": run["actuation_scorer"],
                "training_seed": run["training_seed"],
                "assignment_seed": run["assignment_seed"],
                "assignment_domain": run["assignment_domain"],
                "scientific_outcome_acquisition": True,
                "training_quality_analyzed": True,
                "systems": {
                    "qualification_status": (
                        "PASS_CONTROLLED_FRONTIER_SYSTEMS_READY"
                        if is_fifo
                        else "PASS_OARS_V2_ACTUATION_QUALIFIED"
                    ),
                    "checks": {"complete": True, "policy": True},
                    "time_to_update_64_seconds": 100.0 + arm_offset / 10,
                    "gradient_observer_duty": 0.001,
                    "scheduler_decision_duty": 0.002,
                },
                "mechanism": {
                    "retained_l1_per_selected_group": 1.0 + arm_offset / 100,
                    "valid_actor_tokens_per_update": 1000.0 + arm_offset,
                    "mean_fifo_overlap": 4.0 if is_fifo else 3.0,
                },
                "outcomes": {
                    "terminal_gsm8k_accuracy": correct / 1319,
                    "terminal_gsm8k_correct": correct,
                    "terminal_gsm8k_prompt_count": 1319,
                    "terminal_gsm8k_prompt_manifest_sha256": "a" * 64,
                },
            }
        )
    return rows, manifest


def test_registered_primary_and_secondary_numbers() -> None:
    rows, manifest = fixture_rows()
    result = MODULE.analyze(rows, manifest)
    primary = result["primary_absolute_m4_minus_fifo_terminal_gsm8k_accuracy"]
    assert result["status"] == "PASS_FROZEN_ANALYSIS_COMPLETE"
    assert "classification" not in result
    assert primary["mean"] == pytest.approx(66 / 1319)
    assert primary["ci95_lower"] == pytest.approx(66 / 1319)
    assert primary["ci95_upper"] == pytest.approx(66 / 1319)
    assert primary["exact_two_sided_sign_flip_p"] == pytest.approx(2 / 262144)
    assert primary["evidence_criterion_lower_bound_above_zero"] is True
    secondary = result["secondary_terminal_gsm8k_accuracy"]
    assert secondary["reward_variance_vs_fifo"]["mean"] == pytest.approx(33 / 1319)
    assert secondary["absolute_m4_vs_reward_variance"]["mean"] == pytest.approx(
        33 / 1319
    )
    assert len(result["block_rows"]) == 18


def test_primary_criterion_can_be_false_without_classification() -> None:
    rows, manifest = fixture_rows()
    for row in rows:
        if row["arm"] == "absolute_m4":
            correct = row["outcomes"]["terminal_gsm8k_correct"] - 100
            row["outcomes"]["terminal_gsm8k_correct"] = correct
            row["outcomes"]["terminal_gsm8k_accuracy"] = correct / 1319
    result = MODULE.analyze(rows, manifest)
    primary = result["primary_absolute_m4_minus_fifo_terminal_gsm8k_accuracy"]
    assert primary["mean"] < 0
    assert primary["evidence_criterion_lower_bound_above_zero"] is False
    assert "classification" not in result


def test_incomplete_results_are_rejected() -> None:
    rows, manifest = fixture_rows()
    with pytest.raises(MODULE.QualityPrimaryAnalysisError, match="all 54"):
        MODULE.analyze(rows[:-1], manifest)


def test_prompt_manifest_mismatch_is_rejected() -> None:
    rows, manifest = fixture_rows()
    rows[-1]["outcomes"]["terminal_gsm8k_prompt_manifest_sha256"] = "b" * 64
    with pytest.raises(MODULE.QualityPrimaryAnalysisError, match="prompt manifests"):
        MODULE.analyze(rows, manifest)


def test_frozen_identity_field_mismatch_is_rejected() -> None:
    rows, manifest = fixture_rows()
    rows[-1]["assignment_seed"] += 1
    with pytest.raises(MODULE.QualityPrimaryAnalysisError, match="assignment_seed"):
        MODULE.analyze(rows, manifest)


def test_failed_systems_gate_is_rejected() -> None:
    rows, manifest = fixture_rows()
    rows[-1]["systems"]["checks"]["policy"] = False
    with pytest.raises(MODULE.QualityPrimaryAnalysisError, match="systems gate"):
        MODULE.analyze(rows, manifest)
