"""Tests for the frozen paired OARS confirmatory analysis."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "m4_oars_confirmatory_analysis", ROOT / "analyze_confirmatory_pairs.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture_rows() -> tuple[list[dict], dict]:
    manifest = json.loads((ROOT / "confirmatory_run_manifest.json").read_bytes())
    rows = []
    for run in manifest["runs"]:
        is_oars = run["arm"] == "oars"
        rows.append(
            {
                "schema": "m4-oars-randomized-confirmatory-arm-result-v1",
                "status": "PASS",
                "protocol_sha256": manifest["protocol_sha256"],
                "run_manifest_sha256": MODULE.RUN_MANIFEST_SHA256,
                "source_commit": manifest["runtime_source"]["commit"],
                "identity": run["identity"],
                "pair": run["pair"],
                "arm": run["arm"],
                "mode": run["mode"],
                "training_seed": run["training_seed"],
                "assignment_seed": run["assignment_seed"],
                "assignment_domain": run["assignment_domain"],
                "scientific_outcome_acquisition": True,
                "training_quality_analyzed": True,
                "systems": {
                    "qualification_status": "PASS",
                    "checks": {"complete": True, "policy": True},
                    "time_to_update_64_seconds": 105.0 if is_oars else 100.0,
                },
                "outcomes": {
                    "retained_l1_per_selected_group": 2.0 if is_oars else 1.0,
                    "valid_actor_tokens_per_update": 980.0 if is_oars else 1000.0,
                    "terminal_gsm8k_accuracy": (726 if is_oars else 660) / 1319,
                    "terminal_gsm8k_correct": 726 if is_oars else 660,
                    "terminal_gsm8k_prompt_count": 1319,
                    "terminal_gsm8k_prompt_manifest_sha256": "a" * 64,
                },
            }
        )
    return rows, manifest


def test_material_result() -> None:
    rows, manifest = fixture_rows()
    result = MODULE.analyze(rows, manifest)
    assert result["classification"] == "MATERIAL"
    assert result["primary"]["mean"] == pytest.approx(1.0)
    assert result["primary"]["exact_two_sided_sign_flip_p"] == pytest.approx(
        2 / 1024
    )
    assert result["wall_time_ratio"]["ci95_upper"] < 1.10


def test_non_material_result() -> None:
    rows, manifest = fixture_rows()
    for row in rows:
        if row["arm"] == "oars":
            row["outcomes"]["retained_l1_per_selected_group"] = 0.5
    result = MODULE.analyze(rows, manifest)
    assert result["classification"] == "NON_MATERIAL"


def test_slow_oars_is_inconclusive_even_with_primary_gain() -> None:
    rows, manifest = fixture_rows()
    for row in rows:
        if row["arm"] == "oars":
            row["systems"]["time_to_update_64_seconds"] = 120.0
    result = MODULE.analyze(rows, manifest)
    assert result["classification"] == "INCONCLUSIVE"
    assert result["gates"]["primary_superiority"]
    assert not result["gates"]["wall_time_utility"]


def test_incomplete_results_are_rejected_before_analysis() -> None:
    rows, manifest = fixture_rows()
    with pytest.raises(MODULE.ConfirmatoryAnalysisError, match="all 20"):
        MODULE.analyze(rows[:-1], manifest)


def test_prompt_manifest_mismatch_is_rejected() -> None:
    rows, manifest = fixture_rows()
    rows[-1]["outcomes"]["terminal_gsm8k_prompt_manifest_sha256"] = "b" * 64
    with pytest.raises(MODULE.ConfirmatoryAnalysisError, match="prompt manifests"):
        MODULE.analyze(rows, manifest)
