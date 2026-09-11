# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Dependency-light tests for the frozen downstream-quality implementation."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

# Permit direct dependency-light execution without pytest installation.
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

# The dated report directory is not a Python package name. Load its builder by path
# while keeping the production analyzer import conventional.
import importlib.util

from tools.m4_downstream_quality_analysis import (
    DownstreamQualityAnalysisError,
    analyze,
)


REPORT = REPO / "reports/auto_research/2026-09-11-m4-downstream-quality"
PROTOCOL = REPORT / "trained_paired_acquisition_protocol.json"
AUTHORIZATION = REPORT / "trained_paired_local_implementation_authorization.json"
EMBARGO = REPORT / "trained_paired_outcome_embargo.json"
IMMEDIATE_TEMPLATE = REPORT / "templates/trained_paired_immediate.yaml.tmpl"
MIXED_TEMPLATE = REPORT / "templates/trained_paired_mixed_d5.yaml.tmpl"


def _load_builder():
    path = REPORT / "build_trained_paired_run_configs.py"
    spec = importlib.util.spec_from_file_location("m4_downstream_config_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _load_builder()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class DownstreamQualityImplementationTest(unittest.TestCase):
    def _build_configs(self, root: Path) -> tuple[Path, dict[str, object]]:
        manifest_path = root / "run-manifest.json"
        manifest = BUILDER.build(
            protocol_path=PROTOCOL,
            authorization_path=AUTHORIZATION,
            immediate_template=IMMEDIATE_TEMPLATE,
            mixed_template=MIXED_TEMPLATE,
            output_dir=root / "configs",
            manifest_path=manifest_path,
        )
        return manifest_path, manifest

    def _build_complete_fixture(
        self, root: Path
    ) -> tuple[Path, Path, dict[str, object]]:
        manifest_path, manifest = self._build_configs(root)
        results = root / "results"
        prompt_hashes = [
            hashlib.sha256(f"prompt-{index}".encode()).hexdigest()
            for index in range(1024)
        ]
        base = {
            "schema": "m4-downstream-quality-v8-base-prompt-covariate-v1",
            "prompt_manifest_sha256": json.loads(PROTOCOL.read_bytes())["evaluation"][
                "prompt_manifest_sha256"
            ],
            "scores": [
                {"prompt_row_sha256": prompt_hash, "reward": int(index < 369)}
                for index, prompt_hash in enumerate(prompt_hashes)
            ],
        }
        base_data = _canonical(base)
        base_path = results / "v8-base-covariate.json"
        base_path.parent.mkdir(parents=True)
        base_path.write_bytes(base_data)

        gate_runs = []
        for index, run in enumerate(manifest["runs"]):
            block_index = int(run["block_id"][1:]) - 1
            immediate_correct = 520 + block_index % 3
            correct = (
                immediate_correct
                if run["regime"] == "immediate"
                else immediate_correct - 35 - block_index % 5
            )
            result = {
                "schema": "m4-downstream-quality-trained-run-result-v1",
                "protocol_sha256": manifest["protocol_sha256"],
                "block_id": run["block_id"],
                "regime": run["regime"],
                "training_seed": run["training_seed"],
                "assignment_seed": run["assignment_seed"],
                "assignment_domain": run["assignment_domain"],
                "config_sha256": run["config_sha256"],
                "train_steps": 448,
                "trainer_version": 448,
                "prompt_manifest_sha256": base["prompt_manifest_sha256"],
                "optimizer_exported": False,
                "resumable_checkpoint": False,
                "scores": [
                    {
                        "prompt_row_sha256": prompt_hash,
                        "reward": int(prompt_index < correct),
                    }
                    for prompt_index, prompt_hash in enumerate(prompt_hashes)
                ],
                "systems": {"wall_seconds": 5000.0 + index},
                "mechanism": {"status": "SYNTHETIC_TEST_ONLY"},
            }
            data = _canonical(result)
            path = results / run["expected_result_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            gate_runs.append(
                {
                    "block_id": run["block_id"],
                    "regime": run["regime"],
                    "training_seed": run["training_seed"],
                    "assignment_seed": run["assignment_seed"],
                    "assignment_domain": run["assignment_domain"],
                    "config_sha256": run["config_sha256"],
                    "train_steps": 448,
                    "trainer_version": 448,
                    "prompt_manifest_sha256": base["prompt_manifest_sha256"],
                    "pipeline_id": 1000 + index,
                    "job_id": 2000 + index,
                    "result_path": run["expected_result_path"],
                    "result_sha256": _sha(data),
                    "terminal_export_path": f"{run['block_id']}/{run['regime']}/terminal-policy",
                    "terminal_export_sha256": hashlib.sha256(
                        f"terminal-{index}".encode()
                    ).hexdigest(),
                    "converted_weights_sha256": hashlib.sha256(
                        f"weights-{index}".encode()
                    ).hexdigest(),
                    "evaluation_data_sha256": hashlib.sha256(
                        f"evaluation-{index}".encode()
                    ).hexdigest(),
                    "wall_hours": 1.5,
                    "h100_gpu_hours": 3.0,
                }
            )
        gate = {
            "schema": "m4-downstream-quality-trained-paired-completion-gate-v1",
            "status": "COMPLETE_AUTHENTICATED_16_PAIRS",
            "protocol_sha256": manifest["protocol_sha256"],
            "run_manifest_sha256": _sha(manifest_path.read_bytes()),
            "all_runs_terminal": True,
            "all_artifacts_authenticated": True,
            "resource_caps_passed": True,
            "outcome_release_authorized": True,
            "complete_pair_count": 16,
            "complete_run_count": 32,
            "summed_wall_hours": 48.0,
            "aggregate_h100_gpu_hours": 96.0,
            "base_covariate_path": "v8-base-covariate.json",
            "base_covariate_sha256": _sha(base_data),
            "runs": gate_runs,
        }
        gate_path = root / "completion-gate.json"
        gate_path.write_bytes(_canonical(gate))
        return manifest_path, gate_path, manifest

    def test_materializes_32_unique_pair_checked_configs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            manifest_path, manifest = self._build_configs(Path(raw))
            self.assertEqual(manifest["run_count"], 32)
            self.assertEqual(manifest["block_count"], 16)
            self.assertEqual(
                len({run["config_sha256"] for run in manifest["runs"]}), 32
            )
            self.assertEqual(len(set(manifest["pair_invariant_sha256"].values())), 16)
            self.assertTrue(manifest_path.is_file())
            for run in manifest["runs"]:
                text = (Path(raw) / "configs" / run["config_filename"]).read_text()
                self.assertNotIn("__", text)
                self.assertIn(str(run["training_seed"]), text)
                self.assertIn(run["assignment_domain"], text)

    def test_incomplete_gate_blocks_before_any_result_read(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest_path, manifest = self._build_configs(root)
            gate = {
                "schema": "m4-downstream-quality-trained-paired-completion-gate-v1",
                "status": "INCOMPLETE",
                "protocol_sha256": manifest["protocol_sha256"],
                "run_manifest_sha256": _sha(manifest_path.read_bytes()),
                "runs": [],
            }
            gate_path = root / "completion-gate.json"
            gate_path.write_bytes(_canonical(gate))
            with self.assertRaisesRegex(
                DownstreamQualityAnalysisError, "embargo remains locked"
            ):
                analyze(
                    protocol_path=PROTOCOL,
                    run_manifest_path=manifest_path,
                    embargo_path=EMBARGO,
                    completion_gate_path=gate_path,
                    results_root=root / "missing-results",
                )

    def test_complete_fixture_uses_16_run_level_differences(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest_path, gate_path, _ = self._build_complete_fixture(root)
            result = analyze(
                protocol_path=PROTOCOL,
                run_manifest_path=manifest_path,
                embargo_path=EMBARGO,
                completion_gate_path=gate_path,
                results_root=root / "results",
            )
            self.assertEqual(result["primary"]["block_count"], 16)
            self.assertEqual(result["primary"]["degrees_of_freedom"], 15)
            self.assertEqual(result["primary"]["conclusion"], "MATERIALLY_WORSE")
            self.assertEqual(
                result["primary"]["exact_sign_flip"]["enumerated_assignments"],
                65_536,
            )
            self.assertFalse(result["prompt_count_is_degrees_of_freedom"])
            self.assertFalse(result["historical_v8_enters_primary_estimator"])

    def test_authenticated_identity_mutation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest_path, gate_path, _ = self._build_complete_fixture(root)
            gate = json.loads(gate_path.read_bytes())
            gate["runs"][0]["training_seed"] += 1
            gate_path.write_bytes(_canonical(gate))
            with self.assertRaisesRegex(
                DownstreamQualityAnalysisError, "authenticated identity differs"
            ):
                analyze(
                    protocol_path=PROTOCOL,
                    run_manifest_path=manifest_path,
                    embargo_path=EMBARGO,
                    completion_gate_path=gate_path,
                    results_root=root / "results",
                )

    def test_aggregate_resource_cap_mutation_keeps_embargo_locked(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest_path, gate_path, _ = self._build_complete_fixture(root)
            gate = json.loads(gate_path.read_bytes())
            gate["aggregate_h100_gpu_hours"] = 256.01
            gate_path.write_bytes(_canonical(gate))
            with self.assertRaisesRegex(
                DownstreamQualityAnalysisError, "embargo remains locked"
            ):
                analyze(
                    protocol_path=PROTOCOL,
                    run_manifest_path=manifest_path,
                    embargo_path=EMBARGO,
                    completion_gate_path=gate_path,
                    results_root=root / "results",
                )

    def test_unregistered_result_path_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest_path, gate_path, _ = self._build_complete_fixture(root)
            gate = json.loads(gate_path.read_bytes())
            gate["runs"][0]["result_path"] = "substituted/result.json"
            gate_path.write_bytes(_canonical(gate))
            with self.assertRaisesRegex(
                DownstreamQualityAnalysisError, "result path differs"
            ):
                analyze(
                    protocol_path=PROTOCOL,
                    run_manifest_path=manifest_path,
                    embargo_path=EMBARGO,
                    completion_gate_path=gate_path,
                    results_root=root / "results",
                )


if __name__ == "__main__":
    unittest.main()
