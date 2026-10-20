"""Offline capture finalization and runtime packaging regressions."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from research.m4_eligible_replay.contracts import canonical_json
from research.m4_paired_replay.capture_finalize import finish, inventory
from research.m4_paired_replay.test_integration import synthetic_window

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/auto_research/2026-10-07-m4-causal-utility"


class CaptureFinalizationTests(unittest.TestCase):
    def fixture(self, root: Path, *, native_exception: bool = True) -> dict:
        _, sources = synthetic_window(native_exception=native_exception)
        for name, data in sources.items():
            (root / f"m4-eligible-live-v2a01-{name}").write_bytes(data)
        return dict(
            assets=root,
            identity="v2a01",
            seed=20261201,
            run_exit_code=int(native_exception),
            protocol=root / "protocol.json",
            analyzer=root / "analyzer.py",
        )

    def test_native_boundary_end_to_end_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            result = finish(**args)
            self.assertEqual(result["status"], "NO_ELIGIBLE_DECISION")
            self.assertEqual(result["run_exit_code"], 1)
            inventory(root, "v2a01")
            rows = (root / "artifacts.sha256").read_text().splitlines()
            self.assertEqual(len(rows), 5)
            for line in rows:
                digest, name = line.split("  ", 1)
                self.assertEqual(
                    digest, hashlib.sha256(Path(name).read_bytes()).hexdigest()
                )

    def test_false_no_event_boundaries_rejected(self):
        for mutation in ("cuda", "watchdog", "partial", "eligible", "capsule"):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                args = self.fixture(root)
                if mutation == "cuda":
                    (root / "m4-eligible-live-v2a01-run.log").write_text(
                        "RuntimeError: CUDA error\n"
                    )
                elif mutation == "watchdog":
                    (root / "m4-eligible-live-v2a01-watchdog.json").write_text("{}")
                elif mutation == "partial":
                    path = root / "m4-eligible-live-v2a01-lifecycle.jsonl"
                    path.write_bytes(path.read_bytes().splitlines(keepends=True)[0])
                elif mutation == "eligible":
                    path = root / "m4-eligible-live-v2a01-oars-v2.jsonl"
                    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
                    rows[1]["proposals"]["reward_variance_risk"][
                        "proposed_group_ids"
                    ] = ["different"]
                    path.write_bytes(b"".join(canonical_json(row) for row in rows))
                elif mutation == "capsule":
                    (root / "m4-eligible-live-capsule").mkdir()
                with self.assertRaises(ValueError):
                    finish(**args)
                self.assertFalse(
                    (root / "m4-eligible-live-v2a01-capture-result.json").exists()
                )

    def test_inventory_excludes_mutable_wrappers_and_includes_nested_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / "jet_log.zip").write_text("mutable")
            (root / "jet_assets").mkdir()
            (root / "jet_assets/output.log").write_text("mutable")
            checkpoint = root / "m4-eligible-live-capsule/checkpoint/native/shard"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"test-only-checkpoint")
            inventory(root, "v2a01")
            record = (root / "artifacts.sha256").read_text()
            self.assertIn(str(checkpoint), record)
            self.assertNotIn("jet_log", record)
            self.assertNotIn("output.log", record)

    def test_symlink_and_inventory_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / "linked").symlink_to(root / "m4-eligible-live-v2a01-run.log")
            with self.assertRaises(ValueError):
                inventory(root, "v2a01")
            (root / "linked").unlink()
            inventory(root, "v2a01")
            with self.assertRaises(FileExistsError):
                inventory(root, "v2a01")

    def test_result_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            finish(**args)
            with self.assertRaises(FileExistsError):
                finish(**args)

    def test_capsule_analyzer_assignment_and_integrity_gates(self):
        for valid in (True, False):
            with self.subTest(valid=valid), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = self.fixture(root, native_exception=False)
                capsule = root / "m4-eligible-live-capsule"
                capsule.mkdir()
                (capsule / "manifest.json").write_text("{}")
                result = {
                    "status": "PASS_OUTCOME_EXCLUDED_ELIGIBLE_CAPTURE",
                    "checks": {"test_only": valid},
                    "trajectory_identity": "v2a01",
                    "trajectory_seed": 20261201,
                    "assignment_domain": "m4-eligible-live-causal-acquisition-v2",
                    "post_update_outcomes_opened": False,
                }
                args["analyzer"].write_text(
                    "def analyze(capsule, protocol):\n    return " + repr(result) + "\n"
                )
                if valid:
                    self.assertEqual(finish(**args), result)
                else:
                    with self.assertRaises(ValueError):
                        finish(**args)


class CaptureRuntimeTemplateTests(unittest.TestCase):
    def test_qualified_native_entrypoint_and_frozen_overrides(self):
        path = REPORT / "build_eligible_live_v2_capture_package.py"
        spec = importlib.util.spec_from_file_location("v2capture", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        runtime = module.native_prefix() + module.RUNTIME
        subprocess.run(["bash", "-n"], input=runtime, text=True, check=True)
        self.assertIn("config.async_rl.sampler.max_staleness_versions==1", runtime)
        self.assertIn("suppress_capture_for_liveness_qualification=false", runtime)
        self.assertIn("progress_timeout_seconds=1200", runtime)
        self.assertIn("readonly CAPSULE=$ASSETS/m4-eligible-live-capsule", runtime)
        self.assertIn("readonly HASHES=$ASSETS/artifacts.sha256", runtime)
        self.assertIn("python", runtime.lower())
        for forbidden in (
            "run_arm",
            "run_probe",
            "optimizer_zero",
            "MCORE_REPLAY",
            "suppress_capture_for_liveness_qualification=true",
        ):
            self.assertNotIn(forbidden, runtime)


class CaptureOneShotLauncherTests(unittest.TestCase):
    def load_launcher(self):
        spec = importlib.util.spec_from_file_location(
            "capture_launcher", REPORT / "submit_eligible_live_v2_captures_once.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_missing_credentials_do_not_consume_guard(self):
        module = self.load_launcher()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(module, "GUARDS", root / "guards"),
                patch.object(
                    module.subprocess,
                    "run",
                    side_effect=subprocess.CalledProcessError(1, "credential-check"),
                ),
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    module.submit(
                        {"identity": "v2a01", "seed": 20261201, "sha256": "frozen"},
                        {"package_sha256": "package"},
                        root,
                    )
            self.assertFalse((root / "guards/v2a01.json").exists())

    def test_single_invocation_and_resumption_does_not_resubmit(self):
        module = self.load_launcher()
        candidate = {
            "identity": "v2a01",
            "seed": 20261201,
            "sha256": "frozen",
            "manifest": "test-only/v2a01.json",
        }
        ready = {"package_sha256": "package", "ci_commit_sha": "ci-frozen"}

        def fake_run(command, **kwargs):
            if "stdout" in kwargs:
                kwargs["stdout"].write(
                    b"12345: https://gitlab-master.nvidia.com/dl/jet/ci/-/pipelines/12345\n"
                )
                self.assertIn("--no_wait", command)
                self.assertNotIn("timeout", kwargs)
            return subprocess.CompletedProcess(command, 0, stdout="ci-frozen\n")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(module, "GUARDS", root / "guards"),
                patch.object(module.subprocess, "run", side_effect=fake_run) as run,
            ):
                self.assertEqual(module.submit(candidate, ready, root), 0)
                self.assertEqual(module.submit(candidate, ready, root), 0)
                self.assertEqual(run.call_count, 2)  # refresh/check and one runllm call
            guard = json.loads((root / "guards/v2a01.json").read_bytes())
            self.assertEqual(guard["attempts_consumed"], 1)
            self.assertEqual(guard["parent_pipeline_id"], 12345)

    def test_ambiguous_upload_never_reinvoked(self):
        module = self.load_launcher()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "v2a01.json"
            record = {
                "identity": "v2a01",
                "manifest_sha256": "frozen",
                "package_sha256": "package",
                "attempts_consumed": 1,
                "status": "ATTEMPT_CONSUMED_BEFORE_INVOCATION",
                "parent_pipeline_id": None,
            }
            path.write_bytes(canonical_json(record))
            with self.assertRaises(ValueError):
                module.read_guard(
                    path,
                    candidate={"identity": "v2a01", "sha256": "frozen"},
                    package_hash="package",
                )
            self.assertEqual(json.loads(path.read_bytes()), record)


if __name__ == "__main__":
    unittest.main()
