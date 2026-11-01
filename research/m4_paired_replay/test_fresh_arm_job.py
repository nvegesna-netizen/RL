"""Offline tests for fixed dispatch, independent processes, and metric quarantine."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research.m4_paired_replay import fresh_arm_job as job

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/auto_research/2026-10-07-m4-causal-utility"


class FreshArmTests(unittest.TestCase):
    def test_failure_diagnostic_never_exports_exception_messages(self):
        text = 'Traceback (most recent call last):\n  File "/workspace/source.py", line 27, in update\nValueError: loss=1234.56 secret=test-only\n'
        row = job.sanitized_failure(text)
        self.assertEqual(row["exception_classes"], ["ValueError"])
        self.assertEqual(
            row["frames"],
            [{"path": "/workspace/source.py", "line": 27, "function": "update"}],
        )
        self.assertNotIn("1234.56", json.dumps(row))
        self.assertNotIn("secret", json.dumps(row))

    def test_all_sixteen_capsules_have_exactly_two_fixed_arms(self):
        spec = importlib.util.spec_from_file_location(
            "builder", REPORT / "build_fresh_paired_execution.py"
        )
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        gate = json.loads(
            (
                REPORT / "eligible_live_v2_capture_retry_r1_capture_gate.json"
            ).read_bytes()
        )
        rows = m.fixed_arms(gate)
        self.assertEqual(len(rows), 32)
        self.assertEqual(len({r["identity"] for r in rows}), 32)
        self.assertFalse(
            {r["capture_identity"] for r in rows} & {"v2a09", "v2a11", "v2a17", "v2a19"}
        )
        self.assertTrue(
            all(
                rows[i]["capture_identity"] == rows[i + 1]["capture_identity"]
                and rows[i]["arm"] == "reward_variance_base"
                and rows[i + 1]["arm"] == "m4_shield"
                for i in range(0, 32, 2)
            )
        )

    def test_three_fresh_processes_and_encrypted_only_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = root / "inputs"
            public = root / "public"
            inputs.mkdir()
            public.mkdir()
            candidate = {
                "identity": "v2a01-base",
                "capture_identity": "v2a01",
                "seed": 20261201,
                "arm": "reward_variance_base",
                "manifest_sha256": "a" * 64,
                "source_eos_job_id": 123,
            }
            plan = {"test_only": True}
            captures = []
            for name, value in [
                ("candidate.json", candidate),
                ("plan.json", plan),
                ("captures.json", captures),
            ]:
                (inputs / name).write_text(json.dumps(value))
            lock = {
                "status": "REVIEWED_FRESH_PAIRED_EXECUTION_READY",
                "fixed_arms": [candidate] * 32,
                "plan_file_sha256": job.digest_file(inputs / "plan.json"),
                "captures_file_sha256": job.digest_file(inputs / "captures.json"),
            }
            (inputs / "lock.json").write_text(json.dumps(lock))
            calls = []

            def fetch(**kwargs):
                out = kwargs["output"]
                out.mkdir(parents=True)
                (out / "transport-receipt.json").write_text("{}")

            def run(command, **kwargs):
                calls.append(command)
                if command[2].endswith("run_arm"):
                    receipt = {
                        "status": "COMPLETE_OUTCOME_SEALED",
                        "identity": "v2a01",
                        "arm": "reward_variance_base",
                        "manifest_sha256": "a" * 64,
                        "optimizer_updates": 1,
                        "outcomes_opened": False,
                    }
                    (public / "arm-receipt.json").write_text(json.dumps(receipt))
                    data = b"test-only ciphertext"
                    (public / "encrypted-envelope.json").write_bytes(data)
                    (public / "encrypted-receipt.json").write_text(
                        json.dumps(
                            {
                                "status": "COMPLETE_ENCRYPTED_PRIVATE_EXPORT",
                                "plaintext_exported": False,
                                "envelope_sha256": hashlib.sha256(data).hexdigest(),
                            }
                        )
                    )
                return SimpleNamespace(returncode=0)

            gate = {
                "valid_capsules": 16,
                "admitted": [
                    {
                        "identity": "v2a01",
                        "manifest_sha256": "a" * 64,
                        "eos_job_id": 123,
                        "seed": 20261201,
                    }
                ],
            }
            with (
                patch.object(job, "PRIVATE_BASE", root / "private"),
                patch.object(
                    job,
                    "create_private_root",
                    lambda p: p.mkdir(parents=True, mode=0o700),
                ),
                patch.object(job, "verify_admission_bundle"),
                patch.object(job, "capture_gate", return_value=gate),
                patch.object(job, "fetch_arm", side_effect=fetch),
                patch.object(job.subprocess, "run", side_effect=run),
                patch.dict(
                    os.environ,
                    {
                        "CI_JOB_TOKEN": "test-only",
                        "M4_PAIRED_SYNTHETIC_ENGINEERING": "0",
                    },
                ),
            ):
                self.assertEqual(job.run(inputs, public), 0)
            self.assertEqual(len(calls), 3)
            self.assertEqual(
                [c[2] for c in calls],
                ["research.m4_paired_replay.run_probe"] * 2
                + ["research.m4_paired_replay.run_arm"],
            )
            self.assertIn("selected", calls[0])
            self.assertIn("heldout", calls[1])
            self.assertIn("--gradient-selected", calls[2])
            self.assertEqual(
                json.loads((public / "completion.json").read_bytes())[
                    "completed_stages"
                ],
                ["selected", "heldout", "primary"],
            )
            self.assertNotIn("outcome.json", [p.name for p in public.iterdir()])


if __name__ == "__main__":
    unittest.main()
