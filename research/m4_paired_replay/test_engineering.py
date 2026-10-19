"""Synthetic engineering-gate fixtures; no model imports, network, or GPU."""

from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import zipfile
from unittest.mock import patch

from research.m4_paired_replay.engineering import coordinate, execute_stages
from research.m4_paired_replay.engineering_contracts import (
    DOMAIN,
    IDENTITY,
    PRIMARIES,
    PROBES,
    SEED,
    STAGES,
    compare_primaries,
    synthetic_decision,
    validate_synthetic_manifest,
)
from research.m4_paired_replay.quarantine import generate_keypair, unseal
from research.m4_paired_replay.quarantine import export_files
from research.m4_eligible_replay.contracts import canonical_json, digest_file
from research.m4_paired_replay.engineering_terminal import FILES, NAME, verify
from research.m4_paired_replay.test_package import REPORT, ROOT, fixture


def synthetic_fixture() -> tuple[dict, dict]:
    _, reference, _, _, _ = fixture()
    before = copy.deepcopy(reference["state_before_common"])
    after = copy.deepcopy(reference["state_after_common"])
    before.update(optimizer_step=0, scheduler_num_steps=0)
    after.update(optimizer_step=1, scheduler_num_steps=32)
    runtime = copy.deepcopy(reference["runtime_common"])
    amendment = json.loads((REPORT / "paired_integration_protocol.json").read_bytes())
    for name, digest in amendment["actor_modules"].items():
        runtime["module_origins"][name] = {
            "path": "/workspace/m4-duplicate-update-repo/"
            + name.replace(".", "/")
            + ".py",
            "sha256": digest,
        }
    manifest = {
        "trajectory_identity": IDENTITY,
        "trajectory_seed": SEED,
        "assignment_domain": DOMAIN,
        "learner_version": 0,
        "engineering_synthetic_only": True,
        "decision": synthetic_decision(),
        "parameter_sha256_before": before["parameter_sha256"],
    }
    records = {
        name: {
            "stage": name,
            "arm": arm,
            "observed": observed,
            "status": "COMPLETE_ENGINEERING_PRIMARY",
            "secondaries_complete": observed,
            "state_before": copy.deepcopy(before),
            "state_after": copy.deepcopy(after),
            "runtime": copy.deepcopy(runtime),
        }
        for name, (arm, observed) in PRIMARIES.items()
    }
    return manifest, records


class SyntheticGateTests(unittest.TestCase):
    def test_exact_bounded_process_set(self) -> None:
        self.assertEqual(len(STAGES), 10)
        self.assertEqual(len(PROBES), 4)
        self.assertEqual(len(PRIMARIES), 5)
        self.assertEqual(STAGES[0], "bootstrap")

    def test_complete_synthetic_gate_does_not_unlock_science(self) -> None:
        manifest, records = synthetic_fixture()
        result = compare_primaries(records, manifest)
        self.assertTrue(result["observer_post_state_equal"])
        self.assertFalse(result["fresh_acquisition_unlocked"])
        self.assertFalse(result["collector_plaintext_exclusion_verified"])

    def test_no_scientific_or_historical_assignment(self) -> None:
        manifest, _ = synthetic_fixture()
        for key, value in (
            ("trajectory_identity", "v2a01"),
            ("assignment_domain", "m4-eligible-live-causal-acquisition-v2"),
            ("learner_version", 7),
            ("engineering_synthetic_only", False),
        ):
            row = copy.deepcopy(manifest)
            row[key] = value
            with self.assertRaises(ValueError):
                validate_synthetic_manifest(row)

    def test_missing_duplicate_rejected(self) -> None:
        manifest, records = synthetic_fixture()
        records.pop("shield-b")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            compare_primaries(records, manifest)

    def test_parameter_optimizer_rng_and_scheduler_mismatches_rejected(self) -> None:
        for component in (
            "parameter_sha256",
            "optimizer_sha256",
            "rng_sha256",
            "scheduler_sha256",
        ):
            manifest, records = synthetic_fixture()
            records["base-a"]["state_after"][component] = "f" * 64
            with self.assertRaises(ValueError):
                compare_primaries(records, manifest)

    def test_alternate_action_duplicate_must_match(self) -> None:
        manifest, records = synthetic_fixture()
        records["shield-b"]["state_after"]["rng_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "alternate"):
            compare_primaries(records, manifest)

    def test_noop_or_extra_update_rejected(self) -> None:
        for step in (0, 2):
            manifest, records = synthetic_fixture()
            records["base-a"]["state_after"]["optimizer_step"] = step
            with self.assertRaises(ValueError):
                compare_primaries(records, manifest)

    def test_first_stage_failure_is_not_retried(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "research.m4_paired_replay.engineering.subprocess.run",
                return_value=SimpleNamespace(returncode=1),
            ) as run:
                result = execute_stages(Path(directory))
            self.assertEqual(run.call_count, 1)
            self.assertEqual(result["failed_stage"], "bootstrap")
            self.assertEqual(result["completed_stages"], [])


class ColdStartHookTests(unittest.TestCase):
    def setUp(self) -> None:
        tree = ast.parse((ROOT / "research/m4_eligible_replay/worker.py").read_text())
        hook = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_pre_load_checkpoint_hook"
        )
        namespace = {"Any": object, "os": os}
        exec(
            compile(
                ast.Module(body=[hook], type_ignores=[]), "actual-cold-hook", "exec"
            ),
            namespace,
        )
        self.hook = namespace["_pre_load_checkpoint_hook"]
        self.worker = SimpleNamespace(
            cfg={
                "eligible_replay_paired_integration": True,
                "eligible_replay_engineering_cold_start": True,
            }
        )
        self.checkpoint = SimpleNamespace(
            load=None,
            load_optim=False,
            pretrained_checkpoint="/native-pretrained",
            finetune=False,
            load_rng=False,
        )
        self.state = SimpleNamespace(cfg=SimpleNamespace(checkpoint=self.checkpoint))

    def test_explicit_cold_path_preserves_native_flags(self) -> None:
        with patch.dict(os.environ, {"M4_PAIRED_SYNTHETIC_ENGINEERING": "1"}):
            self.hook(self.worker, self.state, None)
        self.assertFalse(self.checkpoint.load_optim)
        self.assertFalse(self.checkpoint.load_rng)

    def test_cold_path_cannot_override_real_resume(self) -> None:
        self.checkpoint.load = "/real-resume"
        self.checkpoint.load_optim = True
        with (
            patch.dict(os.environ, {"M4_PAIRED_SYNTHETIC_ENGINEERING": "1"}),
            self.assertRaises(ValueError),
        ):
            self.hook(self.worker, self.state, None)

    def test_missing_engineering_marker_rejected(self) -> None:
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            self.hook(self.worker, self.state, None)

    def test_normal_full_resume_still_restores_rng(self) -> None:
        self.worker.cfg["eligible_replay_engineering_cold_start"] = False
        self.checkpoint.load = "/real-resume"
        self.checkpoint.load_optim = True
        self.hook(self.worker, self.state, None)
        self.assertTrue(self.checkpoint.load_rng)


class FailurePrivacyTests(unittest.TestCase):
    def test_failed_comparison_is_sealed_not_echoed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public_key = root / "public.pem"
            public_key.write_bytes(generate_keypair(root / "private.pem"))
            private = root / "staging"
            public = root / "public"
            with (
                patch.dict(os.environ, {"M4_PAIRED_SYNTHETIC_ENGINEERING": "1"}),
                patch(
                    "research.m4_paired_replay.engineering.create_private_root",
                    side_effect=lambda path: path.mkdir(mode=0o700),
                ),
                patch(
                    "research.m4_paired_replay.engineering.execute_stages",
                    side_effect=ValueError("SYNTHETIC_PRIVATE_QUALITY_319"),
                ),
            ):
                code = coordinate(private, public, public_key)
            self.assertEqual(code, 1)
            for path in public.iterdir():
                self.assertNotIn(b"SYNTHETIC_PRIVATE_QUALITY_319", path.read_bytes())
            payload = unseal(
                (public / "engineering-private-envelope.json").read_bytes(),
                private_path=root / "private.pem",
                expected_identity={"identity": IDENTITY, "domain": DOMAIN},
            )
            self.assertIn("controller-error.log", json.loads(payload)["files"])


class CollectorGateTests(unittest.TestCase):
    def archive_fixture(self, root: Path, *, leak: bool = False) -> dict:
        private = root / "private"
        private.mkdir(mode=0o700)
        canary = "123456789abcdef0" * 4
        (private / "canary.json").write_bytes(canonical_json({"canary": canary}))
        pem = generate_keypair(root / "escrow.pem")
        public = root / "public"
        public.mkdir()
        export_files(
            private_root=private,
            names=["canary.json"],
            public_pem=pem,
            identity={"identity": IDENTITY, "domain": DOMAIN},
            envelope_path=public / "engineering-private-envelope.json",
            receipt_path=public / "engineering-encrypted-receipt.json",
        )
        runtime = {"status": "FAILED_PRIVATE_DIAGNOSTICS", "failed_stage": "bootstrap"}
        (public / "engineering-runtime-result.json").write_bytes(
            canonical_json(runtime)
        )
        for name in FILES:
            if not (public / name).exists():
                (public / name).write_bytes(b"synthetic public preflight log")
        inventory = "".join(
            f"{digest_file(public / name)}  /jet/assets/basic/{NAME}/{name}\n"
            for name in sorted(FILES)
        ).encode()
        archives = []
        for label in ("eos", "collector"):
            path = root / f"{label}.zip"
            with zipfile.ZipFile(path, "w") as archive:
                for name in FILES:
                    archive.writestr(
                        f"workspace/assets/basic/{NAME}/{name}",
                        (public / name).read_bytes(),
                    )
                archive.writestr(
                    f"workspace/assets/basic/{NAME}/artifacts.sha256", inventory
                )
                if leak and label == "collector":
                    archive.writestr(
                        "workspace/logs/ray-actor.log",
                        b"unexpected private marker=" + canary.encode(),
                    )
            archives.append(path)
        traces = {}
        for job in (101, 102):
            traces[job] = root / f"{job}.log"
            traces[job].write_bytes(b"synthetic public trace")
        metadata = {
            "schema": "m4-authenticated-engineering-download-v1",
            "workload_name": NAME,
            "all_chains_terminal": True,
            "job_ids": [101, 102],
            "eos_archive_sha256": digest_file(archives[0]),
            "collector_archive_sha256": digest_file(archives[1]),
            "trace_sha256": {str(j): digest_file(p) for j, p in traces.items()},
        }
        return dict(
            eos_archive=archives[0],
            collector_archive=archives[1],
            traces=traces,
            metadata=metadata,
            private_key=root / "escrow.pem",
        )

    def test_terminal_runtime_failure_can_be_inspected_without_science_unlock(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = verify(**self.archive_fixture(Path(directory)))
            self.assertTrue(result["collector_canary_excluded"])
            self.assertFalse(result["fresh_acquisition_unlocked"])

    def test_extra_unlisted_ray_log_leak_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "canary leaked"):
                verify(**self.archive_fixture(Path(directory), leak=True))

    def test_running_chain_never_opens_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.archive_fixture(Path(directory))
            args["metadata"]["all_chains_terminal"] = False
            with (
                patch(
                    "research.m4_paired_replay.engineering_terminal.unseal"
                ) as decrypt,
                self.assertRaisesRegex(ValueError, "terminal"),
            ):
                verify(**args)
            decrypt.assert_not_called()

    def test_trace_hash_tamper_precedes_key_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.archive_fixture(Path(directory))
            args["traces"][102].write_bytes(b"changed")
            with (
                patch(
                    "research.m4_paired_replay.engineering_terminal.unseal"
                ) as decrypt,
                self.assertRaisesRegex(ValueError, "trace changed"),
            ):
                verify(**args)
            decrypt.assert_not_called()


if __name__ == "__main__":
    unittest.main()
