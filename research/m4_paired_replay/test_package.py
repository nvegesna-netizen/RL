"""Clean-room synthetic evidence only: no model, historical outcome or network."""

from __future__ import annotations

import ast
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from research.m4_eligible_replay.contracts import canonical_json
from research.m4_paired_replay.contracts import (
    ARMS,
    DOMAIN,
    admit,
    arm_groups,
    capture_gate,
    members,
)
from research.m4_paired_replay.execution import (
    execute_arm,
    release_outcomes,
    strict_runtime,
)
from research.m4_paired_replay.transport import fetch_arm


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/auto_research/2026-10-07-m4-causal-utility"


def fixture() -> tuple:
    plan = json.loads((REPORT / "eligible_live_acquisition_v2_plan.json").read_bytes())
    reference = json.loads(
        (REPORT / "eligible_live_minimal_observer_terminal_result.json").read_bytes()
    )
    groups = []
    base, shield = ["g0", "g1", "g2", "g3"], ["g0", "g1", "g4", "g5"]
    payload = {"checkpoint/policy/weights/state.bin": b"synthetic checkpoint"}
    for i in range(12):
        name = f"groups/group-{i:02d}.pt"
        payload[name] = f"synthetic tensor {i}".encode()
        groups.append(
            {
                "group_id": f"g{i}",
                "index": i,
                "raw_file": name,
                "raw_file_sha256": hashlib.sha256(payload[name]).hexdigest(),
                "tensor_content_sha256": "a" * 64,
                "sample_ids": [f"s{i}-{j}" for j in range(8)],
                "roles": (["base"] if f"g{i}" in base else [])
                + (["shield"] if f"g{i}" in shield else [])
                + (["heldout"] if i >= 8 else []),
            }
        )
    manifest = {
        "schema": "m4-shield-eligible-live-capsule-v1",
        "status": "QUALIFIED_OUTCOME_EXCLUDED_ELIGIBLE_CAPTURE",
        "trajectory_identity": "v2a01",
        "trajectory_seed": 20261201,
        "assignment_domain": DOMAIN,
        "captured_update_executed": False,
        "post_update_outcomes_opened": False,
        "parameter_hash_unchanged": True,
        "parameter_sha256_before": reference["state_before_common"]["parameter_sha256"],
        "parameter_sha256_after": reference["state_before_common"]["parameter_sha256"],
        "learner_version": 7,
        "train_steps_before_capture": 7,
        "base_group_ids": base,
        "shield_group_ids": shield,
        "groups": groups,
        "checkpoint_root": "checkpoint",
        "checkpoint_files": [
            {
                "path": "policy/weights/state.bin",
                "bytes": len(payload["checkpoint/policy/weights/state.bin"]),
                "sha256": hashlib.sha256(
                    payload["checkpoint/policy/weights/state.bin"]
                ).hexdigest(),
            }
        ],
    }
    manifests, captures = {}, []
    for i, row in enumerate(plan["trajectories"]):
        m = copy.deepcopy(manifest)
        m["trajectory_identity"] = row["identity"]
        m["trajectory_seed"] = row["seed"]
        manifests[row["identity"]] = canonical_json(m)
        captures.append(
            {
                "identity": row["identity"],
                "seed": row["seed"],
                "assignment_domain": DOMAIN,
                "terminal_status": "success",
                "eos_job_id": i + 1,
                "classification": "CAPSULE" if i < 16 else "NO_ELIGIBLE",
                "manifest_sha256": hashlib.sha256(
                    manifests[row["identity"]]
                ).hexdigest()
                if i < 16
                else None,
                "evidence_sha256": "b" * 64,
            }
        )
    return plan, reference, captures, manifests, payload


class FakeBackend:
    def __init__(self, reference: dict, arm: str = ARMS[0]):
        self.state = copy.deepcopy(reference["state_before_common"])
        self.after = copy.deepcopy(reference["state_after_common"])
        self.modes = copy.deepcopy(reference["runtime_common"])
        self.arm = arm
        self.updates = 0
        self.closed = False
        self.mutate_pre = False
        self.mutate_post = False
        self.bad_inputs = False
        self.close_failure = False

    def snapshot(self) -> dict:
        return copy.deepcopy(self.state)

    def runtime(self) -> dict:
        return copy.deepcopy(self.modes)

    def verify_inputs(self) -> None:
        if self.bad_inputs:
            raise ValueError("mutated input")

    def evaluate(self) -> float:
        if (not self.updates and self.mutate_pre) or (
            self.updates and self.mutate_post
        ):
            self.state["rng_sha256"] = "f" * 64
        return 1.0 if not self.updates else (0.9 if self.arm == ARMS[0] else 0.8)

    def update(self) -> None:
        self.updates += 1
        self.state = copy.deepcopy(self.after)

    def close(self) -> None:
        self.closed = True
        if self.close_failure:
            raise OSError("close failed")


class FixtureBase(unittest.TestCase):
    def setUp(self) -> None:
        self.plan, self.ref, self.captures, self.manifests, self.payload = fixture()


class GateTests(FixtureBase):
    def test_fixed_sixteen_of_twenty(self) -> None:
        self.assertEqual(capture_gate(self.plan, self.captures)["valid_capsules"], 16)

    def test_incomplete_fifteen_or_unclassified_never_admitted(self) -> None:
        cases = [
            self.captures[:-1],
            copy.deepcopy(self.captures),
            copy.deepcopy(self.captures),
            copy.deepcopy(self.captures),
        ]
        cases[1][15].update(classification="FAILED", manifest_sha256=None)
        cases[2][-1]["terminal_status"] = "running"
        cases[3][-1]["classification"] = "unknown"
        for rows in cases:
            with self.assertRaises(ValueError):
                capture_gate(self.plan, rows)

    def test_historical_or_seed_substitution_rejected(self) -> None:
        for key, val in [
            ("identity", "qualification"),
            ("seed", 20261101),
            ("assignment_domain", "v1"),
        ]:
            rows = copy.deepcopy(self.captures)
            rows[0][key] = val
            with self.assertRaises(ValueError):
                capture_gate(self.plan, rows)

    def test_terminal_pre_workload_failure_is_classified_not_replaced(self) -> None:
        rows = copy.deepcopy(self.captures)
        rows[-1].update(
            classification="FAILED", terminal_status="failed", eos_job_id=None
        )
        self.assertEqual(capture_gate(self.plan, rows)["valid_capsules"], 16)
        rows[-1]["classification"] = "NO_ELIGIBLE"
        with self.assertRaises(ValueError):
            capture_gate(self.plan, rows)

    def test_reused_job_or_capsule_rejected(self) -> None:
        for key in ("eos_job_id", "manifest_sha256"):
            rows = copy.deepcopy(self.captures)
            rows[1][key] = rows[0][key]
            with self.assertRaises(ValueError):
                capture_gate(self.plan, rows)

    def test_plan_mutations_rejected(self) -> None:
        p = copy.deepcopy(self.plan)
        p["capture_phase"]["minimum_authenticated_capsules_to_open_replay"] = 15
        with self.assertRaises(ValueError):
            capture_gate(p, self.captures)

    def test_manifest_hash_and_fresh_domain_bind(self) -> None:
        admit(self.plan, self.captures, "v2a01", self.manifests["v2a01"])
        with self.assertRaises(ValueError):
            admit(self.plan, self.captures, "v2a01", self.manifests["v2a02"])

    def test_unadmitted_identity_rejected(self) -> None:
        with self.assertRaises(ValueError):
            admit(self.plan, self.captures, "v2a20", self.manifests["v2a20"])

    def test_manifest_action_order_preserves_frontier_and_membership(self) -> None:
        m = json.loads(self.manifests["v2a01"])
        m["base_group_ids"].reverse()
        self.assertEqual(
            [r["group_id"] for r in arm_groups(m, ARMS[0])], ["g0", "g1", "g2", "g3"]
        )
        self.assertEqual(
            [r["group_id"] for r in arm_groups(m, ARMS[1])], ["g0", "g1", "g4", "g5"]
        )

    def test_roles_duplicates_paths_and_unknown_arm_rejected(self) -> None:
        m = json.loads(self.manifests["v2a01"])
        for mutation in (
            lambda x: x["groups"][0].update(roles=[]),
            lambda x: x["groups"][1].update(group_id="g0"),
            lambda x: x["checkpoint_files"][0].update(path="../escape"),
            lambda x: x.update(shield_group_ids=x["base_group_ids"]),
        ):
            candidate = copy.deepcopy(m)
            mutation(candidate)
            with self.assertRaises(ValueError):
                members(candidate, ARMS[0])
        with self.assertRaises(ValueError):
            members(m, "other")

    def test_runtime_must_match_qualified_worker_source_and_modes(self) -> None:
        for key, value in [
            ("deterministic_algorithms", False),
            ("torch_version", "changed"),
        ]:
            runtime = copy.deepcopy(self.ref["runtime_common"])
            runtime[key] = value
            with self.assertRaises(ValueError):
                strict_runtime(runtime)


class ExecutionTests(FixtureBase):
    def run_arm(
        self,
        root: Path,
        arm: str = ARMS[0],
        identity: str = "v2a01",
        backend: FakeBackend | None = None,
    ) -> tuple:
        backend = FakeBackend(self.ref, arm) if backend is None else backend
        name = identity + "-" + arm
        receipt = execute_arm(
            backend=backend,
            plan=self.plan,
            captures=self.captures,
            identity=identity,
            arm=arm,
            manifest_bytes=self.manifests[identity],
            private_dir=root / name,
            receipt_path=root / (name + ".json"),
        )
        return backend, receipt, root / name / "outcome.json"

    def test_one_update_no_public_endpoint_and_private_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            backend, receipt, path = self.run_arm(root)
            self.assertEqual(backend.updates, 1)
            self.assertTrue(backend.closed)
            self.assertFalse(any("loss" in k or k == "change" for k in receipt))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(path.parent.stat().st_mode & 0o777, 0o700)

    def test_evaluation_mutation_and_input_changes_rejected(self) -> None:
        for flag in ("mutate_pre", "mutate_post", "bad_inputs"):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                backend = FakeBackend(self.ref)
                setattr(backend, flag, True)
                with self.assertRaises(ValueError):
                    self.run_arm(root, backend=backend)
                self.assertTrue(backend.closed)
                self.assertFalse((root / ("v2a01-" + ARMS[0] + ".json")).exists())

    def test_close_failure_does_not_produce_complete_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            backend = FakeBackend(self.ref)
            backend.close_failure = True
            with self.assertRaises(OSError):
                self.run_arm(root, backend=backend)
            self.assertFalse((root / ("v2a01-" + ARMS[0] + ".json")).exists())

    def test_primary_error_survives_cleanup_error(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            backend = FakeBackend(self.ref)
            backend.bad_inputs = True
            backend.close_failure = True
            with self.assertRaisesRegex(ValueError, "mutated input") as caught:
                self.run_arm(Path(d), backend=backend)
            self.assertTrue(
                any("close failed" in note for note in caught.exception.__notes__)
            )

    def test_no_overwrite_or_second_update(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self.run_arm(Path(d))
            backend = FakeBackend(self.ref)
            with self.assertRaises(FileExistsError):
                self.run_arm(Path(d), backend=backend)
            self.assertEqual(backend.updates, 0)

    def cohort(self, root: Path) -> tuple:
        receipts, paths = {}, {}
        for identity in list(self.manifests)[:16]:
            for arm in ARMS:
                _, r, p = self.run_arm(root, arm, identity)
                receipts[identity, arm] = r
                paths[identity, arm] = p
        return receipts, paths

    def release(self, receipts: dict, paths: dict) -> list:
        return release_outcomes(
            plan=self.plan,
            captures=self.captures,
            manifests={k: self.manifests[k] for k in list(self.manifests)[:16]},
            receipts=receipts,
            outcome_paths=paths,
        )

    def test_complete_synthetic_cohort_releases_all_contrasts(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            receipts, paths = self.cohort(Path(d))
            values = self.release(receipts, paths)
            self.assertEqual(len(values), 16)
            self.assertAlmostEqual(values[0]["contrast"], -0.1)

    def test_missing_arm_does_not_read_any_outcome_json(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            receipts, paths = self.cohort(Path(d))
            receipts.pop(("v2a16", ARMS[1]))
            original = Path.read_bytes
            opened = []

            def read(path: Path) -> bytes:
                if path.name == "outcome.json":
                    opened.append(path)
                return original(path)

            with patch.object(Path, "read_bytes", read), self.assertRaises(ValueError):
                self.release(receipts, paths)
            self.assertEqual(opened, [])

    def test_bad_final_receipt_does_not_read_any_outcome_json(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            receipts, paths = self.cohort(Path(d))
            receipts["v2a16", ARMS[1]]["state_before"]["rng_sha256"] = "e" * 64
            original = Path.read_bytes
            opened = []

            def read(path: Path) -> bytes:
                if path.name == "outcome.json":
                    opened.append(path)
                return original(path)

            with patch.object(Path, "read_bytes", read), self.assertRaises(ValueError):
                self.release(receipts, paths)
            self.assertEqual(opened, [])

    def test_unexpected_endpoint_in_receipt_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            receipts, paths = self.cohort(Path(d))
            receipts["v2a01", ARMS[0]]["loss"] = 1.0
            with self.assertRaises(ValueError):
                self.release(receipts, paths)

    def test_sealed_payload_tampering_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            receipts, paths = self.cohort(Path(d))
            paths["v2a16", ARMS[1]].write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                self.release(receipts, paths)


class TransportTests(FixtureBase):
    def test_direct_members_streaming_and_role_allowlist(self) -> None:
        urls, sizes = [], []
        manifest = self.manifests["v2a01"]
        payload = self.payload

        class Response(io.BytesIO):
            def read(self, count: int = -1) -> bytes:
                sizes.append(count)
                return super().read(count)

        class Opener:
            def open(self, request: object, timeout: int) -> Response:
                urls.append(request.full_url)
                name = request.full_url.split("/m4-eligible-live-capsule/", 1)[1]
                return Response(manifest if name == "manifest.json" else payload[name])

        with tempfile.TemporaryDirectory() as d:
            receipt = fetch_arm(
                plan=self.plan,
                captures=self.captures,
                identity="v2a01",
                arm=ARMS[1],
                artifact_prefix="workspace/assets/basic/m4-eligible-live-acquisition-v2a01/m4-eligible-live-capsule",
                output=Path(d) / "capsule",
                token="synthetic-token",
                job_token=True,
                opener=Opener(),
            )
            self.assertFalse(receipt["large_archive_downloaded"])
            names = {r["path"] for r in receipt["members"]}
            self.assertNotIn("groups/group-02.pt", names)
            self.assertIn("groups/group-04.pt", names)
            self.assertTrue(
                all("/jobs/1/artifacts/" in u and not u.endswith(".zip") for u in urls)
            )
            self.assertTrue(all(0 < n <= (1 << 20) + 1 for n in sizes))

    def test_incomplete_gate_or_collector_prefix_never_requests_network(self) -> None:
        class Reject:
            def open(self, *args, **kwargs):
                raise AssertionError("network must not run")

        for captures, prefix in [
            (self.captures[:-1], "x"),
            (self.captures, "logs_after/capsule"),
        ]:
            with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
                fetch_arm(
                    plan=self.plan,
                    captures=captures,
                    identity="v2a01",
                    arm=ARMS[0],
                    artifact_prefix=prefix,
                    output=Path(d) / "capsule",
                    token="fake",
                    job_token=True,
                    opener=Reject(),
                )

    def test_actual_backend_evaluation_uses_only_heldout_and_token_loss(self) -> None:
        tree = ast.parse(Path(__file__).with_name("backend.py").read_text())
        evaluate = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "evaluate"
        )
        update = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "update"
        )
        self.assertIn("self.metas[0]", ast.unparse(evaluate))
        self.assertIn("eval_mode=True", ast.unparse(evaluate))
        self.assertIn("self.metas[1]", ast.unparse(update))
        self.assertIn("eval_mode=False", ast.unparse(update))
        self.assertIn(
            'self.config.loss_fn["token_level_loss"]',
            Path(__file__).with_name("backend.py").read_text(),
        )

    def test_native_and_python_output_redirected_in_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "private.log"
            code = (
                "from pathlib import Path; import os; from research.m4_paired_replay.run_arm import private_process_output\nwith private_process_output(Path("
                + repr(str(path))
                + ")):\n print('synthetic-loss=1',flush=True)\n os.write(2,b'native-private-metric\\n')\nprint('public-complete')\n"
            )
            result = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, check=True
            )
            self.assertEqual(result.stdout.strip(), "public-complete")
            self.assertEqual(result.stderr, "")
            self.assertIn("native-private-metric", path.read_text())

    def test_cli_backend_failure_details_stay_private(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            capsule = root / "capsule"
            capsule.mkdir()
            (capsule / "manifest.json").write_bytes(self.manifests["v2a01"])
            manifest = json.loads(self.manifests["v2a01"])
            for row in members(manifest, ARMS[0]):
                target = capsule / row["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(self.payload[row["path"]])
            (root / "plan.json").write_bytes(canonical_json(self.plan))
            (root / "captures.json").write_bytes(canonical_json(self.captures))
            (root / "lock.json").write_bytes(
                canonical_json(
                    {
                        "status": "REVIEWED_FRESH_PAIRED_EXECUTION_READY",
                        "plan_file_sha256": hashlib.sha256(
                            (root / "plan.json").read_bytes()
                        ).hexdigest(),
                    }
                )
            )
            arguments = [
                "run_arm",
                "--plan",
                str(root / "plan.json"),
                "--captures",
                str(root / "captures.json"),
                "--identity",
                "v2a01",
                "--arm",
                ARMS[0],
                "--capsule",
                str(capsule),
                "--config",
                "unused",
                "--private-root",
                str(root / "private"),
                "--receipt",
                str(root / "receipt.json"),
                "--execution-lock",
                str(root / "lock.json"),
            ]
            code = (
                "import sys,types\n"
                "fake=types.ModuleType('research.m4_paired_replay.backend')\n"
                "def fail(**kwargs): raise RuntimeError('SYNTHETIC_PRIVATE_LOSS_123')\n"
                "fake.NeMoBackend=fail\n"
                "sys.modules['research.m4_paired_replay.backend']=fake\n"
                f"sys.argv={arguments!r}\n"
                "from research.m4_paired_replay.run_arm import main\nmain()\n"
            )
            result = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 1)
            self.assertNotIn(
                "SYNTHETIC_PRIVATE_LOSS_123", result.stdout + result.stderr
            )
            self.assertIn(
                "SYNTHETIC_PRIVATE_LOSS_123", (root / "private/process.log").read_text()
            )
            self.assertFalse((root / "receipt.json").exists())


if __name__ == "__main__":
    unittest.main()
