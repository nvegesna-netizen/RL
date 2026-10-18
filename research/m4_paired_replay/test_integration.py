"""Synthetic integration tests; no GPU, network, or scientific outcomes."""

from __future__ import annotations

import ast
import base64
import copy
import hashlib
import json
import os
import math
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from research.m4_eligible_replay.contracts import canonical_json
from research.m4_paired_replay.capture_authentication import (
    CAPTURE_CHECKS,
    CAUSAL_PROTOCOL_HASH,
    classify,
    collect_cohort,
    verify_admission_bundle,
)
from research.m4_paired_replay import quarantine
from research.m4_paired_replay.runtime import verify_reference_runtime
from research.m4_paired_replay.secondary import (
    clipping_record,
    exact_gradient_dot,
    tensor_squared_difference,
    update_kl,
    zero_step_if_uninitialized,
)
from research.m4_paired_replay.test_package import REPORT, ROOT, FakeBackend, fixture
from research.m4_paired_replay.execution import (
    execute_arm,
    release_encrypted_outcomes,
    validate_secondaries,
)


def capture_fixture() -> dict:
    plan, _, _, manifests, _ = fixture()
    assignment = plan["trajectories"][0]
    manifest = manifests[assignment["identity"]]
    result = canonical_json(
        {
            "trajectory_identity": assignment["identity"],
            "trajectory_seed": assignment["seed"],
            "assignment_domain": plan["assignment_domain"],
            "status": "PASS_OUTCOME_EXCLUDED_ELIGIBLE_CAPTURE",
            "protocol_sha256": CAUSAL_PROTOCOL_HASH,
            "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "checks": dict.fromkeys(CAPTURE_CHECKS, True),
            "post_update_outcomes_opened": False,
        }
    )
    prefix = "/jet/assets/basic/m4-eligible-live-acquisition-v2a01/"
    inventory = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {prefix}{name}\n"
        for name, data in (
            ("m4-eligible-live-v2a01-capture-result.json", result),
            ("m4-eligible-live-capsule/manifest.json", manifest),
        )
    ).encode()
    return dict(
        assignment=assignment,
        parent={"id": 1, "status": "success"},
        child={"id": 2, "status": "success"},
        jobs=[
            {
                "id": 3,
                "name": "basic/m4-eligible-live-acquisition-v2a01 dgxh100_eos 00 [2 dgxh100_eos]",
                "status": "success",
            },
            {"id": 4, "name": "logs-after", "status": "success"},
        ],
        expected_parent_id=1,
        expected_child_id=2,
        expected_eos_job_id=3,
        result_bytes=result,
        manifest_bytes=manifest,
        inventory_bytes=inventory,
        progress_bytes=None,
    )


class CaptureAuthenticationTests(unittest.TestCase):
    def test_checks_match_frozen_analyzer_exactly(self) -> None:
        tree = ast.parse(
            (REPORT / "analyze_eligible_live_decision_capture.py").read_text()
        )
        keys = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key in node.keys:
                    if isinstance(key, ast.Constant) and key.value in CAPTURE_CHECKS:
                        keys.add(key.value)
        self.assertEqual(keys, CAPTURE_CHECKS)
        self.assertEqual(len(keys), 37)

    def test_authenticated_capsule(self) -> None:
        row, proof = classify(**capture_fixture())
        self.assertEqual(row["classification"], "CAPSULE")
        self.assertEqual(
            row["evidence_sha256"], hashlib.sha256(canonical_json(proof)).hexdigest()
        )

    def test_running_collector_blocks_classification(self) -> None:
        args = capture_fixture()
        args["jobs"][1]["status"] = "running"
        with self.assertRaisesRegex(ValueError, "terminal"):
            classify(**args)

    def test_success_without_independent_capture_proof_rejected(self) -> None:
        args = capture_fixture()
        args["result_bytes"] = None
        with self.assertRaisesRegex(ValueError, "missing"):
            classify(**args)

    def test_manifest_and_inventory_tampering_rejected(self) -> None:
        for key in ("manifest_bytes", "result_bytes", "inventory_bytes"):
            with self.subTest(key=key):
                args = capture_fixture()
                args[key] += b" "
                with self.assertRaises(ValueError):
                    classify(**args)

    def test_retried_or_wrong_exact_job_rejected(self) -> None:
        args = capture_fixture()
        duplicate = copy.deepcopy(args["jobs"][0])
        duplicate["id"] = 5
        args["jobs"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "exact capture"):
            classify(**args)

    def test_failed_workload_never_reads_outcomes(self) -> None:
        args = capture_fixture()
        args["jobs"][0]["status"] = "failed"
        args["parent"]["status"] = args["child"]["status"] = "failed"
        args["result_bytes"] = b"invalid outcome-like bytes must not be parsed"
        row, _ = classify(**args)
        self.assertEqual(row["classification"], "FAILED")
        self.assertIsNone(row["manifest_sha256"])

    def test_generator_failure_without_child_is_recorded_not_replaced(self) -> None:
        args = capture_fixture()
        args.update(child=None, expected_child_id=None, expected_eos_job_id=None)
        args["parent"]["status"] = "failed"
        args["jobs"] = [{"id": 10, "name": "generate", "status": "failed"}]
        row, proof = classify(**args)
        self.assertEqual(row["classification"], "FAILED")
        self.assertIsNone(row["eos_job_id"])
        self.assertIsNone(proof["child"])

    def test_no_eligibility_requires_authenticated_full_window(self) -> None:
        args = capture_fixture()
        result = json.loads(args["result_bytes"])
        result.update(
            status="NO_ELIGIBLE_DECISION",
            maximum_completed_updates=64,
            paired_replay_started=False,
        )
        args["result_bytes"] = canonical_json(result)
        args["manifest_bytes"] = None
        args["progress_bytes"] = canonical_json(
            dict(
                train_steps=64,
                learner_version=64,
                run_exit_code=0,
                capsule_written=False,
            )
        )
        prefix = "/jet/assets/basic/m4-eligible-live-acquisition-v2a01/"
        args["inventory_bytes"] = "".join(
            f"{hashlib.sha256(data).hexdigest()}  {prefix}{name}\n"
            for name, data in (
                ("m4-eligible-live-v2a01-capture-result.json", args["result_bytes"]),
                ("m4-eligible-live-v2a01-progress.json", args["progress_bytes"]),
            )
        ).encode()
        row, _ = classify(**args)
        self.assertEqual(row["classification"], "NO_ELIGIBLE")
        args["progress_bytes"] = None
        with self.assertRaisesRegex(ValueError, "missing"):
            classify(**args)

    def test_evidence_packets_rechecked_before_dispatch(self) -> None:
        plan, _, _, _, _ = fixture()

        class Client:
            def collect(self, assignment: dict, binding: dict) -> tuple:
                args = capture_fixture()
                args["assignment"] = assignment
                args["parent"]["status"] = args["child"]["status"] = "failed"
                args["jobs"][0].update(
                    name=f"basic/m4-eligible-live-acquisition-{assignment['identity']} dgxh100_eos 00 [2 dgxh100_eos]",
                    status="failed",
                    id=assignment["seed"],
                )
                args["expected_eos_job_id"] = assignment["seed"]
                return classify(**args)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "proof"
            rows = collect_cohort(
                Client(),
                plan=plan,
                bindings=[{"identity": r["identity"]} for r in plan["trajectories"]],
                output=output,
            )
            verify_admission_bundle(plan, rows, output)
            packet = output / "v2a01.json"
            packet.write_bytes(packet.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "packet differs"):
                verify_admission_bundle(plan, rows, output)


class EncryptedExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        cls.key = Path(cls.directory.name) / "escrow.pem"
        cls.public = quarantine.generate_keypair(cls.key)
        cls.identity = {"cohort": "synthetic-engineering", "arm": "base"}

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def test_roundtrip_randomized_and_no_plaintext(self) -> None:
        payload = b"random synthetic endpoint canary 930148"
        one = quarantine.seal(payload, public_pem=self.public, identity=self.identity)
        two = quarantine.seal(payload, public_pem=self.public, identity=self.identity)
        self.assertNotEqual(one, two)
        self.assertNotIn(payload, one)
        self.assertEqual(
            quarantine.unseal(
                one, private_path=self.key, expected_identity=self.identity
            ),
            payload,
        )
        self.assertEqual(self.key.stat().st_mode & 0o777, 0o600)

    def test_ciphertext_corruption_rejected(self) -> None:
        envelope = json.loads(
            quarantine.seal(
                b"synthetic", public_pem=self.public, identity=self.identity
            )
        )
        ciphertext = bytearray(base64.b64decode(envelope["ciphertext"]))
        ciphertext[-1] ^= 1
        envelope["ciphertext"] = base64.b64encode(ciphertext).decode()
        # InvalidTag is the crypto provider's explicit authentication failure.
        from cryptography.exceptions import InvalidTag

        with self.assertRaises(InvalidTag):
            quarantine.unseal(
                canonical_json(envelope),
                private_path=self.key,
                expected_identity=self.identity,
            )

    def test_wrong_identity_or_weak_public_key_rejected(self) -> None:
        envelope = quarantine.seal(
            b"synthetic", public_pem=self.public, identity=self.identity
        )
        with self.assertRaisesRegex(ValueError, "identity"):
            quarantine.unseal(
                envelope, private_path=self.key, expected_identity={"arm": "shield"}
            )
        with patch.object(quarantine, "version", return_value="48.0.0"):
            with self.assertRaisesRegex(ValueError, "patched"):
                quarantine.seal(
                    b"synthetic", public_pem=self.public, identity=self.identity
                )

    def test_incomplete_gate_never_loads_private_key(self) -> None:
        with patch.object(quarantine, "unseal") as decrypt:
            with self.assertRaisesRegex(ValueError, "not all"):
                quarantine.gated_open(
                    {},
                    {},
                    required={("synthetic", "base")},
                    expected_identities={},
                    private_path=self.key,
                )
            decrypt.assert_not_called()

    def test_export_allowlist_and_complete_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private = root / "private"
            private.mkdir(mode=0o700)
            (private / "outcome.json").write_bytes(b"synthetic endpoint")
            (private / "gradient.pt").write_bytes(b"not exported")
            envelope, receipt = (
                root / "public/envelope.json",
                root / "public/receipt.json",
            )
            row = quarantine.export_files(
                private_root=private,
                names=["outcome.json"],
                public_pem=self.public,
                identity=self.identity,
                envelope_path=envelope,
                receipt_path=receipt,
            )
            subject = ("synthetic", "base")
            opened = quarantine.gated_open(
                {subject: row},
                {subject: envelope},
                required={subject},
                expected_identities={subject: self.identity},
                private_path=self.key,
            )
            bundle = json.loads(opened[subject])
            self.assertEqual(set(bundle["files"]), {"outcome.json"})
            self.assertEqual(
                base64.b64decode(bundle["files"]["outcome.json"]), b"synthetic endpoint"
            )
            envelope.write_bytes(envelope.read_bytes() + b" ")
            with patch.object(quarantine, "unseal") as decrypt:
                with self.assertRaisesRegex(ValueError, "hash differs"):
                    quarantine.gated_open(
                        {subject: row},
                        {subject: envelope},
                        required={subject},
                        expected_identities={subject: self.identity},
                        private_path=self.key,
                    )
                decrypt.assert_not_called()

    def test_private_symlink_export_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source").write_bytes(b"secret")
            (root / "link").symlink_to(root / "source")
            with self.assertRaisesRegex(ValueError, "symlink"):
                quarantine.export_files(
                    private_root=root,
                    names=["link"],
                    public_pem=self.public,
                    identity=self.identity,
                    envelope_path=root / "envelope",
                    receipt_path=root / "receipt",
                )

    def test_key_permission_and_symlink_rejected(self) -> None:
        envelope = quarantine.seal(
            b"synthetic", public_pem=self.public, identity=self.identity
        )
        with tempfile.TemporaryDirectory() as directory:
            key = Path(directory) / "key"
            key.write_bytes(self.key.read_bytes())
            os.chmod(key, 0o644)
            with self.assertRaisesRegex(ValueError, "owner-only"):
                quarantine.unseal(
                    envelope, private_path=key, expected_identity=self.identity
                )


class SecondaryBoundaryTests(unittest.TestCase):
    def test_only_genuine_empty_v0_has_implicit_zero_step(self) -> None:
        empty = [{"state": {}, "param_groups": [{"params": [0]}]}]
        self.assertEqual(
            zero_step_if_uninitialized(empty, expected_step=0, scheduler_samples=0), 0
        )
        for step, samples in ((None, 0), (False, 0), (1, 0), (0, 32)):
            with self.assertRaises(ValueError):
                zero_step_if_uninitialized(
                    empty, expected_step=step, scheduler_samples=samples
                )
        for states in (
            [],
            [{"state": {1: {"exp_avg": 1}}, "param_groups": [{}]}],
            [{"state": {}, "param_groups": []}],
        ):
            with self.assertRaises(ValueError):
                zero_step_if_uninitialized(states, expected_step=0, scheduler_samples=0)

    def test_clipping_records_finite_native_norm(self) -> None:
        self.assertFalse(clipping_record(0.5, 1.0)["applied"])
        self.assertTrue(clipping_record(2.0, 1.0)["applied"])
        self.assertAlmostEqual(clipping_record(2.0, 1.0)["coefficient"], 1 / 2.000001)
        self.assertFalse(clipping_record(10.0, 0.0)["applied"])
        for norm, threshold in ((-1, 1), (float("nan"), 1), (1, -1), (1, float("inf"))):
            with self.assertRaises(ValueError):
                clipping_record(norm, threshold)

    def test_actual_backend_requires_integrated_worker(self) -> None:
        _, reference, _, _, _ = fixture()
        with self.assertRaisesRegex(ValueError, "not loaded"):
            verify_reference_runtime(
                reference["runtime_common"], require_integrated=True
            )

    def test_only_hash_bound_actor_changes_are_allowed(self) -> None:
        _, reference, _, _, _ = fixture()
        amendment = json.loads(
            (REPORT / "paired_integration_protocol.json").read_bytes()
        )
        runtime = reference["runtime_common"]
        for module_name, digest in amendment["actor_modules"].items():
            runtime["module_origins"][module_name] = {
                "path": "/workspace/m4-duplicate-update-repo/"
                + module_name.replace(".", "/")
                + ".py",
                "sha256": digest,
            }
        verify_reference_runtime(runtime, require_integrated=True)
        runtime["cudnn_deterministic"] = False
        with self.assertRaisesRegex(ValueError, "runtime differs"):
            verify_reference_runtime(runtime, require_integrated=True)

    def test_primary_native_train_matches_accepted_ast(self) -> None:
        current = (ROOT / "research/m4_eligible_replay/worker.py").read_text()
        tree = ast.parse(current)
        train = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "train"
        )
        self.assertEqual(
            hashlib.sha256(
                ast.dump(train, include_attributes=False).encode()
            ).hexdigest(),
            "8a394e9ed2a56a795a9ba989d64352340e22edeacb79fe387222a8575b2090e5",
        )


def synthetic_secondaries() -> dict:
    return {
        "parameter_update_l2_norm": 0.25,
        "train_batch_gradient_l2_norm": 0.5,
        "heldout_gradient_dot_train_gradient": -0.125,
        "gradient_clipping": clipping_record(0.5, 1.0),
        "policy_kl": 0.001,
        "definitions": {
            "policy_kl": "sampled_K3_pre_update_to_post_update_on_frozen_heldout_tokens",
            "gradient_dot": "exact_FP64_dot_of_native_preclipping_gradients_in_separate_restored_processes",
            "probe_norm_absolute_tolerance": 1e-5,
            "probe_norm_relative_tolerance": 1e-4,
        },
    }


class SecondaryExecutionTests(unittest.TestCase):
    def test_registered_secondary_payload_is_private(self) -> None:
        plan, reference, captures, manifests, _ = fixture()

        class ObservedBackend(FakeBackend):
            def begin_measurements(self) -> None:
                if self.updates:
                    raise ValueError("diagnostics started after update")

            def finish_measurements(self) -> dict:
                if self.updates != 1:
                    raise ValueError("diagnostics require one primary update")
                return synthetic_secondaries()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = ObservedBackend(reference)
            receipt = execute_arm(
                backend=backend,
                plan=plan,
                captures=captures,
                identity="v2a01",
                arm="reward_variance_base",
                manifest_bytes=manifests["v2a01"],
                private_dir=root / "private",
                receipt_path=root / "receipt",
                measure_secondaries=True,
            )
            self.assertNotIn("policy_kl", receipt)
            self.assertNotIn("secondaries", receipt)
            private = json.loads((root / "private/outcome.json").read_bytes())
            self.assertEqual(private["schema"], "m4-paired-private-outcome-v2")
            self.assertEqual(private["secondaries"], synthetic_secondaries())
            self.assertTrue(backend.closed)

    def test_invalid_or_approximate_secondary_rejected(self) -> None:
        for field in (
            "policy_kl",
            "parameter_update_l2_norm",
            "train_batch_gradient_l2_norm",
        ):
            row = synthetic_secondaries()
            row[field] = float("nan")
            with self.assertRaises(ValueError):
                validate_secondaries(row)
        row = synthetic_secondaries()
        row["definitions"]["gradient_dot"] = "countsketch"
        with self.assertRaisesRegex(ValueError, "definitions"):
            validate_secondaries(row)


class EncryptedCohortReleaseTests(unittest.TestCase):
    def build_cohort(self, root: Path) -> tuple:
        plan, reference, captures, manifests, _ = fixture()
        key = root / "escrow.pem"
        public = quarantine.generate_keypair(key)
        receipts, encrypted_receipts, paths = {}, {}, {}

        class Backend(FakeBackend):
            def begin_measurements(self) -> None:
                pass

            def finish_measurements(self) -> dict:
                return synthetic_secondaries()

        for row in captures[:16]:
            identity = row["identity"]
            for arm in ("reward_variance_base", "m4_shield"):
                subject = (identity, arm)
                directory = root / identity / arm
                directory.mkdir(parents=True)
                receipts[subject] = execute_arm(
                    backend=Backend(reference, arm),
                    plan=plan,
                    captures=captures,
                    identity=identity,
                    arm=arm,
                    manifest_bytes=manifests[identity],
                    private_dir=directory / "private/sealed",
                    receipt_path=directory / "receipt",
                    measure_secondaries=True,
                )
                (directory / "private/process.log").write_bytes(
                    b"synthetic-only private log"
                )
                paths[subject] = directory / "public/envelope"
                encrypted_receipts[subject] = quarantine.export_files(
                    private_root=directory / "private",
                    names=["sealed/outcome.json", "process.log"],
                    public_pem=public,
                    identity={
                        "identity": identity,
                        "arm": arm,
                        "manifest_sha256": receipts[subject]["manifest_sha256"],
                        "arm_receipt_sha256": hashlib.sha256(
                            canonical_json(receipts[subject])
                        ).hexdigest(),
                    },
                    envelope_path=paths[subject],
                    receipt_path=directory / "public/receipt",
                )
        return dict(
            plan=plan,
            captures=captures,
            evidence_dir=root / "producer-tested-separately",
            manifests={i: manifests[i] for i in list(manifests)[:16]},
            receipts=receipts,
            encrypted_receipts=encrypted_receipts,
            encrypted_paths=paths,
            private_key_path=key,
        )

    def test_all_thirty_two_encrypted_arms_release_with_secondaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_cohort(Path(directory))
            # Producer packet verification has independent tests; this isolates the
            # real RSA/AES and all-arm state/endpoint release path using fake updates.
            with patch("research.m4_paired_replay.execution.verify_admission_bundle"):
                result = release_encrypted_outcomes(**args)
            self.assertEqual(len(result), 16)
            self.assertAlmostEqual(result[0]["contrast"], -0.1)
            self.assertEqual(result[0]["base_secondaries"], synthetic_secondaries())

    def test_last_arm_provenance_failure_never_loads_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_cohort(Path(directory))
            args["receipts"]["v2a16", "m4_shield"]["optimizer_updates"] = 2
            with (
                patch("research.m4_paired_replay.execution.verify_admission_bundle"),
                patch.object(quarantine, "unseal") as decrypt,
            ):
                with self.assertRaisesRegex(ValueError, "provenance"):
                    release_encrypted_outcomes(**args)
                decrypt.assert_not_called()

    def test_last_ciphertext_failure_never_loads_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_cohort(Path(directory))
            last = args["encrypted_paths"]["v2a16", "m4_shield"]
            last.write_bytes(last.read_bytes() + b" ")
            with (
                patch("research.m4_paired_replay.execution.verify_admission_bundle"),
                patch.object(quarantine, "unseal") as decrypt,
            ):
                with self.assertRaisesRegex(ValueError, "hash differs"):
                    release_encrypted_outcomes(**args)
                decrypt.assert_not_called()


class Number:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class Vector:
    """Tiny Python numeric stand-in, not a real Torch tensor smoke test."""

    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.shape = (len(values),)
        self.dtype = "synthetic-float64"

    def double(self) -> Vector:
        return self

    def reshape(self, size: int) -> Vector:
        return self

    def detach(self) -> Vector:
        return self

    def cpu(self) -> Vector:
        return self

    def contiguous(self) -> Vector:
        return self

    def numel(self) -> int:
        return len(self.values)

    def __getitem__(self, item: slice) -> Vector:
        return Vector(self.values[item])

    def __sub__(self, other: Vector) -> Vector:
        return Vector([a - b for a, b in zip(self.values, other.values)])

    def __mul__(self, other: Vector) -> Vector:
        return Vector([a * b for a, b in zip(self.values, other.values)])

    def __lt__(self, value: float) -> Vector:
        return Vector([a < value for a in self.values])

    def sum(self) -> Number:
        return Number(sum(self.values))

    def any(self) -> Number:
        return Number(any(self.values))


class SecondaryMathTests(unittest.TestCase):
    def test_parameter_delta_and_sampled_update_kl_formula(self) -> None:
        torch = SimpleNamespace(
            sum=lambda x: x.sum(),
            expm1=lambda x: Vector([math.expm1(v) for v in x.values]),
        )
        self.assertEqual(
            tensor_squared_difference(Vector([1, 2]), Vector([3, -2]), torch), 20
        )
        value = update_kl(
            Vector([0, 0]), Vector([math.log(2), 100]), Vector([1, 0]), torch
        )
        self.assertAlmostEqual(value, 1 - math.log(2))
        with self.assertRaises(ValueError):
            update_kl(Vector([0]), Vector([0]), Vector([0]), torch)

    def test_exact_dot_requires_same_restored_state_and_role(self) -> None:
        subject = {"identity": "synthetic", "arm": "base", "manifest_sha256": "a" * 64}
        state = {
            "state_before": {"parameter_sha256": "b" * 64},
            "parameter_sha256": "b" * 64,
        }
        left = {
            **state,
            "subject": {**subject, "role": "selected"},
            "gradients": [("weight", Vector([1, 2]))],
        }
        right = {
            **state,
            "subject": {**subject, "role": "heldout"},
            "gradients": [("weight", Vector([3, -4]))],
        }
        rows = {Path("left"): left, Path("right"): right}
        torch = SimpleNamespace(
            load=lambda path, **kwargs: rows[path], sum=lambda x: x.sum()
        )
        result = exact_gradient_dot(
            Path("left"), Path("right"), torch, expected_subject=subject
        )
        self.assertEqual(result["heldout_gradient_dot_train_gradient"], -5)
        self.assertAlmostEqual(
            result["train_batch_gradient_l2_norm_probe"], math.sqrt(5)
        )
        self.assertEqual(result["heldout_gradient_l2_norm"], 5)
        right["subject"]["role"] = "selected"
        with self.assertRaisesRegex(ValueError, "role differs"):
            exact_gradient_dot(
                Path("left"), Path("right"), torch, expected_subject=subject
            )


if __name__ == "__main__":
    unittest.main()
