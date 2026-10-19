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

from nemo_rl.utils.config import load_config
from research.m4_paired_replay.engineering import (
    CONFIG,
    coordinate,
    execute_stages,
    save_bootstrap_checkpoint,
)
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


class BootstrapCheckpointTests(unittest.TestCase):
    def exercise(
        self,
        *,
        asynchronous: bool,
        state: object,
        finalize_error: bool = False,
        missing_metadata: bool = False,
        save_error: bool = False,
    ) -> list[str]:
        events = []
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            metadata = checkpoint / "latest_train_state.pt"

            class NativePolicy:
                def save_checkpoint(
                    self, weights_path: str, *, optimizer_path: str
                ) -> None:
                    events.append("save")
                    if save_error:
                        raise RuntimeError("native save failed")
                    if weights_path != str(checkpoint) or optimizer_path != str(
                        checkpoint
                    ):
                        raise AssertionError("full native checkpoint paths changed")
                    if not asynchronous and not missing_metadata:
                        metadata.touch()

                def finalize_async_save(self) -> None:
                    events.append("finalize")
                    if finalize_error:
                        raise RuntimeError("native writer failed")
                    if asynchronous and not missing_metadata:
                        metadata.touch()

            def load(path: Path) -> object:
                events.append("load")
                self.assertEqual(path, metadata)
                self.assertTrue(metadata.is_file())
                return state

            save_bootstrap_checkpoint(NativePolicy(), checkpoint, load_train_state=load)
        return events

    def test_delayed_async_metadata_waits_for_native_finalization(self) -> None:
        self.assertEqual(
            self.exercise(asynchronous=True, state={"step": 0}),
            ["save", "finalize", "load"],
        )

    def test_actual_inherited_config_retains_native_async_save(self) -> None:
        config = load_config(ROOT / CONFIG)
        self.assertIs(config.policy.megatron_cfg.checkpoint.async_save, True)
        self.assertEqual(
            self.exercise(
                asynchronous=config.policy.megatron_cfg.checkpoint.async_save,
                state={"step": 0},
            ),
            ["save", "finalize", "load"],
        )

    def test_failed_save_never_advances_to_metadata(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "native save failed"):
            self.exercise(asynchronous=True, state={"step": 0}, save_error=True)

    def test_synchronous_save_keeps_same_completion_boundary(self) -> None:
        self.assertEqual(
            self.exercise(asynchronous=False, state={"step": 0}),
            ["save", "finalize", "load"],
        )

    def test_failed_finalization_never_reads_metadata(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "native writer failed"):
            self.exercise(asynchronous=True, state={"step": 0}, finalize_error=True)

    def test_missing_native_metadata_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "metadata missing"):
            self.exercise(asynchronous=True, state={"step": 0}, missing_metadata=True)

    def test_nonzero_missing_or_malformed_version_is_rejected(self) -> None:
        for state in ({"step": 1}, {"step": 0.5}, {}, [], None):
            with (
                self.subTest(state=state),
                self.assertRaisesRegex(ValueError, "version zero"),
            ):
                self.exercise(asynchronous=True, state=state)

    def test_bootstrap_uses_helper_without_rejecting_native_async_config(self) -> None:
        tree = ast.parse(
            (ROOT / "research/m4_paired_replay/engineering.py").read_text()
        )
        bootstrap = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "bootstrap"
        )
        self.assertIn("save_bootstrap_checkpoint", ast.unparse(bootstrap))
        self.assertNotIn("async_save", ast.unparse(bootstrap))


class CanaryDispatchContractTests(unittest.TestCase):
    def setUp(self) -> None:
        tree = ast.parse((ROOT / "nemo_rl/distributed/worker_groups.py").read_text())
        function = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
            and n.name == "run_all_workers_single_data"
        )
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                function,
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)
        namespace = {"ray": SimpleNamespace(put=lambda value: value)}
        exec(compile(module, "actual_native_worker_group.py", "exec"), namespace)
        self.native_dispatch = namespace["run_all_workers_single_data"]
        self.delivered = []

        def remote(*args: object, **kwargs: object) -> str:
            self.assertEqual(args, ())
            self.delivered.append(kwargs)
            return "synthetic-future"

        self.group = SimpleNamespace(
            workers=[
                SimpleNamespace(
                    paired_engineering_canary=SimpleNamespace(remote=remote)
                )
            ],
            sharding_annotations=SimpleNamespace(
                names=[], get_worker_coords=lambda index: {}
            ),
        )

    def test_native_dispatcher_rejects_original_positional_payload(self) -> None:
        with self.assertRaisesRegex(AssertionError, "Please use kwargs"):
            self.native_dispatch(self.group, "paired_engineering_canary", "fixture")
        self.assertEqual(self.delivered, [])

    def test_both_actual_canary_calls_reach_native_keyword_dispatch(self) -> None:
        tree = ast.parse(
            (ROOT / "research/m4_paired_replay/engineering.py").read_text()
        )
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "run_all_workers_single_data"
            and n.args
            and isinstance(n.args[0], ast.Constant)
            and n.args[0].value == "paired_engineering_canary"
        ]
        self.assertEqual(len(calls), 2)

        def dispatch(method: str, *args: object, **kwargs: object) -> list:
            return self.native_dispatch(self.group, method, *args, **kwargs)

        policy = SimpleNamespace(run_all_workers_single_data=dispatch)
        for call in calls:
            result = eval(
                compile(ast.Expression(body=call), "actual_canary_call.py", "eval"),
                {
                    "policy": policy,
                    "backend": SimpleNamespace(policy=policy),
                    "canary": "synthetic-argument-contract-fixture",
                },
            )
            self.assertEqual(result, ["synthetic-future"])
        self.assertEqual(
            self.delivered, [{"canary": "synthetic-argument-contract-fixture"}] * 2
        )

    def test_all_paired_runtime_single_data_calls_use_keyword_payloads(self) -> None:
        for path in (ROOT / "research/m4_paired_replay").glob("*.py"):
            if path.name.startswith("test_"):
                continue
            for call in ast.walk(ast.parse(path.read_text())):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "run_all_workers_single_data"
                ):
                    with self.subTest(file=path.name, line=call.lineno):
                        self.assertLessEqual(len(call.args), 1)

    def test_paired_rpc_keyword_names_match_actual_worker_signatures(self) -> None:
        worker = ast.parse((ROOT / "research/m4_eligible_replay/worker.py").read_text())
        signatures = {
            node.name: {a.arg for a in node.args.args[1:] + node.args.kwonlyargs}
            for node in ast.walk(worker)
            if isinstance(node, ast.FunctionDef) and node.name.startswith("paired_")
        }
        for path in (ROOT / "research/m4_paired_replay").glob("*.py"):
            if path.name.startswith("test_"):
                continue
            for call in ast.walk(ast.parse(path.read_text())):
                if not (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "run_all_workers_single_data"
                    and call.args
                    and isinstance(call.args[0], ast.Constant)
                ):
                    continue
                name = call.args[0].value
                if name in signatures:
                    with self.subTest(file=path.name, method=name):
                        self.assertTrue(
                            {k.arg for k in call.keywords} <= signatures[name]
                        )


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
