"""Negative bridge gates and actual worker dispatch without CUDA dependencies."""

import ast
import copy
import json
from pathlib import Path
from unittest.mock import Mock
import unittest

from research.m4_eligible_replay.minimal_observer import validate_bridge, WORKER_MODULE
from research.m4_eligible_replay.test_contracts import record
from research.m4_eligible_replay import test_evaluation_rng as rng_tests


class MinimalObserverTests(unittest.TestCase):
    def setUp(self) -> None:
        report = (
            Path(__file__).resolve().parents[2]
            / "reports/auto_research/2026-10-07-m4-causal-utility"
        )
        self.reference = json.loads(
            (
                report
                / "eligible_live_duplicate_update_deterministic_backward_terminal_result.json"
            ).read_bytes()
        )
        self.records = [record(x) for x in ("a", "b")]
        self.reference["capsule_manifest_sha256"] = "manifest"
        for row in self.records:
            row["state_before"] = copy.deepcopy(self.reference["state_before_common"])
            row["state_before_update"] = copy.deepcopy(
                self.reference["state_before_common"]
            )
            row["state_after"] = copy.deepcopy(self.reference["state_after_common"])
        runtime = copy.deepcopy(self.reference["runtime"])
        runtime["module_origins"][WORKER_MODULE]["sha256"] = "a" * 64
        self.boundaries = [
            {
                "schema": "m4-minimal-observer-boundaries-v1",
                "replicate": x,
                "protocol_sha256": "protocol",
                "capsule_manifest_sha256": "manifest",
                "before": {
                    "training_trace_enabled": False,
                    "runtime": copy.deepcopy(runtime),
                },
                "after": {
                    "training_trace_enabled": False,
                    "runtime": copy.deepcopy(runtime),
                },
            }
            for x in ("a", "b")
        ]

    def validate(self) -> dict:
        return validate_bridge(
            self.records,
            self.boundaries,
            reference=self.reference,
            protocol_hash="protocol",
            worker_hash="a" * 64,
        )

    def test_complete_bridge_pass(self) -> None:
        self.assertTrue(self.validate()["cross_instrumentation_full_state_equal"])

    def test_reference_post_state_mismatch_rejected_even_when_duplicates_match(
        self,
    ) -> None:
        for row in self.records:
            row["state_after"]["parameter_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "cross-instrumentation"):
            self.validate()

    def test_duplicate_mismatch_rejected(self) -> None:
        self.records[1]["state_after"]["optimizer_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "duplicate replay"):
            self.validate()

    def test_trace_downgrade_rejected(self) -> None:
        self.boundaries[0]["before"]["training_trace_enabled"] = True
        with self.assertRaisesRegex(ValueError, "disabled"):
            self.validate()

    def test_environment_or_mode_changes_rejected(self) -> None:
        for key in ("deterministic_algorithms", "cudnn_deterministic"):
            with self.subTest(key=key):
                old = self.boundaries[0]["before"]["runtime"][key]
                self.boundaries[0]["before"]["runtime"][key] = False
                with self.assertRaisesRegex(ValueError, "runtime changed"):
                    self.validate()
                self.boundaries[0]["before"]["runtime"][key] = old

    def test_worker_hash_must_authenticate(self) -> None:
        self.boundaries[0]["before"]["runtime"]["module_origins"][WORKER_MODULE][
            "sha256"
        ] = "b" * 64
        with self.assertRaisesRegex(ValueError, "worker source"):
            self.validate()

    def test_capsule_identity_rejected(self) -> None:
        self.boundaries[0]["capsule_manifest_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "identity"):
            self.validate()

    def test_no_outcome_permitted(self) -> None:
        self.records[0]["post_update_heldout_loss_computed"] = True
        with self.assertRaisesRegex(ValueError, "embargo"):
            self.validate()

    def test_actual_minimal_worker_bypasses_deep_trace(self) -> None:
        fixture = rng_tests.EvaluationRngTests()
        fixture.setUp()
        fixture.worker.cfg = {
            "eligible_replay_deterministic": True,
            "eligible_replay_minimal_observer": True,
        }
        namespace = type(fixture.worker).train.__globals__
        namespace["verify_worker"] = Mock()
        namespace["trace_training"] = Mock(
            side_effect=AssertionError("deep observer entered")
        )
        fixture.worker.train("batch", "loss")
        namespace["verify_worker"].assert_called_once()
        namespace["trace_training"].assert_not_called()
        fixture.call.assert_called_once()

    def test_actual_worker_rejects_trace_before_update(self) -> None:
        fixture = rng_tests.EvaluationRngTests()
        fixture.setUp()
        fixture.worker.cfg = {
            "eligible_replay_deterministic": True,
            "eligible_replay_minimal_observer": True,
            "eligible_replay_training_trace": {},
        }
        type(fixture.worker).train.__globals__["verify_worker"] = Mock()
        with self.assertRaisesRegex(ValueError, "no training trace"):
            fixture.worker.train("batch", "loss")
        fixture.call.assert_not_called()

    def test_actual_runner_rejects_invalid_modes_before_ray(self) -> None:
        tree = ast.parse(Path(__file__).with_name("run_qualification.py").read_text())
        fn = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"
        )
        # Execute the actual pre-setup guards; stop before any Ray allocation.
        end = next(
            i
            for i, n in enumerate(fn.body)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "root" for t in n.targets)
        )
        fn.body = fn.body[:end]
        fn.body.append(ast.Return(ast.Constant("PASS_MODE_GUARDS")))
        ns = {
            "Path": Path,
            "Any": object,
            "apply_torch_controls": Mock(),
            "torch": object(),
            "os": __import__("os"),
        }
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                "actual-mode-guards",
                "exec",
            ),
            ns,
        )
        common = dict(
            capsule=Path("."),
            config_path=Path("."),
            protocol_path=Path("."),
            replicate="a",
            minimal_observer=True,
        )
        for options in (
            {},
            {"deterministic_backward": True},
            {
                "deterministic_backward": True,
                "training_diagnostics": True,
                "state_diagnostics_dir": Path("."),
            },
        ):
            with self.assertRaises(ValueError):
                ns["run"](**common, **options)
        self.assertEqual(
            ns["run"](
                **common, deterministic_backward=True, state_diagnostics_dir=Path(".")
            ),
            "PASS_MODE_GUARDS",
        )


if __name__ == "__main__":
    unittest.main()
