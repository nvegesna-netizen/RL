"""Verify deterministic initialization against pinned native validation points."""

import ast
from contextlib import contextmanager
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, patch

from research.m4_eligible_replay import test_evaluation_rng as rng_tests
from research.m4_eligible_replay.deterministic_backward import (
    MARKER,
    REQUIRED_ENV,
    apply_torch_controls,
    model_setup_controls,
    validate_environment,
    verify_torch_controls,
    verify_worker,
)


ENV = {MARKER: "1", **REQUIRED_ENV}


class FakeTorch:
    def __init__(self) -> None:
        self.strict, self.warn = False, False
        self.backends = SimpleNamespace(
            cudnn=SimpleNamespace(deterministic=False, benchmark=True),
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
        )

    def use_deterministic_algorithms(
        self, enabled: bool, *, warn_only: bool = False
    ) -> None:
        self.strict, self.warn = enabled, warn_only

    def are_deterministic_algorithms_enabled(self) -> bool:
        return self.strict

    def is_deterministic_algorithms_warn_only_enabled(self) -> bool:
        return self.warn


class DeterministicBackwardTests(unittest.TestCase):
    def test_required_environment_exact(self) -> None:
        validate_environment(ENV)
        for key in ENV:
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, "missing|mismatch"),
            ):
                validate_environment({**ENV, key: "wrong"})

    def test_missing_environment_rejected_before_torch_change(self) -> None:
        torch = FakeTorch()
        with self.assertRaisesRegex(ValueError, "marker"):
            apply_torch_controls(torch, {})
        self.assertFalse(torch.strict)

    def test_flags_are_strict_not_warn_only(self) -> None:
        torch = FakeTorch()
        apply_torch_controls(torch, ENV)
        self.assertTrue(torch.strict)
        self.assertFalse(torch.warn)
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)

    def test_warn_only_or_tf32_downgrade_rejected(self) -> None:
        torch = FakeTorch()
        apply_torch_controls(torch, ENV)
        torch.warn = True
        with self.assertRaisesRegex(ValueError, "not active"):
            verify_torch_controls(torch, ENV)
        torch.warn = False
        torch.backends.cuda.matmul.allow_tf32 = True
        with self.assertRaisesRegex(ValueError, "not active"):
            verify_torch_controls(torch, ENV)

    def test_model_controls_follow_native_performance_before_validation(self) -> None:
        torch = FakeTorch()
        events = []
        original = Mock(side_effect=lambda model, config: events.append("native"))
        setup = SimpleNamespace(_apply_performance_config=original)
        config = {"eligible_replay_deterministic": True}
        model = SimpleNamespace(
            deterministic_mode=False,
            cross_entropy_loss_fusion=True,
            attention_backend="auto",
        )
        with (
            patch.dict(os.environ, ENV),
            model_setup_controls(setup=setup, config=config, torch=torch),
        ):
            self.assertTrue(torch.strict)
            setup._apply_performance_config(model, config)
            self.assertTrue(model.deterministic_mode)
            self.assertFalse(model.cross_entropy_loss_fusion)
            self.assertEqual(model.attention_backend, "auto")
            events.append("native_validation_and_model_construction")
        self.assertIs(setup._apply_performance_config, original)
        self.assertEqual(events, ["native", "native_validation_and_model_construction"])

    def test_missing_native_hook_fails_closed_and_restores_binding(self) -> None:
        original = Mock()
        setup = SimpleNamespace(_apply_performance_config=original)
        with (
            patch.dict(os.environ, ENV),
            self.assertRaisesRegex(ValueError, "not reached"),
        ):
            with model_setup_controls(
                setup=setup,
                config={"eligible_replay_deterministic": True},
                torch=FakeTorch(),
            ):
                pass
        self.assertIs(setup._apply_performance_config, original)

    def test_native_initialization_error_preserved(self) -> None:
        original = Mock()
        setup = SimpleNamespace(_apply_performance_config=original)
        failure = RuntimeError("native model construction failed")
        with patch.dict(os.environ, ENV), self.assertRaises(RuntimeError) as caught:
            with model_setup_controls(
                setup=setup,
                config={"eligible_replay_deterministic": True},
                torch=FakeTorch(),
            ):
                raise failure
        self.assertIs(caught.exception, failure)
        self.assertIs(setup._apply_performance_config, original)

    def test_actual_worker_model_config_verified(self) -> None:
        torch = FakeTorch()
        apply_torch_controls(torch, ENV)
        model = SimpleNamespace(
            deterministic_mode=True, cross_entropy_loss_fusion=False
        )
        worker = SimpleNamespace(_get_model_config=lambda: model)
        with patch.dict(os.environ, ENV):
            verify_worker(worker, torch)
            model.deterministic_mode = False
            with self.assertRaisesRegex(ValueError, "actual model"):
                verify_worker(worker, torch)

    def test_actual_train_rechecks_modes_for_eval_and_training(self) -> None:
        fixture = rng_tests.EvaluationRngTests()
        fixture.setUp()
        fixture.worker.cfg = {"eligible_replay_deterministic": True}
        namespace = fixture.worker.train.__globals__
        check = Mock(side_effect=ValueError("mode downgraded"))
        namespace["verify_worker"] = check
        for eval_mode in (False, True):
            with self.assertRaisesRegex(ValueError, "downgraded"):
                fixture.worker.train("data", "loss", eval_mode=eval_mode)
        self.assertEqual(check.call_count, 2)
        fixture.call.assert_not_called()

    def test_actual_constructor_wraps_native_init_and_forwards_arguments(self) -> None:
        fixture = rng_tests.EvaluationRngTests()
        fixture.setUp()
        worker_class = type(fixture.worker)
        namespace = worker_class.__init__.__globals__
        events = []

        @contextmanager
        def initialize(config: dict):
            events.append("before_native_model")
            yield
            events.append("after_native_model")

        namespace["initialize_worker"] = initialize
        namespace["verify_worker"] = lambda *args: events.append("verify_actual_model")
        base = worker_class.__bases__[0]
        original = base.__init__

        def native(instance: Any, config: dict, *args: Any, **kwargs: Any) -> None:
            events.append("native")
            self.assertEqual(args, ("tokenizer",))
            self.assertEqual(
                kwargs, {"weights_path": "weights", "init_reference_model": False}
            )
            original(instance, config)

        with patch.object(base, "__init__", native):
            worker_class(
                {"eligible_replay_deterministic": True},
                "tokenizer",
                weights_path="weights",
                init_reference_model=False,
            )
        self.assertEqual(
            events,
            [
                "before_native_model",
                "native",
                "after_native_model",
                "verify_actual_model",
            ],
        )

    def test_actual_configure_worker_preserves_native_resources_and_propagates_env(
        self,
    ) -> None:
        fixture = rng_tests.EvaluationRngTests()
        fixture.setUp()
        worker_class = type(fixture.worker)
        namespace = worker_class.configure_worker.__globals__
        base = worker_class.__bases__[0]
        namespace.update(
            {
                "MegatronPolicyWorkerImpl": base,
                "os": os,
                "MARKER": MARKER,
                "REQUIRED_ENV": REQUIRED_ENV,
                "validate_environment": validate_environment,
            }
        )
        values = (
            {"num_gpus": 1},
            {"original": "kept"},
            {"native": True},
            {"native_runtime": True},
        )
        marker = "M4_PAIRED_SYNTHETIC_ENGINEERING"
        for value in (None, "0", "1", "unexpected"):
            with (
                self.subTest(engineering_marker=value),
                patch.object(
                    base,
                    "configure_worker",
                    Mock(return_value=(values[0], {"original": "kept"}, *values[2:])),
                    create=True,
                ),
                patch.dict(os.environ, ENV),
            ):
                os.environ.pop(marker, None)
                if value is not None:
                    os.environ[marker] = value
                result = worker_class.configure_worker(
                    1, bundle_indices=(0, [0]), num_gpus_per_node=1
                )
            expected = {"original": "kept", **ENV}
            if value == "1":
                expected[marker] = "1"
            self.assertIs(result[0], values[0])
            self.assertIs(result[2], values[2])
            self.assertIs(result[3], values[3])
            self.assertEqual(result[1], expected)

    def test_native_setup_hook_precedes_model_validation(self) -> None:
        root = Path(__file__).resolve().parents[2]
        tree = ast.parse((root / "nemo_rl/models/megatron/setup.py").read_text())
        function = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "setup_model_config"
        )
        calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)]
        hook = next(
            n.lineno
            for n in calls
            if isinstance(n.func, ast.Name) and n.func.id == "_apply_performance_config"
        )
        validation = next(
            n.lineno
            for n in calls
            if isinstance(n.func, ast.Attribute) and n.func.attr == "__post_init__"
        )
        bridge = next(
            n.lineno
            for n in calls
            if isinstance(n.func, ast.Name) and n.func.id == "_create_megatron_config"
        )
        self.assertLess(hook, validation)
        self.assertLess(validation, bridge)

    def test_actual_pinned_bridge_deterministic_validator_accepts_settings(
        self,
    ) -> None:
        path = (
            Path(__file__).resolve().parents[2]
            / "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src/megatron/bridge/training/config.py"
        )
        if not path.is_file():
            self.skipTest(
                "pinned Bridge validator is tested in the exact extracted package"
            )
        node = next(
            n
            for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.FunctionDef)
            and n.name == "_validate_and_apply_deterministic_mode"
        )
        namespace = {"os": os, "torch": FakeTorch()}
        exec(
            compile(
                ast.Module(body=[node], type_ignores=[]),
                "pinned-bridge-validator",
                "exec",
            ),
            namespace,
        )
        cfg = SimpleNamespace(
            model=SimpleNamespace(
                deterministic_mode=True, cross_entropy_loss_fusion=False
            )
        )
        with patch.dict(os.environ, ENV):
            namespace[node.name](cfg)
        self.assertTrue(namespace["torch"].strict)
        cfg.model.cross_entropy_loss_fusion = True
        with patch.dict(os.environ, ENV), self.assertRaises(AssertionError):
            namespace[node.name](cfg)

    def test_pinned_te_attention_checks_deterministic_environment(self) -> None:
        path = (
            Path(__file__).resolve().parents[2]
            / "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/extensions/transformer_engine.py"
        )
        if not path.is_file():
            self.skipTest("pinned TE wrapper is tested in the extracted package")
        node = next(
            n
            for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.If)
            and isinstance(n.test, ast.Attribute)
            and n.test.attr == "deterministic_mode"
        )
        code = compile(
            ast.Module(body=[node], type_ignores=[]),
            "pinned-te-deterministic-check",
            "exec",
        )
        namespace = {
            "os": os,
            "self": SimpleNamespace(config=SimpleNamespace(deterministic_mode=True)),
        }
        with patch.dict(os.environ, {"NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0"}):
            exec(code, namespace)
        with (
            patch.dict(os.environ, {"NVTE_ALLOW_NONDETERMINISTIC_ALGO": "1"}),
            self.assertRaises(RuntimeError),
        ):
            exec(code, namespace)


if __name__ == "__main__":
    unittest.main()
