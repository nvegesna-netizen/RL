"""Lightweight checks for the full-environment preflight contract."""

import ast
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, patch

from research.m4_eligible_replay import preflight
from research.m4_eligible_replay.preflight import MCORE_PREFIX, inspect_environment
from research.m4_eligible_replay.ray_startup import ray_cleanup, validate_ray_root
from research.m4_eligible_replay.deterministic_backward import MARKER, REQUIRED_ENV


class PreflightTests(unittest.TestCase):
    def test_base_interpreter_rejected_before_backend_import(self) -> None:
        with (
            patch.object(sys, "prefix", "/m4-test-deliberately-wrong-prefix"),
            patch.object(preflight, "initialize_bundled_megatron") as initialize,
            self.assertRaisesRegex(ValueError, "expected proven MCORE prefix"),
        ):
            inspect_environment()
        initialize.assert_not_called()

    def test_full_imports_and_actor_gate_are_not_path_only_checks(self) -> None:
        source = Path(__file__).with_name("preflight.py").read_text()
        for required in (
            "transformer_engine.pytorch",
            "megatron.bridge.training.checkpointing",
            "research.m4_eligible_replay.worker",
            "research.m4_eligible_replay.run_qualification",
            "importlib.import_module",
            "probe.options",
            '"py_executable"',
            "driver/actor mismatch",
        ):
            self.assertIn(required, source)

    def exercise_gate(
        self,
        *,
        startup_error: OSError | None = None,
        actor_changes: dict[str, Any] | None = None,
        expected_error: str = "driver/actor mismatch",
    ) -> ModuleType:
        driver = {
            "prefix": str(MCORE_PREFIX),
            "torch_version": "test-torch",
            "ray_version": "test-ray",
            "python_version": "test-python",
            "core_path": "test-core",
            "bridge_checkpointing_path": "test-bridge",
            "transferqueue_commit": "test-commit",
            "strict_deterministic_process": True,
        }
        actor = {**copy.deepcopy(driver), **(actor_changes or {})}
        ray = ModuleType("ray")
        ray.shutdown = Mock()
        ray.remote = Mock()
        ray.get = Mock(return_value=actor)
        cluster = ModuleType("nemo_rl.distributed.virtual_cluster")
        cluster.init_ray = Mock(side_effect=startup_error)
        modules = {
            "ray": ray,
            "nemo_rl": ModuleType("nemo_rl"),
            "nemo_rl.distributed": ModuleType("nemo_rl.distributed"),
            "nemo_rl.distributed.virtual_cluster": cluster,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            with (
                redirect_stdout(io.StringIO()),
                patch.dict(sys.modules, modules),
                patch.object(preflight, "inspect_environment", return_value=driver),
                patch.object(
                    preflight,
                    "allocate_ray_root",
                    return_value=Path("/tmp/m4pf-abcdefgh"),
                ),
            ):
                if startup_error:
                    with self.assertRaises(OSError) as caught:
                        preflight.run(output)
                    self.assertIs(caught.exception, startup_error)
                    self.assertFalse(output.exists())
                    ray.remote.assert_not_called()
                elif actor_changes:
                    with self.assertRaisesRegex(ValueError, expected_error):
                        preflight.run(output)
                    self.assertFalse(output.exists())
                else:
                    preflight.run(output)
                    self.assertIn(
                        "PASS_FULL_DRIVER_AND_ACTOR_IMPORT_PREFLIGHT",
                        output.read_text(),
                    )
            ray.shutdown.assert_called_once_with()
            cluster.init_ray.assert_called_once_with(log_dir="/tmp/m4pf-abcdefgh")
            return ray

    def test_actual_preflight_cleanup_covers_initialization(self) -> None:
        self.exercise_gate(startup_error=OSError("init failed"))

    def test_actual_preflight_checks_actor_identity(self) -> None:
        for key in ("ray_version", "python_version", "prefix", "core_path"):
            with self.subTest(key=key):
                self.exercise_gate(actor_changes={key: "wrong"})

    def test_actual_preflight_success_retains_isolated_probe(self) -> None:
        ray = self.exercise_gate()
        ray.remote.assert_called_once_with(num_cpus=1, num_gpus=0)
        probe = ray.remote.return_value.return_value
        env = probe.options.call_args.kwargs["runtime_env"]
        self.assertEqual(env["py_executable"], str(preflight.MCORE_PYTHON))
        self.assertEqual(ray.get.call_args.kwargs["timeout"], 300)

    def test_strict_probe_environment_is_explicitly_propagated(self) -> None:
        with patch.dict(preflight.os.environ, {MARKER: "1", **REQUIRED_ENV}):
            ray = self.exercise_gate()
        environment = ray.remote.return_value.return_value.options.call_args.kwargs[
            "runtime_env"
        ]["env_vars"]
        for name, value in {MARKER: "1", **REQUIRED_ENV}.items():
            self.assertEqual(environment[name], value)

    def test_strict_probe_downgrade_is_rejected(self) -> None:
        with patch.dict(preflight.os.environ, {MARKER: "1", **REQUIRED_ENV}):
            self.exercise_gate(
                actor_changes={"strict_deterministic_process": False},
                expected_error="strict deterministic driver/actor",
            )

    def test_replay_wrapper_cleans_setup_failure_for_both_replicates(self) -> None:
        # Execute the actual wrapper without importing optional CUDA dependencies.
        source = Path(__file__).with_name("run_qualification.py").read_text()
        function = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == "run"
        )
        for replicate in ("a", "b"):
            shutdown = Mock()
            failure = OSError("replay setup failed")
            inner = Mock(side_effect=failure)
            allocate = Mock(return_value=Path("/tmp/m4r" + replicate + "-abcdefgh"))
            namespace = {
                "Path": Path,
                "Any": Any,
                "json": __import__("json"),
                "allocate_ray_root": allocate,
                "validate_ray_root": validate_ray_root,
                "ray_cleanup": ray_cleanup,
                "ray": SimpleNamespace(shutdown=shutdown),
                "_run": inner,
            }
            exec(
                compile(
                    ast.Module(body=[function], type_ignores=[]), "wrapper", "exec"
                ),
                namespace,
            )
            with redirect_stdout(io.StringIO()), self.assertRaises(OSError) as caught:
                namespace["run"](
                    capsule=Path("/unused"),
                    config_path=Path("/unused"),
                    protocol_path=Path("/unused"),
                    replicate=replicate,
                )
            self.assertIs(caught.exception, failure)
            allocate.assert_called_once_with(replicate)
            shutdown.assert_called_once_with()
            self.assertEqual(
                inner.call_args.kwargs["ray_log_dir"], allocate.return_value
            )


if __name__ == "__main__":
    unittest.main()
