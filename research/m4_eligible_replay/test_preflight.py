"""Lightweight checks for the full-environment preflight contract."""

from pathlib import Path
import sys
import unittest

from research.m4_eligible_replay.preflight import MCORE_PREFIX, inspect_environment


class PreflightTests(unittest.TestCase):
    def test_base_interpreter_rejected_before_backend_import(self) -> None:
        if Path(sys.prefix) == MCORE_PREFIX:
            self.skipTest("the real MCORE interpreter is checked by the full preflight")
        with self.assertRaisesRegex(ValueError, "expected proven MCORE prefix"):
            inspect_environment()

    def test_full_imports_and_actor_gate_are_not_path_only_checks(self) -> None:
        source = Path(__file__).with_name("preflight.py").read_text()
        for required in (
            "transformer_engine.pytorch",
            "megatron.bridge.training.checkpointing",
            "research.m4_eligible_replay.worker",
            "importlib.import_module",
            "probe.options",
            '"py_executable"',
            "driver/actor mismatch",
        ):
            self.assertIn(required, source)


if __name__ == "__main__":
    unittest.main()
