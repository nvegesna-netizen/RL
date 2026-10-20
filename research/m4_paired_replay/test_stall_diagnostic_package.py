"""Qualification must block training; excluded finalization must fail closed."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/auto_research/2026-10-07-m4-causal-utility"


def builder():
    spec = importlib.util.spec_from_file_location(
        "diagnostic_builder", REPORT / "build_watchdog_stall_diagnostic.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def block(template: str) -> str:
    return (
        template.split("<<'PY'\n", 1)[1]
        .rsplit("\nPY", 1)[0]
        .replace("@PROTOCOL_HASH@", "0" * 64)
    )


class StallDiagnosticPackageTests(unittest.TestCase):
    def test_native_qualification_rejects_skips_missing_commands_wrong_counts(self):
        for text, valid in [
            ("2 passed\n7 passed\n17 passed\n", True),
            ("2 passed\n7 passed\n17 passed\n1 skipped\n", False),
            ("2 passed\n7 passed\n", False),
            ("2 passed\n7 passed\n16 passed\n", False),
        ]:
            with self.subTest(text=text), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / "m4-eligible-live-v2a09-tests.log").write_text(text)
                with patch("sys.argv", ["-", temp]):
                    if valid:
                        exec(
                            compile(
                                block(builder().QUALIFICATION),
                                "native-test-gate",
                                "exec",
                            ),
                            {},
                        )
                    else:
                        with self.assertRaises(AssertionError):
                            exec(
                                compile(
                                    block(builder().QUALIFICATION),
                                    "native-test-gate",
                                    "exec",
                                ),
                                {},
                            )
                self.assertEqual((root / "watchdog-qualification.json").exists(), valid)

    def test_watchdog_terminal_record_excludes_scientific_admission(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "watchdog-qualification.json").write_text(
                json.dumps({"status": "PASS_BEFORE_DIAGNOSTIC_TRAINING"})
            )
            (root / "m4-eligible-live-v2a09-run.log").write_text(
                "eligible live capture made no learner progress"
            )
            watchdog = {
                "schema": "eligible-live-progress-timeout-v1",
                "trajectory_identity": "v2a09",
                "trajectory_seed": 20261209,
                "timeout_seconds": 1200,
                "seconds_without_learner_progress": 1201,
                "controller_state": {"buffer_end_weight_version_counts": {}},
                "tasks": {
                    "rollout_pump": {"stack": [], "done": True, "cancelled": False},
                    "train_pump": {"stack": ["test"]},
                },
            }
            (root / "m4-eligible-live-v2a09-watchdog.json").write_text(
                json.dumps(watchdog)
            )
            with patch("sys.argv", ["-", temp, "1"]):
                exec(
                    compile(block(builder().FINALIZE), "excluded-finalizer", "exec"), {}
                )
            record = json.loads((root / "stall-diagnostic-result.json").read_bytes())
            self.assertEqual(record["status"], "WATCHDOG_DIAGNOSTIC_COLLECTED")
            self.assertFalse(record["scientific_admission_permitted"])
            self.assertFalse(record["primary_outcomes_opened"])

    def test_unexpected_failure_is_recorded_but_not_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "watchdog-qualification.json").write_text(
                json.dumps({"status": "PASS_BEFORE_DIAGNOSTIC_TRAINING"})
            )
            (root / "m4-eligible-live-v2a09-run.log").write_text("unrelated exception")
            with patch("sys.argv", ["-", temp, "1"]), self.assertRaises(AssertionError):
                exec(
                    compile(block(builder().FINALIZE), "excluded-finalizer", "exec"), {}
                )
            record = json.loads((root / "stall-diagnostic-result.json").read_bytes())
            self.assertEqual(record["status"], "UNEXPECTED_RUNTIME_FAILURE")


if __name__ == "__main__":
    unittest.main()
