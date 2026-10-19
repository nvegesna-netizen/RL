"""Regression checks for environment parity and credential-safe shell setup."""

import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from build_paired_engineering_manifest import ENGINEERING_TEST_ENV, quiet_shell_prefix
from research.m4_eligible_replay.preflight import MCORE_PREFIX
from research.m4_eligible_replay.test_preflight import PreflightTests


class EngineeringPreflightRepairTests(unittest.TestCase):
    def test_inherited_xtrace_cannot_print_credential_check(self) -> None:
        sentinel = "m4-fake-credential-must-never-appear-in-log"
        script = quiet_shell_prefix('set -euo pipefail\ntest -n "$HF_TOKEN"\n')
        result = subprocess.run(
            ["bash", "-x"],
            input=script,
            env={**os.environ, "HF_TOKEN": sentinel},
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertNotIn(sentinel, result.stdout + result.stderr)
        self.assertNotIn("test -n", result.stderr)

    def test_engineering_marker_is_in_clean_room_environment(self) -> None:
        self.assertEqual(ENGINEERING_TEST_ENV["M4_PAIRED_SYNTHETIC_ENGINEERING"], "1")
        self.assertEqual(ENGINEERING_TEST_ENV["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(ENGINEERING_TEST_ENV["NEMO_RL_PY_EXECUTABLES_SYSTEM"], "1")

    def test_wrong_prefix_test_runs_even_from_mcore_interpreter(self) -> None:
        case = PreflightTests("test_base_interpreter_rejected_before_backend_import")
        result = unittest.TestResult()
        with patch.object(sys, "prefix", str(MCORE_PREFIX)):
            case.run(result)
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(result.skipped, [])
        self.assertTrue(result.wasSuccessful(), result.errors + result.failures)

    def test_shell_suppression_is_first_statement(self) -> None:
        self.assertTrue(quiet_shell_prefix("echo setup\n").startswith("set +x\n"))


if __name__ == "__main__":
    unittest.main()
