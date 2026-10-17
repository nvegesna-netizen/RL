"""Negative tests for the submitted-shell startup boundary; no backend imports."""

from pathlib import Path
import runpy
import unittest


VALIDATE = runpy.run_path(
    str(Path(__file__).with_name("validate_eligible_live_duplicate_update_package.py"))
)["validate_repaired_shell"]
VALIDATE_DETERMINISTIC = runpy.run_path(
    str(Path(__file__).with_name("validate_eligible_live_duplicate_update_package.py"))
)["validate_deterministic_shell"]
LAYOUT = '"$PYTHON" -m research.m4_eligible_replay.ray_startup > layout.json\n'
PREFLIGHT = '"$PYTHON" -m research.m4_eligible_replay.preflight --output env.json\n'
TRANSPORT = '"$PYTHON" -m research.m4_eligible_replay.transport --output capsule\n'
REPLAY = 'for replicate in a b; do\n"$PYTHON" -m research.m4_eligible_replay.run_qualification --replicate "$replicate"\ndone\n'


class PackageShellTests(unittest.TestCase):
    def test_repaired_shell_passes(self) -> None:
        VALIDATE(LAYOUT + PREFLIGHT + TRANSPORT + REPLAY)

    def test_old_root_override_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit Ray root"):
            VALIDATE(
                LAYOUT + PREFLIGHT + TRANSPORT + REPLAY + "--ray-log-dir /tmp/unsafe"
            )

    def test_missing_layout_rejected(self) -> None:
        with self.assertRaises(ValueError):
            VALIDATE(PREFLIGHT + TRANSPORT + REPLAY)

    def test_transport_before_import_gate_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "precede checkpoint"):
            VALIDATE(LAYOUT + TRANSPORT + PREFLIGHT + REPLAY)

    def test_missing_second_replicate_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate replay loop"):
            VALIDATE(LAYOUT + PREFLIGHT + TRANSPORT + REPLAY.replace("a b;", "a;"))

    def test_duplicate_invocation_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate replay loop"):
            VALIDATE(LAYOUT + PREFLIGHT + TRANSPORT + REPLAY + REPLAY)


class DeterministicShellTests(unittest.TestCase):
    def shell(self) -> str:
        return (
            "export M4_ELIGIBLE_REPLAY_DETERMINISTIC=1\nexport CUBLAS_WORKSPACE_CONFIG=:4096:8\n"
            "export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0\nexport NCCL_ALGO=Ring\n"
            '"$PYTHON" - "$RUN_REPO"\n'
            '"$PYTHON" -m unittest research.m4_eligible_replay.test_deterministic_backward -v\n'
            "replay --training-diagnostics --deterministic-backward --output result.json\n"
            "analyze_training_trace --require-identical --assets /assets\n"
        )

    def test_early_environment_and_replay_flag(self) -> None:
        VALIDATE_DETERMINISTIC(self.shell())

    def test_missing_environment_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "environment"):
            VALIDATE_DETERMINISTIC(
                self.shell().replace("export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0\n", "")
            )

    def test_late_environment_rejected(self) -> None:
        control = "export CUBLAS_WORKSPACE_CONFIG=:4096:8\n"
        with self.assertRaisesRegex(ValueError, "environment"):
            VALIDATE_DETERMINISTIC(self.shell().replace(control, "") + control)

    def test_duplicate_environment_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "environment"):
            VALIDATE_DETERMINISTIC("export NCCL_ALGO=Ring\n" + self.shell())

    def test_missing_replay_flag_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "replay flag"):
            VALIDATE_DETERMINISTIC(
                self.shell().replace(" --deterministic-backward", "")
            )

    def test_runtime_unit_gate_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "tests missing"):
            VALIDATE_DETERMINISTIC(
                self.shell().replace(
                    " research.m4_eligible_replay.test_deterministic_backward -v", ""
                )
            )


if __name__ == "__main__":
    unittest.main()
