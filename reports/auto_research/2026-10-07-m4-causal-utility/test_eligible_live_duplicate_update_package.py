"""Negative tests for the submitted-shell startup boundary; no backend imports."""

from pathlib import Path
import runpy
import unittest


VALIDATE = runpy.run_path(
    str(Path(__file__).with_name("validate_eligible_live_duplicate_update_package.py"))
)["validate_repaired_shell"]
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


if __name__ == "__main__":
    unittest.main()
