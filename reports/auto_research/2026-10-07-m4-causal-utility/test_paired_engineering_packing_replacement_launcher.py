"""Offline tests: validation never consumes authority; mismatches fail closed."""

import contextlib
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).with_name(
    "submit_paired_engineering_packing_replacement_once.py"
)
SPEC = importlib.util.spec_from_file_location("replacement_launcher", SCRIPT)
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


class ReplacementLauncherTests(unittest.TestCase):
    def test_valid_inputs_do_not_consume_guard(self) -> None:
        with patch.object(LAUNCHER.subprocess, "check_output", return_value=""):
            _, _, _, _, guard = LAUNCHER.validate()
        self.assertFalse(guard.exists())

    def test_changed_manifest_rejected(self) -> None:
        original = LAUNCHER.digest_file

        def changed(path: Path) -> str:
            return (
                "0" * 64
                if path.name == "m4-synthetic-paired-packing-repaired-manifest.json"
                else original(path)
            )

        with patch.object(LAUNCHER, "digest_file", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "authority/package"):
                LAUNCHER.validate()

    def test_changed_old_guard_rejected(self) -> None:
        original = LAUNCHER.digest_file

        def changed(path: Path) -> str:
            return (
                "0" * 64
                if path.name == "m4-synthetic-paired-engineering-one-shot-guard.json"
                else original(path)
            )

        with patch.object(LAUNCHER, "digest_file", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "consumed original guard"):
                LAUNCHER.validate()

    def test_consumed_replacement_rejected(self) -> None:
        original = Path.exists

        def exists(path: Path) -> bool:
            return path.name == f"{LAUNCHER.STEM}-one-shot-guard.json" or original(path)

        with patch.object(LAUNCHER.subprocess, "check_output", return_value=""):
            with patch.object(Path, "exists", exists):
                with self.assertRaisesRegex(FileExistsError, "never retry"):
                    LAUNCHER.validate()

    def test_validate_only_cannot_launch(self) -> None:
        with patch.object(
            LAUNCHER.subprocess, "check_output", return_value="test-commit"
        ):
            with patch.object(
                LAUNCHER,
                "validate",
                return_value=(
                    {"manifest_sha256": "test"},
                    Path(),
                    Path(),
                    Path(),
                    Path(),
                ),
            ):
                with patch.dict(
                    LAUNCHER.os.environ,
                    {"JET_GITLAB_TOKEN": "test", "HF_TOKEN": "test"},
                ):
                    with patch("sys.argv", [str(SCRIPT), "--validate-only"]):
                        with patch.object(LAUNCHER.subprocess, "run") as run:
                            with contextlib.redirect_stdout(io.StringIO()):
                                LAUNCHER.main()
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
