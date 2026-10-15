"""Offline regressions for all replay socket roots and partial-start cleanup."""

from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from research.m4_eligible_replay.ray_startup import (
    ROOT_PREFIXES,
    SOCKET_SUFFIX,
    allocate_ray_root,
    ray_cleanup,
    validate_ray_root,
)


class RayStartupTests(unittest.TestCase):
    def test_exact_failed_root_is_rejected(self) -> None:
        root = Path("/workspace/m4-full-environment-preflight-ray")
        observed = str(root) + (
            "/session_2026-10-15_03-53-54_937526_3164759/sockets/plasma_store"
        )
        self.assertEqual(len(observed.encode()), 108)
        with self.assertRaisesRegex(ValueError, "socket budget exceeded"):
            validate_ray_root(root)

    def test_all_unique_short_roots_have_headroom(self) -> None:
        for prefix in ROOT_PREFIXES.values():
            with self.subTest(prefix=prefix):
                self.assertEqual(
                    validate_ray_root(Path("/tmp") / (prefix + "abcdefgh")), 92
                )

    def test_utf8_bytes_not_character_count(self) -> None:
        root = Path("/tmp/" + "é" * 15)
        self.assertLessEqual(len(str(root) + SOCKET_SUFFIX), 107)
        with self.assertRaisesRegex(ValueError, "socket budget exceeded"):
            validate_ray_root(root)

    def test_exact_budget_boundary(self) -> None:
        overhead = len(("/tmp/" + SOCKET_SUFFIX).encode())
        self.assertEqual(validate_ray_root(Path("/tmp/" + "x" * (107 - overhead))), 107)
        with self.assertRaises(ValueError):
            validate_ray_root(Path("/tmp/" + "x" * (108 - overhead)))

    def test_invalid_paths_rejected(self) -> None:
        for name in ("relative", "/tmp/../bad", "/tmp/bad\0"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_ray_root(Path(name))

    def test_stage_validation_precedes_allocation(self) -> None:
        with patch("tempfile.mkdtemp") as allocator:
            with self.assertRaises(ValueError):
                allocate_ray_root("unexpected")
            allocator.assert_not_called()

    def test_allocator_uses_explicit_tmp_and_checks_returned_path(self) -> None:
        for stage, prefix in ROOT_PREFIXES.items():
            with patch(
                "tempfile.mkdtemp", return_value="/tmp/" + prefix + "abcdefgh"
            ) as allocator:
                root = allocate_ray_root(stage)
                allocator.assert_called_once_with(prefix=prefix, dir="/tmp")
                self.assertEqual(validate_ray_root(root), 92)
        with patch("tempfile.mkdtemp", return_value="/tmp/" + "x" * 100):
            with self.assertRaisesRegex(ValueError, "socket budget exceeded"):
                allocate_ray_root("preflight")

    def test_real_allocation_is_unique(self) -> None:
        roots = [allocate_ray_root(stage) for stage in ("preflight", "a", "b", "a")]
        try:
            self.assertEqual(len(set(roots)), 4)
            for root in roots:
                self.assertTrue(root.is_dir())
                self.assertLessEqual(validate_ray_root(root), 107)
        finally:
            # These are exact, newly-created empty test directories only.
            for root in roots:
                root.rmdir()

    def test_cleanup_on_success(self) -> None:
        shutdown = Mock()
        with ray_cleanup(shutdown):
            pass
        shutdown.assert_called_once_with()

    def test_cleanup_on_startup_failure(self) -> None:
        shutdown = Mock()
        failure = OSError("startup failed")
        with self.assertRaises(OSError) as caught:
            with ray_cleanup(shutdown):
                raise failure
        self.assertIs(caught.exception, failure)
        shutdown.assert_called_once_with()

    def test_primary_failure_survives_cleanup_failure(self) -> None:
        failure = OSError("startup failed")
        shutdown = Mock(side_effect=RuntimeError("cleanup failed"))
        with self.assertRaises(OSError) as caught:
            with ray_cleanup(shutdown):
                raise failure
        self.assertIs(caught.exception, failure)
        self.assertIn("cleanup failed", failure.__notes__[0])

    def test_cleanup_failure_without_primary_stays_fatal(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            with ray_cleanup(Mock(side_effect=RuntimeError("cleanup failed"))):
                pass


if __name__ == "__main__":
    unittest.main()
