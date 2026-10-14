"""Small clean-room tests without Ray, CUDA, or training dependencies."""

import copy
import hashlib
import io
import tempfile
import unittest
from pathlib import Path

from research.m4_eligible_replay.contracts import (
    allowed_members,
    safe_member,
    stream_member,
    validate_duplicate_replays,
)


def record(label: str) -> dict:
    before = {
        "parameter_sha256": "old",
        "optimizer_sha256": "optim",
        "scheduler_sha256": "sched",
        "rng_sha256": "rng",
        "optimizer_step": 7,
        "scheduler_num_steps": 224,
    }
    return {
        "replicate": label,
        "protocol_sha256": "protocol",
        "capsule_manifest_sha256": "manifest",
        "action": "reward_variance_base",
        "optimizer_updates": 1,
        "scheduler_sample_increment": 32,
        "heldout_files_unchanged": True,
        "post_update_heldout_loss_computed": False,
        "optimizer_restored": True,
        "scheduler_restored": True,
        "rng_restored": True,
        "state_before": before,
        "state_before_update": copy.deepcopy(before),
        "state_after": {
            **before,
            "parameter_sha256": "new",
            "optimizer_step": 8,
            "scheduler_num_steps": 256,
        },
        "base_group_ids": ["b"],
        "heldout_group_ids": ["h"],
        "pre_update_heldout_loss": 0.142,
    }


class ContractTests(unittest.TestCase):
    def test_duplicate_update_pass(self) -> None:
        result = validate_duplicate_replays(
            [record("a"), record("b")],
            protocol_hash="protocol",
            manifest_hash="manifest",
        )
        self.assertEqual(result["status"], "PASS_DUPLICATE_ONE_UPDATE_REPLAY")
        self.assertFalse(result["causal_contrast_computed"])

    def test_bad_state_and_outcome_controls(self) -> None:
        mutations = [
            ("rng_restored", False),
            ("heldout_files_unchanged", False),
            ("action", "m4_shield"),
            ("optimizer_updates", 2),
            ("post_update_heldout_loss_computed", True),
        ]
        for key, value in mutations:
            with self.subTest(key=key):
                rows = [record("a"), record("b")]
                rows[1][key] = value
                with self.assertRaises(ValueError):
                    validate_duplicate_replays(
                        rows, protocol_hash="protocol", manifest_hash="manifest"
                    )

    def test_observed_counters_not_declared_counts(self) -> None:
        rows = [record("a"), record("b")]
        rows[1]["state_after"]["optimizer_step"] = 9
        with self.assertRaises(ValueError):
            validate_duplicate_replays(
                rows, protocol_hash="protocol", manifest_hash="manifest"
            )

    def test_pre_evaluation_mutation_rejected(self) -> None:
        rows = [record("a"), record("b")]
        rows[0]["state_before_update"]["rng_sha256"] = "changed"
        with self.assertRaises(ValueError):
            validate_duplicate_replays(
                rows, protocol_hash="protocol", manifest_hash="manifest"
            )

    def test_full_state_mismatch_rejected(self) -> None:
        rows = [record("a"), record("b")]
        rows[1]["state_after"]["optimizer_sha256"] = "different moments"
        with self.assertRaises(ValueError):
            validate_duplicate_replays(
                rows, protocol_hash="protocol", manifest_hash="manifest"
            )

    def test_paths_fail_closed(self) -> None:
        for value in (
            "../weights",
            "/weights",
            "a/../weights",
            "a//weights",
            "a\\weights",
            "",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                safe_member(value)

    def test_stream_size_hash_and_no_overwrite(self) -> None:
        payload = b"checkpoint" * 500000
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "member"
            stream_member(
                io.BytesIO(payload),
                destination,
                expected_bytes=len(payload),
                expected_hash=hashlib.sha256(payload).hexdigest(),
            )
            self.assertEqual(destination.stat().st_size, len(payload))
            with self.assertRaises(FileExistsError):
                stream_member(
                    io.BytesIO(payload),
                    destination,
                    expected_bytes=len(payload),
                    expected_hash=hashlib.sha256(payload).hexdigest(),
                )
            for name, size, digest in (
                ("bad_size", 1, hashlib.sha256(payload).hexdigest()),
                ("bad_hash", len(payload), "0" * 64),
            ):
                with self.assertRaises(ValueError):
                    stream_member(
                        io.BytesIO(payload),
                        Path(directory) / name,
                        expected_bytes=size,
                        expected_hash=digest,
                    )

    def test_member_allowlist_excludes_shield_only_groups(self) -> None:
        groups = [
            {
                "index": index,
                "group_id": str(index),
                "roles": ["base"]
                if index < 4
                else (["heldout"] if index >= 8 else ["shield"]),
                "raw_file": f"groups/group-{index:02d}.pt",
                "raw_file_sha256": "0" * 64,
            }
            for index in range(12)
        ]
        manifest = {
            "checkpoint_root": "checkpoint",
            "checkpoint_files": [
                {"path": "policy/weights/file", "bytes": 14 << 30, "sha256": "0" * 64}
            ],
            "base_group_ids": [str(index) for index in range(4)],
            "groups": groups,
        }
        members = allowed_members(manifest)
        self.assertEqual(len(members), 9)
        self.assertFalse(
            any(
                f"group-{index:02d}.pt" in row["path"]
                for row in members
                for index in range(4, 8)
            )
        )
        self.assertEqual(members[0]["bytes"], 14 << 30)


if __name__ == "__main__":
    unittest.main()
