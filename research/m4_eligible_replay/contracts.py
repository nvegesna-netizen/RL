"""Dependency-free integrity and engineering-result contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO


def canonical_json(value: Any) -> bytes:
    """Serialize records without accepting NaN values."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def digest_file(path: Path) -> str:
    """Hash large files with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def safe_member(value: str) -> PurePosixPath:
    """Reject ambiguous or escaping artifact paths before any network request."""
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or str(path) != value
        or "\\" in value
        or not path.parts
    ):
        raise ValueError(f"unsafe artifact member: {value!r}")
    return path


def stream_member(
    stream: BinaryIO,
    destination: Path,
    *,
    expected_bytes: int | None,
    expected_hash: str,
) -> None:
    """Write a new member with bounded reads and exact size/hash validation."""
    digest = hashlib.sha256()
    count = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output:
        while chunk := stream.read(1 << 20):
            count += len(chunk)
            if count > (expected_bytes if expected_bytes is not None else 256 << 20):
                raise ValueError("artifact exceeds its authenticated size")
            output.write(chunk)
            digest.update(chunk)
    if (
        expected_bytes is not None and count != expected_bytes
    ) or digest.hexdigest() != expected_hash:
        raise ValueError("artifact size or SHA-256 mismatch")


def allowed_members(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Allow only checkpoint files, comparator tensors, and held-out tensors."""
    root = safe_member(manifest["checkpoint_root"])
    if str(root) != "checkpoint":
        raise ValueError("unexpected checkpoint root")
    rows = []
    for item in manifest["checkpoint_files"]:
        relative = safe_member(item["path"])
        rows.append(
            {
                "path": str(root / relative),
                "bytes": item["bytes"],
                "sha256": item["sha256"],
            }
        )
    base_ids = set(manifest["base_group_ids"])
    selected = [
        item
        for item in manifest["groups"]
        if item["group_id"] in base_ids or item["roles"] == ["heldout"]
    ]
    if len(base_ids) != 4 or len(selected) != 8:
        raise ValueError(
            "qualification requires four comparator and four held-out groups"
        )
    for item in selected:
        path = safe_member(item["raw_file"])
        if str(path) != f"groups/group-{item['index']:02d}.pt":
            raise ValueError("noncanonical group path")
        # Group file size is not present in the historical manifest. Hash it
        # independently; the transport binds actual byte counts in its receipt.
        rows.append(
            {"path": str(path), "bytes": None, "sha256": item["raw_file_sha256"]}
        )
    paths = [row["path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate capsule member")
    return rows


def validate_duplicate_replays(
    records: list[dict[str, Any]], *, protocol_hash: str, manifest_hash: str
) -> dict[str, Any]:
    """Require two identical-action one-update replays, never a treatment contrast."""
    if len(records) != 2 or [row["replicate"] for row in records] != ["a", "b"]:
        raise ValueError("both fixed fresh-process replicates are required")
    for record in records:
        if (
            record["protocol_sha256"] != protocol_hash
            or record["capsule_manifest_sha256"] != manifest_hash
        ):
            raise ValueError("replay provenance differs")
        if (
            record["action"] != "reward_variance_base"
            or record["optimizer_updates"] != 1
            or record["scheduler_sample_increment"] != 32
        ):
            raise ValueError("qualification action or update count differs")
        if (
            record["heldout_files_unchanged"] is not True
            or record["post_update_heldout_loss_computed"] is not False
        ):
            raise ValueError("held-out integrity or outcome embargo violated")
        if not all(
            record[key]
            for key in ("optimizer_restored", "scheduler_restored", "rng_restored")
        ):
            raise ValueError("full training state not restored")
        if (
            record["state_before"]["parameter_sha256"]
            == record["state_after"]["parameter_sha256"]
        ):
            raise ValueError("training update made no parameter change")
        before, after = record["state_before_update"], record["state_after"]
        if (
            after["optimizer_step"] - before["optimizer_step"] != 1
            or after["scheduler_num_steps"] - before["scheduler_num_steps"] != 32
        ):
            raise ValueError(
                "observed training counters do not show exactly one update"
            )
        if record["state_before"] != before:
            raise ValueError("pre-update evaluation mutated training state")
    for key in (
        "state_before",
        "state_before_update",
        "state_after",
        "base_group_ids",
        "heldout_group_ids",
    ):
        if records[0][key] != records[1][key]:
            raise ValueError(f"duplicate replay mismatch: {key}")
    if (
        abs(
            records[0]["pre_update_heldout_loss"]
            - records[1]["pre_update_heldout_loss"]
        )
        > 1e-6
    ):
        raise ValueError("pre-update held-out losses differ")
    return {
        "status": "PASS_DUPLICATE_ONE_UPDATE_REPLAY",
        "replicates": 2,
        "action": "reward_variance_base",
        "historical_capsule_excluded": True,
        "causal_contrast_computed": False,
        "post_update_heldout_loss_computed": False,
    }
