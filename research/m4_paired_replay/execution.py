"""Backend-independent one-arm execution and logical outcome embargo."""

from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
from typing import Any, Protocol

from research.m4_eligible_replay.contracts import canonical_json, digest_file
from research.m4_eligible_replay.analyze_training_trace import validate_runtime
from research.m4_paired_replay.contracts import (
    ARMS,
    admit,
    arm_groups,
    capture_gate,
    object_hash,
    validate_transition,
)
from research.m4_paired_replay.runtime import verify_reference_runtime
from research.m4_eligible_replay.training_trace import observer_cleanup


RECEIPT_FIELDS = {
    "schema",
    "status",
    "identity",
    "arm",
    "seed",
    "manifest_sha256",
    "capture_gate_sha256",
    "selected_group_ids",
    "heldout_group_ids",
    "state_before",
    "state_after",
    "runtime",
    "heldout_files_unchanged",
    "pre_evaluation_state_unchanged",
    "post_evaluation_state_unchanged",
    "optimizer_updates",
    "scheduler_sample_increment",
    "deep_training_trace_enabled",
    "outcome_sha256",
    "outcomes_opened",
}


class Backend(Protocol):
    """One freshly restored process; the caller must never reuse it for another arm."""

    def snapshot(self) -> dict: ...
    def runtime(self) -> dict: ...
    def evaluate(self) -> float: ...
    def update(self) -> None: ...
    def verify_inputs(self) -> None: ...
    def close(self) -> None: ...


def strict_runtime(runtime: dict) -> None:
    validate_runtime(runtime)
    verify_reference_runtime(runtime)
    if (
        runtime["deterministic_algorithms"] is not True
        or runtime["deterministic_warn_only"] is not False
        or runtime["cudnn_deterministic"] is not True
        or runtime["cudnn_benchmark"] is not False
        or runtime["matmul_allow_tf32"] is not False
        or runtime["model_modes"]["deterministic_mode"] != "True"
        or runtime["model_modes"]["cross_entropy_loss_fusion"] != "False"
        or any(
            runtime["environment"][k] != v
            for k, v in {
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
                "NCCL_ALGO": "Ring",
            }.items()
        )
    ):
        raise ValueError("actual strict runtime downgraded")


def finite(value: Any) -> float:
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError("nonfinite or non-scalar endpoint")
    return float(value)


def execute_arm(
    *,
    backend: Backend,
    plan: dict,
    captures: list[dict],
    identity: str,
    arm: str,
    manifest_bytes: bytes,
    private_dir: Path,
    receipt_path: Path,
) -> dict:
    """Write no loss into public receipts; close the backend on every failure.

    Sealing is a logical access embargo and private file permissions, not
    encryption. Logs and Ray files are outcome-bearing and must be kept private.
    """
    with observer_cleanup() as cleanup:
        cleanup.callback(backend.close)
        gate, manifest = admit(plan, captures, identity, manifest_bytes)
        selected = arm_groups(manifest, arm)
        if private_dir.exists() or receipt_path.exists():
            raise FileExistsError("arm output already exists")
        private_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
        backend.verify_inputs()
        runtime_before = backend.runtime()
        strict_runtime(runtime_before)
        before = backend.snapshot()
        pre = finite(backend.evaluate())
        before_update = backend.snapshot()
        if before != before_update:
            raise ValueError("pre-update evaluation mutated state")
        backend.update()
        after = backend.snapshot()
        post = finite(backend.evaluate())
        after_eval = backend.snapshot()
        validate_transition(before, before_update, after, after_eval, manifest)
        backend.verify_inputs()
        runtime_after = backend.runtime()
        strict_runtime(runtime_after)
        if runtime_before != runtime_after:
            raise ValueError("runtime changed during arm")
        outcome = {
            "schema": "m4-paired-private-outcome-v1",
            "identity": identity,
            "arm": arm,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "pre_loss": pre,
            "post_loss": post,
            "change": post - pre,
        }
        sealed = private_dir / "outcome.json"
        fd = os.open(sealed, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical_json(outcome))
        receipt = {
            "schema": "m4-paired-arm-receipt-v1",
            "status": "COMPLETE_OUTCOME_SEALED",
            "identity": identity,
            "arm": arm,
            "seed": manifest["trajectory_seed"],
            "manifest_sha256": outcome["manifest_sha256"],
            "capture_gate_sha256": object_hash(gate),
            "selected_group_ids": [r["group_id"] for r in selected],
            "heldout_group_ids": [r["group_id"] for r in manifest["groups"][8:]],
            "state_before": before,
            "state_after": after,
            "runtime": runtime_before,
            "heldout_files_unchanged": True,
            "pre_evaluation_state_unchanged": True,
            "post_evaluation_state_unchanged": True,
            "optimizer_updates": 1,
            "scheduler_sample_increment": 32,
            "deep_training_trace_enabled": False,
            "outcome_sha256": digest_file(sealed),
            "outcomes_opened": False,
        }
    # A close/cleanup failure prevents the public completion receipt.
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("xb") as stream:
        stream.write(canonical_json(receipt))
    return receipt


def release_outcomes(
    *,
    plan: dict,
    captures: list[dict],
    manifests: dict[str, bytes],
    receipts: dict[tuple[str, str], dict],
    outcome_paths: dict[tuple[str, str], Path],
) -> list[dict]:
    """Authenticate every required arm and file before deserializing any loss."""
    gate = capture_gate(plan, captures)
    required = {(r["identity"], arm) for r in gate["admitted"] for arm in ARMS}
    if (
        set(receipts) != required
        or set(outcome_paths) != required
        or set(manifests) != {i for i, _ in required}
    ):
        raise ValueError("missing, duplicate or unexpected replay arm")
    for identity, arm in sorted(required):
        _, manifest = admit(plan, captures, identity, manifests[identity])
        r = receipts[identity, arm]
        if (
            set(r) != RECEIPT_FIELDS
            or r["schema"] != "m4-paired-arm-receipt-v1"
            or r["status"] != "COMPLETE_OUTCOME_SEALED"
            or r["identity"] != identity
            or r["arm"] != arm
            or r["seed"] != manifest["trajectory_seed"]
            or r["manifest_sha256"]
            != next(
                x["manifest_sha256"]
                for x in gate["admitted"]
                if x["identity"] == identity
            )
            or r["capture_gate_sha256"] != object_hash(gate)
            or r["selected_group_ids"]
            != [x["group_id"] for x in arm_groups(manifest, arm)]
            or r["heldout_group_ids"] != [x["group_id"] for x in manifest["groups"][8:]]
            or r["heldout_files_unchanged"] is not True
            or r["pre_evaluation_state_unchanged"] is not True
            or r["post_evaluation_state_unchanged"] is not True
            or r["optimizer_updates"] != 1
            or r["scheduler_sample_increment"] != 32
            or r["deep_training_trace_enabled"] is not False
            or r["outcomes_opened"] is not False
        ):
            raise ValueError("arm provenance or invariant differs")
        validate_transition(
            r["state_before"],
            r["state_before"],
            r["state_after"],
            r["state_after"],
            manifest,
        )
        strict_runtime(r["runtime"])
        other = receipts[identity, ARMS[1] if arm == ARMS[0] else ARMS[0]]
        if (
            r["state_before"] != other["state_before"]
            or r["runtime"] != other["runtime"]
        ):
            raise ValueError("paired arm initial state or runtime differs")
        if (
            outcome_paths[identity, arm].is_symlink()
            or not 1 <= outcome_paths[identity, arm].stat().st_size <= 8192
        ):
            raise ValueError("sealed outcome path or size invalid")
        if digest_file(outcome_paths[identity, arm]) != r["outcome_sha256"]:
            raise ValueError("sealed outcome hash differs")
    # Only the complete successful first pass permits JSON outcome loading.
    opened = {}
    for key in sorted(required):
        row = json.loads(outcome_paths[key].read_bytes())
        r = receipts[key]
        if set(row) != {
            "schema",
            "identity",
            "arm",
            "manifest_sha256",
            "pre_loss",
            "post_loss",
            "change",
        } or (
            row["schema"] != "m4-paired-private-outcome-v1"
            or (row["identity"], row["arm"]) != key
            or row["manifest_sha256"] != r["manifest_sha256"]
            or finite(row["change"])
            != finite(row["post_loss"]) - finite(row["pre_loss"])
        ):
            raise ValueError("sealed outcome schema or identity differs")
        opened[key] = row
    result = []
    for identity in sorted({i for i, _ in required}):
        base, shield = [opened[identity, arm] for arm in ARMS]
        if abs(base["pre_loss"] - shield["pre_loss"]) > 1e-6:
            raise ValueError("paired pre-update heldout losses differ")
        result.append(
            {
                "identity": identity,
                "base_change": base["change"],
                "shield_change": shield["change"],
                "contrast": shield["change"] - base["change"],
            }
        )
    return result
