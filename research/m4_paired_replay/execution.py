"""Backend-independent one-arm execution and logical outcome embargo."""

from __future__ import annotations

import json
import base64
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
from research.m4_paired_replay.quarantine import gated_open
from research.m4_paired_replay.capture_authentication import verify_admission_bundle


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
    def begin_measurements(self) -> None: ...
    def finish_measurements(self) -> dict: ...
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
    measure_secondaries: bool = False,
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
        if measure_secondaries:
            backend.begin_measurements()
        before_update = backend.snapshot()
        if before != before_update:
            raise ValueError("pre-update evaluation mutated state")
        backend.update()
        after = backend.snapshot()
        post = finite(backend.evaluate())
        secondaries = backend.finish_measurements() if measure_secondaries else None
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
        if measure_secondaries:
            validate_secondaries(secondaries)
            outcome["schema"] = "m4-paired-private-outcome-v2"
            outcome["secondaries"] = secondaries
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


def validate_secondaries(row: dict) -> None:
    """Reject missing or nonfinite registered diagnostics before sealing."""
    if not isinstance(row, dict) or set(row) != {
        "parameter_update_l2_norm",
        "train_batch_gradient_l2_norm",
        "heldout_gradient_dot_train_gradient",
        "gradient_clipping",
        "policy_kl",
        "definitions",
    }:
        raise ValueError("registered secondary endpoints incomplete")
    for name in (
        "parameter_update_l2_norm",
        "train_batch_gradient_l2_norm",
        "policy_kl",
    ):
        if finite(row[name]) < 0:
            raise ValueError("negative norm or KL diagnostic")
    finite(row["heldout_gradient_dot_train_gradient"])
    clipping = row["gradient_clipping"]
    if set(clipping) != {"threshold", "coefficient", "applied"} or (
        finite(clipping["threshold"]) < 0
        or not 0 <= finite(clipping["coefficient"]) <= 1
        or type(clipping["applied"]) is not bool
        or clipping["applied"] != (clipping["coefficient"] < 1)
    ):
        raise ValueError("invalid gradient clipping diagnostic")
    if row["definitions"] != {
        "policy_kl": "sampled_K3_pre_update_to_post_update_on_frozen_heldout_tokens",
        "gradient_dot": "exact_FP64_dot_of_native_preclipping_gradients_in_separate_restored_processes",
        "probe_norm_absolute_tolerance": 1e-5,
        "probe_norm_relative_tolerance": 1e-4,
    }:
        raise ValueError("secondary definitions differ")


def authenticate_arm_receipts(
    *,
    plan: dict,
    captures: list[dict],
    manifests: dict[str, bytes],
    receipts: dict[tuple[str, str], dict],
) -> set[tuple[str, str]]:
    """Authenticate complete arm provenance before file decryption or loss parsing."""
    gate = capture_gate(plan, captures)
    required = {(r["identity"], arm) for r in gate["admitted"] for arm in ARMS}
    if set(receipts) != required or set(manifests) != {i for i, _ in required}:
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
    return required


def release_outcomes(
    *,
    plan: dict,
    captures: list[dict],
    manifests: dict[str, bytes],
    receipts: dict[tuple[str, str], dict],
    outcome_paths: dict[tuple[str, str], Path],
    require_secondaries: bool = False,
) -> list[dict]:
    """Authenticate every required arm and file before deserializing any loss."""
    required = authenticate_arm_receipts(
        plan=plan, captures=captures, manifests=manifests, receipts=receipts
    )
    if set(outcome_paths) != required:
        raise ValueError("missing, duplicate or unexpected replay arm")
    payloads = {}
    for identity, arm in sorted(required):
        if (
            outcome_paths[identity, arm].is_symlink()
            or not 1 <= outcome_paths[identity, arm].stat().st_size <= 8192
        ):
            raise ValueError("sealed outcome path or size invalid")
        payload = outcome_paths[identity, arm].read_bytes()
        if (
            hashlib.sha256(payload).hexdigest()
            != receipts[identity, arm]["outcome_sha256"]
        ):
            raise ValueError("sealed outcome hash differs")
        payloads[identity, arm] = payload
    return decode_outcomes(payloads, receipts, require_secondaries=require_secondaries)


def release_encrypted_outcomes(
    *,
    plan: dict,
    captures: list[dict],
    evidence_dir: Path,
    manifests: dict[str, bytes],
    receipts: dict[tuple[str, str], dict],
    encrypted_receipts: dict[tuple[str, str], dict],
    encrypted_paths: dict[tuple[str, str], Path],
    private_key_path: Path,
) -> list[dict]:
    """Open fresh outcomes only after full capture, arm, and ciphertext authentication."""
    verify_admission_bundle(plan, captures, evidence_dir)
    required = authenticate_arm_receipts(
        plan=plan, captures=captures, manifests=manifests, receipts=receipts
    )
    identities = {
        key: {
            "identity": key[0],
            "arm": key[1],
            "manifest_sha256": receipts[key]["manifest_sha256"],
            "arm_receipt_sha256": hashlib.sha256(
                canonical_json(receipts[key])
            ).hexdigest(),
        }
        for key in required
    }
    bundles = gated_open(
        encrypted_receipts,
        encrypted_paths,
        required=required,
        expected_identities=identities,
        private_path=private_key_path,
    )
    payloads = {}
    for key in sorted(required):
        bundle = json.loads(bundles[key])
        if (
            set(bundle) != {"schema", "files"}
            or bundle["schema"] != "m4-private-file-bundle-v1"
            or set(bundle["files"]) != {"sealed/outcome.json", "process.log"}
        ):
            raise ValueError("encrypted private bundle schema or member list differs")
        payload = base64.b64decode(
            bundle["files"]["sealed/outcome.json"], validate=True
        )
        if (
            len(payload) > 8192
            or hashlib.sha256(payload).hexdigest() != receipts[key]["outcome_sha256"]
        ):
            raise ValueError("encrypted outcome and arm receipt differ")
        payloads[key] = payload
    return decode_outcomes(payloads, receipts, require_secondaries=True)


def decode_outcomes(
    payloads: dict[tuple[str, str], bytes],
    receipts: dict[tuple[str, str], dict],
    *,
    require_secondaries: bool,
) -> list[dict]:
    """Interpret already authenticated private endpoint bytes, never raw artifacts."""
    required = set(receipts)
    if set(payloads) != required:
        raise ValueError("endpoint set differs from authenticated arm receipts")
    # Only the complete successful first pass permits JSON outcome loading.
    opened = {}
    for key in sorted(required):
        row = json.loads(payloads[key])
        r = receipts[key]
        expected_fields = {
            "schema",
            "identity",
            "arm",
            "manifest_sha256",
            "pre_loss",
            "post_loss",
            "change",
        }
        if row["schema"] == "m4-paired-private-outcome-v2":
            expected_fields.add("secondaries")
            validate_secondaries(row["secondaries"])
        elif require_secondaries:
            raise ValueError("required registered secondary endpoints absent")
        if set(row) != expected_fields or (
            row["schema"]
            not in {"m4-paired-private-outcome-v1", "m4-paired-private-outcome-v2"}
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
        pair = {
            "identity": identity,
            "base_change": base["change"],
            "shield_change": shield["change"],
            "contrast": shield["change"] - base["change"],
        }
        if "secondaries" in base and "secondaries" in shield:
            pair["base_secondaries"] = base["secondaries"]
            pair["shield_secondaries"] = shield["secondaries"]
        result.append(pair)
    return result
