"""Locate earliest observed duplicate-update divergence, never a quality effect."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research.m4_eligible_replay.contracts import canonical_json, digest_file
from research.m4_eligible_replay.state_diagnostics import validate_hash
from research.m4_eligible_replay.training_trace import (
    ENVIRONMENT_FIELDS,
    MODEL_MODE_FIELDS,
)


RUNTIME_KEYS = {
    "torch_version",
    "deterministic_algorithms",
    "deterministic_warn_only",
    "cudnn_deterministic",
    "cudnn_benchmark",
    "matmul_allow_tf32",
    "cudnn_allow_tf32",
    "model_modes",
    "environment",
    "module_origins",
    "callable_origins",
}
COUNTS = {
    "microbatches",
    "model_inputs",
    "model_outputs",
    "forward_backward",
    "optimizer_steps",
    "expected_microbatches",
}
OPTIMIZER_HASHES = {
    "inner_optimizer_sha256",
    "master_parameters_sha256",
    "parameter_sha256",
}


def validate_runtime(payload: dict[str, Any]) -> None:
    """Accept only bounded, explicitly selected backend metadata."""
    if set(payload) != RUNTIME_KEYS:
        raise ValueError("unexpected runtime diagnostic fields")
    if (
        not isinstance(payload["torch_version"], str)
        or len(payload["torch_version"]) > 128
    ):
        raise ValueError("invalid Torch version metadata")
    for key in RUNTIME_KEYS - {
        "torch_version",
        "model_modes",
        "environment",
        "module_origins",
        "callable_origins",
    }:
        if type(payload[key]) is not bool:
            raise ValueError("invalid backend flag metadata")
    for key, allowed in (
        ("model_modes", MODEL_MODE_FIELDS),
        ("environment", ENVIRONMENT_FIELDS),
    ):
        values = payload[key]
        if not isinstance(values, dict) or set(values) != set(allowed):
            raise ValueError("unexpected runtime mode/environment fields")
        if any(
            value is not None and (not isinstance(value, str) or len(value) > 256)
            for value in values.values()
        ):
            raise ValueError("invalid runtime mode/environment value")
    origins = payload["module_origins"]
    if not isinstance(origins, dict) or len(origins) > 16:
        raise ValueError("invalid module origins")
    for name, origin in origins.items():
        if not isinstance(name, str) or set(origin) != {"path", "sha256"}:
            raise ValueError("unexpected module origin fields")
        if origin["path"] is None:
            if origin["sha256"] is not None:
                raise ValueError("missing module path with hash")
        else:
            if not isinstance(origin["path"], str) or len(origin["path"]) > 4096:
                raise ValueError("invalid module origin path")
            validate_hash(origin["sha256"])
    if set(payload["callable_origins"]) != {
        "microbatch_iterator",
        "forward_backward",
    } or any(
        not isinstance(value, str) or len(value) > 4096
        for value in payload["callable_origins"].values()
    ):
        raise ValueError("invalid callable origins")


def load_trace(
    path: Path, *, replicate: str, protocol_hash: str, capsule_hash: str
) -> list[dict[str, Any]]:
    """Reject incomplete, mismatched or outcome-bearing records before comparison."""
    validate_hash(protocol_hash)
    validate_hash(capsule_hash)
    if path.stat().st_size > 1 << 20:
        raise ValueError("oversized training trace")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not 10 <= len(rows) <= 256:
        raise ValueError("incomplete or oversized training trace")
    for index, row in enumerate(rows):
        if set(row) != {
            "schema",
            "sequence",
            "replicate",
            "protocol_sha256",
            "capsule_manifest_sha256",
            "phase",
            "payload",
        }:
            raise ValueError("unexpected training trace fields")
        if (
            row["schema"] != "m4-eligible-live-training-trace-v1"
            or type(row["sequence"]) is not int
            or row["sequence"] != index
            or row["replicate"] != replicate
            or row["protocol_sha256"] != protocol_hash
            or row["capsule_manifest_sha256"] != capsule_hash
        ):
            raise ValueError("training trace identity/sequence mismatch")
        payload, phase = row["payload"], row["phase"]
        if phase == "runtime":
            validate_runtime(payload)
        else:
            expected = {
                "worker_input": {"sha256"},
                "forward_backward_start": {"sha256"},
                "microbatch": {"index", "sha256"},
                "model_input": {"index", "sha256"},
                "model_output": {"index", "sha256"},
                "forward_backward_end": {"gradient_sha256"},
                "optimizer_before": OPTIMIZER_HASHES | {"gradient_sha256"},
                "optimizer_after": OPTIMIZER_HASHES,
                "complete": COUNTS,
            }.get(phase)
            if expected is None or set(payload) != expected:
                raise ValueError("unexpected training phase fields")
            for name, value in payload.items():
                if name == "index" or phase == "complete":
                    if type(value) is not int or value < 0:
                        raise ValueError("invalid training trace counter")
                else:
                    validate_hash(value)
    complete = rows[-1]
    if complete["phase"] != "complete":
        raise ValueError("training trace lacks completion")
    counts = complete["payload"]
    size = counts["expected_microbatches"]
    if (
        size < 1
        or counts["forward_backward"] != 1
        or counts["optimizer_steps"] != 1
        or any(
            counts[key] != size
            for key in ("microbatches", "model_inputs", "model_outputs")
        )
    ):
        raise ValueError("incomplete training trace counts")
    phases = (
        ["runtime", "worker_input", "forward_backward_start"]
        + [
            phase
            for _ in range(size)
            for phase in ("microbatch", "model_input", "model_output")
        ]
        + ["forward_backward_end", "optimizer_before", "optimizer_after", "complete"]
    )
    if [row["phase"] for row in rows] != phases:
        raise ValueError("unexpected training execution ordering")
    for phase in ("microbatch", "model_input", "model_output"):
        if [row["payload"]["index"] for row in rows if row["phase"] == phase] != list(
            range(size)
        ):
            raise ValueError("unexpected microbatch ordering")
    return rows


def compare_traces(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> dict[str, Any]:
    """Describe observations without declaring a kernel-level root cause."""
    runtime_changes = sorted(
        key
        for key in RUNTIME_KEYS
        if left[0]["payload"][key] != right[0]["payload"][key]
    )
    mismatches = []
    for index in range(1, max(len(left), len(right))):
        a = left[index] if index < len(left) else None
        b = right[index] if index < len(right) else None
        if (
            a is None
            or b is None
            or (a["phase"], a["payload"]) != (b["phase"], b["payload"])
        ):
            mismatches.append(
                {
                    "sequence": index,
                    "a_phase": a["phase"] if a else None,
                    "b_phase": b["phase"] if b else None,
                    "changed_fields": sorted(
                        key
                        for key in set(a["payload"]) | set(b["payload"])
                        if a["payload"].get(key) != b["payload"].get(key)
                    )
                    if a and b and a["phase"] == b["phase"]
                    else ["event_structure"],
                }
            )
    return {
        "schema": "m4-eligible-live-training-trace-comparison-v1",
        "status": "OBSERVED_EXECUTION_DIVERGENCE"
        if mismatches
        else "NO_OBSERVED_EXECUTION_DIVERGENCE",
        "runtime_changed_fields": runtime_changes,
        "earliest_observed_divergence": mismatches[0] if mismatches else None,
        "observed_mismatches": mismatches,
        "qualification_passed": False,
        "exact_root_cause_established": False,
        "causal_contrast_computed": False,
        "post_update_heldout_loss_computed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_bytes())
    protocol_hash = digest_file(args.protocol)
    paths = [
        args.assets / f"replay-{replicate}-training-trace.jsonl"
        for replicate in ("a", "b")
    ]
    traces = [
        load_trace(
            path,
            replicate=replicate,
            protocol_hash=protocol_hash,
            capsule_hash=protocol["source_capsule"]["manifest_sha256"],
        )
        for path, replicate in zip(paths, ("a", "b"))
    ]
    result = compare_traces(*traces)
    result["protocol_sha256"] = protocol_hash
    result["trace_sha256"] = {
        replicate: digest_file(path) for replicate, path in zip(("a", "b"), paths)
    }
    with (args.assets / "training-trace-comparison.json").open("xb") as stream:
        stream.write(canonical_json(result))
    print(result["status"])


if __name__ == "__main__":
    main()
