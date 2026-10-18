"""Validate the minimal-observer bridge, never a scientific treatment contrast."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from research.m4_eligible_replay.analyze_training_trace import validate_runtime
from research.m4_eligible_replay.contracts import (
    canonical_json,
    digest_file,
    validate_duplicate_replays,
)


WORKER_MODULE = "research.m4_eligible_replay.worker"


def comparable_runtime(runtime: dict[str, Any]) -> dict[str, Any]:
    """Exclude only the intentionally edited custom worker's source hash."""
    validate_runtime(runtime)
    result = copy.deepcopy(runtime)
    result["module_origins"][WORKER_MODULE]["sha256"] = None
    return result


def validate_bridge(
    records: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
    *,
    reference: dict[str, Any],
    protocol_hash: str,
    worker_hash: str,
) -> dict[str, Any]:
    """Require exact duplicate and cross-instrumentation state agreement."""
    if reference["qualification_passed"] is not True:
        raise ValueError("reference qualification did not pass")
    capsule_hash = reference["capsule_manifest_sha256"]
    ordinary = validate_duplicate_replays(
        records, protocol_hash=protocol_hash, manifest_hash=capsule_hash
    )
    if len(boundaries) != 2:
        raise ValueError("both execution boundaries required")
    for label, record, boundary in zip(("a", "b"), records, boundaries):
        if set(boundary) != {
            "schema",
            "replicate",
            "protocol_sha256",
            "capsule_manifest_sha256",
            "before",
            "after",
        } or (
            boundary["schema"] != "m4-minimal-observer-boundaries-v1"
            or boundary["replicate"] != label
            or boundary["protocol_sha256"] != protocol_hash
            or boundary["capsule_manifest_sha256"] != capsule_hash
        ):
            raise ValueError("execution boundary identity differs")
        for stage in ("before", "after"):
            observed = boundary[stage]
            if (
                set(observed) != {"training_trace_enabled", "runtime"}
                or observed["training_trace_enabled"] is not False
            ):
                raise ValueError("training trace must be disabled")
            runtime = observed["runtime"]
            if runtime["module_origins"][WORKER_MODULE]["sha256"] != worker_hash:
                raise ValueError("actual worker source differs")
            if comparable_runtime(runtime) != comparable_runtime(reference["runtime"]):
                raise ValueError("runtime changed beyond custom worker hash")
        if boundary["before"] != boundary["after"]:
            raise ValueError("runtime changed across update")
        if (
            record["state_before"] != reference["state_before_common"]
            or record["state_after"] != reference["state_after_common"]
        ):
            raise ValueError("cross-instrumentation training state differs")
    return {
        "schema": "m4-minimal-observer-comparison-v1",
        "status": "PASS_MINIMAL_OBSERVER_DUPLICATE_AND_REFERENCE_EQUALITY",
        "original_duplicate_gate": ordinary["status"],
        "strict_runtime_matches_reference": True,
        "worker_hash_only_runtime_exception": True,
        "cross_instrumentation_full_state_equal": True,
        "deep_training_trace_enabled": False,
        "historical_capsule_excluded": True,
        "causal_contrast_computed": False,
        "post_update_heldout_loss_computed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--bridge-protocol", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.bridge_protocol.read_bytes())
    if digest_file(args.protocol) != plan["original_protocol_sha256"]:
        raise ValueError("original protocol differs from frozen bridge")
    reference_path = args.bridge_protocol.parent / plan["reference_file"]
    if digest_file(reference_path) != plan["reference_sha256"]:
        raise ValueError("reference evidence differs from frozen bridge")
    if (
        list(args.assets.glob("*training-trace.jsonl"))
        or (args.assets / "training-trace-comparison.json").exists()
    ):
        raise ValueError("minimal-observer workload produced deep trace artifacts")
    records = [
        json.loads((args.assets / f"replay-{x}.json").read_bytes()) for x in ("a", "b")
    ]
    boundaries = [
        json.loads((args.assets / f"replay-{x}-execution-boundaries.json").read_bytes())
        for x in ("a", "b")
    ]
    result = validate_bridge(
        records,
        boundaries,
        reference=json.loads(reference_path.read_bytes()),
        protocol_hash=digest_file(args.protocol),
        worker_hash=digest_file(Path(__file__).with_name("worker.py")),
    )
    result["bridge_protocol_sha256"] = digest_file(args.bridge_protocol)
    result["reference_sha256"] = digest_file(reference_path)
    result["records_sha256"] = {
        x: digest_file(args.assets / f"replay-{x}.json") for x in ("a", "b")
    }
    with (args.assets / "minimal-observer-comparison.json").open("xb") as stream:
        stream.write(canonical_json(result))
    print(result["status"])


if __name__ == "__main__":
    main()
