"""Authenticate synthetic engineering exports and test collector canary exclusion."""

from __future__ import annotations

import argparse
import base64
import hashlib
import gzip
import json
from pathlib import Path
from typing import BinaryIO
import zipfile

from research.m4_eligible_replay.contracts import canonical_json, digest_file
from research.m4_paired_replay.engineering_contracts import (
    DOMAIN,
    IDENTITY,
    PRIMARIES,
    PROBES,
    STAGES,
    SEED,
    compare_primaries,
    synthetic_decision,
)
from research.m4_paired_replay.quarantine import unseal


NAME = "m4-synthetic-paired-v0-engineering"
FILES = {
    "secondary-preflight.json",
    "tests.log",
    "environment-preflight.json",
    "environment-preflight.log",
    "engineering-private-envelope.json",
    "engineering-encrypted-receipt.json",
    "engineering-runtime-result.json",
}


def compact_exports(path: Path) -> dict[str, bytes]:
    """Verify inventory before returning bounded individual export bytes."""
    with zipfile.ZipFile(path) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("duplicate archive members")

        def read(name: str) -> bytes:
            matches = [
                i
                for i in archive.infolist()
                if i.filename.endswith(f"/basic/{NAME}/{name}")
            ]
            if len(matches) != 1 or matches[0].file_size > 128 << 20:
                raise ValueError("missing or oversized engineering compact member")
            return archive.read(matches[0])

        inventory = read("artifacts.sha256")
        expected = {}
        prefix = f"/jet/assets/basic/{NAME}/"
        for line in inventory.decode().splitlines():
            digest, filename = line.split("  ", 1)
            if not filename.startswith(prefix) or filename[len(prefix) :] in expected:
                raise ValueError("unexpected inventory path or duplicate")
            expected[filename[len(prefix) :]] = digest
        if set(expected) != FILES:
            raise ValueError("engineering public member allowlist differs")
        exports = {name: read(name) for name in FILES}
        if any(
            hashlib.sha256(data).hexdigest() != expected[name]
            for name, data in exports.items()
        ):
            raise ValueError("engineering member hash differs")
        return exports


def scan(stream: BinaryIO, needle: bytes) -> None:
    """Scan complete uncompressed content, including chunk-boundary matches."""
    tail = b""
    total = 0
    while block := stream.read(1 << 20):
        total += len(block)
        if total > 512 << 20:
            raise ValueError("collector scan exceeds bounded uncompressed member size")
        value = tail + block
        if needle in value:
            raise ValueError(
                "private canary leaked into a collected archive or job trace"
            )
        tail = value[-(len(needle) - 1) :]


def scan_member(
    stream: BinaryIO, filename: str, canary: bytes, *, depth: int = 0
) -> None:
    """Inspect ordinary and gzip/nested-ZIP logs; reject unsupported containers."""
    if depth > 3:
        raise ValueError("collector archive nesting exceeds scan bound")
    header = stream.read(4)
    stream.seek(0)
    if header[:2] == b"\x1f\x8b":
        with gzip.GzipFile(fileobj=stream) as decoded:
            scan(decoded, canary)
    elif header == b"PK\x03\x04":
        with zipfile.ZipFile(stream) as nested:
            for member in nested.infolist():
                if canary in member.filename.encode() or member.file_size > 512 << 20:
                    raise ValueError("unexpected nested collector member")
                with nested.open(member) as decoded:
                    scan_member(decoded, member.filename, canary, depth=depth + 1)
    elif filename.endswith((".xz", ".bz2", ".zst", ".7z")):
        raise ValueError("unsupported compressed collector member")
    else:
        scan(stream, canary)


def verify(
    *,
    eos_archive: Path,
    collector_archive: Path,
    traces: dict[int, Path],
    metadata: dict,
    private_key: Path,
) -> dict:
    """Only authenticated terminal chains permit opening the synthetic canary."""
    if (
        metadata["schema"] != "m4-authenticated-engineering-download-v1"
        or metadata["workload_name"] != NAME
        or metadata["all_chains_terminal"] is not True
    ):
        raise ValueError("engineering source topology is not terminal/authenticated")
    if (
        len(set(metadata["job_ids"])) != len(metadata["job_ids"])
        or set(traces) != set(metadata["job_ids"])
        or len(traces) < 2
    ):
        raise ValueError("full terminal job trace set required")
    for path, expected in (
        (eos_archive, metadata["eos_archive_sha256"]),
        (collector_archive, metadata["collector_archive_sha256"]),
    ):
        if digest_file(path) != expected:
            raise ValueError("authenticated archive changed")
    for job, path in traces.items():
        if digest_file(path) != metadata["trace_sha256"][str(job)]:
            raise ValueError("authenticated trace changed")
    left, right = compact_exports(eos_archive), compact_exports(collector_archive)
    if left != right:
        raise ValueError("EOS and collector compact copies differ")
    runtime = json.loads(left["engineering-runtime-result.json"])
    receipt = json.loads(left["engineering-encrypted-receipt.json"])
    envelope = left["engineering-private-envelope.json"]
    identity = {"identity": IDENTITY, "domain": DOMAIN}
    if (
        receipt["status"] != "COMPLETE_ENCRYPTED_PRIVATE_EXPORT"
        or receipt["identity"] != identity
        or receipt["plaintext_exported"] is not False
        or receipt["envelope_sha256"] != hashlib.sha256(envelope).hexdigest()
    ):
        raise ValueError("engineering envelope receipt authentication failed")
    payload = unseal(envelope, private_path=private_key, expected_identity=identity)
    bundle = json.loads(payload)
    if bundle["schema"] != "m4-private-file-bundle-v1":
        raise ValueError("unexpected synthetic private bundle")
    files = {
        name: base64.b64decode(data, validate=True)
        for name, data in bundle["files"].items()
    }
    canary = json.loads(files["canary.json"])["canary"].encode()
    if len(canary) != 64 or any(c not in b"0123456789abcdef" for c in canary):
        raise ValueError("synthetic canary invalid")
    for path in (eos_archive, collector_archive):
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                if canary in entry.filename.encode() or entry.file_size > 512 << 20:
                    raise ValueError("unexpected or canary-bearing collector member")
                with archive.open(entry) as stream:
                    scan_member(stream, entry.filename, canary)
    for path in traces.values():
        with path.open("rb") as stream:
            scan(stream, canary)
    if runtime[
        "status"
    ] != "PASS_RUNTIME_ENGINEERING_COLLECTOR_CHECK_PENDING" or runtime[
        "completed_stages"
    ] != list(STAGES):
        return {
            "status": "ENGINEERING_RUNTIME_FAILED",
            "collector_canary_excluded": True,
            "failed_stage": runtime.get("failed_stage"),
            "fresh_acquisition_unlocked": False,
        }
    secondary_preflight = json.loads(left["secondary-preflight.json"])
    if (
        secondary_preflight["status"] != "PASS_REAL_TORCH_SECONDARY_MECHANICS"
        or not secondary_preflight["checks"]
        or not all(v is True for v in secondary_preflight["checks"].values())
    ):
        raise ValueError("real-Torch secondary preflight incomplete")
    records = {stage: json.loads(files[f"records/{stage}.json"]) for stage in PRIMARIES}
    initial = records["base-control"]["state_before"]
    for stage in PROBES:
        probe = json.loads(files[f"records/{stage}.json"])
        if (
            probe["status"] != "CAPTURED_NO_OPTIMIZER_UPDATE"
            or probe["optimizer_updates"] != 0
            or probe["state_before"] != initial
        ):
            raise ValueError("separate gradient probe restore differs")
    # The exact synthetic manifest is not exported; independently reconstruct its
    # gate-only fields from fixed engineering identities and authenticated state.
    manifest = {
        "trajectory_identity": IDENTITY,
        "trajectory_seed": SEED,
        "assignment_domain": DOMAIN,
        "engineering_synthetic_only": True,
        "learner_version": 0,
        "decision": synthetic_decision(),
        "parameter_sha256_before": initial["parameter_sha256"],
    }
    checked = compare_primaries(records, manifest)
    for stage in PRIMARIES:
        if (
            hashlib.sha256(files[f"outcomes/{stage}.json"]).hexdigest()
            != records[stage]["outcome_sha256"]
        ):
            raise ValueError("private engineering endpoint hash differs")
    return {
        **checked,
        "status": "PASS_SYNTHETIC_ENGINEERING_QUALIFICATION",
        "collector_plaintext_exclusion_verified": True,
        "authenticated_archives": 2,
        "authenticated_traces": len(traces),
        "fresh_acquisition_unlocked": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eos-archive", type=Path, required=True)
    parser.add_argument("--collector-archive", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_bytes())
    result = verify(
        eos_archive=args.eos_archive,
        collector_archive=args.collector_archive,
        traces={job: args.trace_dir / f"{job}.log" for job in metadata["job_ids"]},
        metadata=metadata,
        private_key=args.private_key,
    )
    with args.output.open("xb") as stream:
        stream.write(canonical_json(result))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
