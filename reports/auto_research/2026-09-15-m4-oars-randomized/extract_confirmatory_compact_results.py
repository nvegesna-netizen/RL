#!/usr/bin/env python3
"""Extract gate-authenticated compact confirmatory results after outcome release."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


EXPECTED_RUNS = 20
GATE_SCHEMA = "m4-oars-randomized-confirmatory-terminal-authentication-v1"
GATE_STATUS = "PASS_ALL_20_AUTHENTICATED_RESULTS_UNOPENED"


class ExtractionError(ValueError):
    """Raised when an authenticated compact result cannot be released safely."""


def require(condition: bool, message: str) -> None:
    """Raise an extraction error when a required condition is false."""
    if not condition:
        raise ExtractionError(message)


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 digest of bytes."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without interpreting it."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    """Load a compact JSON provenance object."""
    value = json.loads(path.read_bytes())
    require(isinstance(value, dict), f"{path}: expected an object")
    return value


def validate_zip_names(archive: zipfile.ZipFile) -> None:
    """Reject duplicate, absolute, and parent-traversing members."""
    names = archive.namelist()
    require(len(names) == len(set(names)), "duplicate ZIP member")
    for name in names:
        pure = PurePosixPath(name)
        require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe ZIP member: {name}")


def display_path(path: Path, repository: Path) -> str:
    """Return a repository-relative path when possible."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repository.resolve()))
    except ValueError:
        return str(resolved)


def extract(
    repository: Path,
    gate_path: Path,
    archive_root: Path,
    output_dir: Path,
    extracted_at: str,
) -> dict[str, Any]:
    """Release all compact results after authenticating every input boundary."""
    gate = load_object(gate_path)
    require(gate.get("schema") == GATE_SCHEMA, "completion-gate schema differs")
    require(gate.get("status") == GATE_STATUS, "completion gate did not pass")
    require(gate.get("scientific_outcome_accessed") is False, "gate was not sealed")
    identities = gate.get("ordered_identities")
    results = gate.get("results")
    require(
        isinstance(identities, list)
        and len(identities) == EXPECTED_RUNS
        and len(set(identities)) == EXPECTED_RUNS
        and isinstance(results, dict)
        and set(results) == set(identities),
        "completion-gate identity coverage differs",
    )
    partial_dir = output_dir.with_name(f"{output_dir.name}.partial")
    require(not output_dir.exists(), f"output directory already exists: {output_dir}")
    require(not partial_dir.exists(), f"partial output directory already exists: {partial_dir}")
    partial_dir.mkdir(parents=True)

    released: dict[str, Any] = {}
    for identity in identities:
        gate_row = results[identity]
        authentication_record = gate_row.get("authentication_receipt")
        require(
            isinstance(authentication_record, dict),
            f"{identity}: authentication record missing",
        )
        authentication_path = repository / authentication_record["path"]
        require(
            sha256_file(authentication_path) == authentication_record.get("sha256"),
            f"{identity}: authentication receipt hash differs",
        )
        authentication = load_object(authentication_path)
        require(
            authentication.get("identity") == identity
            and authentication.get("status") == "PASS_AUTHENTICATED_OUTCOME_EMBARGOED"
            and authentication.get("result_opened") is False
            and authentication.get("scientific_outcome_accessed") is False,
            f"{identity}: terminal authentication contract differs",
        )
        main_job = authentication["jobs"]["main"]
        archive_path = (
            archive_root
            / f"confirmatory-{identity}-terminal-archives"
            / f"main-{main_job}.zip"
        )
        require(archive_path.is_file(), f"{identity}: main archive missing")
        archive_sha256 = sha256_file(archive_path)
        require(
            archive_sha256 == authentication["archives"]["main"]["sha256"],
            f"{identity}: main archive hash differs",
        )
        member = (
            "workspace/assets/basic/"
            f"m4-oars-confirmatory-{identity}-acquisition/arm-result.json"
        )
        with zipfile.ZipFile(archive_path) as archive:
            validate_zip_names(archive)
            require(member in archive.namelist(), f"{identity}: compact result missing")
            payload = archive.read(member)
        result_sha256 = sha256_bytes(payload)
        require(
            result_sha256 == gate_row.get("sha256")
            and result_sha256
            == authentication["artifact_sha256"]["arm-result.json"],
            f"{identity}: compact result hash differs",
        )
        destination = partial_dir / f"{identity}.json"
        destination.write_bytes(payload)
        released[identity] = {
            "archive": {
                "path": display_path(archive_path, repository),
                "sha256": archive_sha256,
            },
            "authentication_receipt_sha256": authentication_record["sha256"],
            "bytes": len(payload),
            "member": member,
            "sha256": result_sha256,
        }

    partial_dir.rename(output_dir)
    return {
        "schema": "m4-oars-randomized-confirmatory-result-extraction-v1",
        "status": "PASS_20_RESULTS_EXTRACTED_AND_HASH_AUTHENTICATED",
        "extracted_at_utc": extracted_at,
        "completion_gate": {
            "path": display_path(gate_path, repository),
            "sha256": sha256_file(gate_path),
            "status": gate["status"],
        },
        "result_count": len(released),
        "scientific_outcome_accessed": True,
        "evaluation_data_accessed": False,
        "results": released,
    }


def main() -> None:
    """Parse arguments, release compact results, and write an audit receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--completion-gate", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extracted-at", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    require(not args.receipt.exists(), f"receipt already exists: {args.receipt}")
    receipt = extract(
        args.repository.resolve(),
        args.completion_gate.resolve(),
        args.archive_root.resolve(),
        args.output_dir.resolve(),
        args.extracted_at,
    )
    with args.receipt.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
