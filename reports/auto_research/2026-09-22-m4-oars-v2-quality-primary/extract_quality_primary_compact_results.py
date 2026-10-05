#!/usr/bin/env python3
"""Release authenticated compact quality-primary results after the embargo gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

EXPECTED_RUNS = 54
RUN_MANIFEST_SHA256 = "44fe686d444545965afd6964387ae662c2a7f640512b1262da19b90ddfabd20f"
COMPLETION_AUDIT_SHA256 = (
    "d1f5dd6166408f5db1586ba0d0bec1d9c231058fcf2706d969d247dffb887268"
)
COMPLETION_AUDIT_SCHEMA = "m4-oars-v2-quality-primary-acquisition-completion-audit-v1"
COMPLETION_AUDIT_STATUS = "PASS_COMPLETE_OUTCOME_EMBARGOED"


class ExtractionError(ValueError):
    """Raised when a compact result cannot be released safely."""


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
    require(isinstance(value, dict), f"{path}: expected a JSON object")
    return value


def validate_zip_names(archive: zipfile.ZipFile) -> None:
    """Reject duplicate, absolute, and parent-traversing ZIP members."""
    names = archive.namelist()
    require(len(names) == len(set(names)), "duplicate ZIP member")
    for name in names:
        pure = PurePosixPath(name)
        require(
            not pure.is_absolute() and ".." not in pure.parts,
            f"unsafe ZIP member: {name}",
        )


def display_path(path: Path, repository: Path) -> str:
    """Return a repository-relative path when possible."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repository.resolve()))
    except ValueError:
        return str(resolved)


def authentication_path(authentication_dir: Path, identity: str) -> Path:
    """Return the canonical terminal-authentication path for an identity."""
    stem = identity.replace("-", "_")
    return authentication_dir / f"quality_primary_{stem}_terminal_authentication.json"


def validate_release_gate(
    *,
    completion_audit_path: Path,
    run_manifest_path: Path,
    expected_completion_audit_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the frozen manifest and all-54 embargo completion gate."""
    require(
        sha256_file(run_manifest_path) == RUN_MANIFEST_SHA256,
        "frozen run-manifest hash differs",
    )
    require(
        sha256_file(completion_audit_path) == expected_completion_audit_sha256,
        "completion-audit hash differs",
    )
    manifest = load_object(run_manifest_path)
    audit = load_object(completion_audit_path)
    runs = manifest.get("runs")
    require(
        manifest.get("schema") == "m4-oars-v2-quality-primary-run-manifest-v1"
        and manifest.get("run_count") == EXPECTED_RUNS
        and isinstance(runs, list)
        and len(runs) == EXPECTED_RUNS,
        "run-manifest contract differs",
    )
    identities = [run.get("identity") for run in runs]
    require(
        all(isinstance(identity, str) for identity in identities)
        and len(set(identities)) == EXPECTED_RUNS,
        "run-manifest identities differ",
    )
    completion = audit.get("completion_gate")
    embargo = audit.get("embargo")
    frozen_hashes = audit.get("frozen_evidence_sha256")
    require(
        audit.get("schema") == COMPLETION_AUDIT_SCHEMA
        and audit.get("status") == COMPLETION_AUDIT_STATUS
        and isinstance(completion, dict)
        and completion.get("status") == "PASS_ALL_54_AUTHENTICATED_OUTCOME_EMBARGOED"
        and completion.get("expected_identities") == EXPECTED_RUNS
        and completion.get("actual_authenticated_identities") == EXPECTED_RUNS
        and completion.get("missing_identities") == []
        and completion.get("unexpected_identities") == []
        and isinstance(embargo, dict)
        and embargo.get("complete_outcome_embargo_preserved") is True
        and embargo.get("scientific_outcome_accessed") is False
        and embargo.get("result_opened") is False
        and embargo.get("evaluation_data_opened") is False
        and isinstance(frozen_hashes, dict)
        and frozen_hashes.get("run_manifest") == RUN_MANIFEST_SHA256,
        "completion gate does not authorize controlled outcome release",
    )
    return manifest, audit


def extract(
    *,
    repository: Path,
    completion_audit_path: Path,
    run_manifest_path: Path,
    archive_root: Path,
    authentication_dir: Path,
    output_dir: Path,
    extracted_at: str,
    expected_completion_audit_sha256: str = COMPLETION_AUDIT_SHA256,
) -> dict[str, Any]:
    """Release all 54 compact results after authenticating every boundary."""
    manifest, audit = validate_release_gate(
        completion_audit_path=completion_audit_path,
        run_manifest_path=run_manifest_path,
        expected_completion_audit_sha256=expected_completion_audit_sha256,
    )
    partial_dir = output_dir.with_name(f"{output_dir.name}.partial")
    require(not output_dir.exists(), f"output directory already exists: {output_dir}")
    require(
        not partial_dir.exists(),
        f"partial output directory already exists: {partial_dir}",
    )
    partial_dir.mkdir(parents=True)

    released: dict[str, Any] = {}
    for run in manifest["runs"]:
        identity = str(run["identity"])
        auth_path = authentication_path(authentication_dir, identity)
        require(auth_path.is_file(), f"{identity}: authentication record missing")
        auth_sha256 = sha256_file(auth_path)
        authentication = load_object(auth_path)
        require(
            authentication.get("schema")
            == "m4-oars-v2-quality-primary-arm-terminal-authentication-v1"
            and authentication.get("identity") == identity
            and authentication.get("status") == "PASS_AUTHENTICATED_OUTCOME_EMBARGOED"
            and authentication.get("successor_gate")
            == "PASS_AUTHENTICATED_TERMINAL_PREDECESSOR"
            and authentication.get("complete_outcome_embargo_preserved") is True
            and authentication.get("scientific_outcome_accessed") is False
            and authentication.get("result_opened") is False
            and authentication.get("evaluation_data_opened") is False
            and authentication.get("declared_hashes_match") is True
            and authentication.get("terminal_copies_byte_identical") is True,
            f"{identity}: terminal authentication contract differs",
        )
        main_job = authentication["jobs"]["main"]
        archive_path = (
            archive_root
            / f"oars-v2-quality-primary-{identity}-terminal-archives"
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
            f"m4-oars-v2-quality-primary-{identity}-acquisition/arm-result.json"
        )
        with zipfile.ZipFile(archive_path) as archive:
            validate_zip_names(archive)
            require(archive.testzip() is None, f"{identity}: ZIP integrity failed")
            require(member in archive.namelist(), f"{identity}: compact result missing")
            payload = archive.read(member)
        result_sha256 = sha256_bytes(payload)
        require(
            result_sha256 == authentication["artifact_sha256"]["arm-result.json"],
            f"{identity}: compact result hash differs",
        )
        destination = partial_dir / f"{identity}.json"
        destination.write_bytes(payload)
        released[identity] = {
            "archive": {
                "path": display_path(archive_path, repository),
                "sha256": archive_sha256,
            },
            "authentication": {
                "path": display_path(auth_path, repository),
                "sha256": auth_sha256,
            },
            "bytes": len(payload),
            "member": member,
            "sha256": result_sha256,
        }

    require(len(released) == EXPECTED_RUNS, "released result count differs")
    partial_dir.rename(output_dir)
    return {
        "schema": "m4-oars-v2-quality-primary-result-extraction-v1",
        "status": "PASS_54_RESULTS_EXTRACTED_AND_HASH_AUTHENTICATED",
        "extracted_at": extracted_at,
        "completion_audit": {
            "path": display_path(completion_audit_path, repository),
            "sha256": sha256_file(completion_audit_path),
            "status": audit["status"],
        },
        "run_manifest_sha256": RUN_MANIFEST_SHA256,
        "result_count": len(released),
        "scientific_outcome_accessed": True,
        "evaluation_data_accessed": False,
        "results": released,
    }


def main() -> None:
    """Parse arguments, release compact results, and write an audit receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--completion-audit", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--authentication-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extracted-at", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    require(not args.receipt.exists(), f"receipt already exists: {args.receipt}")
    receipt = extract(
        repository=args.repository.resolve(),
        completion_audit_path=args.completion_audit.resolve(),
        run_manifest_path=args.run_manifest.resolve(),
        archive_root=args.archive_root.resolve(),
        authentication_dir=args.authentication_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        extracted_at=args.extracted_at,
    )
    with args.receipt.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
