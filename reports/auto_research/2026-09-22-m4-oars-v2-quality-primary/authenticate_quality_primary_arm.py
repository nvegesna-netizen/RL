#!/usr/bin/env python3
"""Authenticate one quality-primary arm without opening scientific outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

EXPECTED = {
    "arm-result.json",
    "converted-hf-files.sha256",
    "evaluation/config.json",
    "evaluation/evaluation_data.json",
    "gsm8k-prompt-manifest.json",
    "lifecycle.jsonl",
    "oars.jsonl",
    "observer-duty.json",
    "opportunity.jsonl",
    "systems-result.json",
    "terminal-policy-export.json",
    "terminal-policy-files.sha256",
    "terminal-resolved-config.json",
    "training.log",
}


def digest_stream(stream: object) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(1024 * 1024):  # type: ignore[attr-defined]
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def archive_digest(path: Path) -> tuple[str, int]:
    with path.open("rb") as stream:
        return digest_stream(stream)


def safe_names(archive: zipfile.ZipFile) -> list[str]:
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate ZIP member")
    for name in names:
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts:
            raise RuntimeError(f"unsafe ZIP member: {name}")
    return names


def parse_manifest(payload: bytes) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in payload.decode("ascii").splitlines():
        digest, relative = line.split("  ./", maxsplit=1)
        if len(digest) != 64 or relative in rows:
            raise RuntimeError("invalid checksum manifest")
        int(digest, 16)
        rows[relative] = digest
    if set(rows) != EXPECTED:
        raise RuntimeError("checksum manifest coverage differs from frozen contract")
    return rows


def copies_equal(
    left: zipfile.ZipFile, right: zipfile.ZipFile, member: str
) -> bool:
    with left.open(member) as first, right.open(member) as second:
        while True:
            first_chunk = first.read(1024 * 1024)
            second_chunk = second.read(1024 * 1024)
            if first_chunk != second_chunk:
                return False
            if not first_chunk:
                return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--upstream", required=True, type=int)
    parser.add_argument("--child", required=True, type=int)
    parser.add_argument("--main-job", required=True, type=int)
    parser.add_argument("--logs-after-job", required=True, type=int)
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    copies = {
        "main": args.archive_dir / f"main-{args.main_job}.zip",
        "logs_after": args.archive_dir / f"logs-after-{args.logs_after_job}.zip",
    }
    asset_root = (
        "workspace/assets/basic/"
        f"m4-oars-v2-quality-primary-{args.identity}-acquisition/"
    )
    manifest_name = f"{asset_root}artifacts.sha256"
    rank_exit = f"{asset_root}jet_assets/exit_codes/exit_code_rank-0"
    archives = {label: zipfile.ZipFile(path) for label, path in copies.items()}
    try:
        archive_records: dict[str, object] = {}
        for label, archive in archives.items():
            names = safe_names(archive)
            corrupt = archive.testzip()
            if corrupt is not None:
                raise RuntimeError(f"corrupt ZIP member: {corrupt}")
            digest, size = archive_digest(copies[label])
            archive_records[label] = {
                "bytes": size,
                "job_id": (
                    args.main_job if label == "main" else args.logs_after_job
                ),
                "member_count": len(names),
                "sha256": digest,
                "zip_integrity_passed": True,
            }
        manifests = {
            label: archive.read(manifest_name) for label, archive in archives.items()
        }
        if manifests["main"] != manifests["logs_after"]:
            raise RuntimeError("terminal checksum manifests differ")
        declared = parse_manifest(manifests["main"])
        artifact_sha256: dict[str, str] = {}
        for relative, expected_digest in sorted(declared.items()):
            member = f"{asset_root}{relative}"
            for label, archive in archives.items():
                if member not in archive.namelist():
                    raise RuntimeError(f"missing {label} artifact: {relative}")
                with archive.open(member) as stream:
                    digest, _ = digest_stream(stream)
                if digest != expected_digest:
                    raise RuntimeError(
                        f"declared hash mismatch: {label}/{relative}"
                    )
            if not copies_equal(archives["main"], archives["logs_after"], member):
                raise RuntimeError(f"terminal copies differ: {relative}")
            artifact_sha256[relative] = expected_digest
        exit_codes = {
            label: int(archive.read(rank_exit).decode("ascii").strip())
            for label, archive in archives.items()
        }
        if set(exit_codes.values()) != {0}:
            raise RuntimeError("nonzero workload exit code")
        receipt = {
            "archives": archive_records,
            "artifact_sha256": artifact_sha256,
            "checksum_manifest_sha256": hashlib.sha256(
                manifests["main"]
            ).hexdigest(),
            "complete_outcome_embargo_preserved": True,
            "declared_artifact_count": len(declared),
            "declared_hashes_match": True,
            "evaluation_data_opened": False,
            "identity": args.identity,
            "jobs": {
                "logs_after": args.logs_after_job,
                "main": args.main_job,
            },
            "pipeline_ids": {
                "child": args.child,
                "upstream": args.upstream,
            },
            "result_opened": False,
            "schema": "m4-oars-v2-quality-primary-arm-terminal-authentication-v1",
            "scientific_outcome_accessed": False,
            "status": "PASS_AUTHENTICATED_OUTCOME_EMBARGOED",
            "successor_gate": "PASS_AUTHENTICATED_TERMINAL_PREDECESSOR",
            "terminal_copies_byte_identical": True,
            "workload_exit_codes": exit_codes,
        }
        args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps(receipt, indent=2, sort_keys=True))
    finally:
        for archive in archives.values():
            archive.close()


if __name__ == "__main__":
    main()
