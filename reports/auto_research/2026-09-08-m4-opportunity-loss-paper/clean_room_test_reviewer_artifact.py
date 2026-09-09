#!/usr/bin/env python3
"""Extract and test the M4 reviewer artifact with a credential-free environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


FORBIDDEN_CONTENT = (
    b"gitlab-master",
    b"gitlab.nvidia",
    b"private-research",
    b" eos",
    b"nvidia corporation",
    b"/Users/",
    b"/home/",
    b"slurm_job_id",
    b"pipeline_id",
    b"job_id",
    b"bearer ",
    b"api_key",
    b"access_token",
    b"password",
    b"c0d12e61f",
    b"5940059c8",
    b"766351123",
    b"4e6f6a993",
    b"9cc2e9c6e",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], cwd: Path, environment: dict[str, str]) -> str:
    result = subprocess.run(command, cwd=cwd, env=environment, check=True, capture_output=True, text=True)
    return result.stdout + result.stderr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    archive = args.archive.resolve()
    assert sha256(archive) == args.expected_sha256
    receipt = json.loads(args.receipt.read_bytes())
    assert receipt["archive_sha256"] == args.expected_sha256

    with tempfile.TemporaryDirectory(prefix="m4-clean-room-") as temporary:
        temporary_path = Path(temporary)
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle.getmembers():
                target = (temporary_path / member.name).resolve()
                assert target.is_relative_to(temporary_path.resolve())
                assert not member.issym() and not member.islnk()
            bundle.extractall(temporary_path, filter="data")
        root = temporary_path / "m4-reviewer-artifact"
        assert not (root / ".git").exists()
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            lowered = path.read_bytes().lower()
            for pattern in FORBIDDEN_CONTENT:
                assert pattern.lower() not in lowered, f"forbidden content {pattern!r} in {path.relative_to(root)}"

        environment = {
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": os.environ["PATH"],
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        replay = run([sys.executable, "analysis/replay.py", "--verify"], root, environment)
        tests = run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], root, environment)
        result = {
            "archive_sha256": args.expected_sha256,
            "credential_environment_keys": sorted(environment),
            "extracted_file_count": sum(path.is_file() for path in root.rglob("*")),
            "forbidden_content_scan": "PASS",
            "manifest_and_replay": "PASS" if '"status": "PASS"' in replay else "FAIL",
            "stdlib_unit_tests": "PASS" if "OK" in tests else "FAIL",
        }
        assert result["manifest_and_replay"] == "PASS"
        assert result["stdlib_unit_tests"] == "PASS"
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
