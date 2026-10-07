#!/usr/bin/env python3
"""Validate the eligible-live capture package without credentials or GPUs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from build_eligible_live_capture_manifest import (
    AUTHORIZATION_SHA256,
    FILE_HASHES,
    LOCK_SHA256,
    MEGATRON_SHA256,
    PROTOCOL_SHA256,
    SOURCE_COMMIT,
    SOURCE_SHA256,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def safe_archive_facts(path: Path) -> tuple[int, int]:
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
    names = [PurePosixPath(member.name) for member in members]
    if len(names) != len(set(names)):
        raise RuntimeError("archive contains duplicate members")
    if any(name.is_absolute() or ".." in name.parts for name in names):
        raise RuntimeError("archive contains unsafe path")
    symlinks = {name for name, member in zip(names, members) if member.issym()}
    if any(any(parent in symlinks for parent in name.parents) for name in names):
        raise RuntimeError("archive traverses a symlink parent")
    return len(members), len(symlinks)


def embedded_python_blocks(script: str) -> list[str]:
    return re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", script, flags=re.DOTALL)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected_inputs = {
        args.source: SOURCE_SHA256,
        args.megatron: MEGATRON_SHA256,
        args.protocol: PROTOCOL_SHA256,
        args.authorization: AUTHORIZATION_SHA256,
        args.execution_lock: LOCK_SHA256,
    }
    for path, expected in expected_inputs.items():
        if sha256(path) != expected:
            raise RuntimeError(f"frozen input moved: {path}")

    manifest = json.loads(args.manifest.read_bytes())
    script = manifest["spec"]["script"]
    source_members, source_symlinks = safe_archive_facts(args.source)
    megatron_members, _ = safe_archive_facts(args.megatron)
    if megatron_members != 726:
        raise RuntimeError("Megatron archive member count changed")

    with tempfile.TemporaryDirectory(prefix="m4-eligible-live-validate-") as tmp:
        temp = Path(tmp)
        extracted = temp / "source"
        with tarfile.open(args.source, "r:gz") as archive:
            archive.extractall(extracted)
        for relative, expected in FILE_HASHES.items():
            actual = sha256(extracted / relative)
            if actual != expected:
                raise RuntimeError(f"source file hash changed: {relative}")
        capsule_source = (
            extracted
            / "nemo_rl/algorithms/async_utils/eligible_live_decision_capsule.py"
        ).read_text()

        script_path = temp / "run.sh"
        script_path.write_text(script)
        subprocess.run(["bash", "-n", str(script_path)], check=True)
        blocks = embedded_python_blocks(script)
        if len(blocks) != 3:
            raise RuntimeError(
                f"expected three embedded Python blocks, got {len(blocks)}"
            )
        for index, block in enumerate(blocks):
            ast.parse(block, filename=f"embedded-{index}.py")

        rebuilt = temp / "rebuilt.json"
        subprocess.run(
            [
                sys.executable,
                str(args.builder),
                "--source",
                str(args.source),
                "--megatron",
                str(args.megatron),
                "--protocol",
                str(args.protocol),
                "--authorization",
                str(args.authorization),
                "--execution-lock",
                str(args.execution_lock),
                "--output",
                str(rebuilt),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        deterministic = rebuilt.read_bytes() == args.manifest.read_bytes()

    forbidden_credentials = (
        "glpat-",
        "github_pat_",
        "BEGIN OPENSSH PRIVATE KEY",
    )
    shell_without_embedded_payloads = re.sub(
        r"printf %s '[A-Za-z0-9+/=]+'", "printf %s '<embedded>'", script
    )
    credential_free = not any(
        token in shell_without_embedded_payloads for token in forbidden_credentials
    )
    no_queue_deadline = (
        "--deadline" not in script
        and "queue_deadline" not in manifest["spec"]
        and "deadline" not in manifest["spec"]
    )
    jet_runtime_expansions_escaped = (
        "${{NEMO_RL_COMMIT:-unknown}}" in script
        and "${{PIPESTATUS[0]}}" in script
        and "${NEMO_RL_COMMIT:-unknown}"
        not in script.replace("${{NEMO_RL_COMMIT:-unknown}}", "")
        and "${PIPESTATUS[0]}" not in script.replace("${{PIPESTATUS[0]}}", "")
    )
    runtime_tests_before_run = script.index('"$PYTHON" -m pytest') < script.index(
        '"$PYTHON" examples/run_grpo_single_controller.py'
    )
    outcome_exclusion = (
        '"post_update_outcomes_opened":false' in script
        and "analyze_eligible_live_decision_capture.py" in script
        and "paired_replay" not in script
        and "terminal_policy_export.output_dir" not in script
    )
    protocol_2 = "pickle_protocol=2" in capsule_source
    checks = {
        "source_hash": sha256(args.source) == SOURCE_SHA256,
        "megatron_hash": sha256(args.megatron) == MEGATRON_SHA256,
        "protocol_hash": sha256(args.protocol) == PROTOCOL_SHA256,
        "authorization_hash": sha256(args.authorization) == AUTHORIZATION_SHA256,
        "execution_lock_hash": sha256(args.execution_lock) == LOCK_SHA256,
        "source_commit_bound": SOURCE_COMMIT
        == json.loads(args.execution_lock.read_bytes())["source_commit"],
        "safe_source_archive": source_members > 0,
        "credential_free": credential_free,
        "bash_syntax": True,
        "embedded_python_syntax": True,
        "deterministic_rebuild": deterministic,
        "manifest_time_limit": manifest["spec"].get("time_limit") == 14400,
        "no_client_queue_deadline": no_queue_deadline,
        "jet_runtime_expansions_escaped": jet_runtime_expansions_escaped,
        "runtime_tests_before_collection": runtime_tests_before_run,
        "outcome_exclusion": outcome_exclusion,
        "protocol_2_safe_serialization": protocol_2,
    }
    passed = all(checks.values())
    result = {
        "schema": "m4-shield-eligible-live-capture-package-validation-v1",
        "status": "PASS_AUTHORIZED_UNSUBMITTED" if passed else "FAIL_PACKAGE_GATE",
        "source_commit": SOURCE_COMMIT,
        "source_archive_sha256": SOURCE_SHA256,
        "source_archive_members": source_members,
        "source_archive_symlinks": source_symlinks,
        "protocol_sha256": PROTOCOL_SHA256,
        "authorization_sha256": AUTHORIZATION_SHA256,
        "execution_lock_sha256": LOCK_SHA256,
        "manifest_sha256": sha256(args.manifest),
        "builder_sha256": sha256(args.builder),
        "checks": checks,
        "post_update_outcomes_opened": False,
        "submission_attempt_limit": 1,
    }
    args.output.write_bytes(canonical(result))
    print(json.dumps({"output": str(args.output), "status": result["status"]}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
