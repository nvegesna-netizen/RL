#!/usr/bin/env python3
"""Clean-room validate all six eligible-live v2 liveness packages."""

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

from build_eligible_live_v2_liveness_qualification_manifests import (
    ASSIGNMENT_DOMAIN,
    AUTHORIZATION_SHA256,
    DIAGNOSTIC_TERMINAL_SHA256,
    FILE_HASHES,
    IDENTITIES,
    LOCK_SHA256,
    MEGATRON_SHA256,
    PROTOCOL_SHA256,
    SOURCE_COMMIT,
    SOURCE_SHA256,
    V1_TERMINAL_SHA256,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def safe_archive_facts(path: Path) -> tuple[int, int]:
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [PurePosixPath(member.name) for member in members]
        require(len(names) == len(set(names)), "source archive has duplicate paths")
        require(
            not any(name.is_absolute() or ".." in name.parts for name in names),
            "source archive has an unsafe path",
        )
        symlinks = {
            name for name, member in zip(names, members) if member.issym()
        }
        require(
            not any(
                any(parent in symlinks for parent in name.parents) for name in names
            ),
            "source archive traverses a symlink parent",
        )
        require(
            set(FILE_HASHES).issubset({str(name) for name in names}),
            "runtime files are absent",
        )
    return len(members), len(symlinks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--v1-terminal", type=Path, required=True)
    parser.add_argument("--diagnostic-terminal", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--custom-config", type=Path, required=True)
    parser.add_argument("--variables", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected_hashes = {
        args.source: SOURCE_SHA256,
        args.megatron: MEGATRON_SHA256,
        args.protocol: PROTOCOL_SHA256,
        args.authorization: AUTHORIZATION_SHA256,
        args.execution_lock: LOCK_SHA256,
        args.v1_terminal: V1_TERMINAL_SHA256,
        args.diagnostic_terminal: DIAGNOSTIC_TERMINAL_SHA256,
    }
    for path, expected in expected_hashes.items():
        require(sha256(path) == expected, f"frozen input moved: {path}")
    member_count, symlink_count = safe_archive_facts(args.source)

    with tempfile.TemporaryDirectory() as temporary_directory:
        rebuilt = Path(temporary_directory) / "manifests"
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
                "--v1-terminal",
                str(args.v1_terminal),
                "--diagnostic-terminal",
                str(args.diagnostic_terminal),
                "--output-dir",
                str(rebuilt),
            ],
            check=True,
        )
        for identity, _ in IDENTITIES:
            require(
                (rebuilt / f"{identity}.json").read_bytes()
                == (args.manifest_dir / f"{identity}.json").read_bytes(),
                f"manifest is not deterministic: {identity}",
            )

    manifests: list[dict[str, Any]] = []
    for identity, seed in IDENTITIES:
        path = args.manifest_dir / f"{identity}.json"
        manifest = json.loads(path.read_bytes())
        spec = manifest["spec"]
        require(
            spec["name"]
            == f"m4-eligible-live-v2-liveness-qualification-{identity}",
            f"workload identity moved: {identity}",
        )
        require(spec["time_limit"] == 14400, "allocated runtime moved")
        require(
            "sbatch_additional_flags" not in spec,
            "manifest added scheduler flags",
        )
        script = spec["script"].format(assets_dir="/tmp/m4-v2-liveness-assets")
        syntax = subprocess.run(
            ["bash", "-n"], input=script, text=True, capture_output=True, check=False
        )
        require(syntax.returncode == 0, f"invalid shell {identity}: {syntax.stderr}")
        embedded_python = re.findall(r"<<'PY'\n(.*?)\nPY", script, flags=re.DOTALL)
        require(len(embedded_python) == 4, "embedded Python block count moved")
        for index, source in enumerate(embedded_python):
            try:
                ast.parse(source)
            except SyntaxError as error:
                raise RuntimeError(
                    f"embedded Python {identity}:{index} is invalid"
                ) from error
        required_markers = (
            f"readonly IDENTITY={identity}",
            f"readonly SEED={seed}",
            f"readonly ASSIGNMENT_DOMAIN={ASSIGNMENT_DOMAIN}",
            "suppress_capture_for_liveness_qualification=true",
            "progress_timeout_seconds=1200",
            "PASS_FULL_64_UPDATE_LIVENESS",
            'post_update_heldout_outcomes_opened":False',
            'causal_effect_estimated":False',
        )
        require(
            all(marker in script for marker in required_markers),
            f"runtime marker moved: {identity}",
        )
        require(
            script.count("examples/run_grpo_single_controller.py") == 1,
            f"workload invocation count moved: {identity}",
        )
        require(
            script.index("tests/unit/single_controller/test_single_controller.py")
            < script.index("examples/run_grpo_single_controller.py"),
            f"focused tests no longer gate workload: {identity}",
        )
        require("--deadline" not in script, "queue deadline leaked into shell")
        require("CI_JOB_TOKEN" not in script, "credential leaked into manifest")
        require(
            not re.search(r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----", script),
            "private key leaked into manifest",
        )
        manifests.append(
            {
                "identity": identity,
                "seed": seed,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )

    custom = args.custom_config.read_text()
    require("deadline: false" in custom, "queue deadline suppression is absent")
    require(json.loads(args.variables.read_text()) == {}, "variables are not empty")
    protocol = json.loads(args.protocol.read_bytes())
    authorization = json.loads(args.authorization.read_bytes())
    lock = json.loads(args.execution_lock.read_bytes())
    checks = {
        "safe_source_archive": member_count > 0,
        "source_commit_bound": lock["source_commit"] == SOURCE_COMMIT,
        "source_archive_bound": lock["source_archive_sha256"] == SOURCE_SHA256,
        "six_unique_identities": len({row[0] for row in IDENTITIES}) == 6,
        "six_unique_seeds": len({row[1] for row in IDENTITIES}) == 6,
        "six_authorized_submissions": (
            authorization["authorized"]["qualification_eos_submission_attempts"]
            == 6
        ),
        "all_six_required": protocol["joint_gate"]["required_passes"] == 6,
        "focused_tests_precede_workload": True,
        "allocated_runtime_is_four_hours": True,
        "queue_deadline_disabled": "deadline: false" in custom,
        "no_automatic_retry": not lock["execution"]["automatic_retry"],
        "capsules_forbidden": lock["qualification"]["capsule_forbidden"],
        "outcomes_remain_excluded": not lock["outcome_boundary"][
            "post_update_heldout_outcomes_opened"
        ],
        "v2_acquisition_not_started": not lock["outcome_boundary"][
            "v2_acquisition_started"
        ],
    }
    require(all(checks.values()), "qualification package gate failed")
    result: dict[str, Any] = {
        "schema": "m4-shield-eligible-live-v2-liveness-qualification-package-validation-v1",
        "status": "PASS_AUTHORIZED_SIX_IDENTITY_QUALIFICATION_UNSUBMITTED",
        "checks": checks,
        "source_commit": SOURCE_COMMIT,
        "source_archive_sha256": SOURCE_SHA256,
        "source_archive_members": member_count,
        "source_archive_symlinks": symlink_count,
        "protocol_sha256": PROTOCOL_SHA256,
        "authorization_sha256": AUTHORIZATION_SHA256,
        "execution_lock_sha256": LOCK_SHA256,
        "v1_terminal_authentication_sha256": V1_TERMINAL_SHA256,
        "diagnostic_terminal_authentication_sha256": DIAGNOSTIC_TERMINAL_SHA256,
        "manifests": manifests,
        "runtime_time_limit_seconds": 14400,
        "progress_timeout_seconds": 1200,
        "client_queue_deadline": None,
        "automatic_retry": False,
        "automatic_replacement": False,
        "automatic_extension": False,
        "post_update_heldout_outcomes_opened": False,
        "v2_acquisition_started": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output, sha256(args.output))


if __name__ == "__main__":
    main()
