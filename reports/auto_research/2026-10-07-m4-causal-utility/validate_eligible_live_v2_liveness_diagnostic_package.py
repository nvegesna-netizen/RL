#!/usr/bin/env python3
"""Clean-room validate the eligible-live v2 liveness diagnostic package."""

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

from build_eligible_live_v2_liveness_diagnostic_manifest import (
    AUTHORIZATION_SHA256,
    FILE_HASHES,
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
            name for name, member in zip(names, members, strict=True) if member.issym()
        }
        require(
            not any(
                any(parent in symlinks for parent in name.parents) for name in names
            ),
            "source archive traverses a symlink parent",
        )
        require(
            set(FILE_HASHES).issubset({str(name) for name in names}),
            "runtime files absent",
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
    parser.add_argument("--manifest", type=Path, required=True)
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
    }
    for path, expected in expected_hashes.items():
        require(sha256(path) == expected, f"frozen input moved: {path}")
    member_count, symlink_count = safe_archive_facts(args.source)

    with tempfile.TemporaryDirectory() as temporary_directory:
        rebuilt = Path(temporary_directory) / "manifest.json"
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
                "--output",
                str(rebuilt),
            ],
            check=True,
        )
        require(
            rebuilt.read_bytes() == args.manifest.read_bytes(),
            "diagnostic manifest is not deterministic",
        )

    manifest = json.loads(args.manifest.read_bytes())
    spec = manifest["spec"]
    require(
        spec["name"]
        == "m4-eligible-live-v2-liveness-diagnostic-d01-preflight-replacement",
        "workload identity moved",
    )
    require(spec["time_limit"] == 14400, "four-hour allocated runtime moved")
    require("sbatch_additional_flags" not in spec, "manifest added scheduler flags")
    script = spec["script"].format(assets_dir="/tmp/m4-v2-assets")
    syntax = subprocess.run(
        ["bash", "-n"], input=script, text=True, capture_output=True, check=False
    )
    require(syntax.returncode == 0, f"runtime shell is invalid: {syntax.stderr}")
    embedded_python = re.findall(r"<<'PY'\n(.*?)\nPY", script, flags=re.DOTALL)
    require(len(embedded_python) == 4, "embedded Python block count moved")
    for index, source in enumerate(embedded_python):
        try:
            ast.parse(source)
        except SyntaxError as error:
            raise RuntimeError(f"embedded Python block {index} is invalid") from error

    required_markers = (
        "readonly IDENTITY=d01",
        "readonly SEED=20261102",
        "readonly ASSIGNMENT_DOMAIN=m4-eligible-live-v2-liveness-diagnostic-v1",
        "eligible_live_decision_capture.progress_timeout_seconds=1200",
        "eligible_live_decision_capture.liveness_diagnostic_path=$LIVENESS",
        "PASS_DIAGNOSTIC_TIMEOUT_CAPTURED",
        "PASS_NO_STALL_REPRODUCED_CAPTURE_EXCLUDED",
        "PASS_NO_STALL_REPRODUCED_NO_ELIGIBILITY",
        'post_update_outcomes_opened":False',
    )
    require(
        all(marker in script for marker in required_markers), "runtime marker moved"
    )
    require(
        script.count("examples/run_grpo_single_controller.py") == 1,
        "diagnostic workload invocation count moved",
    )
    require(
        script.index("tests/unit/single_controller/test_single_controller.py")
        < script.index("examples/run_grpo_single_controller.py"),
        "focused tests no longer gate workload execution",
    )
    require("--deadline" not in script, "queue deadline leaked into runtime shell")
    require("CI_JOB_TOKEN" not in script, "credential use leaked into diagnostic")
    require(
        not re.search(r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----", script),
        "private key leaked into diagnostic manifest",
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
        "single_authorized_submission": (
            authorization["authorized"][
                "preflight_replacement_eos_submission_attempts"
            ]
            == 1
        ),
        "focused_tests_precede_workload": True,
        "timeout_is_1200_seconds": (
            protocol["instrument"]["progress_timeout_seconds"] == 1200
        ),
        "allocated_runtime_is_four_hours": spec["time_limit"] == 14400,
        "queue_deadline_disabled": "deadline: false" in custom,
        "no_automatic_retry": not lock["execution"]["automatic_retry"],
        "outcomes_remain_excluded": not lock["outcome_boundary"][
            "post_update_loss_computed"
        ],
        "v2_acquisition_not_started": not lock["outcome_boundary"][
            "v2_acquisition_started"
        ],
    }
    require(all(checks.values()), "diagnostic package gate failed")
    result: dict[str, Any] = {
        "schema": "m4-shield-eligible-live-v2-liveness-diagnostic-preflight-replacement-package-validation-v1",
        "status": "PASS_AUTHORIZED_PREFLIGHT_REPLACEMENT_UNSUBMITTED",
        "checks": checks,
        "source_commit": SOURCE_COMMIT,
        "source_archive_sha256": SOURCE_SHA256,
        "source_archive_members": member_count,
        "source_archive_symlinks": symlink_count,
        "protocol_sha256": PROTOCOL_SHA256,
        "authorization_sha256": AUTHORIZATION_SHA256,
        "execution_lock_sha256": LOCK_SHA256,
        "v1_terminal_authentication_sha256": V1_TERMINAL_SHA256,
        "manifest_sha256": sha256(args.manifest),
        "manifest_bytes": args.manifest.stat().st_size,
        "runtime_time_limit_seconds": 14400,
        "progress_timeout_seconds": 1200,
        "client_queue_deadline": None,
        "automatic_retry": False,
        "automatic_replacement": False,
        "automatic_extension": False,
        "post_update_outcomes_opened": False,
        "v2_acquisition_started": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output, sha256(args.output))


if __name__ == "__main__":
    main()
