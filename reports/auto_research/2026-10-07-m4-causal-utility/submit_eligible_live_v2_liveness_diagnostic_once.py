#!/usr/bin/env python3
"""Submit the authorized eligible-live v2 preflight replacement exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
JET_ROOT = Path("/Users/nvegesna/Documents/JET-LLM")
JET_PYTHON = JET_ROOT / ".venv/bin/python"
RUNLLM = JET_ROOT / "runllm.py"
REPORTS = REPO / "reports/auto_research/2026-10-07-m4-causal-utility"
SESSION = REPO / "session/20260914_m4_opportunity_control"
MANIFEST = (
    SESSION
    / "m4-eligible-live-v2-liveness-diagnostic-preflight-replacement-manifest.json"
)
SOURCE = SESSION / "m4-eligible-live-v2-diagnostic-source-5b68a8e4f.tar.gz"
PROTOCOL = REPORTS / "eligible_live_v2_liveness_diagnostic_protocol.json"
AUTHORIZATION = (
    REPORTS
    / "eligible_live_v2_liveness_diagnostic_preflight_replacement_authorization.json"
)
EXECUTION_LOCK = (
    REPORTS
    / "eligible_live_v2_liveness_diagnostic_preflight_replacement_execution_lock.json"
)
PACKAGE = (
    REPORTS
    / "eligible_live_v2_liveness_diagnostic_preflight_replacement_package_validation.json"
)
RECEIPT = (
    REPORTS
    / "eligible_live_v2_liveness_diagnostic_preflight_replacement_package_receipt.json"
)
CUSTOM = REPORTS / "eligible_live_restore_eos_custom_config_deadline_suppressed.yaml"
VARIABLES = REPORTS / "eligible_live_capture_variables.yaml"
GUARD = (
    SESSION
    / "eligible-live-v2-liveness-diagnostic-d01-preflight-replacement-one-shot-guard.json"
)
LOG = (
    SESSION
    / "eligible-live-v2-liveness-diagnostic-d01-preflight-replacement-launch.log"
)
EXPECTED = {
    RUNLLM: "7e5ca7cb2b9807404123b19009d8d975f390f38690bf0a691f65f291b49c3cbb",
    MANIFEST: "a997d440ac871df67141f7d932b635d43db4b800fe5f81bd32e110e2d7c6979e",
    SOURCE: "49a96f60d66c7a29cf82bd76e06f02dcd13146fc9cdfcaef2fbaa16451a5b7f9",
    PROTOCOL: "59f99259d56d8bc993c59e3291e4ccf3f2ecf9832729e72e81222ee01f4dbc9c",
    AUTHORIZATION: "c1da23a5daa196050ea8aebb100dbebe89e05a3282bf8c3d9f6b15dc6bbae285",
    EXECUTION_LOCK: "d83c9f792db15c632b00707e74582575ffacc23be996e69c60275b8c5954a9bd",
    PACKAGE: "6b80a1d0d41888d857153ecab3429a7fcc053577be89e9de754ad993c57dc081",
    RECEIPT: "b70d44c3103c15922e54e277ce7ef08fca87ff466a2be53c2d4f863a8a31586c",
    CUSTOM: "0b72764d2ff41c690bab0f6a28408917f3f04faa311b581c94c56cef9ab35168",
    VARIABLES: "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_boundary() -> None:
    subprocess.run(["git", "diff-index", "--quiet", "HEAD", "--"], cwd=REPO, check=True)
    for path, expected in EXPECTED.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen launch input moved: {path}: {actual}")
    protocol = json.loads(PROTOCOL.read_bytes())
    authorization = json.loads(AUTHORIZATION.read_bytes())
    lock = json.loads(EXECUTION_LOCK.read_bytes())
    package = json.loads(PACKAGE.read_bytes())
    manifest = json.loads(MANIFEST.read_bytes())
    if (
        protocol["status"] != "FROZEN_BEFORE_DIAGNOSTIC_EXECUTION"
        or authorization["status"] != "AUTHORIZED_BEFORE_PREFLIGHT_REPLACEMENT"
        or authorization["authorized"][
            "preflight_replacement_eos_submission_attempts"
        ]
        != 1
        or authorization["authorized"]["required_launcher"] != "runllm.py --no_wait"
        or lock["execution"]["submission_attempt_limit"] != 1
        or lock["execution"]["client_queue_deadline"] is not None
        or lock["execution"]["automatic_retry"]
        or package["status"]
        != "PASS_AUTHORIZED_PREFLIGHT_REPLACEMENT_UNSUBMITTED"
        or package["client_queue_deadline"] is not None
        or package["post_update_outcomes_opened"]
        or package["v2_acquisition_started"]
        or manifest["spec"]["time_limit"] != 14400
        or "deadline: false" not in CUSTOM.read_text()
    ):
        raise RuntimeError("eligible-live v2 replacement launch boundary differs")
    if GUARD.exists() or LOG.exists():
        raise RuntimeError("eligible-live v2 replacement submission is already consumed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    validate_boundary()
    if args.check_only:
        print("M4_ELIGIBLE_LIVE_V2_DIAGNOSTIC_REPLACEMENT_LAUNCH_BOUNDARY_PASS")
        return 0
    if not os.environ.get("JET_GITLAB_TOKEN"):
        raise RuntimeError("JET_GITLAB_TOKEN is absent; authority not consumed")
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is absent; authority not consumed")
    command = [
        str(JET_PYTHON),
        str(RUNLLM),
        "--prebuilt_manifest",
        str(MANIFEST),
        "--prebuilt_custom_config",
        str(CUSTOM),
        "--prebuilt_additional_variables",
        str(VARIABLES),
        "--prebuilt_config_id",
        "dlalgo-ci/eos",
        "--no_wait",
    ]
    submission: dict[str, Any] = {
        "command_boundary": "runllm.py --no_wait",
        "launch_invoked_at": datetime.now().astimezone().isoformat(),
        "launcher_return_code": None,
        "upstream_pipeline_id": None,
    }
    guard: dict[str, Any] = {
        "schema": "m4-shield-eligible-live-v2-liveness-diagnostic-preflight-replacement-one-shot-guard-v1",
        "status": "ATTEMPT_CONSUMED_LAUNCH_INVOKED",
        "identity": "d01",
        "seed": 20261102,
        "manifest_sha256": EXPECTED[MANIFEST],
        "package_validation_sha256": EXPECTED[PACKAGE],
        "submission_attempt_limit": 1,
        "submission_attempts_consumed": 1,
        "automatic_retry": False,
        "automatic_replacement": False,
        "automatic_extension": False,
        "submissions": [submission],
    }
    write_json(GUARD, guard)
    process = subprocess.Popen(
        command,
        cwd=JET_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    surface_parts: list[str] = []
    assert process.stdout is not None
    with LOG.open("w") as log:
        for line in process.stdout:
            surface_parts.append(line)
            log.write(line)
            log.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
    return_code = process.wait()
    surface = "".join(surface_parts)
    match = re.search(
        r"(?m)^(\d+): "
        r"https://gitlab-master\.nvidia\.com/dl/jet/ci/-/pipelines/\1$",
        surface,
    )
    submission.update(
        {
            "launcher_return_code": return_code,
            "upstream_pipeline_id": int(match.group(1)) if match else None,
        }
    )
    guard["submissions"] = [submission]
    guard["status"] = (
        "SUBMITTED_NO_WAIT"
        if return_code == 0 and match
        else "ATTEMPT_CONSUMED_CLIENT_FAILURE"
    )
    write_json(GUARD, guard)
    return 0 if return_code == 0 and match else return_code or 1


if __name__ == "__main__":
    raise SystemExit(main())
