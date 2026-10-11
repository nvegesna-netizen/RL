#!/usr/bin/env python3
"""Submit the authorized eligible-live v2 liveness diagnostic exactly once."""

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
MANIFEST = SESSION / "m4-eligible-live-v2-liveness-diagnostic-manifest.json"
SOURCE = SESSION / "m4-eligible-live-v2-diagnostic-source-f70361f2c.tar.gz"
PROTOCOL = REPORTS / "eligible_live_v2_liveness_diagnostic_protocol.json"
AUTHORIZATION = REPORTS / "eligible_live_v2_liveness_diagnostic_authorization.json"
EXECUTION_LOCK = REPORTS / "eligible_live_v2_liveness_diagnostic_execution_lock.json"
PACKAGE = REPORTS / "eligible_live_v2_liveness_diagnostic_package_validation.json"
RECEIPT = REPORTS / "eligible_live_v2_liveness_diagnostic_package_receipt.json"
CUSTOM = REPORTS / "eligible_live_restore_eos_custom_config_deadline_suppressed.yaml"
VARIABLES = REPORTS / "eligible_live_capture_variables.yaml"
GUARD = SESSION / "eligible-live-v2-liveness-diagnostic-d01-one-shot-guard.json"
LOG = SESSION / "eligible-live-v2-liveness-diagnostic-d01-launch.log"
EXPECTED = {
    RUNLLM: "7e5ca7cb2b9807404123b19009d8d975f390f38690bf0a691f65f291b49c3cbb",
    MANIFEST: "c84ea2a229c5e9eef4f0ef3b1b206513333cc34f5177a10703f93a4ef1ab5942",
    SOURCE: "649c5e28de8cbba98ba83e50259f3376f64039cf003cec43c5ce2bd19026d304",
    PROTOCOL: "59f99259d56d8bc993c59e3291e4ccf3f2ecf9832729e72e81222ee01f4dbc9c",
    AUTHORIZATION: "796a60d0ab52ef782b91a1390e5ae2502c12db0490ae41ee2c40b9793b2f6de0",
    EXECUTION_LOCK: "113758c56418359bb50ec367369d385a78206651ed7a6d37d9fab7fe40c0d6ce",
    PACKAGE: "6d308bae14d224a4f7e4236d828066351c50430f462782a1698ed73487423650",
    RECEIPT: "7b152437d7d27c06e283a684e1714bdce7218fcc64925ed08696b608e2dc0f1c",
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
        or authorization["status"] != "AUTHORIZED_BEFORE_DIAGNOSTIC_EXECUTION"
        or authorization["authorized"]["diagnostic_eos_submission_attempts"] != 1
        or authorization["authorized"]["required_launcher"] != "runllm.py --no_wait"
        or lock["execution"]["submission_attempt_limit"] != 1
        or lock["execution"]["client_queue_deadline"] is not None
        or lock["execution"]["automatic_retry"]
        or package["status"] != "PASS_AUTHORIZED_UNSUBMITTED"
        or package["client_queue_deadline"] is not None
        or package["post_update_outcomes_opened"]
        or package["v2_acquisition_started"]
        or manifest["spec"]["time_limit"] != 14400
        or "deadline: false" not in CUSTOM.read_text()
    ):
        raise RuntimeError("eligible-live v2 diagnostic launch boundary differs")
    if GUARD.exists() or LOG.exists():
        raise RuntimeError("eligible-live v2 diagnostic submission is already consumed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    validate_boundary()
    if args.check_only:
        print("M4_ELIGIBLE_LIVE_V2_DIAGNOSTIC_LAUNCH_BOUNDARY_PASS")
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
        "schema": "m4-shield-eligible-live-v2-liveness-diagnostic-one-shot-guard-v1",
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
