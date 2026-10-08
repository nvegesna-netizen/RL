#!/usr/bin/env python3
"""Consume exactly one authorized eligible-live restore qualification launch."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
JET_ROOT = Path("/Users/nvegesna/Documents/JET-LLM")
JET_PYTHON = JET_ROOT / ".venv/bin/python"
RUNLLM = JET_ROOT / "runllm.py"
REPORTS = REPO / "reports/auto_research/2026-10-07-m4-causal-utility"
SESSION = REPO / "session/20260914_m4_opportunity_control"
MANIFEST = SESSION / "m4-eligible-live-restore-qualification-manifest.json"
CUSTOM = REPORTS / "eligible_live_capture_eos_custom_config.yaml"
ADDITIONAL = REPORTS / "eligible_live_capture_variables.yaml"
AUTHORIZATION = REPORTS / "eligible_live_restore_qualification_authorization.json"
PACKAGE_VALIDATION = (
    REPORTS / "eligible_live_restore_qualification_package_validation.json"
)
PACKAGE_RECEIPT = REPORTS / "eligible_live_restore_qualification_package_receipt.json"
SUBMISSION_PLAN = REPORTS / "eligible_live_restore_qualification_submission_plan.json"
GUARD = SESSION / "eligible-live-restore-qualification-one-shot-guard.json"
LOG = SESSION / "eligible-live-restore-qualification-launch.log"
EXPECTED = {
    RUNLLM: "7e5ca7cb2b9807404123b19009d8d975f390f38690bf0a691f65f291b49c3cbb",
    MANIFEST: "0d91bc0e113ff9f358837577c6b5f4a3094299bcd58b72c3b51086e82667df1c",
    CUSTOM: "526129dd6a5a60e2e9ab0e8e63f0abd28acf7d444e37b0905b7e74459bc4e167",
    ADDITIONAL: "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356",
    AUTHORIZATION: "d16340f130c4d4d816b5d87d7f20c77a5ff372b020f30d4bd06f6940691c3cf7",
    PACKAGE_VALIDATION: "9d3109cb1ec36d396a66818cbdbc1002ace7a94aad6ff9a6cc0f097f7fa5029b",
    PACKAGE_RECEIPT: "a53204410f7269ae33a9e2102fd62f2130341088fe60f0c1cb781b899da0e66c",
    SUBMISSION_PLAN: "2e9e79b7fc1eac4f12029fc2d229e2456d2d725385ae36863f0c8ff387029367",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_guard(value: dict[str, object]) -> None:
    temporary = GUARD.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(GUARD)


def main() -> int:
    if not os.environ.get("JET_GITLAB_TOKEN"):
        raise RuntimeError("JET_GITLAB_TOKEN is absent; authority not consumed")
    subprocess.run(["git", "diff-index", "--quiet", "HEAD", "--"], cwd=REPO, check=True)
    for path, expected in EXPECTED.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen restore input moved: {path.name}: {actual}")
    authorization = json.loads(AUTHORIZATION.read_bytes())
    validation = json.loads(PACKAGE_VALIDATION.read_bytes())
    receipt = json.loads(PACKAGE_RECEIPT.read_bytes())
    plan = json.loads(SUBMISSION_PLAN.read_bytes())
    guard = json.loads(GUARD.read_bytes())
    if (
        authorization["status"] != "AUTHORIZED_BEFORE_RESTORE_QUALIFICATION"
        or authorization["authorized"]["eos_submission_attempts"] != 1
        or validation["status"] != "PASS_AUTHORIZED_UNSUBMITTED"
        or validation["manifest_sha256"] != EXPECTED[MANIFEST]
        or not all(validation["checks"].values())
        or receipt["status"] != "PASS_AUTHORIZED_UNSUBMITTED"
        or receipt["manifest_sha256"] != EXPECTED[MANIFEST]
        or plan["status"] != "READY_FOR_SINGLE_SUBMISSION"
        or plan["manifest_sha256"] != EXPECTED[MANIFEST]
        or plan["package_receipt_sha256"] != EXPECTED[PACKAGE_RECEIPT]
        or plan["launcher"]["submission_attempt_limit"] != 1
        or plan["launcher"]["automatic_retry"]
        or plan["execution"]["client_queue_deadline"] is not None
        or plan["outcome_boundary"]["base_or_shield_post_update_outcomes_opened"]
        or plan["outcome_boundary"]["paired_acquisition_started"]
        or guard["status"] != "READY_AUTHORIZED_UNSUBMITTED"
        or guard["submission_attempts_consumed"] != 0
        or guard["submission_attempt_limit"] != 1
        or guard["automatic_retry"]
        or guard["automatic_extension"]
    ):
        raise RuntimeError("restore qualification authority is unavailable")
    command = [
        str(JET_PYTHON),
        str(RUNLLM),
        "--prebuilt_manifest",
        str(MANIFEST),
        "--prebuilt_custom_config",
        str(CUSTOM),
        "--prebuilt_additional_variables",
        str(ADDITIONAL),
        "--prebuilt_config_id",
        "dlalgo-ci/eos",
        "--no_wait",
    ]
    submission: dict[str, object] = {
        "command_boundary": "runllm.py --no_wait",
        "launch_invoked_at": datetime.now().astimezone().isoformat(),
        "manifest_sha256": EXPECTED[MANIFEST],
    }
    guard["submission_attempts_consumed"] = 1
    guard["status"] = "ATTEMPT_CONSUMED_CLIENT_RUNNING"
    guard["submissions"].append(submission)
    write_guard(guard)
    completed = subprocess.run(
        command,
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    surface = completed.stdout + completed.stderr
    LOG.write_text(surface, encoding="utf-8")
    sys.stdout.write(surface)
    match = re.search(
        r"(?m)^(\d+): "
        r"https://gitlab-master\.nvidia\.com/dl/jet/ci/-/pipelines/\1$",
        surface,
    )
    submission.update(
        {
            "launcher_return_code": completed.returncode,
            "upstream_pipeline_id": int(match.group(1)) if match else None,
        }
    )
    guard["submissions"][-1] = submission
    guard["status"] = (
        "SUBMITTED_NO_WAIT"
        if completed.returncode == 0 and match
        else "ATTEMPT_CONSUMED_CLIENT_FAILURE"
    )
    write_guard(guard)
    return completed.returncode if match else 1


if __name__ == "__main__":
    raise SystemExit(main())
