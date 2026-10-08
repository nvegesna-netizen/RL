#!/usr/bin/env python3
"""Submit the deadline-suppressed restore qualification exactly once."""

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
CUSTOM = REPORTS / "eligible_live_restore_eos_custom_config_deadline_suppressed.yaml"
VARIABLES = REPORTS / "eligible_live_capture_variables.yaml"
FAILURE = REPORTS / "eligible_live_restore_qualification_deadline_failure.json"
REPAIR = REPORTS / "eligible_live_restore_qualification_deadline_repair_protocol.json"
AUTHORIZATION = (
    REPORTS
    / "eligible_live_restore_qualification_deadline_replacement_authorization.json"
)
VALIDATION = (
    REPORTS / "eligible_live_restore_qualification_deadline_replacement_validation.json"
)
PLAN = (
    REPORTS
    / "eligible_live_restore_qualification_deadline_replacement_submission_plan.json"
)
GUARD = SESSION / "eligible-live-restore-deadline-replacement-one-shot-guard.json"
LOG = SESSION / "eligible-live-restore-deadline-replacement-launch.log"
EXPECTED = {
    RUNLLM: "7e5ca7cb2b9807404123b19009d8d975f390f38690bf0a691f65f291b49c3cbb",
    MANIFEST: "0d91bc0e113ff9f358837577c6b5f4a3094299bcd58b72c3b51086e82667df1c",
    CUSTOM: "0b72764d2ff41c690bab0f6a28408917f3f04faa311b581c94c56cef9ab35168",
    VARIABLES: "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356",
    FAILURE: "c5982623775b63c4a6e4c3b3ca4e20cd87f4c2ddcd976b6146e23360c2bff5f9",
    REPAIR: "1f483ebe08be4b735e76e2b5a2a5c0249b228109f4592ac3023ac435582f0d18",
    AUTHORIZATION: "9274e657453d603962ffc2a3dc2ad6ad4ec8208bbbe92c10c05cdec743b27f50",
    VALIDATION: "fa9a7f5da27fe03177b9da606548a4b4e9ac4c4337a8ab09f6d43ca434fc3168",
    PLAN: "46a1bc7d4189b39eec2acafbd8e1ef8852dd518b0c6afeb34fc1a1ff2b8df902",
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
            raise RuntimeError(f"frozen replacement input moved: {path.name}: {actual}")

    failure = json.loads(FAILURE.read_bytes())
    repair = json.loads(REPAIR.read_bytes())
    authorization = json.loads(AUTHORIZATION.read_bytes())
    validation = json.loads(VALIDATION.read_bytes())
    plan = json.loads(PLAN.read_bytes())
    guard = json.loads(GUARD.read_bytes())
    if (
        failure["status"] != "TERMINAL_INFRASTRUCTURE_FAILURE_ZERO_RUNTIME"
        or failure["terminal"]["slurm_runtime"] != "00:00:00"
        or failure["classification"]["workload_process_started"]
        or repair["status"] != "FROZEN_OPERATIONAL_REPLACEMENT"
        or repair["only_change"]["new_value"] is not False
        or repair["boundaries"]["scientific_design_changed"]
        or authorization["status"]
        != "AUTHORIZED_BEFORE_OPERATIONAL_REPLACEMENT"
        or authorization["authorized"]["replacement_eos_submission_attempts"] != 1
        or validation["status"] != "PASS_AUTHORIZED_UNSUBMITTED"
        or not all(validation["checks"].values())
        or not validation["queue_deadline_suppressed"]
        or validation["workload_time_limit_seconds"] != 14400
        or plan["status"] != "READY_FOR_SINGLE_REPLACEMENT_SUBMISSION"
        or plan["execution"]["client_queue_deadline"] is not None
        or not plan["execution"]["default_deadline_suppressed"]
        or plan["launcher"]["replacement_submission_attempt_limit"] != 1
        or plan["launcher"]["automatic_retry"]
        or plan["launcher"]["automatic_extension"]
        or guard["status"] != "READY_AUTHORIZED_UNSUBMITTED"
        or guard["submission_attempts_consumed"] != 0
        or guard["submission_attempt_limit"] != 1
        or guard["automatic_retry"]
        or guard["automatic_extension"]
    ):
        raise RuntimeError("deadline replacement authority is unavailable")

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
    submission: dict[str, object] = {
        "command_boundary": "runllm.py --no_wait",
        "launch_invoked_at": datetime.now().astimezone().isoformat(),
        "manifest_sha256": EXPECTED[MANIFEST],
        "custom_config_sha256": EXPECTED[CUSTOM],
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
