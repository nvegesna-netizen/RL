#!/usr/bin/env python3
"""Submit the checkpoint-prefix restore replacement exactly once."""

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
MANIFEST = SESSION / "m4-eligible-live-restore-checkpoint-prefix-replacement-manifest.json"
CUSTOM = REPORTS / "eligible_live_restore_eos_custom_config_deadline_suppressed.yaml"
VARIABLES = REPORTS / "eligible_live_capture_variables.yaml"
FAILURE = REPORTS / "eligible_live_restore_qualification_checkpoint_prefix_failure.json"
REPAIR = REPORTS / "eligible_live_restore_qualification_checkpoint_prefix_repair_protocol.json"
AUTHORIZATION = REPORTS / (
    "eligible_live_restore_qualification_checkpoint_prefix_replacement_authorization.json"
)
PACKAGE_VALIDATION = REPORTS / (
    "eligible_live_restore_qualification_checkpoint_prefix_package_validation.json"
)
VALIDATION = REPORTS / (
    "eligible_live_restore_qualification_checkpoint_prefix_replacement_validation.json"
)
PLAN = REPORTS / (
    "eligible_live_restore_qualification_checkpoint_prefix_replacement_submission_plan.json"
)
GUARD = SESSION / (
    "eligible-live-restore-checkpoint-prefix-replacement-one-shot-guard.json"
)
LOG = SESSION / "eligible-live-restore-checkpoint-prefix-replacement-launch.log"
EXPECTED = {
    RUNLLM: "7e5ca7cb2b9807404123b19009d8d975f390f38690bf0a691f65f291b49c3cbb",
    MANIFEST: "c4be912b90151660bc4e93a171d002aa5b1be03215bf125a555f50f6b47682ac",
    CUSTOM: "0b72764d2ff41c690bab0f6a28408917f3f04faa311b581c94c56cef9ab35168",
    VARIABLES: "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356",
    FAILURE: "8d28f46d905c712e9f1eb327f5009de03f1642d380644010d2801ebc65eb85c3",
    REPAIR: "9a635f2f3eb20d9abde91f6cfaf6055ae0fc3783648dae3a6b0df2d21035d148",
    AUTHORIZATION: "eb5ea9a86639d2522193d445f8bd3755e35efdc47bd911d4ddafdb6c9cb394c2",
    PACKAGE_VALIDATION: "f0ce22ef06efb9100df1b59b7bb378adad33a2079a92c10301a3c2b623ebcb7c",
    VALIDATION: "d4ebd0a29884d1ebadcd9837b18c1edd8c7d200b50bb091197fc0950d12421ed",
    PLAN: "fa1c430be9a6d0b9a3ca193281a72e703df0cee122797b4cef1eb8fd72a3f8ac",
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
    package_validation = json.loads(PACKAGE_VALIDATION.read_bytes())
    validation = json.loads(VALIDATION.read_bytes())
    plan = json.loads(PLAN.read_bytes())
    guard = json.loads(GUARD.read_bytes())
    if (
        failure["status"] != "TERMINAL_PRE_RESTORE_TRANSPORT_FAILURE"
        or failure["classification"]["failure_class"]
        != "checkpoint_remote_prefix_omitted"
        or failure["classification"]["checkpoint_restore_started"]
        or repair["status"] != "FROZEN_TRANSPORT_REPLACEMENT"
        or repair["only_change"]["new_template"]
        != "$API/checkpoint/policy/weights/..."
        or repair["boundaries"]["scientific_design_changed"]
        or repair["boundaries"]["runtime_source_changed"]
        or authorization["status"]
        != "AUTHORIZED_BEFORE_TRANSPORT_REPLACEMENT"
        or authorization["authorized"]["replacement_eos_submission_attempts"] != 1
        or package_validation["status"] != "PASS_AUTHORIZED_UNSUBMITTED"
        or not all(package_validation["checks"].values())
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
        raise RuntimeError("checkpoint-prefix replacement authority is unavailable")

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
