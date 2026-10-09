#!/usr/bin/env python3
"""Submit the Ray-dashboard-off restore replacement exactly once."""

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
MANIFEST = SESSION / "m4-eligible-live-restore-ray-dashboard-replacement-manifest.json"
SOURCE = SESSION / "m4-eligible-live-restore-source-1a07dfd7e.tar.gz"
CUSTOM = REPORTS / "eligible_live_restore_eos_custom_config_deadline_suppressed.yaml"
VARIABLES = REPORTS / "eligible_live_capture_variables.yaml"
FAILURE = REPORTS / "eligible_live_restore_qualification_ray_dashboard_failure.json"
REPAIR = REPORTS / "eligible_live_restore_qualification_ray_dashboard_repair_protocol.json"
AUTHORIZATION = REPORTS / (
    "eligible_live_restore_qualification_ray_dashboard_replacement_authorization.json"
)
EXECUTION_LOCK = REPORTS / (
    "eligible_live_restore_qualification_ray_dashboard_execution_lock.json"
)
PACKAGE_VALIDATION = REPORTS / (
    "eligible_live_restore_qualification_ray_dashboard_package_validation.json"
)
VALIDATION = REPORTS / (
    "eligible_live_restore_qualification_ray_dashboard_replacement_validation.json"
)
PLAN = REPORTS / (
    "eligible_live_restore_qualification_ray_dashboard_replacement_submission_plan.json"
)
GUARD = SESSION / "eligible-live-restore-ray-dashboard-replacement-one-shot-guard.json"
LOG = SESSION / "eligible-live-restore-ray-dashboard-replacement-launch.log"
EXPECTED = {
    RUNLLM: "7e5ca7cb2b9807404123b19009d8d975f390f38690bf0a691f65f291b49c3cbb",
    MANIFEST: "53330111d90f6ae984d7e442f21ed2914af0c9abb0c76b76ba8cb63c8b9df9a2",
    SOURCE: "9885b5e448e71d0c4946a55e2a8d4afe533af6a2926f5fbbbf97e22dc4435246",
    CUSTOM: "0b72764d2ff41c690bab0f6a28408917f3f04faa311b581c94c56cef9ab35168",
    VARIABLES: "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356",
    FAILURE: "a8e6292eb0285a6b74d3cc49da7667cc26a26bd11f25e8c1be0ed7a094cd1671",
    REPAIR: "b5525eb18f34011c2208b2fb5416ddfbcb61f46955fed24d73d3d7463e85b5fe",
    AUTHORIZATION: "22f104aca3098b4a87ef43e09e7cc9ee5ac9e486f91034aa72e23e7a35c790a2",
    EXECUTION_LOCK: "0222de8f1280139b7a20b190cb4ac5ab03b4a5d94b51d7456b2e364a89e513f3",
    PACKAGE_VALIDATION: "27d02d72d77f395576e995a64e887de25800859f98923283681a6d3512f7b1ac",
    VALIDATION: "5891b6883853ba99b1726e07d0bb596087557f9a5f61562cbbf625afd3e507b4",
    PLAN: "2b6fa860bcc3a12937f178c55a76b6d07578902bf0fbef2cdd0dba9361310428",
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
    execution_lock = json.loads(EXECUTION_LOCK.read_bytes())
    package_validation = json.loads(PACKAGE_VALIDATION.read_bytes())
    validation = json.loads(VALIDATION.read_bytes())
    plan = json.loads(PLAN.read_bytes())
    guard = json.loads(GUARD.read_bytes())
    if (
        failure["status"] != "TERMINAL_PRE_RESTORE_RUNTIME_INFRASTRUCTURE_FAILURE"
        or failure["classification"]["failure_class"]
        != "optional_ray_dashboard_startup"
        or failure["classification"]["checkpoint_restore_started"]
        or repair["status"] != "FROZEN_OPERATIONAL_RUNTIME_REPLACEMENT"
        or repair["only_change"]["new_value"] != "include_dashboard=False"
        or repair["boundaries"]["scientific_design_changed"]
        or repair["boundaries"]["model_or_optimizer_logic_changed"]
        or authorization["status"]
        != "AUTHORIZED_BEFORE_OPERATIONAL_RUNTIME_REPLACEMENT"
        or authorization["authorized"]["replacement_eos_submission_attempts"] != 1
        or execution_lock["operational_repair"]["include_dashboard"] is not False
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
        raise RuntimeError("Ray-dashboard replacement authority is unavailable")

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
        "source_archive_sha256": EXPECTED[SOURCE],
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
