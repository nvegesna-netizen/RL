#!/usr/bin/env python3
"""Consume one guarded quality-primary replacement after a proven pre-trigger client failure."""

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

REPO = Path(__file__).resolve().parents[2]
JET_ROOT = Path("/Users/nvegesna/Documents/JET-LLM")
JET_PYTHON = JET_ROOT / ".venv/bin/python"
RUNLLM = JET_ROOT / "runllm.py"
SESSION = Path(__file__).resolve().parent
REPORTS = REPO / "reports/auto_research/2026-09-22-m4-oars-v2-quality-primary"
ORIGINAL_AUTHORIZATION = REPORTS / "quality_primary_acquisition_authorization.json"
RUN_MANIFEST = REPORTS / "quality_primary_run_manifest.json"
PACKAGE_VALIDATION = REPORTS / "quality_primary_package_validation.json"
CUSTOM = SESSION / "prebuilt-custom-config-deadline-suppressed.json"
ADDITIONAL = SESSION / "prebuilt-additional-variables.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    path.relative_to(REPO.resolve())
    return path


def write_guard(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_boundary(guard_path: Path) -> tuple[dict[str, object], Path, Path]:
    guard = json.loads(guard_path.read_bytes())
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != guard["launch_commit"]:
        raise RuntimeError(f"quality-primary replacement launch commit moved: {head}")
    subprocess.run(["git", "diff-index", "--quiet", "HEAD", "--"], cwd=REPO, check=True)

    manifest = resolve_repo_path(str(guard["manifest_path"]))
    predecessor_path = resolve_repo_path(str(guard["predecessor_authentication_path"]))
    prelaunch_path = resolve_repo_path(str(guard["prelaunch_validation_path"]))
    failure_path = resolve_repo_path(str(guard["failure_receipt_path"]))
    replacement_authorization_path = resolve_repo_path(
        str(guard["replacement_authorization_path"])
    )
    replacement_validation_path = resolve_repo_path(
        str(guard["replacement_validation_path"])
    )
    original_guard_path = resolve_repo_path(str(guard["original_guard_path"]))
    original_log_path = resolve_repo_path(str(guard["original_launch_log_path"]))
    expected = {
        Path(__file__): guard["submission_script_sha256"],
        RUNLLM: guard["launcher_sha256"],
        manifest: guard["manifest_sha256"],
        CUSTOM: guard["custom_config_sha256"],
        ADDITIONAL: guard["additional_variables_sha256"],
        ORIGINAL_AUTHORIZATION: guard["original_authorization_sha256"],
        RUN_MANIFEST: guard["run_manifest_sha256"],
        PACKAGE_VALIDATION: guard["package_validation_sha256"],
        predecessor_path: guard["predecessor_authentication_sha256"],
        prelaunch_path: guard["prelaunch_validation_sha256"],
        failure_path: guard["failure_receipt_sha256"],
        replacement_authorization_path: guard["replacement_authorization_sha256"],
        replacement_validation_path: guard["replacement_validation_sha256"],
        original_guard_path: guard["original_guard_sha256"],
        original_log_path: guard["original_launch_log_sha256"],
    }
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise RuntimeError(f"frozen quality-primary replacement input moved: {path}")

    original_authorization = json.loads(ORIGINAL_AUTHORIZATION.read_bytes())
    run_manifest = json.loads(RUN_MANIFEST.read_bytes())
    package = json.loads(PACKAGE_VALIDATION.read_bytes())
    predecessor = json.loads(predecessor_path.read_bytes())
    prelaunch = json.loads(prelaunch_path.read_bytes())
    failure = json.loads(failure_path.read_bytes())
    replacement_authorization = json.loads(replacement_authorization_path.read_bytes())
    replacement_validation = json.loads(replacement_validation_path.read_bytes())
    original_guard = json.loads(original_guard_path.read_bytes())
    candidate = json.loads(manifest.read_bytes())
    identity = str(guard["identity"])
    sequence = int(guard["global_sequence"])
    run = run_manifest["runs"][sequence - 1]
    record = package["candidates"][sequence - 1]
    original_submission = original_guard["submissions"][0]
    if (
        guard["schema"]
        != "m4-oars-v2-quality-primary-client-pretrigger-replacement-one-shot-guard-v1"
        or guard["status"] != "READY_AUTHORIZED_UNSUBMITTED"
        or guard["replacement_attempt_limit"] != 1
        or guard["replacement_attempts_consumed"] != 0
        or guard["submissions"]
        or guard["automatic_retry"]
        or guard["automatic_replacement"]
        or guard["automatic_extension"]
        or original_authorization["required_launcher"] != "runllm.py --no_wait"
        or original_authorization["sequence_policy"]
        != "strict_global_sequence_one_run_at_a_time"
        or not original_authorization["complete_outcome_embargo_until_all_54_authenticated"]
        or run["identity"] != identity
        or run["global_sequence"] != sequence
        or run["predecessor"] != guard["predecessor"]
        or record["identity"] != identity
        or record["sha256"] != guard["manifest_sha256"]
        or predecessor["identity"] != guard["predecessor"]
        or predecessor["status"] != "PASS_AUTHENTICATED_OUTCOME_EMBARGOED"
        or predecessor["successor_gate"]
        != "PASS_AUTHENTICATED_TERMINAL_PREDECESSOR"
        or predecessor["scientific_outcome_accessed"]
        or prelaunch["status"] != "PASS_AUTHORIZED_UNSUBMITTED"
        or not prelaunch["launch_permitted"]
        or prelaunch["identity"] != identity
        or prelaunch["scientific_outcome_accessed"]
        or failure["status"] != "PRESERVED_FAILED_IDENTITY_NO_SCIENTIFIC_EXECUTION"
        or failure["identity"] != identity
        or failure["classification"]
        != "PRE_TRIGGER_EXTERNAL_SWIFT_DNS_FAILURE"
        or failure["gitlab_pipeline_trigger_reached"]
        or failure["scientific_execution_started"]
        or failure["scientific_outcome_accessed"]
        or replacement_authorization["status"]
        != "FROZEN_BEFORE_OPERATIONAL_REPLACEMENT"
        or replacement_authorization["identity"] != identity
        or replacement_authorization["operational_replacement_attempt_limit"] != 1
        or replacement_authorization["candidate_change_authorized"]
        or replacement_authorization["scientific_design_change_authorized"]
        or replacement_authorization["scientific_outcome_accessed"]
        or replacement_validation["status"]
        != "PASS_OPERATIONAL_REPLACEMENT_AUTHORIZED_UNSUBMITTED"
        or replacement_validation["identity"] != identity
        or replacement_validation["scientific_outcome_accessed"]
        or original_guard["status"] != "ATTEMPT_CONSUMED_CLIENT_FAILURE"
        or original_guard["submission_attempts_consumed"] != 1
        or len(original_guard["submissions"]) != 1
        or original_submission["upstream_pipeline_id"] is not None
        or original_submission["launcher_return_code"] == 0
        or candidate["spec"]["name"]
        != f"m4-oars-v2-quality-primary-{identity}-acquisition"
        or candidate["spec"]["time_limit"] != 14400
        or "sbatch_additional_flags" in candidate["spec"]
    ):
        raise RuntimeError("quality-primary client-pretrigger replacement boundary differs")
    log_path = SESSION / (
        f"oars-v2-quality-primary-{identity}-client-pretrigger-replacement-launch.log"
    )
    if log_path.exists():
        raise RuntimeError(f"{identity} client-pretrigger replacement launch log already exists")
    return guard, manifest, log_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guard", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    guard_path = args.guard.resolve()
    guard_path.relative_to(SESSION.resolve())
    guard, manifest, log_path = validate_boundary(guard_path)
    identity = str(guard["identity"])
    if args.check_only:
        print(
            "M4_OARS_V2_QUALITY_PRIMARY_"
            f"{identity.upper().replace('-', '_')}_CLIENT_PRETRIGGER_REPLACEMENT_BOUNDARY_PASS"
        )
        return 0
    if not os.environ.get("JET_GITLAB_TOKEN"):
        raise RuntimeError("JET_GITLAB_TOKEN is absent; authority not consumed")
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is absent; authority not consumed")
    command = [
        str(JET_PYTHON),
        str(RUNLLM),
        "--prebuilt_manifest",
        str(manifest),
        "--prebuilt_custom_config",
        str(CUSTOM),
        "--prebuilt_additional_variables",
        str(ADDITIONAL),
        "--prebuilt_config_id",
        "dlalgo-ci/eos",
        "--no_wait",
    ]
    submission = {
        "command_boundary": "runllm.py --no_wait",
        "launch_invoked_at": datetime.now().astimezone().isoformat(),
        "launcher_return_code": None,
        "manifest_sha256": guard["manifest_sha256"],
        "upstream_pipeline_id": None,
    }
    guard["replacement_attempts_consumed"] = 1
    guard["status"] = "ATTEMPT_CONSUMED_LAUNCH_INVOKED"
    guard["submissions"].append(submission)
    write_guard(guard_path, guard)
    completed = subprocess.run(
        command,
        cwd=JET_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    surface = completed.stdout + completed.stderr
    log_path.write_text(surface)
    sys.stdout.write(surface)
    match = re.search(
        r"(?m)^(\d+): https://gitlab-master\.nvidia\.com/dl/jet/ci/-/pipelines/\1$",
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
    write_guard(guard_path, guard)
    return completed.returncode if match else 1


if __name__ == "__main__":
    raise SystemExit(main())
