#!/usr/bin/env python3
"""Submit each sealed eligible-live acquisition capture exactly once."""

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
MANIFEST_DIR = SESSION / "m4-eligible-live-acquisition-capture-manifests"
PLAN = REPORTS / "eligible_live_acquisition_plan.json"
AUTHORIZATION = REPORTS / "eligible_live_acquisition_authorization.json"
EXECUTION_LOCK = REPORTS / "eligible_live_acquisition_execution_lock.json"
PACKAGE = REPORTS / "eligible_live_acquisition_capture_package_validation.json"
CUSTOM = REPORTS / "eligible_live_capture_eos_custom_config.yaml"
VARIABLES = REPORTS / "eligible_live_capture_variables.yaml"
EXPECTED = {
    RUNLLM: "7e5ca7cb2b9807404123b19009d8d975f390f38690bf0a691f65f291b49c3cbb",
    PLAN: "49f51e2e1447febcf7777887911a72c60a34050ad170ab70a35e961cfedf3bae",
    AUTHORIZATION: "3e82b9f069877006738cadaa88516df000c47b5880278b7349372d76645774f5",
    EXECUTION_LOCK: "9aebf27d7ab61a25b823525824b101e2a8cf9540e6529ae82bedbe876c24c94b",
    PACKAGE: "6a5abf0d993342a93098a06d49f30404878d6d1bd2bbdd09df3960e086464501",
    CUSTOM: "526129dd6a5a60e2e9ab0e8e63f0abd28acf7d444e37b0905b7e74459bc4e167",
    VARIABLES: "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_boundary() -> list[dict[str, Any]]:
    subprocess.run(
        ["git", "diff-index", "--quiet", "HEAD", "--"], cwd=REPO, check=True
    )
    for path, expected in EXPECTED.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen launch input moved: {path}: {actual}")
    plan = json.loads(PLAN.read_bytes())
    authorization = json.loads(AUTHORIZATION.read_bytes())
    lock = json.loads(EXECUTION_LOCK.read_bytes())
    package = json.loads(PACKAGE.read_bytes())
    candidates = package["candidates"]
    if (
        plan["status"] != "FROZEN_BEFORE_FIRST_ACQUISITION_TRAJECTORY"
        or authorization["status"]
        != "AUTHORIZED_BEFORE_FIRST_ACQUISITION_TRAJECTORY"
        or authorization["authorized"]["required_launcher"]
        != "runllm.py --no_wait"
        or package["status"] != "PASS_AUTHORIZED_UNSUBMITTED"
        or package["candidate_count"] != 20
        or len(candidates) != 20
        or not package["parallel_capture_submission"]
        or package["client_queue_deadline"] is not None
        or package["runtime_time_limit_seconds"] != 14400
        or package["automatic_retry"]
        or package["automatic_replacement"]
        or package["automatic_extension"]
        or lock["outcome_boundary"]["base_or_shield_replay_started"]
        or lock["outcome_boundary"]["post_update_loss_computed"]
    ):
        raise RuntimeError("eligible-live capture launch boundary differs")
    expected_rows = {
        row["identity"]: row["seed"] for row in plan["trajectories"]
    }
    for candidate in candidates:
        identity = candidate["identity"]
        manifest = REPO / candidate["manifest"]
        if (
            expected_rows.get(identity) != candidate["seed"]
            or manifest != MANIFEST_DIR / f"{identity}.json"
            or sha256(manifest) != candidate["sha256"]
        ):
            raise RuntimeError(f"candidate binding moved: {identity}")
        guard = SESSION / f"eligible-live-acquisition-{identity}-one-shot-guard.json"
        log = SESSION / f"eligible-live-acquisition-{identity}-launch.log"
        if guard.exists() or log.exists():
            raise RuntimeError(f"one-shot state already exists: {identity}")
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    candidates = validate_boundary()
    if args.check_only:
        print("M4_ELIGIBLE_ACQUISITION_CAPTURE_LAUNCH_BOUNDARY_PASS", len(candidates))
        return 0
    if not os.environ.get("JET_GITLAB_TOKEN"):
        raise RuntimeError("JET_GITLAB_TOKEN is absent; authority not consumed")
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is absent; authority not consumed")

    for candidate in candidates:
        identity = candidate["identity"]
        manifest = MANIFEST_DIR / f"{identity}.json"
        guard_path = (
            SESSION / f"eligible-live-acquisition-{identity}-one-shot-guard.json"
        )
        log_path = SESSION / f"eligible-live-acquisition-{identity}-launch.log"
        command = [
            str(JET_PYTHON),
            str(RUNLLM),
            "--prebuilt_manifest",
            str(manifest),
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
            "schema": "m4-shield-eligible-live-acquisition-capture-one-shot-guard-v1",
            "status": "ATTEMPT_CONSUMED_LAUNCH_INVOKED",
            "identity": identity,
            "seed": candidate["seed"],
            "manifest_sha256": candidate["sha256"],
            "package_validation_sha256": EXPECTED[PACKAGE],
            "submission_attempt_limit": 1,
            "submission_attempts_consumed": 1,
            "automatic_retry": False,
            "automatic_replacement": False,
            "automatic_extension": False,
            "submissions": [submission],
        }
        write_json(guard_path, guard)
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
        guard["submissions"] = [submission]
        guard["status"] = (
            "SUBMITTED_NO_WAIT"
            if completed.returncode == 0 and match
            else "ATTEMPT_CONSUMED_CLIENT_FAILURE"
        )
        write_json(guard_path, guard)
        if completed.returncode != 0 or not match:
            return completed.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
