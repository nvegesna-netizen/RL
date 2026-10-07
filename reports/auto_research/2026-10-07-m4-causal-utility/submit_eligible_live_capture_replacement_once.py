#!/usr/bin/env python3
"""Consume exactly one authorized replacement eligible-live qualification launch."""

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
MANIFEST = SESSION / "m4-eligible-live-capture-manifest.json"
CUSTOM = REPORTS / "eligible_live_capture_eos_custom_config.yaml"
ADDITIONAL = REPORTS / "eligible_live_capture_variables.yaml"
AUTHORIZATION = REPORTS / "eligible_live_capture_replacement_authorization.json"
PRIOR_RECEIPT = REPORTS / "eligible_live_capture_submission_attempt_receipt.json"
GUARD = SESSION / "eligible-live-capture-replacement-one-shot-guard.json"
LOG = SESSION / "eligible-live-capture-replacement-launch.log"
EXPECTED = {
    RUNLLM: "7e5ca7cb2b9807404123b19009d8d975f390f38690bf0a691f65f291b49c3cbb",
    MANIFEST: "c8ce2aae1c6cc8a6004bca63688210f946fbc909b8826138c42b851f7f749dc7",
    CUSTOM: "526129dd6a5a60e2e9ab0e8e63f0abd28acf7d444e37b0905b7e74459bc4e167",
    ADDITIONAL: "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356",
    PRIOR_RECEIPT: "ca64fe0ee224e9009404c52f13515abb4879e589a37fdf052ffece33fc131316",
    AUTHORIZATION: "5ab7430d1b97d0ed785c04406c0d11fc0d45437a942204794fff621871dbf6e7",
}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one frozen launch input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_guard(value: dict[str, object]) -> None:
    """Atomically persist one-shot state before and after submission."""
    temporary = GUARD.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(GUARD)


def main() -> int:
    """Validate frozen inputs, consume replacement authority, and submit once."""
    if not os.environ.get("JET_GITLAB_TOKEN"):
        raise RuntimeError("JET_GITLAB_TOKEN is absent; authority not consumed")
    subprocess.run(["git", "diff-index", "--quiet", "HEAD", "--"], cwd=REPO, check=True)
    for path, expected in EXPECTED.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen replacement input moved: {path.name}: {actual}")
    authorization = json.loads(AUTHORIZATION.read_bytes())
    prior = json.loads(PRIOR_RECEIPT.read_bytes())
    guard = json.loads(GUARD.read_bytes())
    if (
        authorization["status"] != "AUTHORIZED_AFTER_NO_PIPELINE_CREATED"
        or authorization["authorized"]["replacement_launcher_invocations"] != 1
        or authorization["authorized"]["unchanged_manifest_sha256"]
        != EXPECTED[MANIFEST]
        or prior["status"] != "ATTEMPT_CONSUMED_NO_PIPELINE_CREATED"
        or prior["external_reconciliation"]["new_pipeline_created"]
        or guard["status"] != "READY_AUTHORIZED_UNSUBMITTED"
        or guard["submission_attempts_consumed"] != 0
        or guard["submission_attempt_limit"] != 1
        or guard["automatic_retry"]
        or guard["automatic_extension"]
    ):
        raise RuntimeError("replacement launch authority is unavailable")
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
    guard["status"] = "ATTEMPT_CONSUMED_LAUNCH_INVOKED"
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
