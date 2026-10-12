#!/usr/bin/env python3
"""Submit each authorized eligible-live v2 liveness identity exactly once."""

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
MANIFEST_DIR = SESSION / "m4-eligible-live-v2-liveness-qualification-manifests"
SOURCE = (
    SESSION
    / "m4-eligible-live-v2-liveness-qualification-source-11143be69.tar.gz"
)
PROTOCOL = REPORTS / "eligible_live_v2_liveness_qualification_protocol.json"
AUTHORIZATION = REPORTS / "eligible_live_v2_liveness_qualification_authorization.json"
EXECUTION_LOCK = (
    REPORTS / "eligible_live_v2_liveness_qualification_execution_lock.json"
)
PACKAGE = REPORTS / "eligible_live_v2_liveness_qualification_package_validation.json"
RECEIPT = REPORTS / "eligible_live_v2_liveness_qualification_package_receipt.json"
CUSTOM = REPORTS / "eligible_live_restore_eos_custom_config_deadline_suppressed.yaml"
VARIABLES = REPORTS / "eligible_live_capture_variables.yaml"
IDENTITIES = (
    ("lq01", 20261102, "0e9084f6821478a53ce770214ea4b64d608743e23ae7dbe7fb3fade048f89da5"),
    ("lq02", 20261103, "4705ae2ea33efed8aaeac0055a98494c08188666523c022b5cc2406a462c7120"),
    ("lq03", 20261109, "1eee41d636886306456dbb78d60e96601495706fa6d78cbab5e792f6a1a5616e"),
    ("lq04", 20261110, "ee33a0b1b3ba283d73cb3dbc72c62b28b37bc7a978d8b0d3886b6738cf967c65"),
    ("lq05", 20261111, "6bccf51e0f4d5a3436bf1e31035091e26faecec174148c069ff22751ad25578e"),
    ("lq06", 20261119, "5abbc183890f1a52da2bc43d2e3ac58a9e7e615f7bbc22b094ef274177889980"),
)
EXPECTED = {
    RUNLLM: "7e5ca7cb2b9807404123b19009d8d975f390f38690bf0a691f65f291b49c3cbb",
    SOURCE: "ff8cb7f09b64b6ff5d781c12c52d566c8c8821ebab559d569e322ec2fabe691f",
    PROTOCOL: "eabfcb0fae3f70206dc57182f7c25661a893d77418f8b8ea05daef6730003479",
    AUTHORIZATION: "aee963ca3dd20951bfba9d71056496e870120b3a52bb68979b272d0d33cb177a",
    EXECUTION_LOCK: "12cf416f41fb814a5a6f9c038c83ce88fbbc87146fc1963ea51b7a49f8e65a2b",
    PACKAGE: "037006822fdab25114b8c7e2a9430f5ba99f6247544776ba0bc4c25fb4f7fa4f",
    RECEIPT: "7b564f9496898c46883b174c27fd2edd3c828aaf83d379345d987e7697439b75",
    CUSTOM: "0b72764d2ff41c690bab0f6a28408917f3f04faa311b581c94c56cef9ab35168",
    VARIABLES: "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def guard_path(identity: str) -> Path:
    return SESSION / f"eligible-live-v2-liveness-{identity}-one-shot-guard.json"


def log_path(identity: str) -> Path:
    return SESSION / f"eligible-live-v2-liveness-{identity}-launch.log"


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
    if (
        protocol["status"] != "FROZEN_BEFORE_IMPLEMENTATION_AND_EXECUTION"
        or protocol["joint_gate"]["required_passes"] != 6
        or authorization["authorized"]["qualification_eos_submission_attempts"]
        != 6
        or authorization["authorized"]["required_launcher"] != "runllm.py --no_wait"
        or lock["execution"]["submission_attempt_limit_total"] != 6
        or lock["execution"]["client_queue_deadline"] is not None
        or lock["execution"]["automatic_retry"]
        or package["status"]
        != "PASS_AUTHORIZED_SIX_IDENTITY_QUALIFICATION_UNSUBMITTED"
        or package["client_queue_deadline"] is not None
        or package["post_update_heldout_outcomes_opened"]
        or package["v2_acquisition_started"]
        or "deadline: false" not in CUSTOM.read_text()
    ):
        raise RuntimeError("eligible-live v2 liveness launch boundary differs")
    for identity, seed, expected in IDENTITIES:
        manifest = MANIFEST_DIR / f"{identity}.json"
        if sha256(manifest) != expected:
            raise RuntimeError(f"frozen manifest moved: {identity}")
        value = json.loads(manifest.read_bytes())
        if (
            value["spec"]["name"]
            != f"m4-eligible-live-v2-liveness-qualification-{identity}"
            or value["spec"]["time_limit"] != 14400
            or f"readonly SEED={seed}" not in value["spec"]["script"]
        ):
            raise RuntimeError(f"manifest boundary differs: {identity}")
        if guard_path(identity).exists() or log_path(identity).exists():
            raise RuntimeError(f"submission is already consumed: {identity}")


def launch(identity: str, seed: int, manifest_sha256: str) -> bool:
    manifest = MANIFEST_DIR / f"{identity}.json"
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
        "schema": "m4-shield-eligible-live-v2-liveness-one-shot-guard-v1",
        "status": "ATTEMPT_CONSUMED_LAUNCH_INVOKED",
        "identity": identity,
        "seed": seed,
        "manifest_sha256": manifest_sha256,
        "package_validation_sha256": EXPECTED[PACKAGE],
        "submission_attempt_limit": 1,
        "submission_attempts_consumed": 1,
        "automatic_retry": False,
        "automatic_replacement": False,
        "automatic_extension": False,
        "submissions": [submission],
    }
    write_json(guard_path(identity), guard)
    process = subprocess.Popen(
        command,
        cwd=JET_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    surface_parts: list[str] = []
    assert process.stdout is not None
    with log_path(identity).open("w") as log:
        for line in process.stdout:
            surface_parts.append(line)
            log.write(line)
            log.flush()
            sys.stdout.write(f"[{identity}] {line}")
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
    success = return_code == 0 and match is not None
    guard["submissions"] = [submission]
    guard["status"] = (
        "SUBMITTED_NO_WAIT" if success else "ATTEMPT_CONSUMED_CLIENT_FAILURE"
    )
    write_json(guard_path(identity), guard)
    return success


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    validate_boundary()
    if args.check_only:
        print("M4_ELIGIBLE_LIVE_V2_LIVENESS_SIX_LAUNCH_BOUNDARY_PASS")
        return 0
    if not os.environ.get("JET_GITLAB_TOKEN"):
        raise RuntimeError("JET_GITLAB_TOKEN is absent; authority not consumed")
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is absent; authority not consumed")
    results = [launch(*row) for row in IDENTITIES]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
