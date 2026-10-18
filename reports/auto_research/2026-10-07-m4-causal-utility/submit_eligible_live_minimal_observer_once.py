"""Consume exactly one authorization for the frozen minimal-observer bridge."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess

from research.m4_eligible_replay.contracts import canonical_json, digest_file


ROOT = Path(__file__).resolve().parents[3]
JET = Path("/Users/nvegesna/Documents/JET-LLM")
REPORT = Path(__file__).resolve().parent
SESSION = ROOT / "session/20260914_m4_opportunity_control"
MANIFEST_HASH = "de2ee7b5c1b4437b1d6ed53b9f930a3bf96dca471f31aee780aea3844fab238e"
PLAN_HASH = "bca546a19be0470aef206b11afa5637053dab3744e0a4db42878a4e5c323b7b8"
STEM = "eligible-live-minimal-observer"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("JET_GITLAB_TOKEN"):
        raise ValueError("refresh rotated setup_jet.sh before submission")
    manifest = SESSION / "m4-eligible-live-minimal-observer-verified-manifest.json"
    plan = REPORT / "eligible_live_minimal_observer_protocol.json"
    package_path = REPORT / "eligible_live_minimal_observer_package_receipt.json"
    authorization_path = (
        REPORT / "eligible_live_minimal_observer_launch_authorization.json"
    )
    if digest_file(manifest) != MANIFEST_HASH or digest_file(plan) != PLAN_HASH:
        raise ValueError("frozen package identity changed")
    package = json.loads(package_path.read_bytes())
    authorization = json.loads(authorization_path.read_bytes())
    if (
        package["manifest_sha256"] != MANIFEST_HASH
        or package["bridge_protocol_sha256"] != PLAN_HASH
        or package["clean_room_tests_run"] != 106
        or package["clean_room_tests_skipped"] != 0
        or package["shell_tests_run"] != 16
        or not package["checks"]
        or not all(value is True for value in package["checks"].values())
    ):
        raise ValueError("offline gates are not satisfied")
    if (
        authorization["manifest_sha256"] != MANIFEST_HASH
        or authorization["bridge_protocol_sha256"] != PLAN_HASH
        or authorization["maximum_launcher_invocations"] != 1
        or authorization["automatic_retry"] is not False
        or authorization["fresh_acquisition_authorized"] is not False
    ):
        raise ValueError("one-shot authorization changed")
    for name, expected in package["prior_guard_sha256"].items():
        if digest_file(SESSION / name) != expected:
            raise ValueError("consumed predecessor guard changed")
    prior = json.loads(
        (
            SESSION
            / "eligible-live-duplicate-update-deterministic-backward-one-shot-guard.json"
        ).read_bytes()
    )
    if (
        prior["parent_pipeline_id"] != 74643360
        or prior["status"] != "SUBMITTED_NO_WAIT"
    ):
        raise ValueError("reference launch identity changed")
    plan_value = json.loads(plan.read_bytes())
    reference = REPORT / plan_value["reference_file"]
    if digest_file(reference) != plan_value["reference_sha256"]:
        raise ValueError("authenticated reference changed")
    if json.loads(reference.read_bytes())["qualification_passed"] is not True:
        raise ValueError("reference qualification not passed")
    for name, expected in package["overlay_sha256"].items():
        if digest_file(ROOT / name) != expected:
            raise ValueError(f"source changed: {name}")
    custom = REPORT / "eligible_live_restore_eos_custom_config_deadline_suppressed.yaml"
    variables = REPORT / "eligible_live_capture_variables.yaml"
    if (
        digest_file(custom) != package["custom_config_sha256"]
        or digest_file(variables) != package["variables_sha256"]
    ):
        raise ValueError("submission configuration changed")
    if "deadline: false" not in custom.read_text():
        raise ValueError("client queue deadline must stay disabled")
    spec = json.loads(manifest.read_bytes())["spec"]
    if spec["time_limit"] != 14400 or spec["name"] != package["workload_name"]:
        raise ValueError("allocated limit or workload identity changed")
    paths = [
        *package["overlay_sha256"],
        str(package_path),
        str(authorization_path),
        str(Path(__file__)),
        str(custom),
        str(variables),
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], cwd=ROOT, text=True
    ).strip():
        raise ValueError("qualification inputs not committed")
    guard = SESSION / f"{STEM}-one-shot-guard.json"
    if guard.exists():
        raise FileExistsError("minimal-observer invocation already consumed")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "PASS_GUARDED_SUBMISSION_INPUTS",
                    "source_commit": commit,
                    "manifest_sha256": MANIFEST_HASH,
                    "new_launcher_invocations": 0,
                }
            )
        )
        return
    record = {
        "status": "ATTEMPT_CONSUMED_BEFORE_INVOCATION",
        "manifest_sha256": MANIFEST_HASH,
        "bridge_protocol_sha256": PLAN_HASH,
        "source_commit": commit,
        "invoked_at": datetime.now().astimezone().isoformat(),
        "maximum_invocations": 1,
        "automatic_retry": False,
        "qualification_passed": False,
        "fresh_acquisition_launched": False,
    }
    with guard.open("xb") as stream:
        stream.write(canonical_json(record))
    process = subprocess.run(
        [
            str(JET / ".venv/bin/python"),
            str(JET / "runllm.py"),
            "--prebuilt_manifest",
            str(manifest),
            "--prebuilt_custom_config",
            str(custom),
            "--prebuilt_additional_variables",
            str(variables),
            "--prebuilt_config_id",
            "dlalgo-ci/eos",
            "--no_wait",
        ],
        cwd=JET,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    (SESSION / f"{STEM}-launch.log").write_text(process.stdout)
    match = re.search(r"(?m)^(\d+): ", process.stdout)
    record.update(
        {
            "status": "SUBMITTED_NO_WAIT"
            if match
            else "CONSUMED_NO_PIPELINE_IDENTIFIED",
            "parent_pipeline_id": int(match.group(1)) if match else None,
            "launcher_return_code": process.returncode,
        }
    )
    guard.write_bytes(canonical_json(record))
    (REPORT / "eligible_live_minimal_observer_launch_receipt.json").write_bytes(
        canonical_json(record)
    )
    print(json.dumps(record, sort_keys=True))
    if process.returncode or not match:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
