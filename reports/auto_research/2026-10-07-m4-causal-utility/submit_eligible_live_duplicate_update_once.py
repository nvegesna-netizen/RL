"""Consume one explicit guard and submit the reviewed engineering workload."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

from research.m4_eligible_replay.contracts import canonical_json, digest_file


ROOT = Path(__file__).resolve().parents[3]
JET = Path("/Users/nvegesna/Documents/JET-LLM")
REPORT = ROOT / "reports/auto_research/2026-10-07-m4-causal-utility"
SESSION = ROOT / "session/20260914_m4_opportunity_control"
MANIFEST = SESSION / "m4-eligible-live-duplicate-update-qualification-manifest.json"
MANIFEST_HASH = "7ecb51c8d232b21131753fcb7eb9bbdda00e6acab089ec84f8df26f7b7488e48"
PROTOCOL_HASH = "a57986017e250fff4140ab3e078e6cc48f1a9a6f2a6a418f9b56c66de860037d"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-replacement", action="store_true")
    parser.add_argument("--full-environment", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_replacement and args.full_environment:
        raise ValueError("select exactly one replacement package")
    manifest = MANIFEST
    manifest_hash = MANIFEST_HASH
    receipt_name = "eligible_live_duplicate_update_package_receipt.json"
    stem = "eligible-live-duplicate-update"
    launch_receipt_name = "eligible_live_duplicate_update_launch_receipt.json"
    if args.bootstrap_replacement:
        manifest = (
            SESSION
            / "m4-eligible-live-duplicate-update-bootstrap-replacement-final-manifest.json"
        )
        manifest_hash = (
            "c37bf75c3e7fa0e6142620a6e911f891a4780812daf6699b5ad24ab3bea28ed7"
        )
        receipt_name = (
            "eligible_live_duplicate_update_bootstrap_replacement_package_receipt.json"
        )
        stem = "eligible-live-duplicate-update-bootstrap-replacement"
        launch_receipt_name = (
            "eligible_live_duplicate_update_bootstrap_replacement_launch_receipt.json"
        )
    if args.full_environment:
        manifest = (
            SESSION / "m4-eligible-live-duplicate-update-full-environment-manifest.json"
        )
        manifest_hash = (
            "67269375ac579d6d9e338c76befb0f47990dacc3b3f02d44a7f8585e8065c06e"
        )
        receipt_name = (
            "eligible_live_duplicate_update_full_environment_package_receipt.json"
        )
        stem = "eligible-live-duplicate-update-full-environment"
        launch_receipt_name = (
            "eligible_live_duplicate_update_full_environment_launch_receipt.json"
        )
    if not os.environ.get("JET_GITLAB_TOKEN"):
        raise ValueError("source rotated setup_jet.sh before launch")
    if (
        digest_file(manifest) != manifest_hash
        or digest_file(
            REPORT / "eligible_live_duplicate_update_qualification_protocol.json"
        )
        != PROTOCOL_HASH
    ):
        raise ValueError("frozen qualification package moved")
    package = json.loads((REPORT / receipt_name).read_bytes())
    if (
        package["manifest_sha256"] != manifest_hash
        or package["clean_room_status"] != "PASS_CLEAN_ROOM_STATIC_PACKAGE"
    ):
        raise ValueError("qualification package not reviewed")
    if args.bootstrap_replacement or args.full_environment:
        authorization = package["replacement_authorization"]
        if (
            authorization["additional_launcher_invocations"] != 1
            or authorization["original_guard_reused"]
            or authorization["qualification_thresholds_changed"]
        ):
            raise ValueError("replacement authorization boundary changed")
        original_guard = json.loads(
            (
                SESSION / "eligible-live-duplicate-update-one-shot-guard.json"
            ).read_bytes()
        )
        if (
            original_guard["parent_pipeline_id"] != 73740465
            or original_guard["status"] != "SUBMITTED_NO_WAIT"
        ):
            raise ValueError("original launch identity changed")
    if args.full_environment:
        prior = json.loads(
            (
                SESSION
                / "eligible-live-duplicate-update-bootstrap-replacement-one-shot-guard.json"
            ).read_bytes()
        )
        if (
            prior["parent_pipeline_id"] != 73836981
            or prior["status"] != "SUBMITTED_NO_WAIT"
        ):
            raise ValueError("bootstrap replacement identity changed")
        if package["environment"]["full_import_preflight_before_transport"] is not True:
            raise ValueError("full environment gate missing")
    paths = list(package["overlay_sha256"]) + [
        "reports/auto_research/2026-10-07-m4-causal-utility/submit_eligible_live_duplicate_update_once.py"
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], cwd=ROOT, text=True
    ).strip():
        raise ValueError("qualification source is not committed")
    for name, expected in package["overlay_sha256"].items():
        if digest_file(ROOT / name) != expected:
            raise ValueError(f"qualification source changed: {name}")
    custom = REPORT / "eligible_live_restore_eos_custom_config_deadline_suppressed.yaml"
    if "deadline: false" not in custom.read_text():
        raise ValueError("queue deadline suppression missing")
    guard_path = SESSION / f"{stem}-one-shot-guard.json"
    record = {
        "status": "ATTEMPT_CONSUMED_BEFORE_INVOCATION",
        "manifest_sha256": manifest_hash,
        "invoked_at": datetime.now().astimezone().isoformat(),
        "maximum_invocations": 1,
        "automatic_retry": False,
    }
    with guard_path.open("xb") as guard:
        guard.write(canonical_json(record))
    command = [
        str(JET / ".venv/bin/python"),
        str(JET / "runllm.py"),
        "--prebuilt_manifest",
        str(manifest),
        "--prebuilt_custom_config",
        str(custom),
        "--prebuilt_additional_variables",
        str(REPORT / "eligible_live_capture_variables.yaml"),
        "--prebuilt_config_id",
        "dlalgo-ci/eos",
        "--no_wait",
    ]
    process = subprocess.run(
        command,
        cwd=JET,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    (SESSION / f"{stem}-launch.log").write_text(process.stdout)
    match = re.search(r"(?m)^(\d+): ", process.stdout)
    record.update(
        {
            "launcher_return_code": process.returncode,
            "parent_pipeline_id": int(match.group(1)) if match else None,
            "status": "SUBMITTED_NO_WAIT"
            if match
            else "CONSUMED_NO_PIPELINE_IDENTIFIED",
            "qualification_passed": False,
            "acquisition_launched": False,
        }
    )
    guard_path.write_bytes(canonical_json(record))
    (REPORT / launch_receipt_name).write_bytes(canonical_json(record))
    print(json.dumps(record, sort_keys=True))
    if process.returncode or not match:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
