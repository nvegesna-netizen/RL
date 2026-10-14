"""Consume one explicit guard and submit the reviewed engineering workload."""

from __future__ import annotations

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
    if not os.environ.get("JET_GITLAB_TOKEN"):
        raise ValueError("source rotated setup_jet.sh before launch")
    if (
        digest_file(MANIFEST) != MANIFEST_HASH
        or digest_file(
            REPORT / "eligible_live_duplicate_update_qualification_protocol.json"
        )
        != PROTOCOL_HASH
    ):
        raise ValueError("frozen qualification package moved")
    package = json.loads(
        (REPORT / "eligible_live_duplicate_update_package_receipt.json").read_bytes()
    )
    if (
        package["manifest_sha256"] != MANIFEST_HASH
        or package["clean_room_status"] != "PASS_CLEAN_ROOM_STATIC_PACKAGE"
    ):
        raise ValueError("qualification package not reviewed")
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
    guard_path = SESSION / "eligible-live-duplicate-update-one-shot-guard.json"
    record = {
        "status": "ATTEMPT_CONSUMED_BEFORE_INVOCATION",
        "manifest_sha256": MANIFEST_HASH,
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
        str(MANIFEST),
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
    (SESSION / "eligible-live-duplicate-update-launch.log").write_text(process.stdout)
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
    (REPORT / "eligible_live_duplicate_update_launch_receipt.json").write_bytes(
        canonical_json(record)
    )
    print(json.dumps(record, sort_keys=True))
    if process.returncode or not match:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
