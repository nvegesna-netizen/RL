"""Consume exactly one authorization for the synthetic paired EOS qualification."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
from zoneinfo import ZoneInfo

from research.m4_eligible_replay.contracts import canonical_json, digest_file


ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
SESSION = ROOT / "session/20260914_m4_opportunity_control"
JET = Path("/Users/nvegesna/Documents/JET-LLM")
STEM = "m4-synthetic-paired-engineering"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("JET_GITLAB_TOKEN") or not os.environ.get("HF_TOKEN"):
        raise ValueError(
            "refresh rotated setup_jet.sh and require model access before submission"
        )
    package_path = REPORT / "paired_engineering_package_receipt.json"
    authorization_path = REPORT / "paired_engineering_launch_authorization.json"
    package = json.loads(package_path.read_bytes())
    authorization = json.loads(authorization_path.read_bytes())
    manifest = SESSION / "m4-synthetic-paired-engineering-ready-manifest.json"
    protocol_path = REPORT / "paired_engineering_protocol.json"
    if (
        digest_file(manifest) != package["manifest_sha256"]
        or digest_file(protocol_path) != package["protocol_sha256"]
        or authorization["manifest_sha256"] != package["manifest_sha256"]
        or authorization["protocol_sha256"] != package["protocol_sha256"]
        or authorization["maximum_launcher_invocations"] != 1
        or authorization["automatic_retry"] is not False
        or authorization["scientific_acquisition_authorized"] is not False
        or authorization["public_key_sha256"] != package["public_key_sha256"]
    ):
        raise ValueError("frozen synthetic engineering authority/package differs")
    if (
        package["clean_room_tests"]
        != {"run": 178, "skipped": 0, "failures": 0, "errors": 0}
        or package["independent_rebuild_identical"] is not True
        or package["jet_schema_validated"] is not True
        or package["private_key_uploaded"] is not False
    ):
        raise ValueError("synthetic engineering offline gates incomplete")
    for name, expected in package["source_sha256"].items():
        if digest_file(ROOT / name) != expected:
            raise ValueError("frozen synthetic source changed")
    for name, expected in package["packaging_tools_sha256"].items():
        if digest_file(ROOT / name) != expected:
            raise ValueError("frozen engineering packaging tool changed")
    for name, expected in package["prior_guard_sha256"].items():
        if digest_file(SESSION / name) != expected:
            raise ValueError("consumed predecessor guard changed")
    custom = REPORT / "eligible_live_restore_eos_custom_config_deadline_suppressed.yaml"
    variables = REPORT / "eligible_live_capture_variables.yaml"
    if (
        digest_file(custom) != package["custom_config_sha256"]
        or digest_file(variables) != package["variables_sha256"]
        or "deadline: false" not in custom.read_text()
    ):
        raise ValueError("allocation/queue configuration changed")
    spec = json.loads(manifest.read_bytes())["spec"]
    if spec["name"] != package["workload_name"] or spec["time_limit"] != 14400:
        raise ValueError("synthetic workload identity or allocated time changed")
    paths = [
        *package["source_sha256"],
        *package["packaging_tools_sha256"],
        str(package_path),
        str(authorization_path),
        str(Path(__file__)),
        str(custom),
        str(variables),
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], cwd=ROOT, text=True
    ).strip():
        raise ValueError(
            "engineering inputs must be committed before invoking the launcher"
        )
    guard = SESSION / f"{STEM}-one-shot-guard.json"
    if guard.exists():
        raise FileExistsError(
            "synthetic engineering launcher invocation already consumed"
        )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "PASS_GUARDED_SYNTHETIC_INPUTS",
                    "source_commit": commit,
                    "manifest_sha256": package["manifest_sha256"],
                    "new_launcher_invocations": 0,
                }
            )
        )
        return
    record = {
        "status": "ATTEMPT_CONSUMED_BEFORE_INVOCATION",
        "source_commit": commit,
        "manifest_sha256": package["manifest_sha256"],
        "protocol_sha256": package["protocol_sha256"],
        "invoked_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
        "maximum_invocations": 1,
        "automatic_retry": False,
        "synthetic_only": True,
        "qualification_passed": False,
        "scientific_acquisition_launched": False,
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
    fd = os.open(
        SESSION / f"{STEM}-launch.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(fd, "w") as stream:
        stream.write(process.stdout)
    match = re.search(r"(?m)^(\d+): ", process.stdout)
    record.update(
        status="SUBMITTED_NO_WAIT" if match else "CONSUMED_NO_PIPELINE_IDENTIFIED",
        parent_pipeline_id=int(match.group(1)) if match else None,
        launcher_return_code=process.returncode,
    )
    guard.write_bytes(canonical_json(record))
    (REPORT / "paired_engineering_launch_receipt.json").write_bytes(
        canonical_json(record)
    )
    print(json.dumps(record, sort_keys=True))
    if process.returncode or not match:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
