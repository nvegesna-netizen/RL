"""Submit the canary-rpc-repaired synthetic package once under a fresh authorization."""

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
STEM = "m4-synthetic-paired-engineering-canary-rpc-replacement"


def validate() -> tuple[dict, Path, Path, Path, Path]:
    """Check committed authority, transport budget, source, and immutable guards."""
    package_path = REPORT / "paired_engineering_canary_rpc_repair_package_receipt.json"
    auth_path = REPORT / "paired_engineering_canary_rpc_replacement_authorization.json"
    package = json.loads(package_path.read_bytes())
    auth = json.loads(auth_path.read_bytes())
    manifest = SESSION / "m4-synthetic-paired-canary-rpc-repaired-ready-manifest.json"
    custom = REPORT / "eligible_live_restore_eos_custom_config_deadline_suppressed.yaml"
    variables = REPORT / "eligible_live_capture_variables.yaml"
    if (
        auth["maximum_launcher_invocations"] != 1
        or auth["automatic_retry"] is not False
        or auth["scientific_acquisition_authorized"] is not False
        or digest_file(package_path) != auth["package_receipt_sha256"]
        or digest_file(manifest) != auth["manifest_sha256"]
        or auth["manifest_sha256"] != package["manifest_sha256"]
        or auth["public_key_sha256"] != package["public_key_sha256"]
    ):
        raise ValueError("replacement authority/package mismatch")
    if (
        package["clean_room_tests"]
        != {"run": 190, "skipped": 0, "failures": 0, "errors": 0, "skip_details": []}
        or package["independent_rebuild_identical"] is not True
        or package["jet_schema_validated"] is not True
        or package["rendered_shell_validated"] is not True
        or package["native_worker_and_scientific_protocol_unchanged"] is not True
        or package["private_key_uploaded"] is not False
    ):
        raise ValueError("offline qualification incomplete")
    budget = package["generator_artifact_budget"]
    if not (
        budget["total_uncompressed_member_bytes"] < budget["limit_bytes"]
        and budget["all_other_members_byte_identical"] is True
        and len(budget["changed_members"]) == 3
    ):
        raise ValueError("artifact budget gate failed")
    for relative, expected in auth["local_evidence_sha256"].items():
        if digest_file(SESSION / relative) != expected:
            raise ValueError("local transport evidence changed")
    for relative, expected in package["prior_guard_sha256"].items():
        if digest_file(SESSION / relative) != expected:
            raise ValueError("historical guard changed")
    for relative, expected in package["consumed_guard_sha256"].items():
        if digest_file(SESSION / relative) != expected:
            raise ValueError("consumed original guard changed")
    inputs = {
        **package["source_sha256"],
        **package["packaging_tools_sha256"],
        **auth["packaging_tools_sha256"],
    }
    for relative, expected in inputs.items():
        if digest_file(ROOT / relative) != expected:
            raise ValueError("frozen source or launcher changed")
    if (
        digest_file(custom) != package["custom_config_sha256"]
        or digest_file(variables) != package["variables_sha256"]
        or "deadline: false" not in custom.read_text()
    ):
        raise ValueError("queue/allocation configuration changed")
    spec = json.loads(manifest.read_bytes())["spec"]
    if spec["name"] != package["workload_name"] or spec["time_limit"] != 14400:
        raise ValueError("workload identity or allocated time changed")
    paths = [*inputs, str(package_path), str(auth_path), str(custom), str(variables)]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], cwd=ROOT, text=True
    ).strip():
        raise ValueError("replacement inputs must be committed")
    guard = SESSION / f"{STEM}-one-shot-guard.json"
    if guard.exists() or (SESSION / f"{STEM}-launch.log").exists():
        raise FileExistsError("replacement already consumed; never retry")
    return package, manifest, custom, variables, guard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("JET_GITLAB_TOKEN") or not os.environ.get("HF_TOKEN"):
        raise ValueError("refresh rotated setup_jet.sh; require model access")
    package, manifest, custom, variables, guard = validate()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "PASS_REPLACEMENT_INPUTS",
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
        "replaces_parent_pipeline_id": 75096695,
        "invoked_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
        "maximum_invocations": 1,
        "automatic_retry": False,
        "synthetic_only": True,
        "qualification_passed": False,
        "scientific_acquisition_launched": False,
    }
    with guard.open("xb") as stream:
        stream.write(canonical_json(record))
    os.chmod(guard, 0o600)
    fd = os.open(
        SESSION / f"{STEM}-launch.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(fd, "w") as stream:
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
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    output = (SESSION / f"{STEM}-launch.log").read_text()
    match = re.search(r"(?m)^(\d+): ", output)
    record.update(
        status="SUBMITTED_NO_WAIT" if match else "CONSUMED_NO_PIPELINE_IDENTIFIED",
        parent_pipeline_id=int(match.group(1)) if match else None,
        launcher_return_code=process.returncode,
    )
    guard.write_bytes(canonical_json(record))
    (
        REPORT / "paired_engineering_canary_rpc_replacement_launch_receipt.json"
    ).write_bytes(canonical_json(record))
    print(json.dumps(record, sort_keys=True))
    if process.returncode or not match:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
