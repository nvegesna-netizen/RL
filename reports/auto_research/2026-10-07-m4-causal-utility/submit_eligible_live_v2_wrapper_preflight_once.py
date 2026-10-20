"""Launch exactly one repaired no-training wrapper preflight, never the cohort."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from research.m4_eligible_replay.contracts import digest_file

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
READY = REPORT / "eligible_live_v2_wrapper_preflight_launch_readiness.json"
GUARDS = (
    ROOT
    / "session/20260914_m4_opportunity_control/m4-v2-wrapper-repair-preflight-guards"
)


def validate() -> dict:
    ready = json.loads(READY.read_bytes())
    if (
        ready["status"] != "PASS_WRAPPER_PREFLIGHT_READY"
        or ready["purpose"] != "wrapper_preflight_no_training"
    ):
        raise ValueError("single preflight readiness required")
    for name, expected in ready["frozen_file_sha256"].items():
        if digest_file(ROOT / name) != expected:
            raise ValueError(f"frozen preflight input changed: {name}")
    protocol = json.loads(
        (
            REPORT / "eligible_live_v2_wrapper_repair_preflight_protocol.json"
        ).read_bytes()
    )
    if (
        protocol["maximum_launcher_invocations"] != 1
        or protocol["gates"]["actual_training_entrypoint_invocations"] != 0
    ):
        raise ValueError("single no-training preflight boundary changed")
    manifest = json.loads((ROOT / ready["candidate"]["manifest"]).read_bytes())
    if (
        digest_file(ROOT / ready["candidate"]["manifest"])
        != ready["candidate"]["sha256"]
        or manifest["spec"]["time_limit"] != 14400
        or "examples/run_grpo_single_controller.py" in manifest["spec"]["script"]
    ):
        raise ValueError("preflight may not contain a training entrypoint")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", ready["source_commit"], "HEAD"],
        cwd=ROOT,
        check=True,
    )
    return ready


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    ready = validate()
    spec = importlib.util.spec_from_file_location(
        "one_shot_submit", REPORT / "submit_eligible_live_v2_captures_once.py"
    )
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    launcher.read_guard(
        GUARDS / f"{ready['candidate']['identity']}.json",
        candidate=ready["candidate"],
        package_hash=ready["package_sha256"],
    )
    if args.check_only:
        print("PASS_SINGLE_NO_TRAINING_PREFLIGHT_BOUNDARY")
        return 0
    return launcher.submit(
        ready["candidate"],
        ready,
        ROOT / ready["package_directory"],
        guard_directory=GUARDS,
    )


if __name__ == "__main__":
    sys.exit(main())
