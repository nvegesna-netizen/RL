"""Submit the frozen 32 fresh paired arms once each, without retries or outcomes."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

from research.m4_eligible_replay.contracts import digest_file
from research.m4_paired_replay.capture_authentication import verify_admission_bundle
from research.m4_paired_replay.contracts import capture_gate

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
SESSION = ROOT / "session/20260914_m4_opportunity_control"
READY = REPORT / "paired_fresh_execution_launch_readiness.json"
GUARDS = SESSION / "m4-paired-causal-replay-arm-guards"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPORT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate() -> tuple[dict, Path]:
    ready = json.loads(READY.read_bytes())
    if (
        ready["status"] != "PASS_FIXED_32_PAIRED_ARMS_READY"
        or ready["purpose"] != "fixed_paired_causal_replay"
    ):
        raise ValueError("reviewed fixed paired readiness required")
    for name, expected in ready["frozen_file_sha256"].items():
        if digest_file(ROOT / name) != expected:
            raise ValueError(f"frozen paired input moved: {name}")
    directory = ROOT / ready["package_directory"]
    package = json.loads((directory / "package.json").read_bytes())
    lock = json.loads((directory / "execution-lock.json").read_bytes())
    if (
        digest_file(directory / "package.json") != ready["package_sha256"]
        or digest_file(directory / "execution-lock.json")
        != package["execution_lock_sha256"]
    ):
        raise ValueError("paired package or lock moved")
    plan = json.loads((REPORT / "eligible_live_acquisition_v2_plan.json").read_bytes())
    captures = json.loads(
        (
            REPORT
            / "eligible_live_v2_capture_retry_r1_authenticated_classifications.json"
        ).read_bytes()
    )
    verify_admission_bundle(
        plan, captures, SESSION / "m4-v2-capture-retry-r1-terminal-authentication"
    )
    expected = load("build_fresh_paired_execution").fixed_arms(
        capture_gate(plan, captures)
    )
    if (
        lock["status"] != "REVIEWED_FRESH_PAIRED_EXECUTION_READY"
        or lock["fixed_arms"] != expected
        or len(package["candidates"]) != 32
    ):
        raise ValueError("fixed 16x2 replay lock differs")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", package["source_commit"], "HEAD"],
        cwd=ROOT,
        check=True,
    )
    launcher = load("submit_eligible_live_v2_captures_once")
    for candidate, row in zip(package["candidates"], expected):
        if (
            any(candidate[k] != v for k, v in row.items())
            or digest_file(ROOT / candidate["manifest"]) != candidate["sha256"]
        ):
            raise ValueError("paired candidate moved or reordered")
        launcher.read_guard(
            GUARDS / f"{candidate['identity']}.json",
            candidate=candidate,
            package_hash=ready["package_sha256"],
        )
    return ready, directory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    ready, directory = validate()
    if args.check_only:
        print("PASS_EXACT_SIXTEEN_CAPSULES_THIRTY_TWO_ARMS")
        return 0
    launcher = load("submit_eligible_live_v2_captures_once")
    for candidate in json.loads((directory / "package.json").read_bytes())[
        "candidates"
    ]:
        validate()
        if launcher.submit(candidate, ready, directory, guard_directory=GUARDS) != 0:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
