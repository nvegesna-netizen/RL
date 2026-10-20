"""Submit one excluded real-buffer watchdog qualification and gated stall probe."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

from research.m4_eligible_replay.contracts import digest_file

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
READY = REPORT / "watchdog_stall_diagnostic_launch_readiness.json"
GUARDS = (
    ROOT / "session/20260914_m4_opportunity_control/m4-watchdog-stall-diagnostic-guards"
)


def launcher() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "native_once", REPORT / "submit_eligible_live_v2_captures_once.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate() -> dict:
    ready = json.loads(READY.read_bytes())
    if (
        ready["status"] != "PASS_EXCLUDED_WATCHDOG_DIAGNOSTIC_READY"
        or ready["purpose"] != "excluded_watchdog_stall_diagnostic"
    ):
        raise ValueError("excluded diagnostic readiness required")
    for name, expected in ready["frozen_file_sha256"].items():
        if digest_file(ROOT / name) != expected:
            raise ValueError(f"frozen diagnostic input changed: {name}")
    protocol = json.loads(
        (REPORT / "watchdog_stall_diagnostic_protocol.json").read_bytes()
    )
    if (
        protocol["maximum_launcher_invocations"] != 1
        or protocol["scientific_acquisition"]
        or protocol["automatic_retry"]
        or protocol["paired_replays"] != 0
    ):
        raise ValueError("single excluded diagnostic boundary changed")
    manifest = json.loads((ROOT / ready["candidate"]["manifest"]).read_bytes())
    if (
        digest_file(ROOT / ready["candidate"]["manifest"])
        != ready["candidate"]["sha256"]
        or manifest["spec"]["time_limit"] != 14400
        or manifest["spec"]["name"] != "m4-v2-watchdog-stall-diagnostic-d01"
    ):
        raise ValueError("diagnostic manifest differs")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", ready["source_commit"], "HEAD"],
        cwd=ROOT,
        check=True,
    )
    launcher().read_guard(
        GUARDS / "watchdog-stall-d01.json",
        candidate=ready["candidate"],
        package_hash=ready["package_sha256"],
    )
    return ready


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    ready = validate()
    if args.check_only:
        print("PASS_ONE_EXCLUDED_WATCHDOG_DIAGNOSTIC_BOUNDARY")
        return 0
    return launcher().submit(
        ready["candidate"],
        ready,
        ROOT / ready["package_directory"],
        guard_directory=GUARDS,
    )


if __name__ == "__main__":
    sys.exit(main())
