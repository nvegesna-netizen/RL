"""Submit the prospectively amended capture retry, once per original seed."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

from research.m4_eligible_replay.contracts import digest_file

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
READY = REPORT / "eligible_live_v2_capture_retry_r1_launch_readiness.json"
SESSION = ROOT / "session/20260914_m4_opportunity_control"


def load_tool(name: str) -> ModuleType:
    """Load an existing report-local tool without altering its global boundaries."""
    spec = importlib.util.spec_from_file_location(name, REPORT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate() -> tuple[dict, Path, Path]:
    """Authenticate the complete frozen package and preserved predecessor guards."""
    ready = json.loads(READY.read_bytes())
    if (
        ready["status"] != "PASS_VERIFIED_CAPTURE_RETRY_R1_READY"
        or ready["purpose"] != "capture_retry_r1"
        or not ready["capture_only"]
        or ready["paired_replay_started"]
        or ready["scientific_outcomes_opened"]
    ):
        raise ValueError("capture-only retry readiness required")
    for name, expected in ready["frozen_file_sha256"].items():
        if digest_file(ROOT / name) != expected:
            raise ValueError(f"frozen retry input changed: {name}")
    builder = load_tool("build_eligible_live_v2_capture_package")
    amendment_path = REPORT / "eligible_live_v2_capture_retry_r1_amendment.json"
    amendment = builder.validate_retry_amendment(amendment_path)
    directory = ROOT / ready["package_directory"]
    guards = SESSION / amendment["attempt_namespace"]
    if guards == SESSION / "m4-v2-capture-fixed-cohort-guards":
        raise ValueError("retry may not consume or reset original guards")
    package = json.loads((directory / "package.json").read_bytes())
    lock = json.loads((directory / "execution-lock.json").read_bytes())
    plan = json.loads((REPORT / "eligible_live_acquisition_v2_plan.json").read_bytes())
    if (
        digest_file(directory / "package.json") != ready["package_sha256"]
        or digest_file(directory / "execution-lock.json")
        != package["execution_lock_sha256"]
        or lock["retry_amendment_sha256"] != digest_file(amendment_path)
        or lock["cohort"] != plan["trajectories"]
        or lock["execution"] != plan["execution"]
        or lock["attempt_namespace"] != amendment["attempt_namespace"]
        or lock["paired_replay_started"]
        or lock["post_update_outcomes_opened"]
        or digest_file(directory / "authorization.json") != lock["authorization_sha256"]
        or [
            {"identity": row["identity"], "seed": row["seed"]}
            for row in package["candidates"]
        ]
        != plan["trajectories"]
        or len(package["candidates"]) != 20
    ):
        raise ValueError("retry package, lock, or fixed cohort differs")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", lock["execution_source_commit"], "HEAD"],
        cwd=ROOT,
        check=True,
    )
    for name, expected in lock["artifact_tool_sha256"].items():
        committed = subprocess.check_output(
            ["git", "show", f"{lock['execution_source_commit']}:{name}"], cwd=ROOT
        )
        if (
            hashlib.sha256(committed).hexdigest() != expected
            or digest_file(ROOT / name) != expected
        ):
            raise ValueError("artifact tools differ from committed source")
    transport = json.loads((directory / "transport-verification.json").read_bytes())
    if (
        transport["status"] != "PASS_OFFLINE_TRANSPORT_AND_NATIVE_EXTRACTION"
        or transport["package_sha256"] != ready["package_sha256"]
        or transport["manifest_count"] != 20
        or transport["native_configuration_probe_count"] != 20
        or not transport["complete_native_cross_section_validator_executed"]
    ):
        raise ValueError("complete retry transport verification required")
    launcher = load_tool("submit_eligible_live_v2_captures_once")
    for candidate in package["candidates"]:
        if digest_file(ROOT / candidate["manifest"]) != candidate["sha256"]:
            raise ValueError("retry manifest changed")
        launcher.read_guard(
            guards / f"{candidate['identity']}.json",
            candidate=candidate,
            package_hash=ready["package_sha256"],
        )
    return ready, directory, guards


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    ready, directory, guards = validate()
    if args.check_only:
        print(
            "PASS_CAPTURE_RETRY_R1_BOUNDARY: twenty original seeds; fresh attempt guards"
        )
        return 0
    launcher = load_tool("submit_eligible_live_v2_captures_once")
    package = json.loads((directory / "package.json").read_bytes())
    for candidate in package["candidates"]:
        # Revalidate all immutable authority before every new invocation.
        validate()
        if launcher.submit(candidate, ready, directory, guard_directory=guards) != 0:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
