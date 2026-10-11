#!/usr/bin/env python3
"""Validate the sealed 20-trajectory acquisition-capture package."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


PLAN_SHA256 = "49f51e2e1447febcf7777887911a72c60a34050ad170ab70a35e961cfedf3bae"
AUTHORIZATION_SHA256 = "3e82b9f069877006738cadaa88516df000c47b5880278b7349372d76645774f5"
LOCK_SHA256 = "9aebf27d7ab61a25b823525824b101e2a8cf9540e6529ae82bedbe876c24c94b"
SOURCE_SHA256 = "4cafd18a9dc16b6921e3e39995633d3a32de46f935e4818e8a177104a3a89548"
ASSIGNMENT_DOMAIN = "m4-eligible-live-causal-acquisition-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(sha256(args.plan) == PLAN_SHA256, "acquisition plan moved")
    require(
        sha256(args.authorization) == AUTHORIZATION_SHA256,
        "acquisition authorization moved",
    )
    require(sha256(args.execution_lock) == LOCK_SHA256, "execution lock moved")
    plan = json.loads(args.plan.read_bytes())
    authorization = json.loads(args.authorization.read_bytes())
    lock = json.loads(args.execution_lock.read_bytes())
    require(
        authorization["authorized"]["required_launcher"] == "runllm.py --no_wait",
        "launcher authority changed",
    )
    require(
        authorization["authorized"]["fixed_outcome_excluded_capture_trajectories"]
        == 20,
        "capture count authority changed",
    )
    require(
        lock["source_archive_sha256"] == SOURCE_SHA256,
        "source archive binding changed",
    )
    require(
        lock["capture_execution"]["client_queue_deadline"] is None,
        "client queue deadline was introduced",
    )

    expected = {row["identity"]: row["seed"] for row in plan["trajectories"]}
    paths = sorted(args.manifest_dir.glob("a*.json"))
    require(len(paths) == 20, "capture manifest count is not 20")
    candidates: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for path in paths:
        identity = path.stem
        require(identity in expected, f"unexpected identity {identity}")
        seed = expected[identity]
        manifest = json.loads(path.read_bytes())
        spec = manifest.get("spec", {})
        script_template = spec.get("script", "")
        require(spec.get("name") == f"m4-eligible-live-capture-{identity}", "name drift")
        require(spec.get("time_limit") == 14400, "four-hour runtime cap drift")
        require(spec.get("nodes") == 1, "node count drift")
        require("sbatch_additional_flags" not in spec, "unexpected scheduler flags")
        require(spec["name"] not in seen_names, "duplicate workload name")
        seen_names.add(spec["name"])
        script = script_template.format(assets_dir="/tmp/m4-assets")
        syntax = subprocess.run(
            ["bash", "-n"], input=script, text=True, capture_output=True, check=False
        )
        require(syntax.returncode == 0, f"shell syntax failed for {identity}: {syntax.stderr}")
        markers = {
            f"readonly IDENTITY={identity}": True,
            f"readonly SEED={seed}": True,
            f"readonly ASSIGNMENT_DOMAIN={ASSIGNMENT_DOMAIN}": True,
            '"grpo.seed=$SEED"': True,
            '"async_rl.controlled_release_delay.seed=$SEED"': True,
            '"async_rl.eligible_live_decision_capture.trajectory_identity=$IDENTITY"': True,
            '"async_rl.eligible_live_decision_capture.trajectory_seed=$SEED"': True,
            "PASS_OUTCOME_EXCLUDED_ELIGIBLE_CAPTURE": True,
            'readonly GATE_RC=0': True,
            "post_update_outcomes_opened": True,
            "paired_replay_started": True,
        }
        require(all(marker in script for marker in markers), f"runtime marker drift: {identity}")
        require(
            script.count("examples/run_grpo_single_controller.py") == 1,
            f"training invocation count drift: {identity}",
        )
        require(
            "run_eligible_live_restore_qualification.py" not in script,
            f"restore qualification leaked into capture: {identity}",
        )
        require(
            "--deadline" not in script and "queue_deadline" not in script,
            f"queue deadline leaked into capture: {identity}",
        )
        candidates.append(
            {
                "identity": identity,
                "seed": seed,
                "manifest": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "shell_syntax": "PASS",
            }
        )

    result = {
        "schema": "m4-shield-eligible-live-acquisition-capture-package-validation-v1",
        "status": "PASS_AUTHORIZED_UNSUBMITTED",
        "plan_sha256": PLAN_SHA256,
        "authorization_sha256": AUTHORIZATION_SHA256,
        "execution_lock_sha256": LOCK_SHA256,
        "source_archive_sha256": SOURCE_SHA256,
        "candidate_count": len(candidates),
        "parallel_capture_submission": True,
        "runtime_time_limit_seconds": 14400,
        "client_queue_deadline": None,
        "minimum_authenticated_capsules_before_replay": 16,
        "outcome_embargo": {
            "capture_post_update_outcomes": False,
            "paired_replay_started": False,
            "partial_outcome_access": False,
        },
        "automatic_retry": False,
        "automatic_replacement": False,
        "automatic_extension": False,
        "candidates": candidates,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output, sha256(args.output))


if __name__ == "__main__":
    main()
