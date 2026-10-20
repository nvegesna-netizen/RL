"""Submit frozen v2 captures once per identity, with durable pre-invocation guards."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from research.m4_eligible_replay.contracts import canonical_json, digest_file

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
SESSION = ROOT / "session/20260914_m4_opportunity_control"
JET = Path("/Users/nvegesna/Documents/JET-LLM")
READY = REPORT / "eligible_live_v2_capture_launch_readiness.json"
GUARDS = SESSION / "m4-v2-capture-fixed-cohort-guards"
PLAN_HASH = "5dbe1ec155a9c3e7d57363820625286290f6803c0bd2e47484847ed3980e141e"


def read_guard(path: Path, *, candidate: dict, package_hash: str) -> dict | None:
    """An ambiguous prior invocation blocks; never reset or silently retry it."""
    if not path.exists():
        return None
    record = json.loads(path.read_bytes())
    if (
        record["identity"] != candidate["identity"]
        or record["manifest_sha256"] != candidate["sha256"]
        or record["package_sha256"] != package_hash
        or record["attempts_consumed"] != 1
        or record["status"] != "SUBMITTED_NO_WAIT"
        or type(record["parent_pipeline_id"]) is not int
        or record["parent_pipeline_id"] < 1
    ):
        raise ValueError(
            "prior submission ambiguous or mismatched; authenticate it before continuing"
        )
    return record


def validate() -> tuple[dict, Path]:
    ready = json.loads(READY.read_bytes())
    if (
        ready["status"] != "PASS_VERIFIED_FIXED_COHORT_READY"
        or ready["plan_sha256"] != PLAN_HASH
    ):
        raise ValueError("verified readiness receipt missing")
    directory = ROOT / ready["package_directory"]
    package = json.loads((directory / "package.json").read_bytes())
    lock = json.loads((directory / "execution-lock.json").read_bytes())
    for name, expected in ready["frozen_file_sha256"].items():
        if digest_file(ROOT / name) != expected:
            raise ValueError(f"frozen launch input moved: {name}")
    if digest_file(directory / "package.json") != ready["package_sha256"]:
        raise ValueError("frozen manifest package moved")
    if (
        digest_file(directory / "execution-lock.json")
        != package["execution_lock_sha256"]
    ):
        raise ValueError("execution lock moved")
    plan = json.loads((REPORT / "eligible_live_acquisition_v2_plan.json").read_bytes())
    if lock["cohort"] != plan["trajectories"] or lock["execution"] != plan["execution"]:
        raise ValueError("cohort or execution boundary changed")
    if (
        len(package["candidates"]) != 20
        or [
            {"identity": row["identity"], "seed": row["seed"]}
            for row in package["candidates"]
        ]
        != plan["trajectories"]
    ):
        raise ValueError("candidate cohort incomplete or reordered")
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
            raise ValueError("artifact tool differs from committed execution source")
    for candidate in package["candidates"]:
        if digest_file(ROOT / candidate["manifest"]) != candidate["sha256"]:
            raise ValueError("manifest changed")
        read_guard(
            GUARDS / f"{candidate['identity']}.json",
            candidate=candidate,
            package_hash=ready["package_sha256"],
        )
    return ready, directory


def submit(candidate: dict, ready: dict, directory: Path) -> int:
    """Perform exactly one native invocation; no timeout or duplicate upload retry."""
    identity = candidate["identity"]
    GUARDS.mkdir(mode=0o700, parents=True, exist_ok=True)
    guard = GUARDS / f"{identity}.json"
    prior = read_guard(guard, candidate=candidate, package_hash=ready["package_sha256"])
    if prior is not None:
        print(
            f"{identity}: already submitted parent {prior['parent_pipeline_id']}",
            flush=True,
        )
        return 0
    # Readiness errors and missing credentials do not consume a submission.
    ci_probe = (
        "import json,os,urllib.request; "
        "assert os.environ.get('HF_TOKEN'); "
        "r=urllib.request.Request('https://gitlab-master.nvidia.com/api/v4/projects/70847/repository/branches/dlalgo-ci%2Feos',"
        "headers={'PRIVATE-TOKEN':os.environ['JET_GITLAB_TOKEN']}); "
        "print(json.load(urllib.request.urlopen(r,timeout=60))['commit']['id'])"
    )
    refresh = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" >/dev/null 2>&1; shift; exec "$@"',
            "m4-refresh",
            str(JET / "setup_jet.sh"),
            "uv",
            "run",
            "--no-project",
            "--offline",
            "--python",
            str(JET / ".venv/bin/python"),
            "python",
            "-c",
            ci_probe,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if refresh.stdout.strip() != ready["ci_commit_sha"]:
        raise ValueError("EOS CI reference moved; authority not consumed")
    record = {
        "schema": "m4-v2-capture-one-shot-guard-v1",
        "identity": identity,
        "seed": candidate["seed"],
        "manifest_sha256": candidate["sha256"],
        "package_sha256": ready["package_sha256"],
        "attempts_consumed": 1,
        "status": "ATTEMPT_CONSUMED_BEFORE_INVOCATION",
        "invoked_at": datetime.now(timezone.utc).isoformat(),
        "parent_pipeline_id": None,
        "launcher_return_code": None,
        "ci_commit_sha": ready["ci_commit_sha"],
        "automatic_retry": False,
    }
    fd = os.open(guard, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(canonical_json(record))
        stream.flush()
        os.fsync(stream.fileno())
    log = GUARDS / f"{identity}.launch.log"
    command = [
        "bash",
        "-c",
        'source "$1" >/dev/null 2>&1; shift; exec "$@"',
        "m4-native-launch",
        str(JET / "setup_jet.sh"),
        "uv",
        "run",
        "--no-project",
        "--offline",
        "--python",
        str(JET / ".venv/bin/python"),
        "python",
        str(JET / "runllm.py"),
        "--prebuilt_manifest",
        str(ROOT / candidate["manifest"]),
        "--prebuilt_custom_config",
        str(REPORT / "eligible_live_v2_capture_eos_custom_config.yaml"),
        "--prebuilt_additional_variables",
        str(REPORT / "eligible_live_capture_variables.yaml"),
        "--prebuilt_config_id",
        "dlalgo-ci/eos",
        "--no_wait",
    ]
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    print(f"{identity}: invoking runllm.py --no_wait (one attempt)", flush=True)
    with os.fdopen(fd, "wb") as stream:
        completed = subprocess.run(
            command, cwd=JET, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    match = re.search(
        r"(?m)^(\d+): https://gitlab-master\.nvidia\.com/dl/jet/ci/-/pipelines/\1$",
        log.read_text(),
    )
    record["launcher_return_code"] = completed.returncode
    record["parent_pipeline_id"] = int(match.group(1)) if match else None
    record["status"] = (
        "SUBMITTED_NO_WAIT"
        if completed.returncode == 0 and match
        else "ATTEMPT_CONSUMED_CLIENT_FAILURE"
    )
    temporary = guard.with_suffix(".completed.json")
    with temporary.open("xb") as stream:
        stream.write(canonical_json(record))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(guard)
    print(
        f"{identity}: {record['status']}; parent={record['parent_pipeline_id']}",
        flush=True,
    )
    return 0 if record["status"] == "SUBMITTED_NO_WAIT" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    ready, directory = validate()
    package = json.loads((directory / "package.json").read_bytes())
    if args.check_only:
        print("PASS_FIXED_COHORT_LAUNCH_BOUNDARY", len(package["candidates"]))
        return 0
    for candidate in package["candidates"]:
        if submit(candidate, ready, directory) != 0:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
