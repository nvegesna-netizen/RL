"""Execute one future admitted arm, with all process output routed privately."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import traceback
from collections.abc import Iterator

from research.m4_eligible_replay.contracts import digest_file
from research.m4_paired_replay.contracts import ARMS, admit, members
from research.m4_paired_replay.execution import execute_arm


@contextmanager
def private_process_output(path: Path) -> Iterator[None]:
    """Redirect Python, native libraries and driver-forwarded Ray stdout/stderr.

    This does not encrypt files or hide node-local Ray logs from an administrator.
    The future collector must keep those logs out of outcome-free compact exports.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.dup2(fd, 1)
            os.dup2(fd, 2)
        finally:
            os.close(fd)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        for destination, source in zip((1, 2), saved):
            os.dup2(source, destination)
            os.close(source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    captures = json.loads(args.captures.read_bytes())
    manifest_bytes = (args.capsule / "manifest.json").read_bytes()
    _, manifest = admit(plan, captures, args.identity, manifest_bytes)
    for row in members(manifest, args.arm):
        if digest_file(args.capsule / row["path"]) != row["sha256"]:
            raise ValueError("arm capsule authentication failed")
    lock = json.loads(args.execution_lock.read_bytes())
    if args.validate_only:
        print(json.dumps({"status": "PASS_OFFLINE_ARM_INPUTS", "gpu_started": False}))
        return
    # No execution-ready lock is produced by this offline preparation step.
    if lock.get("status") != "REVIEWED_FRESH_PAIRED_EXECUTION_READY" or lock.get(
        "plan_file_sha256"
    ) != digest_file(args.plan):
        raise ValueError("fresh paired GPU execution is not unlocked")
    if args.private_root.exists():
        raise FileExistsError("private arm root already exists")
    args.private_root.mkdir(mode=0o700, parents=True)
    with private_process_output(args.private_root / "process.log"):
        # CUDA/backend imports occur only after fresh admission and explicit lock.
        try:
            from research.m4_paired_replay.backend import NeMoBackend

            backend = NeMoBackend(
                capsule=args.capsule,
                manifest=manifest,
                arm=args.arm,
                config_path=args.config,
                expected_manifest_sha256=digest_file(args.capsule / "manifest.json"),
            )
            execute_arm(
                backend=backend,
                plan=plan,
                captures=captures,
                identity=args.identity,
                arm=args.arm,
                manifest_bytes=manifest_bytes,
                private_dir=args.private_root / "sealed",
                receipt_path=args.receipt,
            )
        except Exception:
            # Preserve runtime diagnostics privately; never echo a metric-bearing
            # backend exception/traceback into the public JET wrapper stream.
            traceback.print_exc()
            raise SystemExit("ARM_FAILED_DETAILS_PRIVATE") from None
    print(
        json.dumps(
            {
                "status": "COMPLETE_OUTCOME_SEALED",
                "identity": args.identity,
                "arm": args.arm,
            }
        )
    )


if __name__ == "__main__":
    main()
