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
from research.m4_paired_replay.capture_authentication import verify_admission_bundle
from research.m4_paired_replay.quarantine import export_files, validate_public_key


def create_private_root(path: Path) -> None:
    """Keep plaintext outcomes outside JET asset/log mounts under an owner-only root."""
    root = Path("/workspace/m4-paired-private")
    if (
        not path.is_absolute()
        or path == root
        or ".." in path.parts
        or not path.is_relative_to(root)
        or any(p.is_symlink() for p in (path, *path.parents))
    ):
        raise ValueError(
            "plaintext replay root must be private and outside collector assets"
        )
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.stat().st_mode & 0o077:
        raise ValueError("plaintext replay root is not owner-only")
    path.mkdir(mode=0o700, parents=True, exist_ok=False)


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
    parser.add_argument("--capture-evidence", type=Path)
    parser.add_argument("--gradient-selected", type=Path)
    parser.add_argument("--gradient-heldout", type=Path)
    parser.add_argument("--public-key", type=Path)
    parser.add_argument("--encrypted-envelope", type=Path)
    parser.add_argument("--encrypted-receipt", type=Path)
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
    if any(
        value is None
        for value in (
            args.capture_evidence,
            args.gradient_selected,
            args.gradient_heldout,
            args.public_key,
            args.encrypted_envelope,
            args.encrypted_receipt,
        )
    ):
        raise ValueError(
            "authenticated capture proofs, separate gradient probes, and encrypted export required"
        )
    verify_admission_bundle(plan, captures, args.capture_evidence)
    public_pem = args.public_key.read_bytes()
    validate_public_key(public_pem)
    create_private_root(args.private_root)
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
                gradient_probe_files=(args.gradient_selected, args.gradient_heldout),
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
                measure_secondaries=True,
            )
        except Exception:
            # Preserve runtime diagnostics privately; never echo a metric-bearing
            # backend exception/traceback into the public JET wrapper stream.
            traceback.print_exc()
            raise SystemExit("ARM_FAILED_DETAILS_PRIVATE") from None
    export_files(
        private_root=args.private_root,
        names=["sealed/outcome.json", "process.log"],
        public_pem=public_pem,
        identity={
            "identity": args.identity,
            "arm": args.arm,
            "manifest_sha256": digest_file(args.capsule / "manifest.json"),
            "arm_receipt_sha256": digest_file(args.receipt),
        },
        envelope_path=args.encrypted_envelope,
        receipt_path=args.encrypted_receipt,
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE_ENCRYPTED_PRIVATE_EXPORT",
                "identity": args.identity,
                "arm": args.arm,
            }
        )
    )


if __name__ == "__main__":
    main()
