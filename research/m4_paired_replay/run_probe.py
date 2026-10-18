"""Run one separately restored gradient probe without an optimizer update."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from research.m4_eligible_replay.contracts import digest_file
from research.m4_eligible_replay.training_trace import observer_cleanup
from research.m4_paired_replay.capture_authentication import verify_admission_bundle
from research.m4_paired_replay.contracts import ARMS, admit
from research.m4_paired_replay.run_arm import (
    create_private_root,
    private_process_output,
)


def main() -> None:
    """Authenticate the full fresh cohort before importing the GPU backend."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--capture-evidence", type=Path, required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--role", choices=("selected", "heldout"), required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    captures = json.loads(args.captures.read_bytes())
    verify_admission_bundle(plan, captures, args.capture_evidence)
    _, manifest = admit(
        plan, captures, args.identity, (args.capsule / "manifest.json").read_bytes()
    )
    lock = json.loads(args.execution_lock.read_bytes())
    if lock.get("status") != "REVIEWED_FRESH_PAIRED_EXECUTION_READY" or lock.get(
        "plan_file_sha256"
    ) != digest_file(args.plan):
        raise ValueError("fresh paired GPU execution is not unlocked")
    create_private_root(args.private_root)
    with private_process_output(args.private_root / "process.log"):
        try:
            # Optional model/CUDA imports occur only after admission and lock checks.
            from research.m4_paired_replay.backend import NeMoBackend

            with observer_cleanup() as cleanup:
                backend = NeMoBackend(
                    capsule=args.capsule,
                    manifest=manifest,
                    arm=args.arm,
                    config_path=args.config,
                    expected_manifest_sha256=digest_file(
                        args.capsule / "manifest.json"
                    ),
                    probe_mode=True,
                )
                cleanup.callback(backend.close)
                backend.gradient_probe(
                    role=args.role, output=args.private_root / "gradient.pt"
                )
        except Exception:
            traceback.print_exc()
            raise SystemExit("GRADIENT_PROBE_FAILED_DETAILS_PRIVATE") from None
    print(
        json.dumps(
            {
                "status": "CAPTURED_NO_OPTIMIZER_UPDATE",
                "identity": args.identity,
                "arm": args.arm,
                "role": args.role,
            }
        )
    )


if __name__ == "__main__":
    main()
