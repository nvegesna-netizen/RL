"""Coordinate one admitted arm without exposing metrics to the JET asset stream."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

from research.m4_eligible_replay.contracts import canonical_json, digest_file
from research.m4_paired_replay.capture_authentication import verify_admission_bundle
from research.m4_paired_replay.contracts import ARMS, capture_gate
from research.m4_paired_replay.quarantine import export_files
from research.m4_paired_replay.run_arm import create_private_root
from research.m4_paired_replay.transport import fetch_arm

CONFIG = "examples/configs/grpo_math_1B_megatron_single_controller_m4_eligible_live_capture.yaml"
PRIVATE_BASE = Path("/workspace/m4-paired-private")


def sanitized_failure(text: str) -> dict:
    """Keep only frame locations and exception class names, never messages/metrics."""
    frames = re.findall(r'^\s*File "([^"\n]+)", line (\d+), in ([\w<>]+)', text, re.M)
    classes = re.findall(
        r"^(?:[\w.]+\.)?([A-Za-z]\w*(?:Error|Exception))(?::|\()", text, re.M
    )
    return {
        "exception_classes": classes[-8:],
        "frames": [
            {"path": p, "line": int(n), "function": f} for p, n, f in frames[-24:]
        ],
    }


def write(path: Path, value: dict) -> None:
    with path.open("xb") as stream:
        stream.write(canonical_json(value))


def run(inputs: Path, public: Path) -> int:
    candidate = json.loads((inputs / "candidate.json").read_bytes())
    lock = json.loads((inputs / "lock.json").read_bytes())
    plan = json.loads((inputs / "plan.json").read_bytes())
    captures = json.loads((inputs / "captures.json").read_bytes())
    verify_admission_bundle(plan, captures, inputs / "evidence")
    gate = capture_gate(plan, captures)
    identity, arm = candidate["capture_identity"], candidate["arm"]
    sources = [row for row in gate["admitted"] if row["identity"] == identity]
    if (
        arm not in ARMS
        or candidate not in lock["fixed_arms"]
        or lock["status"] != "REVIEWED_FRESH_PAIRED_EXECUTION_READY"
        or digest_file(inputs / "plan.json") != lock["plan_file_sha256"]
        or digest_file(inputs / "captures.json") != lock["captures_file_sha256"]
        or len(lock["fixed_arms"]) != 32
        or gate["valid_capsules"] != 16
        or os.environ.get("M4_PAIRED_SYNTHETIC_ENGINEERING") == "1"
        or len(sources) != 1
        or candidate["manifest_sha256"] != sources[0]["manifest_sha256"]
        or candidate["source_eos_job_id"] != sources[0]["eos_job_id"]
        or candidate["seed"] != sources[0]["seed"]
    ):
        raise ValueError("fixed scientific arm boundary differs")
    root = PRIVATE_BASE / candidate["identity"]
    create_private_root(root)
    stage = "transport"
    completed = []
    try:
        fetch_arm(
            plan=plan,
            captures=captures,
            identity=identity,
            arm=arm,
            artifact_prefix=f"workspace/assets/basic/m4-eligible-live-acquisition-{identity}/m4-eligible-live-capsule",
            output=root / "input",
            token=os.environ["CI_JOB_TOKEN"],
            job_token=True,
        )
        write(
            public / "transport-receipt.json",
            json.loads((root / "input/transport-receipt.json").read_bytes()),
        )
        common = [
            "--plan",
            str(inputs / "plan.json"),
            "--captures",
            str(inputs / "captures.json"),
            "--capture-evidence",
            str(inputs / "evidence"),
            "--identity",
            identity,
            "--arm",
            arm,
            "--capsule",
            str(root / "input"),
            "--config",
            CONFIG,
            "--execution-lock",
            str(inputs / "lock.json"),
        ]
        for stage in ("selected", "heldout", "primary"):
            if stage != "primary":
                command = [
                    sys.executable,
                    "-m",
                    "research.m4_paired_replay.run_probe",
                    *common,
                    "--role",
                    stage,
                    "--private-root",
                    str(root / stage),
                ]
            else:
                command = [
                    sys.executable,
                    "-m",
                    "research.m4_paired_replay.run_arm",
                    *common,
                    "--private-root",
                    str(root / "primary"),
                    "--receipt",
                    str(public / "arm-receipt.json"),
                    "--gradient-selected",
                    str(root / "selected/gradient.pt"),
                    "--gradient-heldout",
                    str(root / "heldout/gradient.pt"),
                    "--public-key",
                    "engineering-public.pem",
                    "--encrypted-envelope",
                    str(public / "encrypted-envelope.json"),
                    "--encrypted-receipt",
                    str(public / "encrypted-receipt.json"),
                ]
            fd = os.open(
                root / f"{stage}.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            with os.fdopen(fd, "wb") as log:
                process = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, check=False
                )
            if process.returncode:
                raise RuntimeError("private arm stage failed")
            completed.append(stage)
        receipt = json.loads((public / "arm-receipt.json").read_bytes())
        encrypted = json.loads((public / "encrypted-receipt.json").read_bytes())
        if (
            receipt["status"] != "COMPLETE_OUTCOME_SEALED"
            or receipt["identity"] != identity
            or receipt["arm"] != arm
            or receipt["manifest_sha256"] != candidate["manifest_sha256"]
            or receipt["optimizer_updates"] != 1
            or receipt["outcomes_opened"] is not False
            or encrypted["status"] != "COMPLETE_ENCRYPTED_PRIVATE_EXPORT"
            or encrypted["plaintext_exported"] is not False
            or encrypted["envelope_sha256"]
            != digest_file(public / "encrypted-envelope.json")
        ):
            raise ValueError("arm completion or encrypted export differs")
        write(
            public / "completion.json",
            {
                "schema": "m4-fixed-paired-job-completion-v1",
                "status": "COMPLETE_ENCRYPTED_PRIVATE_EXPORT",
                "identity": identity,
                "arm": arm,
                "completed_stages": completed,
                "manifest_sha256": candidate["manifest_sha256"],
                "primary_outcomes_opened": False,
            },
        )
        return 0
    except Exception:
        fd = os.open(
            root / "coordinator-failure.log",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(fd, "w") as stream:
            traceback.print_exc(file=stream)
        logs = [
            root / "coordinator-failure.log",
            root / f"{stage}.log",
            root / stage / "process.log",
        ]
        existing = [p for p in logs if p.is_file()]
        diagnostic = sanitized_failure(
            "\n".join(p.read_text(errors="replace") for p in existing)
        )
        write(
            public / "failure.json",
            {
                "schema": "m4-fixed-paired-failure-v1",
                "identity": identity,
                "arm": arm,
                "stage": stage,
                "completed_stages": completed,
                "diagnostic": diagnostic,
                "primary_outcomes_opened": False,
            },
        )
        export_files(
            private_root=root,
            names=[str(p.relative_to(root)) for p in existing],
            public_pem=Path("engineering-public.pem").read_bytes(),
            identity={"identity": identity, "arm": arm, "kind": "private-failure-logs"},
            envelope_path=public / "failure-envelope.json",
            receipt_path=public / "failure-envelope-receipt.json",
        )
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    args = parser.parse_args()
    return run(args.inputs, args.public)


if __name__ == "__main__":
    sys.exit(main())
