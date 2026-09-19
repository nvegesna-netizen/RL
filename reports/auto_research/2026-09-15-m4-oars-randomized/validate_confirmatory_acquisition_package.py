#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Clean-room validation for the authorized 20-run confirmatory package."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from build_confirmatory_acquisition_candidates import (
    ACQUISITION_AMENDMENT_SHA256,
    ACQUISITION_AUTHORIZATION_SHA256,
    REPAIR_VALIDATION_SHA256,
    validate_authority,
)
from build_confirmatory_candidates import (
    ANALYSIS_PLAN_SHA256,
    AUTHORIZATION_SHA256 as LOCAL_AUTHORIZATION_SHA256,
    IMAGE_PATH,
    MEGATRON_SHA256,
    PROTOCOL_SHA256,
    QUALIFICATION_GATE_SHA256,
    RUN_MANIFEST_SHA256,
    SOURCE_COMMIT,
    SOURCE_SHA256,
)
from validate_confirmatory_package import exercise_finalizer


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exercise_authority(
    block: str,
    decoded: list[bytes],
    authorization: dict[str, object],
    identity: str,
) -> dict[str, bool]:
    """Exercise positive authority and a training-authority negative control."""
    with tempfile.TemporaryDirectory(
        prefix=f"m4-oars-confirmatory-acquisition-authority-{identity}-"
    ) as raw:
        root = Path(raw)
        paths: list[Path] = []
        for name, value in zip(
            ("protocol.json", "runs.json", "analysis.json", "gate.json", "auth.json"),
            decoded[2:],
            strict=True,
        ):
            path = root / name
            path.write_bytes(value)
            paths.append(path)
        positive = subprocess.run(
            [sys.executable, "-", *map(str, paths)],
            input=block,
            text=True,
            capture_output=True,
            check=False,
        )
        if positive.returncode != 0:
            raise RuntimeError(
                f"{identity}: positive authority failed: {positive.stderr[-400:]}"
            )

        denied = dict(authorization)
        denied["training_authorized"] = False
        denied_path = root / "denied-auth.json"
        denied_path.write_text(json.dumps(denied, sort_keys=True) + "\n")
        denied_sha256 = sha256(denied_path)
        if block.count(ACQUISITION_AUTHORIZATION_SHA256) != 1:
            raise RuntimeError(f"{identity}: authority hash assertion differs")
        denied_block = block.replace(ACQUISITION_AUTHORIZATION_SHA256, denied_sha256)
        negative = subprocess.run(
            [sys.executable, "-", *map(str, paths[:-1]), str(denied_path)],
            input=denied_block,
            text=True,
            capture_output=True,
            check=False,
        )
        if negative.returncode == 0:
            raise RuntimeError(f"{identity}: training-authority denial failed")
    return {
        "positive_authority_exercised": True,
        "training_authority_negative_control_exercised": True,
    }


def validate_candidate(
    path: Path,
    repaired_path: Path,
    run: dict[str, object],
    repaired_sha256: str,
    authorization: dict[str, object],
) -> dict[str, object]:
    """Validate one authorized manifest and dynamically exercise its gates."""
    manifest = json.loads(path.read_bytes())
    repaired = json.loads(repaired_path.read_bytes())
    spec = manifest["spec"]
    identity = str(run["identity"])
    if (
        sha256(repaired_path) != repaired_sha256
        or manifest["format_version"] != 1
        or manifest["labels"] != {"target": "silicon"}
        or manifest["launchers"] != {"type:slurm": {"nodes": 1, "ntasks_per_node": 1}}
        or spec["name"] != f"m4-oars-confirmatory-{identity}-acquisition"
        or repaired["spec"]["name"]
        != f"m4-oars-confirmatory-{identity}-local-candidate"
        or spec["workspace"] != "/workspace"
        or spec["nodes"] != 1
        or spec["time_limit"] != 14400
        or "sbatch_additional_flags" in spec
        or spec["image_source"] != {"local_path": IMAGE_PATH}
    ):
        raise RuntimeError(f"{identity}: acquisition JET boundary differs")
    for key in set(manifest).union(repaired):
        if key != "spec" and manifest.get(key) != repaired.get(key):
            raise RuntimeError(f"{identity}: top-level candidate boundary differs")
    for key in set(spec).union(repaired["spec"]):
        if key not in {"name", "script"} and spec.get(key) != repaired["spec"].get(key):
            raise RuntimeError(f"{identity}: non-authority spec field differs: {key}")

    rendered = spec["script"].format(assets_dir=f"/tmp/m4-oars-confirmatory-{identity}")
    subprocess.run(["bash", "-n"], input=rendered, text=True, check=True)
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", rendered, flags=re.DOTALL)
    if len(blocks) != 7:
        raise RuntimeError(f"{identity}: embedded Python block count differs")
    for index, block in enumerate(blocks):
        compile(block, f"<m4-oars-confirmatory-acquisition-{identity}-{index}>", "exec")
    encoded = re.findall(r"printf %s '([A-Za-z0-9+/=]+)' \| base64 -d", rendered)
    decoded = [base64.b64decode(value) for value in encoded]
    payload_hashes = [hashlib.sha256(value).hexdigest() for value in decoded]
    if payload_hashes != [
        SOURCE_SHA256,
        MEGATRON_SHA256,
        PROTOCOL_SHA256,
        RUN_MANIFEST_SHA256,
        ANALYSIS_PLAN_SHA256,
        QUALIFICATION_GATE_SHA256,
        ACQUISITION_AUTHORIZATION_SHA256,
    ]:
        raise RuntimeError(f"{identity}: embedded payload hashes differ")
    authority_test = exercise_authority(blocks[0], decoded, authorization, identity)
    finalizer_test = exercise_finalizer(rendered, identity)

    lowered = rendered.lower()
    forbidden = (
        "ci_job_token",
        "private-token",
        "authorization:",
        "hf_token",
        "nvidia_api_key",
        "queue_deadline",
        "sbatch ",
        "srun ",
    )
    if any(token in lowered for token in forbidden):
        raise RuntimeError(f"{identity}: credential or nested launch surface present")
    required = (
        'a["required_launcher"]=="runllm.py --no_wait"',
        "strict_global_sequence_one_run_at_a_time",
        "M4_OARS_CONFIRMATORY_ACQUISITION_AUTHORITY_MISSING",
        "terminal_gsm8k_accuracy",
        "retained_l1_per_selected_group",
        "valid_actor_tokens_per_update",
    )
    if not all(token in rendered for token in required):
        raise RuntimeError(f"{identity}: authority or result contract missing")
    if "M4_OARS_CONFIRMATORY_LOCAL_PACKAGE_NO_EXECUTION_AUTHORITY" in rendered:
        raise RuntimeError(f"{identity}: local denial marker remains")
    return {
        "bash_syntax": True,
        "bytes": path.stat().st_size,
        "embedded_python_blocks": len(blocks),
        "identity": identity,
        "predecessor": run["predecessor"],
        "repaired_candidate_sha256": repaired_sha256,
        "sha256": sha256(path),
        **authority_test,
        **finalizer_test,
    }


def main() -> None:
    """Deterministically rebuild and validate all 20 authorized candidates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-validation", type=Path, required=True)
    parser.add_argument("--repaired-candidates-dir", type=Path, required=True)
    parser.add_argument("--local-authorization", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--candidates-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path, expected in (
        (args.repair_validation, REPAIR_VALIDATION_SHA256),
        (args.local_authorization, LOCAL_AUTHORIZATION_SHA256),
        (args.authorization, ACQUISITION_AUTHORIZATION_SHA256),
        (args.amendment, ACQUISITION_AMENDMENT_SHA256),
        (args.run_manifest, RUN_MANIFEST_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen acquisition input moved: {path.name}")
    runs = validate_authority(args.authorization, args.amendment, args.run_manifest)
    if len(runs) != 20 or [run["global_sequence"] for run in runs] != list(
        range(1, 21)
    ):
        raise RuntimeError("registered sequence differs")
    for index, run in enumerate(runs):
        expected_predecessor = None if index == 0 else runs[index - 1]["identity"]
        if run["predecessor"] != expected_predecessor:
            raise RuntimeError(f"{run['identity']}: predecessor differs")

    repair_validation = json.loads(args.repair_validation.read_bytes())
    repaired_hashes = {
        record["identity"]: record["sha256"]
        for record in repair_validation["candidates"]
    }
    authorization = json.loads(args.authorization.read_bytes())
    records = []
    with tempfile.TemporaryDirectory(
        prefix="m4-oars-confirmatory-acquisition-cleanroom-"
    ) as raw:
        root = Path(raw)
        for run in runs:
            identity = str(run["identity"])
            repaired = args.repaired_candidates_dir / f"{identity}.json"
            candidate = args.candidates_dir / f"{identity}.json"
            rebuilt = root / f"{identity}.json"
            subprocess.run(
                [
                    sys.executable,
                    str(args.builder),
                    "--identity",
                    identity,
                    "--candidate",
                    str(repaired),
                    "--repair-validation",
                    str(args.repair_validation),
                    "--local-authorization",
                    str(args.local_authorization),
                    "--authorization",
                    str(args.authorization),
                    "--amendment",
                    str(args.amendment),
                    "--run-manifest",
                    str(args.run_manifest),
                    "--output",
                    str(rebuilt),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            if rebuilt.read_bytes() != candidate.read_bytes():
                raise RuntimeError(f"{identity}: deterministic rebuild differs")
            records.append(
                validate_candidate(
                    candidate,
                    repaired,
                    run,
                    repaired_hashes[identity],
                    authorization,
                )
            )
    if len({record["sha256"] for record in records}) != 20:
        raise RuntimeError("authorized candidate manifests are not all distinct")
    result = {
        "acquisition_amendment_sha256": ACQUISITION_AMENDMENT_SHA256,
        "acquisition_authorization_sha256": ACQUISITION_AUTHORIZATION_SHA256,
        "candidate_count": 20,
        "candidates": records,
        "complete_outcome_embargo": True,
        "first_authorized_identity": records[0]["identity"],
        "first_authorized_identity_predecessor": records[0]["predecessor"],
        "repair_validation_sha256": REPAIR_VALIDATION_SHA256,
        "runtime_source_commit": SOURCE_COMMIT,
        "schema": "m4-oars-randomized-confirmatory-acquisition-package-validation-v1",
        "sequence_policy": "strict_global_sequence_one_run_at_a_time",
        "status": "PASS_AUTHORIZED_UNSUBMITTED",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
