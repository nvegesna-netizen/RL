#!/usr/bin/env python3
"""Clean-room validate all 32 authorization-bound acquisition manifests."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BUILDER = ROOT / "build_authorized_trained_paired_manifests.py"
BUILDER_SHA = "2492dddeadaa504075390dec78b05292de17902a612da69f954b4f1b1f3de024"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_builder():
    spec = importlib.util.spec_from_file_location("authorized_package_builder", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--local-authorization", type=Path, required=True)
    parser.add_argument("--local-validation", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--authorized-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    builder = load_builder()
    for path, expected in (
        (BUILDER, BUILDER_SHA),
        (args.local_authorization, builder.LOCAL_AUTH_SHA),
        (args.local_validation, builder.LOCAL_VALIDATION_SHA),
        (args.authorization, builder.AUTHORIZED_AUTH_SHA),
        (args.run_manifest, builder.RUN_MANIFEST_SHA),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"authorized validation input moved: {path.name}")
    authorization = json.loads(args.authorization.read_bytes())
    local_validation = json.loads(args.local_validation.read_bytes())
    runs = json.loads(args.run_manifest.read_bytes())["runs"]
    identities = [f"{run['block_id']}_{run['regime']}" for run in runs]
    if (
        authorization["identities"] != identities
        or len(identities) != 32
        or authorization["submission_attempt_limit_each"] != 1
        or authorization["required_launcher"] != "runllm.py --no_wait"
        or not authorization["submit_all_32_before_inspecting_any_outcome"]
        or not all(
            authorization[key]
            for key in (
                "model_weight_access_authorized",
                "optimizer_initialization_authorized",
                "training_authorized",
                "eos_submission_authorized",
                "scientific_acquisition_authorized",
            )
        )
        or authorization["automatic_retry"]
        or authorization["automatic_replacement"]
        or authorization["automatic_extension"]
    ):
        raise RuntimeError("authorized acquisition boundary differs")

    manifest_hashes: dict[str, str] = {}
    manifest_bytes: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="m4-downstream-authorized-") as raw:
        temp = Path(raw)
        for run in runs:
            identity = f"{run['block_id']}_{run['regime']}"
            stem = identity.replace("_", "-")
            local = args.candidate_dir / f"m4-downstream-quality-{stem}-local-candidate.yaml"
            authorized = args.authorized_dir / f"m4-downstream-quality-{stem}-acquisition-authorized.yaml"
            if sha256(local) != local_validation["candidate_sha256"][identity]:
                raise RuntimeError(f"local candidate moved: {identity}")
            manifest = json.loads(authorized.read_bytes())
            if (
                manifest["spec"]["name"]
                != f"m4-downstream-quality-{stem}-acquisition-authorized"
                or manifest["spec"]["nodes"] != 1
                or manifest["spec"]["time_limit"] != 14400
            ):
                raise RuntimeError(f"authorized scheduler boundary differs: {identity}")
            rendered = manifest["spec"]["script"].format(assets_dir=f"/tmp/{identity}")
            subprocess.run(["bash", "-n"], input=rendered, text=True, check=True)
            blocks = re.findall(r"<<'PY'\n(.*?)\nPY", rendered, flags=re.DOTALL)
            if len(blocks) != 7:
                raise RuntimeError(f"authorized embedded block count differs: {identity}")
            for index, block in enumerate(blocks):
                compile(block, f"<{identity}-authorized-{index}>", "exec")
            encoded = re.findall(
                r"printf %s '([A-Za-z0-9+/=]+)' \| base64 -d", rendered
            )
            payload_hashes = [
                hashlib.sha256(base64.b64decode(value)).hexdigest()
                for value in encoded
            ]
            if (
                len(payload_hashes) != 7
                or payload_hashes[5] != builder.AUTHORIZED_AUTH_SHA
                or payload_hashes[6] != run["config_sha256"]
                or builder.LOCAL_AUTH_SHA in rendered
            ):
                raise RuntimeError(f"authorized payload identity differs: {identity}")
            opened = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    blocks[0],
                    str(args.repo / "reports/auto_research/2026-09-11-m4-downstream-quality/trained_paired_acquisition_protocol.json"),
                    str(args.run_manifest),
                    str(args.authorization),
                ],
                cwd=args.repo,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            marker = f"M4_DOWNSTREAM_QUALITY_{identity.upper()}_ACQUISITION_AUTHORITY_PASS"
            if opened.returncode != 0 or marker not in opened.stdout or opened.stderr:
                raise RuntimeError(f"authorized gate did not open exactly: {identity}")
            if (
                "M4_DOWNSTREAM_QUALITY_LOCAL_PACKAGE_NO_EXECUTION_AUTHORITY" in rendered
                or rendered.index(marker) > rendered.index('mkdir -p "$RUN_REPO"')
                or re.search(r"(?mi)^\s*(?:\S*/)?(?:runllm\.py|sbatch|srun)\b", rendered)
            ):
                raise RuntimeError(f"authorized execution ordering differs: {identity}")
            rebuilt = temp / authorized.name
            subprocess.run(
                [
                    sys.executable,
                    str(BUILDER),
                    "--identity",
                    identity,
                    "--candidate",
                    str(local),
                    "--local-authorization",
                    str(args.local_authorization),
                    "--local-validation",
                    str(args.local_validation),
                    "--authorization",
                    str(args.authorization),
                    "--run-manifest",
                    str(args.run_manifest),
                    "--output",
                    str(rebuilt),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if rebuilt.read_bytes() != authorized.read_bytes():
                raise RuntimeError(f"authorized deterministic rebuild differs: {identity}")
            manifest_hashes[identity] = sha256(authorized)
            manifest_bytes[identity] = authorized.stat().st_size

    result = {
        "schema": "m4-downstream-quality-trained-paired-authorized-package-validation-v1",
        "status": "PASS_32_AUTHORIZED_PACKAGES_NOT_SUBMITTED",
        "authorization_sha256": builder.AUTHORIZED_AUTH_SHA,
        "local_validation_sha256": builder.LOCAL_VALIDATION_SHA,
        "builder_sha256": BUILDER_SHA,
        "manifest_count": 32,
        "manifest_sha256": manifest_hashes,
        "manifest_bytes": manifest_bytes,
        "bash_syntax_passed": True,
        "embedded_python_syntax_passed": True,
        "authorization_gate_opened_exactly": True,
        "local_authority_absent": True,
        "nested_launch_surface_absent": True,
        "deterministic_rebuild": True,
        "launch_attempted": False,
        "scientific_acquisition_started": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
