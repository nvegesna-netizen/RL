#!/usr/bin/env python3
"""Clean-room validate four nonlaunchable Llama 3B acquisition candidates."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import sys
import tarfile
import tempfile
from copy import deepcopy
from pathlib import Path, PurePosixPath

import yaml
from jetclient import JETWorkloadManifest

from build_acquisition_candidates import (
    AUTH_SHA,
    CELLS,
    CONTRACT_SHA,
    MEGATRON_SHA,
    RECOVERY_SHA,
    SOURCE_SHA,
    TERMINAL_SHA,
)


BUILDER_SHA = "94c86c419ea7b165f7061d91564529fa6832b4568eb36f84b36c44cc1269a45d"
CANDIDATE_SHAS = {
    "openmath_r1": "1b1ef30489fbb9fafe368485efc19da7562829d2417f8ab97b2e8a26c3fe311f",
    "openmath_r2": "b04b11e19c9578b84f079f0c556980ee1cd2c014f4eb7b5a3c11ef2efab383c8",
    "gsm8k_r1": "e63303e3f843826efb8cfae9a3bde78cfd62376ff178d912d92b152cd3632a4d",
    "gsm8k_r2": "21c42e14487354226a75c96efeea800f5669c1354caf6f8c317d15ca70704643",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--template-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path, expected in (
        (args.source, SOURCE_SHA),
        (args.megatron, MEGATRON_SHA),
        (args.contract, CONTRACT_SHA),
        (args.terminal, TERMINAL_SHA),
        (args.recovery, RECOVERY_SHA),
        (args.authorization, AUTH_SHA),
        (args.builder, BUILDER_SHA),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen acquisition validation input moved: {path.name}")
    contract = json.loads(args.contract.read_bytes())
    terminal = json.loads(args.terminal.read_bytes())
    recovery = json.loads(args.recovery.read_bytes())
    authorization = json.loads(args.authorization.read_bytes())
    if (
        contract["status"] != "FROZEN_AFTER_QUALIFICATION_BEFORE_CAUSAL_DATA"
        or terminal["status"] != "PIPELINES_FAILED_POST_TRAINING_BOTH_QUALIFIED_OFFLINE"
        or recovery["status"] != "BOTH_QUALIFIED"
        or authorization["local_package_build_authorized"] is not True
        or any(
            authorization[key]
            for key in (
                "eos_submission_authorized",
                "model_weight_access_authorized",
                "training_authorized",
                "scientific_acquisition_authorized",
                "automatic_retry",
                "automatic_extension",
            )
        )
    ):
        raise RuntimeError("local acquisition boundary differs")
    expected_payloads = (
        SOURCE_SHA,
        MEGATRON_SHA,
        CONTRACT_SHA,
        TERMINAL_SHA,
        RECOVERY_SHA,
        AUTH_SHA,
    )
    templates = {
        "openmath_r1": "m4-llama-v5-openmath-r1-acquisition-local-candidate.yaml",
        "openmath_r2": "m4-llama-v5-openmath-r2-acquisition-local-candidate.yaml",
        "gsm8k_r1": "m4-llama-v5-gsm8k-r1-acquisition-local-candidate.yaml",
        "gsm8k_r2": "m4-llama-v5-gsm8k-r2-acquisition-local-candidate.yaml",
    }
    with tempfile.TemporaryDirectory(prefix="m4-llama3b-acquisition-cleanroom-") as raw:
        temp = Path(raw)
        extracted = temp / "source"
        extracted.mkdir()
        with tarfile.open(args.source, "r:gz") as source:
            members = source.getmembers()
            names = [PurePosixPath(member.name) for member in members]
            if len(names) != len(set(names)):
                raise RuntimeError("duplicate source archive member")
            if any(name.is_absolute() or ".." in name.parts for name in names):
                raise RuntimeError("unsafe source archive path")
            symlinks = {
                name
                for name, member in zip(names, members, strict=True)
                if member.issym()
            }
            if any(any(parent in symlinks for parent in name.parents) for name in names):
                raise RuntimeError("source archive traverses symlink parent")
            source.extractall(extracted)
        for cell, registered in CELLS.items():
            path = args.manifest_dir / f"m4-llama3b-{cell.replace('_', '-')}-acquisition-local-candidate.yaml"
            if sha256(path) != CANDIDATE_SHAS[cell]:
                raise RuntimeError(f"{cell} candidate hash differs")
            manifest = yaml.safe_load(path.read_bytes())
            JETWorkloadManifest.model_validate(deepcopy(manifest))
            spec = manifest["spec"]
            if spec["nodes"] != 1 or spec["time_limit"] != 14400:
                raise RuntimeError(f"{cell} scheduler boundary differs")
            rendered = spec["script"].format(
                assets_dir=f"/tmp/m4-llama3b-{cell}-acquisition-local"
            )
            subprocess.run(["bash", "-n"], input=rendered, text=True, check=True)
            blocks = re.findall(r"<<'PY'\n(.*?)\nPY", rendered, flags=re.DOTALL)
            if len(blocks) != 4:
                raise RuntimeError(f"{cell} embedded Python block count differs")
            for index, block in enumerate(blocks):
                compile(block, f"<{cell}-acquisition-{index}>", "exec")
            encoded = re.findall(
                r"printf %s '([A-Za-z0-9+/=]+)' \| base64 -d", rendered
            )
            hashes = tuple(
                hashlib.sha256(base64.b64decode(value)).hexdigest()
                for value in encoded
            )
            if hashes != expected_payloads:
                raise RuntimeError(f"{cell} embedded payload order or hash differs")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    blocks[2],
                    str(args.contract),
                    str(args.terminal),
                    str(args.recovery),
                    str(args.authorization),
                ],
                cwd=args.repo,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            marker = f"M4_LLAMA3B_{cell.upper()}_ACQUISITION_EVIDENCE_PASS"
            if (
                completed.returncode == 0
                or marker not in completed.stdout
                or "M4_LLAMA3B_LOCAL_ACQUISITION_CANDIDATE_NO_LAUNCH_AUTHORITY"
                not in completed.stderr
                or "NameError" in completed.stderr
            ):
                raise RuntimeError(f"{cell} did not fail closed dynamically")
            guard = rendered.index(
                "M4_LLAMA3B_LOCAL_ACQUISITION_CANDIDATE_NO_LAUNCH_AUTHORITY"
            )
            config = rendered.index("from omegaconf import OmegaConf")
            training = rendered.index("examples/run_grpo_single_controller.py")
            if not guard < config < training:
                raise RuntimeError(f"{cell} no-launch ordering differs")
            if (
                registered["config"] not in rendered
                or registered["domain"] not in rendered
                or f'r["seed"]=={registered["seed"]}' not in rendered
            ):
                raise RuntimeError(f"{cell} scientific identity differs")
            rebuilt = temp / path.name
            subprocess.run(
                [
                    sys.executable,
                    str(args.builder),
                    "--cell",
                    cell,
                    "--template",
                    str(args.template_dir / templates[cell]),
                    "--source",
                    str(args.source),
                    "--megatron",
                    str(args.megatron),
                    "--contract",
                    str(args.contract),
                    "--terminal",
                    str(args.terminal),
                    "--recovery",
                    str(args.recovery),
                    "--authorization",
                    str(args.authorization),
                    "--output",
                    str(rebuilt),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            if rebuilt.read_bytes() != path.read_bytes():
                raise RuntimeError(f"{cell} deterministic rebuild differs")
    result = {
        "schema": "m4-llama3p2-3b-four-acquisition-local-package-validation-v1",
        "status": "PASS",
        "builder_sha256": BUILDER_SHA,
        "contract_sha256": CONTRACT_SHA,
        "authorization_sha256": AUTH_SHA,
        "candidate_sha256": CANDIDATE_SHAS,
        "source_archive_member_count": len(members),
        "source_archive_symlink_count": len(symlinks),
        "jet_manifest_schema_green": True,
        "embedded_python_blocks_each": 4,
        "dynamic_no_launch_guard_passed": True,
        "deterministic_rebuild": True,
        "scientific_acquisition_started": False,
        "launch_attempted": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
