#!/usr/bin/env python3
"""Clean-room validate four authorized Llama 3B acquisition manifests."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import yaml
from jetclient import JETWorkloadManifest

from build_authorized_acquisition_manifests import (
    AUTHORIZED_AUTH_SHA,
    CONTRACT_SHA,
    LOCAL_CANDIDATE_SHAS,
)
from build_acquisition_candidates import (
    MEGATRON_SHA,
    RECOVERY_SHA,
    SOURCE_SHA,
    TERMINAL_SHA,
)


BUILDER_SHA = "5485babb4e8fe88b84514860cb5dd3f71e8980e33dd1e37fcf33b7780c15cb70"
AUTHORIZED_SHAS = {
    "openmath_r1": "44b79931bb9de522c67fb05e84b4566c80d447695ca43e2db3e947150304c39e",
    "openmath_r2": "7bcd92f23089f24bef85967a9539fb367e790cbbe8ae32978f50cb7bf3d55a7d",
    "gsm8k_r1": "7670167b574683bf4a1a6016a6ebe0743fa2b662da0ceaf7ccfccc52c07bb25b",
    "gsm8k_r2": "d1d8966e247d1fe527980fee6f2eeeb5232d4ca22e86f972d6ba5d56f90d942d",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--local-authorization", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected_files = (
        (args.contract, CONTRACT_SHA),
        (args.terminal, TERMINAL_SHA),
        (args.recovery, RECOVERY_SHA),
        (args.authorization, AUTHORIZED_AUTH_SHA),
        (args.builder, BUILDER_SHA),
    )
    for path, expected in expected_files:
        if sha256(path) != expected:
            raise RuntimeError(f"authorized validation input moved: {path.name}")
    authorization = json.loads(args.authorization.read_bytes())
    if (
        not all(
            authorization[key]
            for key in (
                "eos_submission_authorized",
                "model_weight_access_authorized",
                "training_authorized",
                "scientific_acquisition_authorized",
                "submit_all_four_before_inspecting_any_causal_outcome",
            )
        )
        or authorization["submission_attempt_limit_each"] != 1
        or authorization["trainer_steps_each"] != 448
        or authorization["required_launcher"] != "runllm.py --no_wait"
        or authorization["automatic_retry"]
        or authorization["automatic_extension"]
    ):
        raise RuntimeError("authorized four-cell boundary differs")

    expected_payloads = (
        SOURCE_SHA,
        MEGATRON_SHA,
        CONTRACT_SHA,
        TERMINAL_SHA,
        RECOVERY_SHA,
        AUTHORIZED_AUTH_SHA,
    )
    with tempfile.TemporaryDirectory(prefix="m4-llama3b-authorized-cleanroom-") as raw:
        temp = Path(raw)
        for cell, expected in AUTHORIZED_SHAS.items():
            stem = cell.replace("_", "-")
            path = args.manifest_dir / f"m4-llama3b-{stem}-acquisition-authorized.yaml"
            if sha256(path) != expected:
                raise RuntimeError(f"{cell} authorized manifest hash differs")
            manifest = yaml.safe_load(path.read_bytes())
            JETWorkloadManifest.model_validate(deepcopy(manifest))
            spec = manifest["spec"]
            if (
                spec["name"] != f"m4-llama3b-{stem}-acquisition-authorized"
                or spec["nodes"] != 1
                or spec["time_limit"] != 14400
            ):
                raise RuntimeError(f"{cell} manifest boundary differs")
            rendered = spec["script"].format(assets_dir=f"/tmp/{stem}")
            subprocess.run(["bash", "-n"], input=rendered, text=True, check=True)
            blocks = re.findall(r"<<'PY'\n(.*?)\nPY", rendered, flags=re.DOTALL)
            if len(blocks) != 4:
                raise RuntimeError(f"{cell} embedded Python block count differs")
            for index, block in enumerate(blocks):
                compile(block, f"<{cell}-authorized-{index}>", "exec")
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
                completed.returncode != 0
                or marker not in completed.stdout
                or "NO_LAUNCH_AUTHORITY" in completed.stderr
                or "NameError" in completed.stderr
            ):
                raise RuntimeError(f"{cell} authorization did not open exactly")
            if (
                'if not auth["eos_submission_authorized"]' not in rendered
                or rendered.index('if not auth["eos_submission_authorized"]')
                > rendered.index("from omegaconf import OmegaConf")
                or any(
                    re.match(r"^\s*(?:\S*/)?(?:runllm\.py|sbatch|srun)\b", line)
                    for line in rendered.splitlines()
                )
            ):
                raise RuntimeError(f"{cell} nested launch surface differs")
            local = args.manifest_dir / f"m4-llama3b-{stem}-acquisition-local-candidate.yaml"
            if sha256(local) != LOCAL_CANDIDATE_SHAS[cell]:
                raise RuntimeError(f"{cell} local source candidate moved")
            rebuilt = temp / path.name
            subprocess.run(
                [
                    sys.executable,
                    str(args.builder),
                    "--cell",
                    cell,
                    "--candidate",
                    str(local),
                    "--local-authorization",
                    str(args.local_authorization),
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
                raise RuntimeError(f"{cell} deterministic authorized rebuild differs")

    result = {
        "schema": "m4-llama3p2-3b-four-authorized-acquisition-package-validation-v1",
        "status": "PASS",
        "authorization_sha256": AUTHORIZED_AUTH_SHA,
        "builder_sha256": BUILDER_SHA,
        "manifest_sha256": AUTHORIZED_SHAS,
        "manifest_count": 4,
        "jet_manifest_schema_green": True,
        "embedded_python_blocks_each": 4,
        "authorization_gate_opened_exactly": True,
        "nested_launch_surface_absent": True,
        "deterministic_rebuild": True,
        "launch_attempted": False,
        "scientific_acquisition_started": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
