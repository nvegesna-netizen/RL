#!/usr/bin/env python3
"""Validate the exact paired Llama 3B neutral qualification packages."""

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

from build_paired_qualification_manifests import (
    AMENDMENT_SHA,
    AUTH_SHA,
    CELLS,
    CONTRACT_SHA,
    MEGATRON_SHA,
    PREFLIGHT_SHA,
    RECEIPT_SHA,
    SOURCE_SHA,
)


MANIFEST_SHAS = {
    "openmath": "55a0ee88d17265ad78d6b37cad3b0edabaa28773c0b517bd95ae23a3a66a94ec",
    "gsm8k": "8d7f521aea8422948f2c43ba7ca07c1c5f9187b542a46cb8802b8ddb0adc73d4",
}
CUSTOM_SHA = "e5111315a4754386375c65baf8d907515f92b4b381fb44bf2641375db757cb56"
ADDITIONAL_SHA = "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--openmath-template", type=Path, required=True)
    parser.add_argument("--gsm8k-template", type=Path, required=True)
    parser.add_argument("--openmath-manifest", type=Path, required=True)
    parser.add_argument("--gsm8k-manifest", type=Path, required=True)
    parser.add_argument("--custom-config", type=Path, required=True)
    parser.add_argument("--additional-variables", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path, expected in (
        (args.source, SOURCE_SHA),
        (args.megatron, MEGATRON_SHA),
        (args.contract, CONTRACT_SHA),
        (args.amendment, AMENDMENT_SHA),
        (args.receipt, RECEIPT_SHA),
        (args.preflight, PREFLIGHT_SHA),
        (args.authorization, AUTH_SHA),
        (args.openmath_manifest, MANIFEST_SHAS["openmath"]),
        (args.gsm8k_manifest, MANIFEST_SHAS["gsm8k"]),
        (args.custom_config, CUSTOM_SHA),
        (args.additional_variables, ADDITIONAL_SHA),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen qualification input moved: {path.name}")
    contract = json.loads(args.contract.read_bytes())
    auth = json.loads(args.authorization.read_bytes())
    if (
        contract["status"] != "FROZEN_BEFORE_QUALIFICATION_DATA"
        or contract["submit_both_before_inspecting_either"] is not True
        or auth["submit_both_before_inspecting_either"] is not True
        or auth["submission_attempt_limit_each_cell"] != 1
        or auth["required_launcher"] != "runllm.py --no_wait"
        or auth["paired_qualification_authorized"] is not True
        or auth["training_authorized"] is not True
        or auth["scientific_acquisition_authorized"] is not False
        or auth["automatic_retry"] is not False
        or auth["automatic_extension"] is not False
    ):
        raise RuntimeError("paired qualification authority differs")
    custom = json.loads(args.custom_config.read_bytes())
    launcher = custom["launchers"]["dgxh100_eos"]
    if (
        custom["environments"]["retrier"]["enabled"] is not False
        or launcher["sbatch_additional_flags"]["time"] != "04:00:00"
        or launcher["srun_additional_flags"]["time"] != "04:00:00"
        or json.loads(args.additional_variables.read_bytes()) != {}
    ):
        raise RuntimeError("scheduler, retry, or variables surface differs")
    payloads = (
        SOURCE_SHA, MEGATRON_SHA, CONTRACT_SHA, AMENDMENT_SHA,
        RECEIPT_SHA, PREFLIGHT_SHA, AUTH_SHA,
    )
    manifests = {
        "openmath": args.openmath_manifest,
        "gsm8k": args.gsm8k_manifest,
    }
    templates = {
        "openmath": args.openmath_template,
        "gsm8k": args.gsm8k_template,
    }
    for cell, path in manifests.items():
        value = CELLS[cell]
        manifest = yaml.safe_load(path.read_bytes())
        JETWorkloadManifest.model_validate(deepcopy(manifest))
        spec = manifest["spec"]
        if (
            spec["name"] != f"m4-llama3b-{cell}-neutral-qualification-v1"
            or spec["nodes"] != 1
            or spec["time_limit"] != 14400
        ):
            raise RuntimeError(f"{cell} topology differs")
        rendered = eval("f" + repr(spec["script"]), {"assets_dir": f"/tmp/{cell}"})
        subprocess.run(["bash", "-n"], input=rendered, text=True, check=True)
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", rendered, flags=re.DOTALL)
        if len(blocks) != 6:
            raise RuntimeError(f"{cell} embedded Python count differs")
        for index, block in enumerate(blocks):
            compile(block, f"<{cell}-qualification-{index}>", "exec")
        encoded = re.findall(r"printf %s '([A-Za-z0-9+/=]+)' \| base64 -d", rendered)
        if tuple(hashlib.sha256(base64.b64decode(item)).hexdigest() for item in encoded) != payloads:
            raise RuntimeError(f"{cell} embedded payloads differ")
        surface = "\n".join(
            line for line in rendered.splitlines() if not line.startswith("printf %s '")
        )
        required = (
            value["config"],
            value["domain"],
            "meta-llama/Llama-3.2-3B-Instruct",
            f"assignment_bootstrap_seed={value['assignment_bootstrap_seed']}",
            f"timing_bootstrap_seed={value['timing_bootstrap_seed']}",
            "assess_llama3b_qualification(",
            "intervals=[(advance_time[version+1]-advance_time[version])/1e9 for version in range(8,56)]",
            '"scientific_acquisition_authorized":false',
            f"M4_LLAMA3B_{cell.upper()}_NEUTRAL_QUALIFICATION_GREEN",
        )
        if not all(token in surface for token in required):
            raise RuntimeError(f"{cell} qualification contract incomplete")
        if surface.count("examples/run_grpo_single_controller.py") != 1:
            raise RuntimeError(f"{cell} training invocation count differs")
        for forbidden in (
            "sbatch ", "srun ", '"scientific_acquisition_authorized":true',
            "trainer_version': 448", "automatic_retry\":true", "automatic_extension\":true",
        ):
            if forbidden in surface:
                raise RuntimeError(f"{cell} unauthorized surface present: {forbidden}")
    with tempfile.TemporaryDirectory(prefix="m4-llama3b-qualification-") as raw:
        root = Path(raw) / "source"
        root.mkdir()
        with tarfile.open(args.source, "r:gz") as archive:
            members = archive.getmembers()
            names = [PurePosixPath(member.name) for member in members]
            if len(names) != len(set(names)) or any(
                name.is_absolute() or ".." in name.parts for name in names
            ):
                raise RuntimeError("unsafe source archive")
            archive.extractall(root)
        for cell, value in CELLS.items():
            config = root / value["config"]
            if sha256(config) != value["config_sha"]:
                raise RuntimeError(f"{cell} extracted config differs")
            rebuilt = Path(raw) / f"{cell}.yaml"
            subprocess.run(
                [
                    sys.executable, str(args.builder), "--cell", cell,
                    "--template", str(templates[cell]), "--source", str(args.source),
                    "--megatron", str(args.megatron), "--contract", str(args.contract),
                    "--amendment", str(args.amendment), "--receipt", str(args.receipt),
                    "--preflight", str(args.preflight), "--authorization", str(args.authorization),
                    "--output", str(rebuilt),
                ],
                check=True, capture_output=True, text=True,
            )
            if rebuilt.read_bytes() != manifests[cell].read_bytes():
                raise RuntimeError(f"{cell} deterministic rebuild differs")
    result = {
        "schema": "m4-llama3p2-3b-paired-neutral-qualification-package-validation-v1",
        "status": "PASS",
        "contract_sha256": CONTRACT_SHA,
        "authorization_sha256": AUTH_SHA,
        "manifest_sha256": MANIFEST_SHAS,
        "source_members": len(members),
        "jet_manifest_schema_green": True,
        "deterministic_rebuild": True,
        "submit_both_before_inspecting_either": True,
        "submission_attempt_limit_each_cell": 1,
        "scheduler_cap_seconds_each": 14400,
        "jet_retrier_enabled": False,
        "acquisition_authorized": False,
        "launch_attempted": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
