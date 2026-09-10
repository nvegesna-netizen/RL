#!/usr/bin/env python3
"""Dynamically validate both nonlaunchable repaired qualification candidates."""

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

from build_repaired_qualification_candidates_v2 import (
    CELLS,
    CONFIG_HEADER,
    EVIDENCE_HEADER,
    LOCAL_AUTH_SHA,
    REPAIR_SHA,
    V1_AUTH_SHA,
    V1_MANIFEST_SHAS,
)


CANDIDATE_SHAS = {
    "openmath": "314dd66ee6257c77bb1a1a26e705e27a147bdc71785c2d67ffee1940cbbf98b9",
    "gsm8k": "8b557a6689614f44318053d32253e4e00d59c34c734e1fe47e949612dbf359ef",
}
CONTRACT_SHA = "4e124c51a2179bd3faff5da8d34a58dc0ce33f98aa58faa2e00f1d6965944e62"
RECEIPT_SHA = "6a9fc16e404d15f3f133c1fef042cfbd5cb53eb01563a4b8c794f97bd21e90c0"
PREFLIGHT_SHA = "eaa4408d27becc95d12fad7e51c89731c6106046acd0b5fe596259061d32899a"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def block(script: str, start_marker: str, end_marker: str) -> str:
    start = script.index(start_marker)
    end = script.index(end_marker, start)
    return script[start:end]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--v1-authorization", type=Path, required=True)
    parser.add_argument("--local-authorization", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--openmath-v1", type=Path, required=True)
    parser.add_argument("--gsm8k-v1", type=Path, required=True)
    parser.add_argument("--openmath-candidate", type=Path, required=True)
    parser.add_argument("--gsm8k-candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path, expected in (
        (args.repair, REPAIR_SHA),
        (args.v1_authorization, V1_AUTH_SHA),
        (args.local_authorization, LOCAL_AUTH_SHA),
        (args.contract, CONTRACT_SHA),
        (args.receipt, RECEIPT_SHA),
        (args.preflight, PREFLIGHT_SHA),
        (args.openmath_v1, V1_MANIFEST_SHAS["openmath"]),
        (args.gsm8k_v1, V1_MANIFEST_SHAS["gsm8k"]),
        (args.openmath_candidate, CANDIDATE_SHAS["openmath"]),
        (args.gsm8k_candidate, CANDIDATE_SHAS["gsm8k"]),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen repair validation input moved: {path.name}")
    repair = json.loads(args.repair.read_bytes())
    local_auth = json.loads(args.local_authorization.read_bytes())
    if (
        repair["scientific_configuration_changes_allowed"] is not False
        or repair["fresh_authorization_required_for_launchable_v2_pair"] is not True
        or local_auth["eos_submission_authorized"] is not False
        or local_auth["training_authorized"] is not False
        or local_auth["paired_qualification_authorized"] is not False
    ):
        raise RuntimeError("repair boundary differs")
    v1_paths = {"openmath": args.openmath_v1, "gsm8k": args.gsm8k_v1}
    candidate_paths = {
        "openmath": args.openmath_candidate,
        "gsm8k": args.gsm8k_candidate,
    }
    old_payload = base64.b64encode(args.v1_authorization.read_bytes()).decode()
    new_payload = base64.b64encode(args.local_authorization.read_bytes()).decode()
    dynamic_results = {}
    for cell in ("openmath", "gsm8k"):
        original = yaml.safe_load(v1_paths[cell].read_bytes())
        candidate = yaml.safe_load(candidate_paths[cell].read_bytes())
        JETWorkloadManifest.model_validate(deepcopy(candidate))
        if candidate["spec"]["name"] != f"m4-llama3b-{cell}-neutral-qualification-v2-local-candidate":
            raise RuntimeError(f"{cell} candidate name differs")
        if candidate["spec"]["nodes"] != 1 or candidate["spec"]["time_limit"] != 14400:
            raise RuntimeError(f"{cell} candidate topology differs")
        old_script = original["spec"]["script"]
        new_script = candidate["spec"]["script"]
        old_evidence = block(old_script, EVIDENCE_HEADER, CONFIG_HEADER)
        new_evidence = block(new_script, EVIDENCE_HEADER, CONFIG_HEADER)
        normalized_script = new_script.replace(new_payload, old_payload).replace(
            LOCAL_AUTH_SHA, V1_AUTH_SHA
        ).replace(new_evidence, old_evidence)
        normalized = deepcopy(candidate)
        normalized["spec"]["name"] = original["spec"]["name"]
        normalized["spec"]["script"] = normalized_script
        if normalized != original:
            raise RuntimeError(f"{cell} repair delta exceeds registered surface")
        rendered = eval(
            "f" + repr(new_script),
            {"assets_dir": f"/tmp/m4-llama3b-{cell}-v2-local"},
        )
        subprocess.run(["bash", "-n"], input=rendered, text=True, check=True)
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", rendered, flags=re.DOTALL)
        if len(blocks) != 6:
            raise RuntimeError(f"{cell} embedded Python block count differs")
        for index, source in enumerate(blocks):
            compile(source, f"<{cell}-v2-{index}>", "exec")
        if "NameError: name 'false'" in rendered or "assert auth=={\"" in rendered:
            raise RuntimeError(f"{cell} JSON-as-Python defect remains")
        evidence_source = blocks[3]
        completed = subprocess.run(
            [
                sys.executable, "-c", evidence_source,
                str(args.contract), str(args.receipt), str(args.preflight),
                str(args.local_authorization),
            ],
            cwd=args.repo, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False,
        )
        expected_pass = f"M4_LLAMA3B_{cell.upper()}_V2_REPAIRED_EVIDENCE_PASS"
        if (
            completed.returncode == 0
            or expected_pass not in completed.stdout
            or "M4_LLAMA3B_V2_LOCAL_CANDIDATE_NO_LAUNCH_AUTHORITY" not in completed.stderr
            or "NameError" in completed.stderr
        ):
            raise RuntimeError(f"{cell} dynamic evidence execution did not fail closed")
        config_index = rendered.index(CONFIG_HEADER)
        guard_index = rendered.index("M4_LLAMA3B_V2_LOCAL_CANDIDATE_NO_LAUNCH_AUTHORITY")
        run_index = rendered.index("examples/run_grpo_single_controller.py")
        if not guard_index < config_index < run_index:
            raise RuntimeError(f"{cell} no-launch guard ordering differs")
        if CELLS[cell]["config"] not in rendered or CELLS[cell]["domain"] not in rendered:
            raise RuntimeError(f"{cell} frozen scientific identity differs")
        dynamic_results[cell] = {
            "authorization_evidence_executed": True,
            "evidence_pass_marker_observed": True,
            "explicit_no_launch_exit_observed": True,
            "name_error_observed": False,
        }
    with tempfile.TemporaryDirectory(prefix="m4-llama3b-v2-rebuild-") as raw:
        for cell in ("openmath", "gsm8k"):
            rebuilt = Path(raw) / f"m4-llama3b-{cell}-neutral-qualification-v2-local-candidate.yaml"
            subprocess.run(
                [
                    sys.executable, str(args.builder), "--cell", cell,
                    "--v1-manifest", str(v1_paths[cell]),
                    "--v1-authorization", str(args.v1_authorization),
                    "--local-authorization", str(args.local_authorization),
                    "--repair", str(args.repair), "--output", str(rebuilt),
                ],
                check=True, capture_output=True, text=True,
            )
            if rebuilt.read_bytes() != candidate_paths[cell].read_bytes():
                raise RuntimeError(f"{cell} v2 rebuild differs")
    result = {
        "schema": "m4-llama3p2-3b-paired-qualification-local-repair-validation-v2",
        "status": "PASS",
        "repair_sha256": REPAIR_SHA,
        "local_authorization_sha256": LOCAL_AUTH_SHA,
        "failed_v1_manifest_sha256": V1_MANIFEST_SHAS,
        "v2_local_candidate_sha256": CANDIDATE_SHAS,
        "exact_registered_delta": True,
        "embedded_python_blocks_each": 6,
        "dynamic_authorization_evidence": dynamic_results,
        "deterministic_rebuild": True,
        "jet_manifest_schema_green": True,
        "scientific_configuration_changed": False,
        "launch_authority_present": False,
        "launch_attempted": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
