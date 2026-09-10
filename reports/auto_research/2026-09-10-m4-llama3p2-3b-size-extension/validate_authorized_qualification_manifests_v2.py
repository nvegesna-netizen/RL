#!/usr/bin/env python3
"""Validate the exact one-shot launchable Llama 3B v2 qualification pair."""

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

from build_authorized_qualification_manifests_v2 import (
    LAUNCH_AUTH_SHA,
    LOCAL_CANDIDATE_SHAS,
)
from build_repaired_qualification_candidates_v2 import (
    CONFIG_HEADER,
    EVIDENCE_HEADER,
    LOCAL_AUTH_SHA,
)


BUILDER_SHA = "f9ed46a51905ec3134c9a31b722b9d33b164bfd743c2ff46fa32074312230994"
REPAIR_RECEIPT_SHA = "6603836231624e441acc50bc917ac93568d61bb83789e699614775c48b54b585"
CONTRACT_SHA = "4e124c51a2179bd3faff5da8d34a58dc0ce33f98aa58faa2e00f1d6965944e62"
DEPENDENCY_RECEIPT_SHA = "6a9fc16e404d15f3f133c1fef042cfbd5cb53eb01563a4b8c794f97bd21e90c0"
PREFLIGHT_SHA = "eaa4408d27becc95d12fad7e51c89731c6106046acd0b5fe596259061d32899a"
MANIFEST_SHAS = {
    "openmath": "23903e358f85e70a373490a95edc3bb18c558deb86fe3055b9345a76c3480a62",
    "gsm8k": "583becb35807ab2231779a576fe460e4f6d8e7255c7f897cdfc6b14af02a40e1",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(script: str, start: str, end: str) -> str:
    left = script.index(start)
    right = script.index(end, left)
    return script[left:right]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--local-authorization", type=Path, required=True)
    parser.add_argument("--launch-authorization", type=Path, required=True)
    parser.add_argument("--repair-receipt", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--dependency-receipt", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--openmath-candidate", type=Path, required=True)
    parser.add_argument("--gsm8k-candidate", type=Path, required=True)
    parser.add_argument("--openmath-manifest", type=Path, required=True)
    parser.add_argument("--gsm8k-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixed = (
        (args.builder, BUILDER_SHA),
        (args.local_authorization, LOCAL_AUTH_SHA),
        (args.launch_authorization, LAUNCH_AUTH_SHA),
        (args.repair_receipt, REPAIR_RECEIPT_SHA),
        (args.contract, CONTRACT_SHA),
        (args.dependency_receipt, DEPENDENCY_RECEIPT_SHA),
        (args.preflight, PREFLIGHT_SHA),
        (args.openmath_candidate, LOCAL_CANDIDATE_SHAS["openmath"]),
        (args.gsm8k_candidate, LOCAL_CANDIDATE_SHAS["gsm8k"]),
        (args.openmath_manifest, MANIFEST_SHAS["openmath"]),
        (args.gsm8k_manifest, MANIFEST_SHAS["gsm8k"]),
    )
    for path, expected in fixed:
        if sha256(path) != expected:
            raise RuntimeError(f"frozen launch validation input moved: {path.name}")
    authorization = json.loads(args.launch_authorization.read_bytes())
    if not all(
        authorization[key]
        for key in (
            "eos_submission_authorized",
            "model_weight_access_authorized",
            "training_authorized",
            "paired_qualification_authorized",
            "submit_both_before_inspecting_either",
        )
    ):
        raise RuntimeError("launch authority is incomplete")
    if any(
        authorization[key]
        for key in (
            "scientific_acquisition_authorized",
            "automatic_retry",
            "automatic_extension",
        )
    ):
        raise RuntimeError("launch authority is not fail-closed")
    candidates = {
        "openmath": args.openmath_candidate,
        "gsm8k": args.gsm8k_candidate,
    }
    manifests = {
        "openmath": args.openmath_manifest,
        "gsm8k": args.gsm8k_manifest,
    }
    old_payload = base64.b64encode(args.local_authorization.read_bytes()).decode()
    new_payload = base64.b64encode(args.launch_authorization.read_bytes()).decode()
    evidence_results = {}
    for cell in ("openmath", "gsm8k"):
        candidate = yaml.safe_load(candidates[cell].read_bytes())
        manifest = yaml.safe_load(manifests[cell].read_bytes())
        JETWorkloadManifest.model_validate(deepcopy(manifest))
        if manifest["spec"]["name"] != f"m4-llama3b-{cell}-neutral-qualification-v2":
            raise RuntimeError(f"{cell} launch name differs")
        if manifest["spec"]["nodes"] != 1 or manifest["spec"]["time_limit"] != 14400:
            raise RuntimeError(f"{cell} scheduler boundary differs")
        old_script = candidate["spec"]["script"]
        new_script = manifest["spec"]["script"]
        old_evidence = extract(old_script, EVIDENCE_HEADER, CONFIG_HEADER)
        new_evidence = extract(new_script, EVIDENCE_HEADER, CONFIG_HEADER)
        normalized_script = (
            new_script.replace(new_payload, old_payload)
            .replace(LAUNCH_AUTH_SHA, LOCAL_AUTH_SHA)
            .replace(new_evidence, old_evidence)
        )
        normalized = deepcopy(manifest)
        normalized["spec"]["name"] = candidate["spec"]["name"]
        normalized["spec"]["script"] = normalized_script
        if normalized != candidate:
            raise RuntimeError(f"{cell} authorized delta exceeds registered surface")
        rendered = eval(
            "f" + repr(new_script),
            {"assets_dir": f"/tmp/m4-llama3b-{cell}-v2-authorized"},
        )
        subprocess.run(["bash", "-n"], input=rendered, text=True, check=True)
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", rendered, flags=re.DOTALL)
        if len(blocks) != 6:
            raise RuntimeError(f"{cell} embedded Python block count differs")
        for index, source in enumerate(blocks):
            compile(source, f"<{cell}-authorized-v2-{index}>", "exec")
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                blocks[3],
                str(args.contract),
                str(args.dependency_receipt),
                str(args.preflight),
                str(args.launch_authorization),
            ],
            cwd=args.repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        marker = f"M4_LLAMA3B_{cell.upper()}_V2_AUTHORIZED_EVIDENCE_PASS"
        if completed.returncode != 0 or marker not in completed.stdout or completed.stderr:
            raise RuntimeError(f"{cell} authorized evidence execution failed")
        if "M4_LLAMA3B_V2_LOCAL_CANDIDATE_NO_LAUNCH_AUTHORITY" in rendered:
            raise RuntimeError(f"{cell} local no-launch guard remains")
        if not rendered.index(EVIDENCE_HEADER) < rendered.index(CONFIG_HEADER) < rendered.index("examples/run_grpo_single_controller.py"):
            raise RuntimeError(f"{cell} evidence/config/training order differs")
        evidence_results[cell] = {
            "authorization_evidence_executed": True,
            "authorization_evidence_passed": True,
            "local_no_launch_guard_absent": True,
        }
    with tempfile.TemporaryDirectory(prefix="m4-llama3b-v2-authorized-rebuild-") as raw:
        for cell in ("openmath", "gsm8k"):
            rebuilt = Path(raw) / f"m4-llama3b-{cell}-neutral-qualification-v2.yaml"
            subprocess.run(
                [
                    sys.executable,
                    str(args.builder),
                    "--cell",
                    cell,
                    "--candidate",
                    str(candidates[cell]),
                    "--local-authorization",
                    str(args.local_authorization),
                    "--launch-authorization",
                    str(args.launch_authorization),
                    "--repair-receipt",
                    str(args.repair_receipt),
                    "--output",
                    str(rebuilt),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            if rebuilt.read_bytes() != manifests[cell].read_bytes():
                raise RuntimeError(f"{cell} authorized rebuild differs")
    result = {
        "schema": "m4-llama3p2-3b-paired-neutral-qualification-package-validation-v2",
        "status": "PASS",
        "authorization_sha256": LAUNCH_AUTH_SHA,
        "builder_sha256": BUILDER_SHA,
        "manifest_sha256": MANIFEST_SHAS,
        "exact_registered_delta_from_validated_local_candidates": True,
        "jet_manifest_schema_green": True,
        "embedded_python_blocks_each": 6,
        "dynamic_authorization_evidence": evidence_results,
        "deterministic_rebuild": True,
        "scheduler_cap_seconds_each": 14400,
        "submission_attempt_limit_each_cell": 1,
        "scientific_acquisition_authorized": False,
        "automatic_retry": False,
        "automatic_extension": False,
        "submitted": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
