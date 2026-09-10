#!/usr/bin/env python3
"""Bind the frozen Llama 3B acquisition candidates to launch authority."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

import yaml


LOCAL_AUTH_SHA = "22bf5d2e2e3d118304aa974fa13a3a2bd3551cc5e478fb3d6e990c5f0dade899"
AUTHORIZED_AUTH_SHA = "a732d48ebae61a11508acbf95a48f10555ee0e973db95abdd128c91ba504b6a9"
CONTRACT_SHA = "f1696fdd2cf628af053c36c37fc7147974afc1d0f5d563853ed237afbe7abeb8"
LOCAL_CANDIDATE_SHAS = {
    "openmath_r1": "1b1ef30489fbb9fafe368485efc19da7562829d2417f8ab97b2e8a26c3fe311f",
    "openmath_r2": "b04b11e19c9578b84f079f0c556980ee1cd2c014f4eb7b5a3c11ef2efab383c8",
    "gsm8k_r1": "e63303e3f843826efb8cfae9a3bde78cfd62376ff178d912d92b152cd3632a4d",
    "gsm8k_r2": "21c42e14487354226a75c96efeea800f5669c1354caf6f8c317d15ca70704643",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=sorted(LOCAL_CANDIDATE_SHAS), required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--local-authorization", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if sha256(args.candidate) != LOCAL_CANDIDATE_SHAS[args.cell]:
        raise RuntimeError("frozen local candidate moved")
    if sha256(args.local_authorization) != LOCAL_AUTH_SHA:
        raise RuntimeError("frozen local authorization moved")
    if sha256(args.authorization) != AUTHORIZED_AUTH_SHA:
        raise RuntimeError("frozen acquisition authorization moved")

    local_auth = json.loads(args.local_authorization.read_bytes())
    authorization = json.loads(args.authorization.read_bytes())
    if (
        authorization["schema"]
        != "m4-llama3p2-3b-four-causal-acquisition-authorization-v1"
        or authorization["acquisition_execution_contract_sha256"] != CONTRACT_SHA
        or authorization["cells"]
        != ["openmath_r1", "openmath_r2", "gsm8k_r1", "gsm8k_r2"]
        or authorization["trainer_steps_each"] != 448
        or authorization["submission_attempt_limit_each"] != 1
        or authorization["required_launcher"] != "runllm.py --no_wait"
        or authorization["wall_clock_cap_hours_each"] != 4.0
        or not authorization["submit_all_four_before_inspecting_any_causal_outcome"]
        or not all(
            authorization[key]
            for key in (
                "eos_submission_authorized",
                "model_weight_access_authorized",
                "training_authorized",
                "scientific_acquisition_authorized",
            )
        )
        or authorization["automatic_retry"]
        or authorization["automatic_extension"]
    ):
        raise RuntimeError("acquisition authority differs from frozen boundary")

    manifest = yaml.safe_load(args.candidate.read_bytes())
    script = manifest["spec"]["script"]
    local_payload = base64.b64encode(args.local_authorization.read_bytes()).decode("ascii")
    authorized_payload = base64.b64encode(args.authorization.read_bytes()).decode("ascii")
    local_literal = repr(local_auth).replace("{", "{{").replace("}", "}}")
    authorized_literal = repr(authorization).replace("{", "{{").replace("}", "}}")
    replacements = (
        (local_payload, authorized_payload),
        (LOCAL_AUTH_SHA, AUTHORIZED_AUTH_SHA),
        (local_literal, authorized_literal),
    )
    for old, new in replacements:
        if script.count(old) != 1:
            raise RuntimeError("candidate authorization surface differs")
        script = script.replace(old, new)
    if any(old in script for old, _ in replacements):
        raise RuntimeError("local authorization remains in launchable manifest")

    manifest["spec"]["name"] = (
        f"m4-llama3b-{args.cell.replace('_', '-')}-acquisition-authorized"
    )
    manifest["spec"]["time_limit"] = 14400
    manifest["spec"]["script"] = script
    args.output.write_text(yaml.safe_dump(manifest, sort_keys=False, width=10**9))
    print(args.output, sha256(args.output))


if __name__ == "__main__":
    main()
