#!/usr/bin/env python3
"""Derive the one-shot launchable v2 pair from validated local candidates."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

import yaml

from build_repaired_qualification_candidates_v2 import (
    CELLS,
    CONFIG_HEADER,
    EVIDENCE_HEADER,
    LOCAL_AUTH_SHA,
)


LAUNCH_AUTH_SHA = "fb64bbd5b92e3fdc536e8fc98c937b575344bbff1370f098c69bf90fed258858"
REPAIR_RECEIPT_SHA = "6603836231624e441acc50bc917ac93568d61bb83789e699614775c48b54b585"
LOCAL_CANDIDATE_SHAS = {
    "openmath": "314dd66ee6257c77bb1a1a26e705e27a147bdc71785c2d67ffee1940cbbf98b9",
    "gsm8k": "8b557a6689614f44318053d32253e4e00d59c34c734e1fe47e949612dbf359ef",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence_block(cell: str, authorization: dict[str, object]) -> str:
    value = CELLS[cell]
    block = f'''"$PYTHON" - "$CONTRACT" "$RECEIPT" "$PREFLIGHT" "$AUTH" <<'PY'
import hashlib,json,sys
from pathlib import Path
contract,receipt,preflight,auth=[json.loads(Path(path).read_bytes()) for path in sys.argv[1:]]
assert contract["schema"]=="m4-llama3p2-3b-paired-neutral-qualification-execution-contract-v1"
assert contract["status"]=="FROZEN_BEFORE_QUALIFICATION_DATA"
assert contract["prospective_protocol_sha256"]=="4f2a5e496030e836b3449697928f4f363d7d36e3cec668105b2cdcc701b9814c"
assert contract["trainer_steps_each"]==64 and contract["evaluable_start_versions"]==[8,55]
assert contract["terminal_guard_start_versions"]==[56,64]
assert contract["minimum_assignment_projection_lower_95"]==5000.0
assert contract["maximum_projected_runtime_upper_95_seconds"]==12600.0
assert contract["scheduler_cap_seconds_each"]==14400
assert contract["qualification_data_allowed_in_causal_estimator"] is False
assert contract["acquisition_authorized"] is False
assert contract["automatic_retry"] is False and contract["automatic_extension"] is False
assert receipt["megatron_lm"]["archive_sha256"]=="98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
assert preflight["schema"]=="m4-llama3p2-3b-no-training-preflight-terminal-result-v1"
assert preflight["status"]=="GREEN" and preflight["result"]["training_started"] is False
assert preflight["result"]["model"]=="meta-llama/Llama-3.2-3B-Instruct"
assert auth=={authorization!r}
assert auth["eos_submission_authorized"] is True
assert auth["training_authorized"] is True and auth["paired_qualification_authorized"] is True
assert auth["scientific_acquisition_authorized"] is False
assert auth["automatic_retry"] is False and auth["automatic_extension"] is False
registered=contract["cells"]["{cell}"]
assert registered["config"]=="{value['config']}"
assert registered["config_sha256"]=="{value['config_sha']}"
assert registered["assignment_domain"]=="{value['domain']}"
assert registered["assignment_seed"]=={value['seed']}
assert registered["assignment_bootstrap_seed"]=={value['assignment_bootstrap_seed']}
assert registered["timing_bootstrap_seed"]=={value['timing_bootstrap_seed']}
assert hashlib.sha256(Path(registered["config"]).read_bytes()).hexdigest()==registered["config_sha256"]
print("M4_LLAMA3B_{cell.upper()}_V2_AUTHORIZED_EVIDENCE_PASS")
PY
'''
    return block.replace("{", "{{").replace("}", "}}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=sorted(CELLS), required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--local-authorization", type=Path, required=True)
    parser.add_argument("--launch-authorization", type=Path, required=True)
    parser.add_argument("--repair-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.name != f"m4-llama3b-{args.cell}-neutral-qualification-v2.yaml":
        raise RuntimeError("authorized builder may emit only the frozen v2 name")
    for path, expected in (
        (args.candidate, LOCAL_CANDIDATE_SHAS[args.cell]),
        (args.local_authorization, LOCAL_AUTH_SHA),
        (args.launch_authorization, LAUNCH_AUTH_SHA),
        (args.repair_receipt, REPAIR_RECEIPT_SHA),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen authorized-package input moved: {path.name}")
    authorization = json.loads(args.launch_authorization.read_bytes())
    receipt = json.loads(args.repair_receipt.read_bytes())
    if (
        receipt["status"] != "VALIDATED_LOCAL_REPAIR_NO_LAUNCH_AUTHORITY"
        or authorization["required_launcher"] != "runllm.py --no_wait"
        or authorization["submission_attempt_limit_each_cell"] != 1
        or authorization["submit_both_before_inspecting_either"] is not True
        or authorization["eos_submission_authorized"] is not True
        or authorization["training_authorized"] is not True
        or authorization["paired_qualification_authorized"] is not True
        or authorization["scientific_acquisition_authorized"] is not False
        or authorization["automatic_retry"] is not False
        or authorization["automatic_extension"] is not False
    ):
        raise RuntimeError("authorized v2 boundary differs")
    manifest = yaml.safe_load(args.candidate.read_bytes())
    script = manifest["spec"]["script"]
    old_payload = base64.b64encode(args.local_authorization.read_bytes()).decode()
    new_payload = base64.b64encode(args.launch_authorization.read_bytes()).decode()
    if script.count(old_payload) != 1 or script.count(LOCAL_AUTH_SHA) != 1:
        raise RuntimeError("local candidate authorization surface differs")
    script = script.replace(old_payload, new_payload).replace(LOCAL_AUTH_SHA, LAUNCH_AUTH_SHA)
    start = script.index(EVIDENCE_HEADER)
    end = script.index(CONFIG_HEADER, start)
    script = script[:start] + evidence_block(args.cell, authorization) + script[end:]
    manifest["spec"]["script"] = script
    manifest["spec"]["name"] = f"m4-llama3b-{args.cell}-neutral-qualification-v2"
    args.output.write_text(yaml.safe_dump(manifest, sort_keys=False, width=10**9))
    print(args.output, sha256(args.output))


if __name__ == "__main__":
    main()
