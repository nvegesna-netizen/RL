#!/usr/bin/env python3
"""Derive nonlaunchable v2 qualification candidates from the failed v1 pair."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

import yaml


REPAIR_SHA = "f70b62e24b4a3f24c7ae1b62ce56c380704b8b75436623650a5428329d772883"
V1_AUTH_SHA = "90997c5ceead304efe0f535512b1c4cce49d88e457fb1fe9ded766f32cf603ff"
LOCAL_AUTH_SHA = "8ea7472092fe8b340373b54f34dce36d57a7ac2ec4cc7a451587d66a5c4bf269"
CONTRACT_SHA = "4e124c51a2179bd3faff5da8d34a58dc0ce33f98aa58faa2e00f1d6965944e62"
PREFLIGHT_SHA = "eaa4408d27becc95d12fad7e51c89731c6106046acd0b5fe596259061d32899a"
V1_MANIFEST_SHAS = {
    "openmath": "55a0ee88d17265ad78d6b37cad3b0edabaa28773c0b517bd95ae23a3a66a94ec",
    "gsm8k": "8d7f521aea8422948f2c43ba7ca07c1c5f9187b542a46cb8802b8ddb0adc73d4",
}
CELLS = {
    "openmath": {
        "config": "examples/configs/grpo_math_3B_megatron_single_controller_m4_llama3p2_3b_openmath_neutral_qualification_v1.yaml",
        "config_sha": "15788df473dca65e96b811fc09790784f9fdb248380d152855b3b6204bc61830",
        "domain": "m4-llama3p2-3b-openmath-neutral-qualification-v1",
        "seed": 20261101,
        "assignment_bootstrap_seed": 20261107,
        "timing_bootstrap_seed": 20261109,
    },
    "gsm8k": {
        "config": "examples/configs/grpo_math_3B_megatron_single_controller_m4_llama3p2_3b_gsm8k_neutral_qualification_v1.yaml",
        "config_sha": "337351dfb177f6ad61b96de4fa4f859a3bbaaaacb966e4ff4d7dad63fb469fd5",
        "domain": "m4-llama3p2-3b-gsm8k-neutral-qualification-v1",
        "seed": 20261102,
        "assignment_bootstrap_seed": 20261108,
        "timing_bootstrap_seed": 20261110,
    },
}
EVIDENCE_HEADER = '"$PYTHON" - "$CONTRACT" "$RECEIPT" "$PREFLIGHT" "$AUTH" <<\'PY\'\n'
CONFIG_HEADER = '"$PYTHON" - <<\'PY\'\nfrom omegaconf import OmegaConf\n'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(value: str, old: str, new: str, label: str) -> str:
    if value.count(old) != 1:
        raise RuntimeError(f"expected exactly one {label}, found {value.count(old)}")
    return value.replace(old, new)


def evidence_block(cell: str, local_auth: dict[str, object]) -> str:
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
assert auth=={local_auth!r}
registered=contract["cells"]["{cell}"]
assert registered["config"]=="{value['config']}"
assert registered["config_sha256"]=="{value['config_sha']}"
assert registered["assignment_domain"]=="{value['domain']}"
assert registered["assignment_seed"]=={value['seed']}
assert registered["assignment_bootstrap_seed"]=={value['assignment_bootstrap_seed']}
assert registered["timing_bootstrap_seed"]=={value['timing_bootstrap_seed']}
assert hashlib.sha256(Path(registered["config"]).read_bytes()).hexdigest()==registered["config_sha256"]
print("M4_LLAMA3B_{cell.upper()}_V2_REPAIRED_EVIDENCE_PASS")
if not auth["eos_submission_authorized"]:
 raise SystemExit("M4_LLAMA3B_V2_LOCAL_CANDIDATE_NO_LAUNCH_AUTHORITY")
PY
'''
    return block.replace("{", "{{").replace("}", "}}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=sorted(CELLS), required=True)
    parser.add_argument("--v1-manifest", type=Path, required=True)
    parser.add_argument("--v1-authorization", type=Path, required=True)
    parser.add_argument("--local-authorization", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.name.endswith("-v2-local-candidate.yaml"):
        raise RuntimeError("repair builder may emit only a v2 local candidate")
    for path, expected in (
        (args.v1_manifest, V1_MANIFEST_SHAS[args.cell]),
        (args.v1_authorization, V1_AUTH_SHA),
        (args.local_authorization, LOCAL_AUTH_SHA),
        (args.repair, REPAIR_SHA),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen repair input moved: {path.name}")
    repair = json.loads(args.repair.read_bytes())
    local_auth = json.loads(args.local_authorization.read_bytes())
    if (
        repair["status"] != "FROZEN_LOCAL_REPAIR_NO_LAUNCH_AUTHORITY"
        or repair["scientific_configuration_changes_allowed"] is not False
        or local_auth["local_package_build_authorized"] is not True
        or any(
            local_auth[key]
            for key in (
                "eos_submission_authorized", "model_weight_access_authorized",
                "training_authorized", "paired_qualification_authorized",
                "scientific_acquisition_authorized", "automatic_retry", "automatic_extension",
            )
        )
    ):
        raise RuntimeError("local repair authority differs")
    manifest = yaml.safe_load(args.v1_manifest.read_bytes())
    script = manifest["spec"]["script"]
    old_payload = base64.b64encode(args.v1_authorization.read_bytes()).decode()
    new_payload = base64.b64encode(args.local_authorization.read_bytes()).decode()
    script = replace_once(script, old_payload, new_payload, "authorization payload")
    script = replace_once(script, V1_AUTH_SHA, LOCAL_AUTH_SHA, "authorization hash")
    start = script.index(EVIDENCE_HEADER)
    end = script.index(CONFIG_HEADER, start)
    script = script[:start] + evidence_block(args.cell, local_auth) + script[end:]
    manifest["spec"]["script"] = script
    manifest["spec"]["name"] = f"m4-llama3b-{args.cell}-neutral-qualification-v2-local-candidate"
    args.output.write_text(yaml.safe_dump(manifest, sort_keys=False, width=10**9))
    print(args.output, sha256(args.output))


if __name__ == "__main__":
    main()
