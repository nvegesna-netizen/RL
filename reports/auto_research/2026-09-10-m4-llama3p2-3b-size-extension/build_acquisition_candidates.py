#!/usr/bin/env python3
"""Build four fail-closed Llama 3B acquisition candidates."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

import yaml


SOURCE_COMMIT = "88424bdb7bfb526e541b4a7b07830bc31028e64f"
SOURCE_SHA = "1ae875a63786d687fcf46bd7fa7cb2f26b1031c3cf3c36e38114305c732820eb"
MEGATRON_SHA = "98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
CONTRACT_SHA = "f1696fdd2cf628af053c36c37fc7147974afc1d0f5d563853ed237afbe7abeb8"
TERMINAL_SHA = "339c1317e12ccae11f8ca5d9f33dd214c8bfe0ca4ee97eaa6b22f7e09288fe74"
RECOVERY_SHA = "5f944114be9b9265ba47108dbb8cd148509323b8d3fbdd2d565d893f16720a87"
AUTH_SHA = "22bf5d2e2e3d118304aa974fa13a3a2bd3551cc5e478fb3d6e990c5f0dade899"
CELLS = {
    "openmath_r1": {
        "template_sha": "6d60507f0ab7741f3722da6251453a7e5c3dfddc469f2b56973e13843895f6ee",
        "config": "examples/configs/grpo_math_3B_megatron_single_controller_m4_llama3p2_3b_openmath_size_extension_v1_r1.yaml",
        "config_sha": "0b7bfb8fa3c9101ad0aea3320a519e9997d2337d000e5bc6728873b8b546e1f5",
        "domain": "m4-llama3p2-3b-openmath-size-extension-v1-r1",
        "seed": 20261103,
    },
    "openmath_r2": {
        "template_sha": "d209e614eb67be10592efb627dfe974eb0957ae1137e3aa4ea5517bd4c8cbf2c",
        "config": "examples/configs/grpo_math_3B_megatron_single_controller_m4_llama3p2_3b_openmath_size_extension_v1_r2.yaml",
        "config_sha": "8a97966d28d9d2d1e5b2a204f7bf81c55a93fadd0248986294dec23f73ef6140",
        "domain": "m4-llama3p2-3b-openmath-size-extension-v1-r2",
        "seed": 20261104,
    },
    "gsm8k_r1": {
        "template_sha": "2ba05dba4b12a9d044363047b526a43a0bf0ca2070b021c3a71be1dcca82494f",
        "config": "examples/configs/grpo_math_3B_megatron_single_controller_m4_llama3p2_3b_gsm8k_size_extension_v1_r1.yaml",
        "config_sha": "50bee58c0d96ea99236790ab74ac9e612ec6ab316754d914cfd0ce50be434932",
        "domain": "m4-llama3p2-3b-gsm8k-size-extension-v1-r1",
        "seed": 20261105,
    },
    "gsm8k_r2": {
        "template_sha": "74678bc7afb2412c249b36e35f4562a7e2ce2203a17735149f0650de3fbd285f",
        "config": "examples/configs/grpo_math_3B_megatron_single_controller_m4_llama3p2_3b_gsm8k_size_extension_v1_r2.yaml",
        "config_sha": "27ec964ca51a41c18813b1804a53864f763efc91b51a410f846deaedda9bb06a",
        "domain": "m4-llama3p2-3b-gsm8k-size-extension-v1-r2",
        "seed": 20261106,
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=sorted(CELLS), required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = CELLS[args.cell]
    for path, expected in (
        (args.template, value["template_sha"]),
        (args.source, SOURCE_SHA),
        (args.megatron, MEGATRON_SHA),
        (args.contract, CONTRACT_SHA),
        (args.terminal, TERMINAL_SHA),
        (args.recovery, RECOVERY_SHA),
        (args.authorization, AUTH_SHA),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen acquisition input moved: {path.name}")
    authorization = json.loads(args.authorization.read_bytes())
    if (
        authorization["local_package_build_authorized"] is not True
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
        raise RuntimeError("local acquisition package authority differs")
    tag = args.cell.upper()
    stem = f"m4-llama3b-{args.cell.replace('_', '-')}-acquisition"
    script = f'''set -euo pipefail
readonly PYTHON=/opt/nemo_rl_venv/bin/python
readonly RUN_REPO=/workspace/{stem}-repo
readonly SOURCE=/workspace/m4-llama3b-acquisition-source.tar.gz
readonly MEGATRON=/workspace/m4-llama3b-acquisition-megatron.tar.gz
readonly CONTRACT=/workspace/m4-llama3b-acquisition-contract.json
readonly TERMINAL=/workspace/m4-llama3b-qualification-terminal.json
readonly RECOVERY=/workspace/m4-llama3b-qualification-recovery.json
readonly AUTH=/workspace/m4-llama3b-acquisition-authorization.json
readonly ASSETS={{assets_dir}}
readonly RUN_LOG=$ASSETS/{stem}-run.log
readonly LIFECYCLE=$ASSETS/{stem}-lifecycle.jsonl
readonly OPPORTUNITY=$ASSETS/{stem}-opportunity.jsonl
readonly DUTY=$ASSETS/{stem}-lifecycle-duty.json
readonly DERIVATION=$ASSETS/{stem}-derivation.json
readonly HASHES=$ASSETS/{stem}-artifacts.sha256
readonly METRICS=$ASSETS/{stem}-metrics
test "${{NEMO_RL_COMMIT:-unknown}}" = "ae07eafe8035b5b2e84efa7234e70e7fd7e493c1"
test ! -e "$RUN_REPO"
mkdir -p "$RUN_REPO" "$METRICS"
printf %s '{payload(args.source)}' | base64 -d > "$SOURCE"
printf %s '{payload(args.megatron)}' | base64 -d > "$MEGATRON"
printf %s '{payload(args.contract)}' | base64 -d > "$CONTRACT"
printf %s '{payload(args.terminal)}' | base64 -d > "$TERMINAL"
printf %s '{payload(args.recovery)}' | base64 -d > "$RECOVERY"
printf %s '{payload(args.authorization)}' | base64 -d > "$AUTH"
test "$(sha256sum "$SOURCE" | cut -d' ' -f1)" = "{SOURCE_SHA}"
test "$(sha256sum "$MEGATRON" | cut -d' ' -f1)" = "{MEGATRON_SHA}"
test "$(sha256sum "$CONTRACT" | cut -d' ' -f1)" = "{CONTRACT_SHA}"
test "$(sha256sum "$TERMINAL" | cut -d' ' -f1)" = "{TERMINAL_SHA}"
test "$(sha256sum "$RECOVERY" | cut -d' ' -f1)" = "{RECOVERY_SHA}"
test "$(sha256sum "$AUTH" | cut -d' ' -f1)" = "{AUTH_SHA}"
"$PYTHON" - "$SOURCE" "$RUN_REPO" <<'PY'
import sys,tarfile
from pathlib import Path,PurePosixPath
archive,output=map(Path,sys.argv[1:])
with tarfile.open(archive,"r:gz") as source:
 members=source.getmembers(); names=[PurePosixPath(m.name) for m in members]
 assert len(names)==len(set(names))
 assert not any(n.is_absolute() or ".." in n.parts for n in names)
 symlinks={{n for n,m in zip(names,members,strict=True) if m.issym()}}
 assert not any(any(p in symlinks for p in n.parents) for n in names)
 source.extractall(output)
print("M4_LLAMA3B_ACQUISITION_SAFE_SOURCE_PASS",len(names),len(symlinks))
PY
"$PYTHON" - "$MEGATRON" "$RUN_REPO" <<'PY'
import sys,tarfile
from pathlib import Path,PurePosixPath
archive,root=map(Path,sys.argv[1:]); dest=root/"3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty"
with tarfile.open(archive,"r:gz") as source:
 members=source.getmembers(); assert len(members)==726
 assert all(not PurePosixPath(m.name).is_absolute() and ".." not in PurePosixPath(m.name).parts for m in members)
 dest.mkdir(parents=True,exist_ok=True); source.extractall(dest)
assert (dest/"Megatron-LM/megatron/core/__init__.py").is_file()
print("M4_LLAMA3B_ACQUISITION_MEGATRON_PASS")
PY
cd "$RUN_REPO"
export PYTHONPATH="$RUN_REPO"
export CUDA_VISIBLE_DEVICES=0,1
"$PYTHON" - "$CONTRACT" "$TERMINAL" "$RECOVERY" "$AUTH" <<'PY'
import json,sys
from pathlib import Path
contract,terminal,recovery,auth=[json.loads(Path(path).read_bytes()) for path in sys.argv[1:]]
assert contract["schema"]=="m4-llama3p2-3b-four-acquisition-execution-contract-v1"
assert contract["status"]=="FROZEN_AFTER_QUALIFICATION_BEFORE_CAUSAL_DATA"
assert terminal["status"]=="PIPELINES_FAILED_POST_TRAINING_BOTH_QUALIFIED_OFFLINE"
assert recovery["status"]=="BOTH_QUALIFIED"
assert all(recovery["cells"][cell]["qualification_complete"] for cell in ("openmath","gsm8k"))
assert contract["qualification_data_allowed_in_causal_estimator"] is False
assert contract["trainer_steps_each"]==448 and contract["primary_start_versions"]==[8,407]
assert contract["terminal_guard_start_versions"]==[408,447]
assert contract["submission_attempt_limit_each"]==1
assert contract["submit_all_four_before_inspecting_any_causal_outcome"] is True
assert contract["automatic_retry"] is False and contract["automatic_extension"] is False
assert auth=={authorization!r}
registered=contract["cells"]["{args.cell}"]
assert registered["config"]=="{value['config']}"
assert registered["config_sha256"]=="{value['config_sha']}"
assert registered["assignment_domain"]=="{value['domain']}"
assert registered["assignment_seed"]=={value['seed']}
print("M4_LLAMA3B_{tag}_ACQUISITION_EVIDENCE_PASS")
if not auth["eos_submission_authorized"]:
 raise SystemExit("M4_LLAMA3B_LOCAL_ACQUISITION_CANDIDATE_NO_LAUNCH_AUTHORITY")
PY
"$PYTHON" - <<'PY'
import hashlib
from pathlib import Path
from omegaconf import OmegaConf
from nemo_rl.algorithms.single_controller_utils.config import MasterConfig,validate_single_controller_config
from nemo_rl.utils.config import load_config,register_omegaconf_resolvers
config="{value['config']}"; assert hashlib.sha256(Path(config).read_bytes()).hexdigest()=="{value['config_sha']}"
register_omegaconf_resolvers(); c=OmegaConf.to_container(load_config(config),resolve=True)
assert c["grpo"]["max_num_steps"]==448
assert c["policy"]["model_name"]=="meta-llama/Llama-3.2-3B-Instruct"
assert c["policy"]["tokenizer"]["name"]=="meta-llama/Llama-3.2-3B-Instruct"
assert c["cluster"]["num_nodes"]==1 and c["cluster"]["gpus_per_node"]==2
r=c["async_rl"]["controlled_release_delay"]
assert r["enabled"] is True and r["seed"]=={value['seed']} and r["assignment_domain"]=="{value['domain']}"
assert r["arms"]==[{{"label":"control","delay_seconds":0.0,"mass":1}},{{"label":"d5","delay_seconds":5.0,"mass":1}}]
assert c["async_rl"]["gradient_opportunity_audit"]["enabled"] is False
assert c["async_rl"]["lifecycle_derived_opportunity_audit"]["enabled"] is True
validate_single_controller_config(MasterConfig(**c))
print("M4_LLAMA3B_{tag}_ACQUISITION_CONFIG_PASS")
PY
"$PYTHON" examples/run_grpo_single_controller.py --config {value['config']} \\
 "logger.log_dir=$METRICS" \\
 "async_rl.lifecycle_audit_path=$LIFECYCLE" \\
 "async_rl.lifecycle_derived_opportunity_audit.output_path=$OPPORTUNITY" \\
 "async_rl.lifecycle_derived_opportunity_audit.lifecycle_duty_path=$DUTY" \\
 "async_rl.lifecycle_derived_opportunity_audit.derivation_summary_path=$DERIVATION" 2>&1 | tee "$RUN_LOG"
grep -Fq "SC run complete: {{'train_steps': 448, 'trainer_version': 448}}" "$RUN_LOG"
(cd "$ASSETS"; sha256sum "$(basename "$RUN_LOG")" "$(basename "$LIFECYCLE")" "$(basename "$OPPORTUNITY")" "$(basename "$DUTY")" "$(basename "$DERIVATION")" > "$HASHES")
echo M4_LLAMA3B_{tag}_ACQUISITION_CAPTURE_COMPLETE_NO_CAUSAL_INSPECTION
'''
    script = script.replace("{", "{{").replace("}", "}}").replace(
        "{{assets_dir}}", "{assets_dir}"
    )
    manifest = yaml.safe_load(args.template.read_bytes())
    manifest["spec"]["name"] = f"m4-llama3b-{args.cell.replace('_', '-')}-acquisition-local-candidate"
    manifest["spec"]["time_limit"] = 14400
    manifest["spec"]["script"] = script
    args.output.write_text(yaml.safe_dump(manifest, sort_keys=False, width=10**9))
    print(args.output, sha256(args.output))


if __name__ == "__main__":
    main()
