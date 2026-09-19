#!/usr/bin/env python3
"""Build one of 20 credential-free, locally nonlaunchable OARS candidates."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

SOURCE_COMMIT = "90dbb632026591f24c966a43edd05b1e4933358b"
SOURCE_SHA256 = "1d9e17a430fc1a184d767413de829ad9ee1375091485b5e7e0b24572f61f7690"
MEGATRON_SHA256 = "98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
PROTOCOL_SHA256 = "3bc49412a212419635e55b616e080708e17cf265c1926c1d93faebf072a87bfd"
RUN_MANIFEST_SHA256 = (
    "1f59d8cf3339d70b58b084af48bd2f192a880d7a43fca8190fb1d54649db5b39"
)
ANALYSIS_PLAN_SHA256 = (
    "66431cf76f3314b1bc1a0d6ece2735f9842e4790ed3eb673322c58b84d268d58"
)
QUALIFICATION_GATE_SHA256 = (
    "d359c5536147b74c4255a3edf090d98460cff3c8fae4507ac9dcdef15ec43ca9"
)
AUTHORIZATION_SHA256 = (
    "644a66c15b83125b70add145cfed4a48daf7f80298a8d565165319cd27d5b1f2"
)
IMAGE_PATH = (
    "/lustre/fsw/coreai_dlalgo_ci/nvegesna/nemo_rl_images/"
    "nemo-rl-nightly-5802754.sqsh"
)
IMAGE_COMMIT = "ae07eafe8035b5b2e84efa7234e70e7fd7e493c1"
TRANSFERQUEUE_COMMIT = "b266d39a15aae114730de36cf8317b6285436f7f"
FIFO_CONFIG = (
    "examples/configs/"
    "grpo_math_1B_megatron_single_controller_m4_oars_qualification_fifo.yaml"
)
OARS_CONFIG = (
    "examples/configs/"
    "grpo_math_1B_megatron_single_controller_m4_oars_qualification_act.yaml"
)
CONFIG_SHA256 = {
    "fifo": "17ad965489aa9595d266295125880fe55edb4af303bd95aed66a61034c943c9b",
    "oars": "9cd92f17e96193f584f206747132c7ced70dd2e1a9eda44f53d4af752b1967fc",
}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload(path: Path) -> str:
    """Return a base64-encoded payload."""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    """Build one frozen local candidate selected by registered identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--analysis-plan", type=Path, required=True)
    parser.add_argument("--qualification-gate", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path, expected in (
        (args.source, SOURCE_SHA256),
        (args.megatron, MEGATRON_SHA256),
        (args.protocol, PROTOCOL_SHA256),
        (args.run_manifest, RUN_MANIFEST_SHA256),
        (args.analysis_plan, ANALYSIS_PLAN_SHA256),
        (args.qualification_gate, QUALIFICATION_GATE_SHA256),
        (args.authorization, AUTHORIZATION_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen package input moved: {path.name}")

    authorization = json.loads(args.authorization.read_bytes())
    forbidden = (
        "model_weight_access_authorized",
        "optimizer_initialization_authorized",
        "training_authorized",
        "eos_submission_authorized",
        "scientific_acquisition_authorized",
        "automatic_retry",
        "automatic_replacement",
        "automatic_extension",
    )
    if (
        authorization["schema"]
        != "m4-oars-randomized-confirmatory-local-package-authorization-v1"
        or authorization["candidate_count"] != 20
        or authorization["runtime_source_commit"] != SOURCE_COMMIT
        or not authorization["runtime_source_archive_build_authorized"]
        or not authorization["credential_free_eos_package_build_authorized"]
        or not authorization["clean_room_validation_authorized"]
        or any(authorization[key] for key in forbidden)
    ):
        raise RuntimeError("local package authority differs")
    run_manifest = json.loads(args.run_manifest.read_bytes())
    runs = {run["identity"]: run for run in run_manifest["runs"]}
    if len(runs) != 20 or args.identity not in runs:
        raise RuntimeError("registered run identity differs")
    run = runs[args.identity]
    arm = run["arm"]
    config_path = FIFO_CONFIG if arm == "fifo" else OARS_CONFIG
    config_sha256 = CONFIG_SHA256[arm]
    stem = f"m4-oars-confirmatory-{run['identity']}"
    tag = run["identity"].replace("-", "_").upper()

    script = f'''set -euo pipefail
readonly PYTHON=/opt/nemo_rl_venv/bin/python
readonly RUN_REPO=/workspace/{stem}-repo
readonly SOURCE=/workspace/{stem}-source.tar.gz
readonly MEGATRON=/workspace/{stem}-megatron.tar.gz
readonly PROTOCOL=/workspace/{stem}-protocol.json
readonly RUN_MANIFEST=/workspace/{stem}-run-manifest.json
readonly ANALYSIS_PLAN=/workspace/{stem}-analysis-plan.json
readonly QUALIFICATION=/workspace/{stem}-qualification-gate.json
readonly AUTH=/workspace/{stem}-authorization.json
readonly ASSETS={{assets_dir}}
readonly RUN_LOG=$ASSETS/training.log
readonly LIFECYCLE=$ASSETS/lifecycle.jsonl
readonly OPPORTUNITY=$ASSETS/opportunity.jsonl
readonly DUTY=$ASSETS/observer-duty.json
readonly OARS=$ASSETS/oars.jsonl
readonly SYSTEMS=$ASSETS/systems-result.json
readonly RESULT=$ASSETS/arm-result.json
readonly PROMPTS=$ASSETS/gsm8k-prompt-manifest.json
readonly EVAL_OUTPUT=$ASSETS/evaluation
readonly EXPORT=$ASSETS/terminal-policy
readonly HF_MODEL=/workspace/{stem}-terminal-hf
readonly HASHES=$ASSETS/artifacts.sha256
readonly METRICS=$ASSETS/metrics
test "${{NEMO_RL_COMMIT:-unknown}}" = "{IMAGE_COMMIT}"
test ! -e "$RUN_REPO"
mkdir -p "$ASSETS" "$METRICS" "$EVAL_OUTPUT"
finalize_artifacts() {{
  set +e
  (cd "$ASSETS"; find . -type f ! -path './jet_assets/output_logs/*' ! -name artifacts.sha256 ! -name .artifacts.sha256.tmp -print0 | sort -z | xargs -0 sha256sum > .artifacts.sha256.tmp && mv .artifacts.sha256.tmp artifacts.sha256)
}}
trap finalize_artifacts EXIT
printf %s '{payload(args.source)}' | base64 -d > "$SOURCE"
printf %s '{payload(args.megatron)}' | base64 -d > "$MEGATRON"
printf %s '{payload(args.protocol)}' | base64 -d > "$PROTOCOL"
printf %s '{payload(args.run_manifest)}' | base64 -d > "$RUN_MANIFEST"
printf %s '{payload(args.analysis_plan)}' | base64 -d > "$ANALYSIS_PLAN"
printf %s '{payload(args.qualification_gate)}' | base64 -d > "$QUALIFICATION"
printf %s '{payload(args.authorization)}' | base64 -d > "$AUTH"
test "$(sha256sum "$SOURCE" | cut -d' ' -f1)" = "{SOURCE_SHA256}"
test "$(sha256sum "$MEGATRON" | cut -d' ' -f1)" = "{MEGATRON_SHA256}"
test "$(sha256sum "$PROTOCOL" | cut -d' ' -f1)" = "{PROTOCOL_SHA256}"
test "$(sha256sum "$RUN_MANIFEST" | cut -d' ' -f1)" = "{RUN_MANIFEST_SHA256}"
test "$(sha256sum "$ANALYSIS_PLAN" | cut -d' ' -f1)" = "{ANALYSIS_PLAN_SHA256}"
test "$(sha256sum "$QUALIFICATION" | cut -d' ' -f1)" = "{QUALIFICATION_GATE_SHA256}"
test "$(sha256sum "$AUTH" | cut -d' ' -f1)" = "{AUTHORIZATION_SHA256}"
"$PYTHON" - "$PROTOCOL" "$RUN_MANIFEST" "$ANALYSIS_PLAN" "$QUALIFICATION" "$AUTH" <<'PY'
import hashlib,json,sys
from pathlib import Path
p_path,r_path,n_path,q_path,a_path=map(Path,sys.argv[1:])
p=json.loads(p_path.read_bytes()); r=json.loads(r_path.read_bytes()); n=json.loads(n_path.read_bytes()); q=json.loads(q_path.read_bytes()); a=json.loads(a_path.read_bytes())
assert hashlib.sha256(p_path.read_bytes()).hexdigest()=="{PROTOCOL_SHA256}"
assert hashlib.sha256(r_path.read_bytes()).hexdigest()=="{RUN_MANIFEST_SHA256}"
assert hashlib.sha256(n_path.read_bytes()).hexdigest()=="{ANALYSIS_PLAN_SHA256}"
assert hashlib.sha256(q_path.read_bytes()).hexdigest()=="{QUALIFICATION_GATE_SHA256}"
assert hashlib.sha256(a_path.read_bytes()).hexdigest()=="{AUTHORIZATION_SHA256}"
assert p["protocol_id"]=="m4-oars-randomized-actuation-v1" and p["confirmatory_design"]["fixed_complete_pairs"]==10
assert r["run_count"]==20 and q["qualification_gate"]=="PASS"
assert n["confidence_intervals"]["df"]==9 and n["exact_randomization"]["enumeration_count"]==1024
assert a["schema"]=="m4-oars-randomized-confirmatory-local-package-authorization-v1"
assert a["scope"]=="build_and_clean_room_validate_credential_free_20_run_confirmatory_candidates_only"
assert a["runtime_source_commit"]=="{SOURCE_COMMIT}" and a["candidate_count"]==20
assert a["runtime_source_archive_build_authorized"] and a["credential_free_eos_package_build_authorized"] and a["clean_room_validation_authorized"]
blocked=("model_weight_access_authorized","optimizer_initialization_authorized","training_authorized","eos_submission_authorized","scientific_acquisition_authorized","automatic_retry","automatic_replacement","automatic_extension")
assert not any(a[key] for key in blocked)
run=next(item for item in r["runs"] if item["identity"]=="{run['identity']}")
assert run["global_sequence"]=={run['global_sequence']} and run["pair"]=={run['pair']}
assert run["arm"]=="{arm}" and run["mode"]=="{run['mode']}"
assert run["training_seed"]=={run['training_seed']} and run["assignment_seed"]=={run['assignment_seed']}
assert run["assignment_domain"]=="{run['assignment_domain']}"
if not (a["eos_submission_authorized"] and a["model_weight_access_authorized"] and a["optimizer_initialization_authorized"] and a["training_authorized"] and a["scientific_acquisition_authorized"]):
 raise SystemExit("M4_OARS_CONFIRMATORY_LOCAL_PACKAGE_NO_EXECUTION_AUTHORITY")
PY
"$PYTHON" - "$SOURCE" "$RUN_REPO" <<'PY'
import sys,tarfile
from pathlib import Path,PurePosixPath
archive,output=map(Path,sys.argv[1:])
with tarfile.open(archive,"r:gz") as source:
 members=source.getmembers(); names=[PurePosixPath(member.name) for member in members]
 assert len(names)==len(set(names)); assert not any(name.is_absolute() or ".." in name.parts for name in names)
 symlinks={{name for name,member in zip(names,members,strict=True) if member.issym()}}
 assert not any(any(parent in symlinks for parent in name.parents) for name in names)
 source.extractall(output)
PY
"$PYTHON" - "$MEGATRON" "$RUN_REPO" <<'PY'
import sys,tarfile
from pathlib import Path,PurePosixPath
archive,root=map(Path,sys.argv[1:]); dest=root/"3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty"
with tarfile.open(archive,"r:gz") as source:
 members=source.getmembers(); names=[PurePosixPath(member.name) for member in members]
 assert len(members)==726 and len(names)==len(set(names)); assert not any(name.is_absolute() or ".." in name.parts for name in names)
 dest.mkdir(parents=True,exist_ok=True); source.extractall(dest)
assert (dest/"Megatron-LM/megatron/core/__init__.py").is_file()
PY
cd "$RUN_REPO"
export PYTHONPATH="$RUN_REPO"
export CUDA_VISIBLE_DEVICES=0,1
readonly CONFIG="{config_path}"
test "$(sha256sum "$CONFIG" | cut -d' ' -f1)" = "{config_sha256}"
"$PYTHON" - "$CONFIG" <<'PY'
import json,sys,tomllib
from importlib.metadata import distribution
from pathlib import Path
from omegaconf import OmegaConf
from nemo_rl.algorithms.single_controller_utils.config import MasterConfig,validate_single_controller_config
from nemo_rl.utils.config import load_config,parse_hydra_overrides,register_omegaconf_resolvers
path=Path(sys.argv[1]); overrides={json.dumps([
        f"grpo.seed={run['training_seed']}",
        f"async_rl.controlled_release_delay.seed={run['assignment_seed']}",
        f"async_rl.controlled_release_delay.assignment_domain={run['assignment_domain']}",
        f"async_rl.opportunity_at_risk_shadow.mode={run['mode']}",
        "data_plane.actor_runtime_env_mode=inherit_baked_single_node",
        "async_rl.terminal_policy_export.enabled=true",
        "async_rl.terminal_policy_export.output_dir={assets_dir}/terminal-policy",
    ])}
register_omegaconf_resolvers(); raw=parse_hydra_overrides(load_config(path),overrides); config=MasterConfig(**OmegaConf.to_container(raw,resolve=True))
assert config.grpo.max_num_steps==64 and config.grpo.max_num_epochs==2 and config.grpo.seed=={run['training_seed']}
assert config.policy["model_name"]=="meta-llama/Llama-3.2-1B-Instruct"
assert config.data["train"]["dataset_name"]=="gsm8k" and config.data["train"]["split"]=="train"
assert config.cluster["num_nodes"]==1 and config.cluster["gpus_per_node"]==2
assert config.async_rl.sampler.name=="weight_fifo" and config.async_rl.sampler.selection_candidate_watermark==8
assert config.async_rl.sampler.max_staleness_versions==1
assert config.async_rl.opportunity_at_risk_shadow.enabled and config.async_rl.opportunity_at_risk_shadow.mode=="{run['mode']}"
assert config.async_rl.opportunity_at_risk_shadow.service_budget_multiplier==1.02
assert config.async_rl.controlled_release_delay.seed=={run['assignment_seed']}
assert config.async_rl.controlled_release_delay.assignment_domain=="{run['assignment_domain']}"
assert config.async_rl.terminal_policy_export.enabled and not config.checkpointing["enabled"]
assert config.data_plane["actor_runtime_env_mode"]=="inherit_baked_single_node"
locked=tomllib.loads(Path("uv.lock").read_text()); tq=[item for item in locked["package"] if item["name"].lower()=="transferqueue"]
assert len(tq)==1 and tq[0]["source"]["git"].endswith("#{TRANSFERQUEUE_COMMIT}")
direct=json.loads(distribution("TransferQueue").read_text("direct_url.json")); assert direct["vcs_info"]["commit_id"]=="{TRANSFERQUEUE_COMMIT}"
validate_single_controller_config(config)
print("M4_OARS_CONFIRMATORY_{tag}_CONFIG_PASS")
PY
readonly START_NS=$("$PYTHON" -c 'import time; print(time.time_ns())')
set +e
"$PYTHON" examples/run_grpo_single_controller.py --config "$CONFIG" \
  "grpo.seed={run['training_seed']}" \
  "async_rl.controlled_release_delay.seed={run['assignment_seed']}" \
  "async_rl.controlled_release_delay.assignment_domain={run['assignment_domain']}" \
  "async_rl.opportunity_at_risk_shadow.mode={run['mode']}" \
  "data_plane.actor_runtime_env_mode=inherit_baked_single_node" \
  "async_rl.lifecycle_audit_path=$LIFECYCLE" \
  "async_rl.gradient_opportunity_audit.output_path=$OPPORTUNITY" \
  "async_rl.gradient_opportunity_audit.observer_duty_path=$DUTY" \
  "async_rl.opportunity_at_risk_shadow.output_path=$OARS" \
  "async_rl.terminal_policy_export.enabled=true" \
  "async_rl.terminal_policy_export.output_dir=$EXPORT" \
  "logger.log_dir=$METRICS" 2>&1 | tee "$RUN_LOG"
readonly TRAIN_RC=${{PIPESTATUS[0]}}
set -e
readonly END_NS=$("$PYTHON" -c 'import time; print(time.time_ns())')
if test "$TRAIN_RC" -ne 0; then
  printf '{{"identity":"{run['identity']}","schema":"m4-oars-randomized-confirmatory-arm-result-v1","scientific_outcome_acquisition":true,"status":"TRAINING_FAILURE","training_exit_code":%s,"training_quality_analyzed":false}}\n' "$TRAIN_RC" > "$RESULT"
  exit "$TRAIN_RC"
fi
grep -Fq "SC run complete: {{'train_steps': 64, 'trainer_version': 64" "$RUN_LOG"
"$PYTHON" tools/m4_oars_actuation_qualification.py --mode "{run['mode']}" --oars "$OARS" --lifecycle "$LIFECYCLE" --observer-duty "$DUTY" --source-commit "{SOURCE_COMMIT}" --run-start-ns "$START_NS" --run-end-ns "$END_NS" --output "$SYSTEMS"
readonly ITER=$("$PYTHON" - "$EXPORT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); manifest=json.loads((root/"terminal_policy_export.json").read_bytes())
assert manifest["train_steps"]==64 and manifest["trainer_version"]==64 and not manifest["optimizer_exported"] and not manifest["resumable_training_checkpoint"]
iterations=[path for path in (root/manifest["weights_path"]).glob("iter_*") if (path/"run_config.yaml").is_file()]
assert len(iterations)==1; print(iterations[0])
PY
)
(cd "$EXPORT"; find . -type f -print0 | sort -z | xargs -0 sha256sum > "$ASSETS/terminal-policy-files.sha256")
readonly MCORE_VENV=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
readonly MCORE_PYTHON="$MCORE_VENV/bin/python"
env VIRTUAL_ENV="$MCORE_VENV" UV_PROJECT_ENVIRONMENT="$MCORE_VENV" "$MCORE_PYTHON" examples/converters/convert_megatron_to_hf.py --config "$EXPORT/resolved_config.json" --hf-model-name meta-llama/Llama-3.2-1B-Instruct --megatron-ckpt-path "$ITER" --hf-ckpt-path "$HF_MODEL"
(cd "$HF_MODEL"; find . -type f -print0 | sort -z | xargs -0 sha256sum > "$ASSETS/converted-hf-files.sha256")
"$PYTHON" - "$HF_MODEL" "$EVAL_OUTPUT" "$PROMPTS" <<'PY'
import hashlib,json,sys
from pathlib import Path
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_response_data
from nemo_rl.evals.eval import MasterConfig as EvalMasterConfig,run_env_eval,setup
from nemo_rl.models.generation import configure_generation_config
hf_model,output,prompts=map(Path,sys.argv[1:]); tokenizer_config={{"name":str(hf_model),"chat_template":"default","chat_template_kwargs":None,"use_fastokens":False}}; tokenizer=get_tokenizer(tokenizer_config)
data={{"dataset_name":"gsm8k","max_input_seq_length":2048,"shuffle":False,"num_workers":1,"use_multiple_dataloader":False,"train":{{"dataset_name":"gsm8k","subset":"main","split":"test","split_validation_size":0.0,"seed":None,"extract_answer":True}},"validation":None,"default":{{"prompt_file":None,"system_prompt_file":"examples/prompts/gsm8k.txt","processor":"math_hf_data_processor","env_name":"math"}}}}
dataset,validation,envs,_=setup_response_data(tokenizer,data,{{"math":{{"num_workers":8,"math_verify_impl":"hf_math_verify"}}}}); assert validation is None and len(dataset)==1319
rows=[dataset.dataset[index] for index in range(len(dataset))]; row_hashes=[hashlib.sha256(json.dumps(row,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest() for row in rows]
prompt_manifest={{"schema":"m4-oars-confirmatory-gsm8k-prompt-manifest-v1","dataset":"openai/gsm8k","subset":"main","split":"test","selection":"all_rows_in_repository_order","count":1319,"row_sha256":row_hashes}}
prompts.write_text(json.dumps(prompt_manifest,sort_keys=True,separators=(",",":"))+"\\n")
generation={{"backend":"vllm","model_name":str(hf_model),"max_new_tokens":1792,"temperature":0.0,"top_p":1.0,"top_k":-1,"val_temperature":0.0,"val_top_p":1.0,"val_top_k":-1,"num_prompts_per_step":32,"stop_token_ids":None,"stop_strings":None,"vllm_cfg":{{"async_engine":False,"precision":"bfloat16","tensor_parallel_size":1,"pipeline_parallel_size":1,"expert_parallel_size":1,"gpu_memory_utilization":0.9,"max_model_len":2048,"enforce_eager":False}},"colocated":{{"enabled":True,"resources":{{"gpus_per_node":None,"num_nodes":None}}}}}}
generation=configure_generation_config(generation,tokenizer,is_eval=True); master=EvalMasterConfig.model_construct(eval={{"metric":"pass@k","num_tests_per_prompt":1,"seed":20260915,"k_value":1,"save_path":str(output)}},generation=generation,tokenizer=tokenizer_config,data=data,env={{"math":{{"num_workers":8,"math_verify_impl":"hf_math_verify"}}}},cluster={{"gpus_per_node":1,"num_nodes":1}})
vllm,loader,master=setup(master,tokenizer,dataset); run_env_eval(vllm,loader,next(iter(envs.values())),master)
PY
"$PYTHON" - "$SYSTEMS" "$OARS" "$EVAL_OUTPUT/evaluation_data.json" "$PROMPTS" "$RESULT" <<'PY'
import hashlib,json,math,sys
from pathlib import Path
systems_path,oars_path,evaluation_path,prompts_path,result_path=map(Path,sys.argv[1:])
systems=json.loads(systems_path.read_bytes()); rows=[json.loads(line) for line in oars_path.read_text().splitlines()]; header,*decisions=rows
assert systems["status"]=="PASS" and len(decisions)==64 and all(systems["checks"].values())
assert header["mode"]=="{run['mode']}" and all(row["mode"]=="{run['mode']}" for row in decisions)
l1_key="proposed_l1" if "{arm}"=="oars" else "baseline_l1"; token_key="proposed_valid_actor_tokens" if "{arm}"=="oars" else "baseline_valid_actor_tokens"
retained_l1=math.fsum(float(row[l1_key]) for row in decisions)/(64*4); tokens=math.fsum(float(row[token_key]) for row in decisions)/64
evaluation=json.loads(evaluation_path.read_bytes())["evaluation_data"]; prompt_manifest=json.loads(prompts_path.read_bytes()); assert len(evaluation)==len(prompt_manifest["row_sha256"])==1319
rewards=[float(row["reward"]) for row in evaluation]; assert all(reward in (0.0,1.0) for reward in rewards)
value={{"schema":"m4-oars-randomized-confirmatory-arm-result-v1","status":"PASS","protocol_sha256":"{PROTOCOL_SHA256}","run_manifest_sha256":"{RUN_MANIFEST_SHA256}","source_commit":"{SOURCE_COMMIT}","identity":"{run['identity']}","pair":{run['pair']},"arm":"{arm}","mode":"{run['mode']}","training_seed":{run['training_seed']},"assignment_seed":{run['assignment_seed']},"assignment_domain":"{run['assignment_domain']}","scientific_outcome_acquisition":True,"training_quality_analyzed":True,"systems":{{"qualification_status":systems["status"],"checks":systems["checks"],"time_to_update_64_seconds":systems["runtime_seconds"],"gradient_observer_duty":systems["gradient_observer_duty"],"oars_decision_duty":systems["oars_decision_duty"]}},"outcomes":{{"retained_l1_per_selected_group":retained_l1,"valid_actor_tokens_per_update":tokens,"terminal_gsm8k_accuracy":math.fsum(rewards)/1319,"terminal_gsm8k_correct":int(sum(rewards)),"terminal_gsm8k_prompt_count":1319,"terminal_gsm8k_prompt_manifest_sha256":hashlib.sha256(prompts_path.read_bytes()).hexdigest()}}}}
result_path.write_text(json.dumps(value,sort_keys=True,separators=(",",":"))+"\\n")
PY
cp "$EXPORT/terminal_policy_export.json" "$ASSETS/terminal-policy-export.json"
cp "$EXPORT/resolved_config.json" "$ASSETS/terminal-resolved-config.json"
rm -rf "$EXPORT" "$HF_MODEL"
echo M4_OARS_CONFIRMATORY_{tag}_CAPTURE_COMPLETE_OUTCOME_EMBARGOED
'''
    script = (
        script.replace("{", "{{")
        .replace("}", "}}")
        .replace("{{assets_dir}}", "{assets_dir}")
    )
    manifest = {
        "type": "basic",
        "format_version": 1,
        "maintainers": ["nvegesna"],
        "loggers": ["stdout"],
        "labels": {"target": "silicon"},
        "launchers": {"type:slurm": {"nodes": 1, "ntasks_per_node": 1}},
        "spec": {
            "name": f"{stem}-local-candidate",
            "workspace": "/workspace",
            "nodes": 1,
            "time_limit": 14400,
            "image_source": {"local_path": IMAGE_PATH},
            "script": script,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    print(args.output, sha256(args.output))


if __name__ == "__main__":
    main()
