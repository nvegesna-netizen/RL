#!/usr/bin/env python3
"""Build one of 32 credential-free, locally nonlaunchable acquisition candidates."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

SOURCE_COMMIT = "e90e400eb2219284b1e0b1726bdfd8e878ed8620"
SOURCE_SHA = "7e10c4252e7f20c877a02b95e3bf5df3937c71bfc8772da2da47350d8db75f64"
MEGATRON_SHA = "98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
BRIDGE_SHA = "1429945d1d50045900e40314fa283fa5f66484e945aad4920da137d7ad2c9313"
PROTOCOL_SHA = "dbe7a4f7d1938ef43d24e536ccf9dd57110a01bab5e6cb511980218e472ef2c0"
RUN_MANIFEST_SHA = "a461e83c7dbccdefa1c5c779062e709f34bfc502639a3e7de663a717fefa93bf"
AUTHORIZATION_SHA = "7b5830d5347125ebcf73d6df6da86ad1b181ef634f230687a516d504cf0fdbc3"
PROMPT_MANIFEST_SHA = "469a34a176febe9d15c5c6d967ff21f135a43cb20a7cb7d5ae1a53066e6b3a0a"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path, expected in (
        (args.source, SOURCE_SHA),
        (args.megatron, MEGATRON_SHA),
        (args.bridge, BRIDGE_SHA),
        (args.protocol, PROTOCOL_SHA),
        (args.run_manifest, RUN_MANIFEST_SHA),
        (args.authorization, AUTHORIZATION_SHA),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen package input moved: {path.name}")
    authorization = json.loads(args.authorization.read_bytes())
    forbidden = (
        "model_weight_access_authorized",
        "optimizer_initialization_authorized",
        "training_authorized",
        "eos_submission_authorized",
        "qualification_authorized",
        "scientific_acquisition_authorized",
        "automatic_retry",
        "automatic_replacement",
        "automatic_extension",
    )
    if (
        authorization["candidate_count"] != 32
        or authorization["runtime_source_commit"] != SOURCE_COMMIT
        or not authorization["credential_free_eos_package_build_authorized"]
        or not authorization["clean_room_validation_authorized"]
        or any(authorization[key] for key in forbidden)
    ):
        raise RuntimeError("local package authority differs")
    run_manifest = json.loads(args.run_manifest.read_bytes())
    runs = {f"{run['block_id']}_{run['regime']}": run for run in run_manifest["runs"]}
    if args.identity not in runs or len(runs) != 32:
        raise RuntimeError("registered run identity differs")
    run = runs[args.identity]
    config = args.config_dir / run["config_filename"]
    if sha256(config) != run["config_sha256"]:
        raise RuntimeError("registered run config moved")

    stem = f"m4-downstream-quality-{run['block_id']}-{run['regime'].replace('_', '-')}"
    tag = f"{run['block_id']}_{run['regime']}".upper()
    script = f'''set -euo pipefail
readonly PYTHON=/opt/nemo_rl_venv/bin/python
readonly RUN_REPO=/workspace/{stem}-repo
readonly SOURCE=/workspace/{stem}-source.tar.gz
readonly MEGATRON=/workspace/{stem}-megatron.tar.gz
readonly BRIDGE=/workspace/{stem}-bridge.tar.gz
readonly PROTOCOL=/workspace/{stem}-protocol.json
readonly RUN_MANIFEST=/workspace/{stem}-run-manifest.json
readonly AUTH=/workspace/{stem}-authorization.json
readonly CONFIG_PAYLOAD=/workspace/{stem}-config.yaml
readonly ASSETS={{assets_dir}}
readonly EVAL_OUTPUT=$ASSETS/evaluation
readonly RESULT=$ASSETS/trained-run-result.json
readonly PROMPTS=$ASSETS/prompt-manifest.json
readonly HASHES=$ASSETS/artifacts.sha256
readonly RUN_LOG=$ASSETS/training.log
test "${{NEMO_RL_COMMIT:-unknown}}" = "ae07eafe8035b5b2e84efa7234e70e7fd7e493c1"
test ! -e "$RUN_REPO"
printf %s '{payload(args.source)}' | base64 -d > "$SOURCE"
printf %s '{payload(args.megatron)}' | base64 -d > "$MEGATRON"
printf %s '{payload(args.bridge)}' | base64 -d > "$BRIDGE"
printf %s '{payload(args.protocol)}' | base64 -d > "$PROTOCOL"
printf %s '{payload(args.run_manifest)}' | base64 -d > "$RUN_MANIFEST"
printf %s '{payload(args.authorization)}' | base64 -d > "$AUTH"
printf %s '{payload(config)}' | base64 -d > "$CONFIG_PAYLOAD"
test "$(sha256sum "$SOURCE" | cut -d' ' -f1)" = "{SOURCE_SHA}"
test "$(sha256sum "$MEGATRON" | cut -d' ' -f1)" = "{MEGATRON_SHA}"
test "$(sha256sum "$BRIDGE" | cut -d' ' -f1)" = "{BRIDGE_SHA}"
test "$(sha256sum "$PROTOCOL" | cut -d' ' -f1)" = "{PROTOCOL_SHA}"
test "$(sha256sum "$RUN_MANIFEST" | cut -d' ' -f1)" = "{RUN_MANIFEST_SHA}"
test "$(sha256sum "$AUTH" | cut -d' ' -f1)" = "{AUTHORIZATION_SHA}"
test "$(sha256sum "$CONFIG_PAYLOAD" | cut -d' ' -f1)" = "{run['config_sha256']}"
"$PYTHON" - "$PROTOCOL" "$RUN_MANIFEST" "$AUTH" <<'PY'
import hashlib,json,sys
from pathlib import Path
protocol_path,manifest_path,auth_path=map(Path,sys.argv[1:])
p=json.loads(protocol_path.read_bytes()); m=json.loads(manifest_path.read_bytes()); a=json.loads(auth_path.read_bytes())
assert hashlib.sha256(protocol_path.read_bytes()).hexdigest()=="{PROTOCOL_SHA}"
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()=="{RUN_MANIFEST_SHA}"
assert a["schema"]=="m4-downstream-quality-trained-paired-local-package-authorization-v1"
assert a["scope"]=="build_and_clean_room_validate_credential_free_32_run_acquisition_candidates_only"
assert a["runtime_source_commit"]=="{SOURCE_COMMIT}" and a["candidate_count"]==32
assert a["source_archive_build_authorized"] and a["credential_free_eos_package_build_authorized"] and a["clean_room_validation_authorized"]
blocked=("model_weight_access_authorized","optimizer_initialization_authorized","training_authorized","eos_submission_authorized","qualification_authorized","scientific_acquisition_authorized","automatic_retry","automatic_replacement","automatic_extension")
assert not any(a[k] for k in blocked)
r=next(r for r in m["runs"] if r["block_id"]=="{run['block_id']}" and r["regime"]=="{run['regime']}")
assert r["config_sha256"]=="{run['config_sha256']}" and r["training_seed"]=={run['training_seed']}
assert r["assignment_seed"]=={run['assignment_seed']} and r["assignment_domain"]=="{run['assignment_domain']}"
print("M4_DOWNSTREAM_QUALITY_{tag}_PACKAGE_EVIDENCE_PASS")
if not (a["eos_submission_authorized"] and a["model_weight_access_authorized"] and a["optimizer_initialization_authorized"] and a["training_authorized"] and a["scientific_acquisition_authorized"]):
 raise SystemExit("M4_DOWNSTREAM_QUALITY_LOCAL_PACKAGE_NO_EXECUTION_AUTHORITY")
PY
mkdir -p "$RUN_REPO" "$EVAL_OUTPUT"
"$PYTHON" - "$SOURCE" "$RUN_REPO" <<'PY'
import sys,tarfile
from pathlib import Path,PurePosixPath
archive,output=map(Path,sys.argv[1:])
with tarfile.open(archive,"r:gz") as source:
 members=source.getmembers(); names=[PurePosixPath(m.name) for m in members]
 assert len(names)==len(set(names)); assert not any(n.is_absolute() or ".." in n.parts for n in names)
 symlinks={{n for n,m in zip(names,members,strict=True) if m.issym()}}
 assert not any(any(parent in symlinks for parent in n.parents) for n in names)
 source.extractall(output)
PY
"$PYTHON" - "$MEGATRON" "$BRIDGE" "$RUN_REPO" <<'PY'
import sys,tarfile
from pathlib import Path,PurePosixPath
megatron,bridge,root=map(Path,sys.argv[1:]); workspace=root/"3rdparty/Megatron-Bridge-workspace"; bridge_root=workspace/"Megatron-Bridge"
for archive,dest,count,prefix in ((bridge,workspace,1060,"Megatron-Bridge/"),(megatron,bridge_root/"3rdparty",726,"Megatron-LM/")):
 with tarfile.open(archive,"r:gz") as source:
  members=source.getmembers(); names=[PurePosixPath(m.name) for m in members]
  assert len(members)==count and len(names)==len(set(names)); assert all(str(n)==prefix.rstrip("/") or str(n).startswith(prefix) for n in names)
  assert not any(n.is_absolute() or ".." in n.parts for n in names); dest.mkdir(parents=True,exist_ok=True); source.extractall(dest)
PY
cd "$RUN_REPO"
cp "$CONFIG_PAYLOAD" "examples/configs/{run['config_filename']}"
export PYTHONPATH="$RUN_REPO/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:$RUN_REPO/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:$RUN_REPO"
export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME=/workspace/{stem}-hf-cache
readonly CONFIG="examples/configs/{run['config_filename']}"
readonly RUN_ROOT="$RUN_REPO/{run['run_root']}"
readonly EXPORT="$RUN_ROOT/terminal-policy"
readonly HF_MODEL=/workspace/{stem}-terminal-hf
readonly START_SECONDS=$SECONDS
"$PYTHON" - "$CONFIG" <<'PY'
import hashlib,sys
from pathlib import Path
from omegaconf import OmegaConf
from nemo_rl.algorithms.single_controller_utils.config import MasterConfig,validate_single_controller_config
from nemo_rl.utils.config import load_config,register_omegaconf_resolvers
path=Path(sys.argv[1]); assert hashlib.sha256(path.read_bytes()).hexdigest()=="{run['config_sha256']}"
register_omegaconf_resolvers(); raw=OmegaConf.to_container(load_config(str(path)),resolve=True); config=MasterConfig(**raw)
assert config.grpo.max_num_steps==448 and config.grpo.seed=={run['training_seed']}
assert config.policy["model_name"]=="Qwen/Qwen3-0.6B" and config.cluster["gpus_per_node"]==2
assert config.async_rl.controlled_release_delay.seed=={run['assignment_seed']}
assert config.async_rl.controlled_release_delay.assignment_domain=="{run['assignment_domain']}"
assert config.async_rl.terminal_policy_export.enabled and not config.checkpointing["enabled"]
validate_single_controller_config(config)
PY
"$PYTHON" examples/run_grpo_single_controller.py --config "$CONFIG" 2>&1 | tee "$RUN_LOG"
grep -Fq "SC run complete: {{'train_steps': 448, 'trainer_version': 448" "$RUN_LOG"
readonly ITER=$("$PYTHON" - "$EXPORT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); m=json.loads((root/"terminal_policy_export.json").read_bytes())
assert m["train_steps"]==448 and m["trainer_version"]==448 and not m["optimizer_exported"] and not m["resumable_training_checkpoint"]
iters=[p for p in (root/m["weights_path"]).glob("iter_*") if (p/"run_config.yaml").is_file()]; assert len(iters)==1; print(iters[0])
PY
)
readonly MCORE_VENV=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
readonly MCORE_PYTHON="$MCORE_VENV/bin/python"
env VIRTUAL_ENV="$MCORE_VENV" UV_PROJECT_ENVIRONMENT="$MCORE_VENV" "$MCORE_PYTHON" examples/converters/convert_megatron_to_hf.py --config "$EXPORT/resolved_config.json" --hf-model-name Qwen/Qwen3-0.6B --megatron-ckpt-path "$ITER" --hf-ckpt-path "$HF_MODEL"
"$PYTHON" - "$HF_MODEL" "$EVAL_OUTPUT" "$PROMPTS" <<'PY'
import hashlib,json,sys
from pathlib import Path
from torch.utils.data import Subset
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_response_data
from nemo_rl.evals.eval import MasterConfig as EvalMasterConfig,run_env_eval,setup
from nemo_rl.models.generation import configure_generation_config
hf_model,output,prompts=map(Path,sys.argv[1:]); tok={{"name":str(hf_model),"chat_template":"default","chat_template_kwargs":None,"use_fastokens":False}}; tokenizer=get_tokenizer(tok)
data={{"dataset_name":"OpenMathInstruct-2","max_input_seq_length":2048,"shuffle":False,"num_workers":1,"use_multiple_dataloader":False,"train":{{"dataset_name":"OpenMathInstruct-2","split_validation_size":0.05,"seed":20260911}},"validation":None,"default":{{"prompt_file":"examples/prompts/cot.txt","system_prompt_file":None,"processor":"math_hf_data_processor","env_name":"math"}}}}
_,dataset,_,envs=setup_response_data(tokenizer,data,{{"math":{{"num_workers":8,"math_verify_impl":"hf_math_verify"}}}}); assert dataset is not None and len(dataset)>=1024
rows=[dataset.dataset[i] for i in range(1024)]; hashes=[hashlib.sha256(json.dumps(r,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest() for r in rows]
value={{"schema":"m4-downstream-quality-openmath-prompt-manifest-v1","dataset":"OpenMathInstruct-2","split_fraction":0.05,"split_seed":20260911,"selection":"first_1024_after_hf_train_test_split","count":1024,"row_sha256":hashes}}
prompts.write_text(json.dumps(value,sort_keys=True,separators=(",",":"))+"\\n"); assert hashlib.sha256(prompts.read_bytes()).hexdigest()=="{PROMPT_MANIFEST_SHA}"
generation={{"backend":"vllm","model_name":str(hf_model),"max_new_tokens":1792,"temperature":0.0,"top_p":1.0,"top_k":-1,"val_temperature":0.0,"val_top_p":1.0,"val_top_k":-1,"num_prompts_per_step":32,"stop_token_ids":None,"stop_strings":None,"vllm_cfg":{{"async_engine":False,"precision":"bfloat16","tensor_parallel_size":1,"pipeline_parallel_size":1,"expert_parallel_size":1,"gpu_memory_utilization":0.9,"max_model_len":2048,"enforce_eager":False}},"colocated":{{"enabled":True,"resources":{{"gpus_per_node":None,"num_nodes":None}}}}}}
generation=configure_generation_config(generation,tokenizer,is_eval=True); master=EvalMasterConfig.model_construct(eval={{"metric":"pass@k","num_tests_per_prompt":1,"seed":20260911,"k_value":1,"save_path":str(output)}},generation=generation,tokenizer=tok,data=data,env={{"math":{{"num_workers":8,"math_verify_impl":"hf_math_verify"}}}},cluster={{"gpus_per_node":1,"num_nodes":1}})
vllm,loader,master=setup(master,tokenizer,Subset(dataset,range(1024))); run_env_eval(vllm,loader,next(iter(envs.values())),master)
PY
readonly WALL_SECONDS=$((SECONDS-START_SECONDS))
"$PYTHON" - "$CONFIG" "$EXPORT" "$EVAL_OUTPUT" "$PROMPTS" "$RESULT" "$WALL_SECONDS" <<'PY'
import hashlib,json,sys
from pathlib import Path
config,export,eval_output,prompts,result=map(Path,sys.argv[1:6]); wall=int(sys.argv[6]); evaluation=json.loads((eval_output/"evaluation_data.json").read_bytes())["evaluation_data"]; prompt=json.loads(prompts.read_bytes())
assert len(evaluation)==1024 and len(prompt["row_sha256"])==1024
scores=[{{"prompt_row_sha256":h,"reward":row["reward"]}} for h,row in zip(prompt["row_sha256"],evaluation,strict=True)]; assert all(row["reward"] in (0,1,0.0,1.0) for row in scores)
manifest=json.loads((export/"terminal_policy_export.json").read_bytes())
value={{"schema":"m4-downstream-quality-trained-run-result-v1","protocol_sha256":"{PROTOCOL_SHA}","source_commit":"{SOURCE_COMMIT}","block_id":"{run['block_id']}","regime":"{run['regime']}","training_seed":{run['training_seed']},"assignment_seed":{run['assignment_seed']},"assignment_domain":"{run['assignment_domain']}","config_sha256":"{run['config_sha256']}","train_steps":448,"trainer_version":448,"prompt_manifest_sha256":"{PROMPT_MANIFEST_SHA}","optimizer_exported":False,"resumable_checkpoint":False,"scores":scores,"systems":{{"wall_seconds":wall,"gpus":2}},"mechanism":{{"lifecycle_path":"lifecycle.jsonl","opportunity_path":"opportunity.jsonl","observer_duty_path":"observer-duty.json"}},"terminal_export":manifest}}
result.write_text(json.dumps(value,sort_keys=True,separators=(",",":"))+"\\n")
PY
cp "$RUN_ROOT/lifecycle.jsonl" "$RUN_ROOT/opportunity.jsonl" "$RUN_ROOT/observer-duty.json" "$ASSETS/"
mv "$EXPORT" "$ASSETS/terminal-policy"
mv "$HF_MODEL" "$ASSETS/converted-hf"
(cd "$ASSETS"; sha256sum trained-run-result.json prompt-manifest.json evaluation/evaluation_data.json lifecycle.jsonl opportunity.jsonl observer-duty.json terminal-policy/terminal_policy_export.json terminal-policy/resolved_config.json > artifacts.sha256)
echo M4_DOWNSTREAM_QUALITY_{tag}_CAPTURE_COMPLETE_NO_OUTCOME_INSPECTION
'''
    script = script.replace("{", "{{").replace("}", "}}").replace(
        "{{assets_dir}}", "{assets_dir}"
    )
    manifest = json.loads(args.base_manifest.read_bytes())
    manifest["spec"]["name"] = f"{stem}-local-candidate"
    manifest["spec"]["time_limit"] = 14400
    manifest["spec"]["script"] = script
    args.output.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    print(args.output, sha256(args.output))


if __name__ == "__main__":
    main()
