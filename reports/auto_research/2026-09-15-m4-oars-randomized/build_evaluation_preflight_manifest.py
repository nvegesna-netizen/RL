#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build the one-shot Llama no-training evaluation preflight manifest."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

SOURCE_COMMIT = "90dbb632026591f24c966a43edd05b1e4933358b"
SOURCE_SHA256 = "1d9e17a430fc1a184d767413de829ad9ee1375091485b5e7e0b24572f61f7690"
BRIDGE_SHA256 = "1429945d1d50045900e40314fa283fa5f66484e945aad4920da137d7ad2c9313"
MEGATRON_SHA256 = "98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
PROTOCOL_SHA256 = "4b17db7713ed579ed1db4a5d6aee125592693d26ae78eb1d69f13af04316a7d2"
AUTHORIZATION_SHA256 = (
    "6e6545403c7052d26379524f747f98d5e79fb20a0e0a73274c1740bc57b89081"
)
CONFIG_PATH = (
    "examples/configs/"
    "grpo_math_1B_megatron_single_controller_m4_oars_qualification_fifo.yaml"
)
CONFIG_SHA256 = "17ad965489aa9595d266295125880fe55edb4af303bd95aed66a61034c943c9b"
IMAGE_PATH = (
    "/lustre/fsw/coreai_dlalgo_ci/nvegesna/nemo_rl_images/nemo-rl-nightly-5802754.sqsh"
)
IMAGE_COMMIT = "ae07eafe8035b5b2e84efa7234e70e7fd7e493c1"


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload(path: Path) -> str:
    """Return a base64-encoded file payload."""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    """Build one credential-free JET manifest from frozen inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path, expected in (
        (args.source, SOURCE_SHA256),
        (args.bridge, BRIDGE_SHA256),
        (args.megatron, MEGATRON_SHA256),
        (args.protocol, PROTOCOL_SHA256),
        (args.authorization, AUTHORIZATION_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen evaluation-preflight input moved: {path.name}")

    authorization = json.loads(args.authorization.read_bytes())
    allowed = (
        "eos_submission_authorized",
        "model_weight_access_authorized",
        "zero_step_terminal_export_authorized",
        "megatron_to_hf_conversion_authorized",
        "full_gsm8k_evaluation_authorized",
    )
    blocked = (
        "optimizer_initialization_authorized",
        "training_authorized",
        "qualification_authorized",
        "pilot_authorized",
        "scientific_acquisition_authorized",
        "automatic_retry",
        "automatic_replacement",
        "automatic_extension",
    )
    if (
        authorization["schema"] != "m4-oars-llama-evaluation-preflight-authorization-v1"
        or authorization["submission_attempt_limit"] != 1
        or authorization["optimizer_steps_allowed"] != 0
        or not all(authorization[key] for key in allowed)
        or any(authorization[key] for key in blocked)
    ):
        raise RuntimeError("evaluation-preflight authority differs")

    script = f'''set -euo pipefail
readonly PYTHON=/opt/nemo_rl_venv/bin/python
readonly RUN_REPO=/workspace/m4-oars-llama-evaluation-preflight-repo
readonly SOURCE=/workspace/m4-oars-llama-evaluation-preflight-source.tar.gz
readonly BRIDGE=/workspace/m4-oars-llama-evaluation-preflight-bridge.tar.gz
readonly MEGATRON=/workspace/m4-oars-llama-evaluation-preflight-megatron.tar.gz
readonly PROTOCOL=/workspace/m4-oars-llama-evaluation-preflight-protocol.json
readonly AUTH=/workspace/m4-oars-llama-evaluation-preflight-authorization.json
readonly EXPORT=/workspace/m4-oars-llama-evaluation-preflight-export
readonly HF_MODEL=/workspace/m4-oars-llama-evaluation-preflight-hf
readonly ASSETS={{assets_dir}}
readonly EVAL_OUTPUT=$ASSETS/evaluation
readonly PROMPTS=$ASSETS/gsm8k-prompt-manifest.json
readonly RESULT=$ASSETS/evaluation-preflight-result.json
readonly HASHES=$ASSETS/artifacts.sha256
readonly CONVERTED_HASHES=$ASSETS/converted-hf-files.sha256
test "${{NEMO_RL_COMMIT:-unknown}}" = "{IMAGE_COMMIT}"
test ! -e "$RUN_REPO" && test ! -e "$EXPORT" && test ! -e "$HF_MODEL"
mkdir -p "$ASSETS" "$EVAL_OUTPUT"
finalize_artifacts() {{
  set +e
  (cd "$ASSETS"; find . -type f ! -name artifacts.sha256 ! -name .artifacts.sha256.tmp -print0 | sort -z | xargs -0 sha256sum > .artifacts.sha256.tmp && mv .artifacts.sha256.tmp artifacts.sha256)
}}
trap finalize_artifacts EXIT
printf %s '{payload(args.source)}' | base64 -d > "$SOURCE"
printf %s '{payload(args.bridge)}' | base64 -d > "$BRIDGE"
printf %s '{payload(args.megatron)}' | base64 -d > "$MEGATRON"
printf %s '{payload(args.protocol)}' | base64 -d > "$PROTOCOL"
printf %s '{payload(args.authorization)}' | base64 -d > "$AUTH"
test "$(sha256sum "$SOURCE" | cut -d' ' -f1)" = "{SOURCE_SHA256}"
test "$(sha256sum "$BRIDGE" | cut -d' ' -f1)" = "{BRIDGE_SHA256}"
test "$(sha256sum "$MEGATRON" | cut -d' ' -f1)" = "{MEGATRON_SHA256}"
test "$(sha256sum "$PROTOCOL" | cut -d' ' -f1)" = "{PROTOCOL_SHA256}"
test "$(sha256sum "$AUTH" | cut -d' ' -f1)" = "{AUTHORIZATION_SHA256}"
"$PYTHON" - "$PROTOCOL" "$AUTH" <<'PY'
import hashlib,json,sys
from pathlib import Path
protocol_path,authorization_path=map(Path,sys.argv[1:])
p=json.loads(protocol_path.read_bytes()); a=json.loads(authorization_path.read_bytes())
assert hashlib.sha256(protocol_path.read_bytes()).hexdigest()=="{PROTOCOL_SHA256}"
assert hashlib.sha256(authorization_path.read_bytes()).hexdigest()=="{AUTHORIZATION_SHA256}"
assert p["schema"]=="m4-oars-llama-evaluation-preflight-protocol-v1"
assert p["status"]=="FROZEN_BEFORE_EOS_SUBMISSION" and p["scientific_endpoint"] is False
assert p["model"]["name"]=="meta-llama/Llama-3.2-1B-Instruct"
assert p["model"]["optimizer_initialized"] is False and p["model"]["optimizer_steps"]==0
assert p["model"]["trainer_steps"]==0 and p["evaluation"]["expected_prompt_count"]==1319
assert p["execution"]["queue_deadline_override"] is None
assert a["schema"]=="m4-oars-llama-evaluation-preflight-authorization-v1"
assert a["scope"]=="exactly_one_credential_free_llama_zero_step_terminal_export_conversion_and_full_gsm8k_eos_preflight"
assert a["required_launcher"]=="runllm.py --no_wait" and a["submission_attempt_limit"]==1
allowed=("eos_submission_authorized","model_weight_access_authorized","zero_step_terminal_export_authorized","megatron_to_hf_conversion_authorized","full_gsm8k_evaluation_authorized")
blocked=("optimizer_initialization_authorized","training_authorized","qualification_authorized","pilot_authorized","scientific_acquisition_authorized","automatic_retry","automatic_replacement","automatic_extension")
assert all(a[key] for key in allowed) and a["optimizer_steps_allowed"]==0
assert not any(a[key] for key in blocked)
print("M4_OARS_LLAMA_EVALUATION_PREFLIGHT_AUTHORITY_PASS")
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
print("M4_OARS_LLAMA_EVALUATION_PREFLIGHT_SAFE_SOURCE_PASS",len(names),len(symlinks))
PY
"$PYTHON" - "$BRIDGE" "$MEGATRON" "$RUN_REPO" <<'PY'
import sys,tarfile
from pathlib import Path,PurePosixPath
bridge,megatron,root=map(Path,sys.argv[1:]); workspace=root/"3rdparty/Megatron-Bridge-workspace"; bridge_root=workspace/"Megatron-Bridge"
for archive,dest,count,prefix in ((bridge,workspace,1060,"Megatron-Bridge/"),(megatron,bridge_root/"3rdparty",726,"Megatron-LM/")):
 with tarfile.open(archive,"r:gz") as source:
  members=source.getmembers(); names=[PurePosixPath(member.name) for member in members]
  assert len(members)==count and len(names)==len(set(names)); assert all(str(name)==prefix.rstrip("/") or str(name).startswith(prefix) for name in names)
  assert not any(name.is_absolute() or ".." in name.parts for name in names)
  symlinks={{name for name,member in zip(names,members,strict=True) if member.issym()}}
  assert not any(any(parent in symlinks for parent in name.parents) for name in names)
  dest.mkdir(parents=True,exist_ok=True); source.extractall(dest)
assert (bridge_root/"src/megatron/bridge/__init__.py").is_file()
assert (bridge_root/"3rdparty/Megatron-LM/megatron/core/__init__.py").is_file()
print("M4_OARS_LLAMA_EVALUATION_PREFLIGHT_DEPENDENCIES_PASS")
PY
cd "$RUN_REPO"
export PYTHONPATH="$RUN_REPO/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:$RUN_REPO/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:$RUN_REPO"
export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME=/workspace/m4-oars-llama-evaluation-preflight-hf-cache
readonly MCORE_VENV=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
readonly MCORE_PYTHON="$MCORE_VENV/bin/python"
test -x "$MCORE_PYTHON"
env VIRTUAL_ENV="$MCORE_VENV" UV_PROJECT_ENVIRONMENT="$MCORE_VENV" "$MCORE_PYTHON" - "$RUN_REPO" "$MCORE_VENV" <<'PY'
import sys
from pathlib import Path
root=Path(sys.argv[1]).resolve(); venv=Path(sys.argv[2]).absolute()
import megatron.bridge
import megatron.core
import transformer_engine.pytorch as te
from megatron.bridge import AutoBridge
bridge_path=Path(megatron.bridge.__file__).absolute(); core_path=Path(megatron.core.__file__).absolute(); te_raw=Path(te.__file__).absolute()
print("M4_OARS_LLAMA_EVALUATION_PREFLIGHT_MCORE_PATHS",sys.executable,bridge_path,core_path,te_raw,te_raw.resolve(),flush=True)
assert bridge_path.resolve().is_relative_to(root) and core_path.resolve().is_relative_to(root)
assert te_raw.is_relative_to(venv) and AutoBridge.__module__.startswith("megatron.bridge.")
PY
env VIRTUAL_ENV="$MCORE_VENV" UV_PROJECT_ENVIRONMENT="$MCORE_VENV" "$MCORE_PYTHON" examples/converters/convert_megatron_to_hf.py --help >/dev/null
echo M4_OARS_LLAMA_EVALUATION_PREFLIGHT_CONVERTER_IMPORT_PASS
"$PYTHON" - "$EXPORT" <<'PY'
import hashlib,sys
from pathlib import Path
import ray
from omegaconf import OmegaConf
from nemo_rl.algorithms.single_controller_utils.config import MasterConfig
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster,init_ray
from nemo_rl.models.policy.tq_policy import TQPolicy
from nemo_rl.utils.config import load_config,register_omegaconf_resolvers
from nemo_rl.utils.terminal_policy_export import export_terminal_policy
export_dir=Path(sys.argv[1]); config_path=Path("{CONFIG_PATH}")
assert hashlib.sha256(config_path.read_bytes()).hexdigest()=="{CONFIG_SHA256}"
register_omegaconf_resolvers(); raw=OmegaConf.to_container(load_config(config_path),resolve=True)
raw["grpo"]["max_num_steps"]=0; raw["policy"]["megatron_cfg"]["train_iters"]=1
config=MasterConfig(**raw)
assert config.policy["model_name"]=="meta-llama/Llama-3.2-1B-Instruct"
assert config.grpo.max_num_steps==0 and config.checkpointing["enabled"] is False
init_ray(); tokenizer=get_tokenizer(config.policy["tokenizer"])
cluster=RayVirtualCluster(name="m4_oars_llama_evaluation_preflight",bundle_ct_per_node_list=[1],use_gpus=True,num_gpus_per_node=1,max_colocated_worker_groups=1)
policy=TQPolicy(cluster=cluster,config=config.policy,tokenizer=tokenizer,processor=None,weights_path=None,optimizer_path=None,init_optimizer=False,init_reference_model=False,dp_cfg=config.data_plane)
try:
 manifest=export_terminal_policy(trainer=policy,master_config=config,output_dir=str(export_dir),train_steps=0,trainer_version=0)
finally:
 policy.shutdown(); ray.shutdown()
assert manifest["train_steps"]==0 and manifest["trainer_version"]==0
assert manifest["optimizer_exported"] is False and manifest["resumable_training_checkpoint"] is False
print("M4_OARS_LLAMA_ZERO_STEP_EXPORT_PASS")
PY
readonly ITER=$("$PYTHON" - "$EXPORT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); manifest=json.loads((root/"terminal_policy_export.json").read_bytes())
assert manifest["train_steps"]==0 and manifest["trainer_version"]==0 and not manifest["optimizer_exported"] and not manifest["resumable_training_checkpoint"]
iterations=[path for path in (root/manifest["weights_path"]).glob("iter_*") if (path/"run_config.yaml").is_file()]
assert len(iterations)==1; print(iterations[0])
PY
)
env VIRTUAL_ENV="$MCORE_VENV" UV_PROJECT_ENVIRONMENT="$MCORE_VENV" "$MCORE_PYTHON" examples/converters/convert_megatron_to_hf.py --config "$EXPORT/resolved_config.json" --hf-model-name meta-llama/Llama-3.2-1B-Instruct --megatron-ckpt-path "$ITER" --hf-ckpt-path "$HF_MODEL"
(cd "$HF_MODEL"; find . -type f -print0 | sort -z | xargs -0 sha256sum > "$CONVERTED_HASHES")
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
prompt_manifest={{"schema":"m4-oars-llama-evaluation-preflight-gsm8k-prompt-manifest-v1","dataset":"openai/gsm8k","subset":"main","split":"test","selection":"all_rows_in_repository_order","count":1319,"row_sha256":row_hashes}}
prompts.write_text(json.dumps(prompt_manifest,sort_keys=True,separators=(",",":"))+"\\n")
generation={{"backend":"vllm","model_name":str(hf_model),"max_new_tokens":1792,"temperature":0.0,"top_p":1.0,"top_k":-1,"val_temperature":0.0,"val_top_p":1.0,"val_top_k":-1,"num_prompts_per_step":32,"stop_token_ids":None,"stop_strings":None,"vllm_cfg":{{"async_engine":False,"precision":"bfloat16","tensor_parallel_size":1,"pipeline_parallel_size":1,"expert_parallel_size":1,"gpu_memory_utilization":0.9,"max_model_len":2048,"enforce_eager":False}},"colocated":{{"enabled":True,"resources":{{"gpus_per_node":None,"num_nodes":None}}}}}}
generation=configure_generation_config(generation,tokenizer,is_eval=True); master=EvalMasterConfig.model_construct(eval={{"metric":"pass@k","num_tests_per_prompt":1,"seed":20260915,"k_value":1,"save_path":str(output)}},generation=generation,tokenizer=tokenizer_config,data=data,env={{"math":{{"num_workers":8,"math_verify_impl":"hf_math_verify"}}}},cluster={{"gpus_per_node":1,"num_nodes":1}})
vllm,loader,master=setup(master,tokenizer,dataset); run_env_eval(vllm,loader,next(iter(envs.values())),master)
PY
"$PYTHON" - "$EXPORT" "$EVAL_OUTPUT/evaluation_data.json" "$PROMPTS" "$CONVERTED_HASHES" "$RESULT" <<'PY'
import hashlib,json,sys
from pathlib import Path
export,evaluation_path,prompts_path,converted_hashes,result_path=map(Path,sys.argv[1:])
terminal=json.loads((export/"terminal_policy_export.json").read_bytes()); evaluation=json.loads(evaluation_path.read_bytes())["evaluation_data"]; prompt=json.loads(prompts_path.read_bytes())
assert len(evaluation)==len(prompt["row_sha256"])==1319 and len(set(prompt["row_sha256"]))==1319
assert all(float(row["reward"]) in (0.0,1.0) for row in evaluation)
converted_lines=converted_hashes.read_text().splitlines(); assert converted_lines
value={{"schema":"m4-oars-llama-evaluation-preflight-result-v1","status":"PASS","scientific_endpoint":False,"source_commit":"{SOURCE_COMMIT}","model":"meta-llama/Llama-3.2-1B-Instruct","optimizer_initialized":False,"optimizer_steps":0,"trainer_steps":0,"learner_version_advances":0,"training_started":False,"qualification_started":False,"pilot_started":False,"scientific_acquisition_started":False,"terminal_export":terminal,"hf_reload_passed":True,"evaluation_prompt_count":1319,"binary_reward_serialization_passed":True,"prompt_manifest_sha256":hashlib.sha256(prompts_path.read_bytes()).hexdigest(),"evaluation_data_sha256":hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),"converted_hf_file_count":len(converted_lines),"converted_hf_hash_manifest_sha256":hashlib.sha256(converted_hashes.read_bytes()).hexdigest(),"automatic_retry":False,"automatic_replacement":False,"automatic_extension":False}}
result_path.write_text(json.dumps(value,sort_keys=True,separators=(",",":"))+"\\n")
print("M4_OARS_LLAMA_EVALUATION_PREFLIGHT_PASS")
PY
cp "$EXPORT/terminal_policy_export.json" "$ASSETS/terminal-policy-export.json"
cp "$EXPORT/resolved_config.json" "$ASSETS/terminal-resolved-config.json"
rm -rf "$EXPORT" "$HF_MODEL"
echo M4_OARS_LLAMA_EVALUATION_PREFLIGHT_CAPTURE_COMPLETE_NO_SCIENTIFIC_ENDPOINT
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
            "name": "m4-oars-llama-evaluation-no-training-preflight",
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
