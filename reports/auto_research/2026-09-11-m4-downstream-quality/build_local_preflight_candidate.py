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
"""Build a fail-closed downstream-quality no-training preflight candidate."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

SOURCE_COMMIT = "1941728f8a636d82126dec8632277cd63172e3b7"
SOURCE_SHA256 = "80741c66b32a7951e4ebd620e66a8043858159e7a871c7fb87d038db068aa714"
MEGATRON_SHA256 = "98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
PROTOCOL_SHA256 = "5ec319043f6e3be90e8e294946eee3596b4c8ab445630de93ae103bbcabb7162"
PLAN_SHA256 = "b160c6610226bf2b2250cbc20500667776bfb1e62df0918994c4221a690918fa"
AUTHORIZATION_SHA256 = (
    "d34ec018f5d75711bcab3efcf4225f0d4a57b75fa338cfcd050b79b821d3c7c2"
)
ANCHOR_CONFIG = (
    "examples/configs/"
    "grpo_math_1B_megatron_single_controller_m4_opportunity_loss_followup.yaml"
)
ANCHOR_CONFIG_SHA256 = (
    "e0c62bd698ffd744c242535a12817e5adf4d893c7efa4dc79af305a2ad70b5e8"
)
IMAGE_PATH = (
    "/lustre/fsw/coreai_dlalgo_ci/nvegesna/nemo_rl_images/nemo-rl-nightly-5802754.sqsh"
)
IMAGE_COMMIT = "ae07eafe8035b5b2e84efa7234e70e7fd7e493c1"


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload(path: Path) -> str:
    """Return a file as a base64 ASCII payload."""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    """Build one JSON-formatted JET workload manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path, expected in (
        (args.source, SOURCE_SHA256),
        (args.megatron, MEGATRON_SHA256),
        (args.protocol, PROTOCOL_SHA256),
        (args.plan, PLAN_SHA256),
        (args.authorization, AUTHORIZATION_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen preflight input moved: {path.name}")

    authorization = json.loads(args.authorization.read_bytes())
    if (
        authorization["local_package_build_authorized"] is not True
        or authorization["clean_room_validation_authorized"] is not True
        or authorization["optimizer_steps_allowed"] != 0
        or any(
            authorization[key]
            for key in (
                "eos_submission_authorized",
                "model_weight_access_authorized",
                "optimizer_initialization_authorized",
                "training_authorized",
                "qualification_authorized",
                "pilot_authorized",
                "scientific_acquisition_authorized",
                "automatic_retry",
                "automatic_extension",
            )
        )
    ):
        raise RuntimeError("local preflight authority differs")

    script = f'''set -euo pipefail
readonly PYTHON=/opt/nemo_rl_venv/bin/python
readonly RUN_REPO=/workspace/m4-downstream-quality-preflight-repo
readonly SOURCE=/workspace/m4-downstream-quality-source.tar.gz
readonly MEGATRON=/workspace/m4-downstream-quality-megatron.tar.gz
readonly PROTOCOL=/workspace/m4-downstream-quality-protocol.json
readonly PLAN=/workspace/m4-downstream-quality-preflight-plan.md
readonly AUTH=/workspace/m4-downstream-quality-preflight-authorization.json
readonly EXPORT=/workspace/m4-downstream-quality-initial-export
readonly HF_MODEL=/workspace/m4-downstream-quality-initial-hf
readonly EVAL_OUTPUT={{assets_dir}}/base-evaluation
readonly RESULT={{assets_dir}}/m4-downstream-quality-no-training-preflight-result.json
readonly PROMPTS={{assets_dir}}/m4-downstream-quality-prompt-manifest.json
readonly HASHES={{assets_dir}}/m4-downstream-quality-no-training-preflight.sha256
test "${{NEMO_RL_COMMIT:-unknown}}" = "{IMAGE_COMMIT}"
test ! -e "$RUN_REPO"
test ! -e "$EXPORT"
test ! -e "$HF_MODEL"
printf %s '{payload(args.source)}' | base64 -d > "$SOURCE"
printf %s '{payload(args.megatron)}' | base64 -d > "$MEGATRON"
printf %s '{payload(args.protocol)}' | base64 -d > "$PROTOCOL"
printf %s '{payload(args.plan)}' | base64 -d > "$PLAN"
printf %s '{payload(args.authorization)}' | base64 -d > "$AUTH"
test "$(sha256sum "$SOURCE" | cut -d' ' -f1)" = "{SOURCE_SHA256}"
test "$(sha256sum "$MEGATRON" | cut -d' ' -f1)" = "{MEGATRON_SHA256}"
test "$(sha256sum "$PROTOCOL" | cut -d' ' -f1)" = "{PROTOCOL_SHA256}"
test "$(sha256sum "$PLAN" | cut -d' ' -f1)" = "{PLAN_SHA256}"
test "$(sha256sum "$AUTH" | cut -d' ' -f1)" = "{AUTHORIZATION_SHA256}"
"$PYTHON" - "$AUTH" <<'PY'
import json,sys
from pathlib import Path
a=json.loads(Path(sys.argv[1]).read_bytes())
assert a["schema"]=="m4-downstream-quality-local-preflight-package-authorization-v1"
assert a["source_commit"]=="{SOURCE_COMMIT}"
assert a["source_archive_sha256"]=="{SOURCE_SHA256}"
assert a["megatron_dependency_sha256"]=="{MEGATRON_SHA256}"
assert a["optimizer_steps_allowed"]==0
assert a["local_package_build_authorized"] is True
assert a["clean_room_validation_authorized"] is True
blocked=("eos_submission_authorized","model_weight_access_authorized","optimizer_initialization_authorized","training_authorized","qualification_authorized","pilot_authorized","scientific_acquisition_authorized","automatic_retry","automatic_extension")
assert not any(a[key] for key in blocked)
print("M4_DOWNSTREAM_QUALITY_LOCAL_PREFLIGHT_AUTHORITY_PASS")
if not a["eos_submission_authorized"]:
 raise SystemExit("M4_DOWNSTREAM_QUALITY_LOCAL_CANDIDATE_NO_EOS_AUTHORITY")
PY
mkdir -p "$RUN_REPO" "$EVAL_OUTPUT"
"$PYTHON" - "$SOURCE" "$RUN_REPO" <<'PY'
import sys,tarfile
from pathlib import Path,PurePosixPath
archive,output=map(Path,sys.argv[1:])
with tarfile.open(archive,"r:gz") as source:
 members=source.getmembers(); names=[PurePosixPath(m.name) for m in members]
 assert len(names)==len(set(names))
 assert not any(n.is_absolute() or ".." in n.parts for n in names)
 symlinks={{n for n,m in zip(names,members,strict=True) if m.issym()}}
 assert not any(any(parent in symlinks for parent in n.parents) for n in names)
 source.extractall(output)
print("M4_DOWNSTREAM_QUALITY_SAFE_SOURCE_PASS",len(names),len(symlinks))
PY
"$PYTHON" - "$MEGATRON" "$RUN_REPO" <<'PY'
import sys,tarfile
from pathlib import Path,PurePosixPath
archive,root=map(Path,sys.argv[1:]); dest=root/"3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty"
with tarfile.open(archive,"r:gz") as source:
 members=source.getmembers(); names=[PurePosixPath(m.name) for m in members]
 assert len(members)==726
 assert len(names)==len(set(names))
 assert not any(n.is_absolute() or ".." in n.parts for n in names)
 dest.mkdir(parents=True,exist_ok=True); source.extractall(dest)
assert (dest/"Megatron-LM/megatron/core/__init__.py").is_file()
print("M4_DOWNSTREAM_QUALITY_MEGATRON_DEPENDENCY_PASS")
PY
cd "$RUN_REPO"
export PYTHONPATH="$RUN_REPO"
export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME=/workspace/m4-downstream-quality-hf-cache
"$PYTHON" - "$PROTOCOL" "$PLAN" <<'PY'
import hashlib,json,sys
from pathlib import Path
protocol_path,plan_path=map(Path,sys.argv[1:])
p=json.loads(protocol_path.read_bytes())
assert p["schema"]=="m4-downstream-quality-protocol-draft-v1"
assert p["status"]=="DRAFT_LOCAL_NO_LAUNCH_AUTHORITY"
assert p["anchor"]["model"]=="Qwen/Qwen3-0.6B"
assert p["anchor"]["workload"]=="OpenMathInstruct-2"
assert p["dataset"]["split_seed"]==20260911
assert p["dataset"]["candidate_evaluation_prompts"]==1024
assert p["evaluation"]["interim_validation_traffic"] is False
assert p["artifact_contract"]["conversion_to_huggingface_required_before_vllm_evaluation"] is True
assert p["automatic_retry"] is False and p["automatic_extension"] is False
assert hashlib.sha256(plan_path.read_bytes()).hexdigest()=="{PLAN_SHA256}"
print("M4_DOWNSTREAM_QUALITY_PROTOCOL_PASS")
PY
"$PYTHON" - "$EXPORT" <<'PY'
import sys
from pathlib import Path
import ray
from omegaconf import OmegaConf
from nemo_rl.algorithms.single_controller_utils.config import MasterConfig
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster,init_ray
from nemo_rl.models.policy.tq_policy import TQPolicy
from nemo_rl.utils.config import load_config
from nemo_rl.utils.terminal_policy_export import export_terminal_policy
export_dir=Path(sys.argv[1])
config_path=Path("{ANCHOR_CONFIG}")
assert __import__("hashlib").sha256(config_path.read_bytes()).hexdigest()=="{ANCHOR_CONFIG_SHA256}"
raw=OmegaConf.to_container(load_config(config_path),resolve=True)
raw["grpo"]["max_num_steps"]=0
raw["data"]["train"]["seed"]=20260911
raw["policy"]["megatron_cfg"]["train_iters"]=1
config=MasterConfig(**raw)
assert config.checkpointing["enabled"] is False
assert config.async_rl.terminal_policy_export.enabled is False
init_ray()
tokenizer=get_tokenizer(config.policy["tokenizer"])
cluster=RayVirtualCluster(name="m4_downstream_quality_export",bundle_ct_per_node_list=[1],use_gpus=True,num_gpus_per_node=1,max_colocated_worker_groups=1)
policy=TQPolicy(cluster=cluster,config=config.policy,tokenizer=tokenizer,processor=None,weights_path=None,optimizer_path=None,init_optimizer=False,init_reference_model=False,dp_cfg=config.data_plane)
try:
 manifest=export_terminal_policy(trainer=policy,master_config=config,output_dir=str(export_dir),train_steps=0,trainer_version=0)
finally:
 policy.shutdown(); ray.shutdown()
assert manifest["train_steps"]==0 and manifest["trainer_version"]==0
assert manifest["optimizer_exported"] is False
assert manifest["resumable_training_checkpoint"] is False
print("M4_DOWNSTREAM_QUALITY_ZERO_STEP_EXPORT_PASS")
PY
readonly ITER=$("$PYTHON" - "$EXPORT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); manifest=json.loads((root/"terminal_policy_export.json").read_bytes())
assert manifest["train_steps"]==0 and manifest["optimizer_exported"] is False
weights=root/manifest["weights_path"]
iters=[path for path in weights.glob("iter_*") if (path/"run_config.yaml").is_file()]
assert len(iters)==1,iters
print(iters[0])
PY
)
"$PYTHON" examples/converters/convert_megatron_to_hf.py --config "$EXPORT/resolved_config.json" --hf-model-name Qwen/Qwen3-0.6B --megatron-ckpt-path "$ITER" --hf-ckpt-path "$HF_MODEL"
"$PYTHON" - "$HF_MODEL" "$EVAL_OUTPUT" "$PROMPTS" <<'PY'
import hashlib,json,sys
from pathlib import Path
from torch.utils.data import Subset
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_response_data
from nemo_rl.evals.eval import MasterConfig as EvalMasterConfig,run_env_eval,setup
from nemo_rl.models.generation import configure_generation_config
hf_model,eval_output,prompt_path=map(Path,sys.argv[1:])
tokenizer_cfg={{"name":str(hf_model),"chat_template":"default","chat_template_kwargs":None,"use_fastokens":False}}
tokenizer=get_tokenizer(tokenizer_cfg)
data={{"max_input_seq_length":2048,"shuffle":False,"num_workers":1,"use_multiple_dataloader":False,"train":{{"dataset_name":"OpenMathInstruct-2","split_validation_size":0.05,"seed":20260911}},"validation":None,"default":{{"prompt_file":"examples/prompts/cot.txt","system_prompt_file":None,"processor":"math_hf_data_processor","env_name":"math"}}}}
_,val_dataset,_,val_envs=setup_response_data(tokenizer,data,{{"math":{{"num_workers":8,"math_verify_impl":"hf_math_verify"}}}})
assert val_dataset is not None and len(val_dataset)>=1024
raw_rows=[val_dataset.dataset[index] for index in range(1024)]
row_hashes=[hashlib.sha256(json.dumps(row,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest() for row in raw_rows]
prompt_manifest={{"schema":"m4-downstream-quality-openmath-prompt-manifest-v1","dataset":"OpenMathInstruct-2","split_fraction":0.05,"split_seed":20260911,"selection":"first_1024_after_hf_train_test_split","count":1024,"row_sha256":row_hashes}}
prompt_path.write_text(json.dumps(prompt_manifest,sort_keys=True,separators=(",",":"))+"\\n")
generation={{"backend":"vllm","model_name":str(hf_model),"max_new_tokens":1792,"temperature":0.0,"top_p":1.0,"top_k":-1,"val_temperature":0.0,"val_top_p":1.0,"val_top_k":-1,"num_prompts_per_step":32,"stop_token_ids":None,"stop_strings":None,"vllm_cfg":{{"async_engine":False,"precision":"bfloat16","tensor_parallel_size":1,"pipeline_parallel_size":1,"expert_parallel_size":1,"gpu_memory_utilization":0.9,"max_model_len":2048,"enforce_eager":False}},"colocated":{{"enabled":True,"resources":{{"gpus_per_node":None,"num_nodes":None}}}}}}
generation=configure_generation_config(generation,tokenizer,is_eval=True)
master=EvalMasterConfig.model_construct(eval={{"metric":"pass@k","num_tests_per_prompt":1,"seed":20260911,"k_value":1,"save_path":str(eval_output)}},generation=generation,tokenizer=tokenizer_cfg,data=data,env={{"math":{{"num_workers":8,"math_verify_impl":"hf_math_verify"}}}},cluster={{"gpus_per_node":1,"num_nodes":1}})
subset=Subset(val_dataset,range(1024))
environment=next(iter(val_envs.values()))
vllm,dataloader,master=setup(master,tokenizer,subset)
run_env_eval(vllm,dataloader,environment,master)
print("M4_DOWNSTREAM_QUALITY_FIXED_EVALUATION_PASS")
PY
"$PYTHON" - "$EXPORT" "$HF_MODEL" "$EVAL_OUTPUT" "$PROMPTS" "$RESULT" <<'PY'
import hashlib,json,sys
from pathlib import Path
export,hf_model,eval_output,prompts,result=map(Path,sys.argv[1:])
rows=json.loads((eval_output/"evaluation_data.json").read_bytes())["evaluation_data"]
assert len(rows)==1024 and all(row["reward"] in (0.0,1.0) for row in rows)
manifest=json.loads((export/"terminal_policy_export.json").read_bytes())
value={{"schema":"m4-downstream-quality-no-training-preflight-result-v1","status":"PASS","source_commit":"{SOURCE_COMMIT}","model":"Qwen/Qwen3-0.6B","optimizer_initialized":False,"optimizer_steps":0,"trainer_steps":0,"learner_version_advances":0,"training_started":False,"qualification_started":False,"pilot_started":False,"acquisition_started":False,"terminal_export":manifest,"hf_reload_passed":True,"evaluation_prompt_count":1024,"base_accuracy":sum(row["reward"] for row in rows)/len(rows),"prompt_manifest_sha256":hashlib.sha256(prompts.read_bytes()).hexdigest(),"evaluation_data_sha256":hashlib.sha256((eval_output/"evaluation_data.json").read_bytes()).hexdigest(),"automatic_retry":False,"automatic_extension":False}}
result.write_text(json.dumps(value,sort_keys=True,separators=(",",":"))+"\\n")
print("M4_DOWNSTREAM_QUALITY_NO_TRAINING_PREFLIGHT_PASS")
PY
sha256sum "$RESULT" "$PROMPTS" "$EVAL_OUTPUT/evaluation_data.json" "$EXPORT/terminal_policy_export.json" "$EXPORT/resolved_config.json" > "$HASHES"
'''
    script = (
        script.replace("{", "{{")
        .replace("}", "}}")
        .replace("{{assets_dir}}", "{assets_dir}")
    )

    template = json.loads(args.template.read_bytes())
    template["spec"]["name"] = "m4-downstream-quality-no-training-preflight-local"
    template["spec"]["time_limit"] = 7200
    template["spec"]["image_source"] = {"local_path": IMAGE_PATH}
    template["spec"]["script"] = script
    args.output.write_text(
        json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output, sha256(args.output))


if __name__ == "__main__":
    main()
