#!/usr/bin/env python3
"""Build one source-bound, deliberately non-launchable 3B preflight candidate."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path


IMAGE = "/lustre/fsw/coreai_dlalgo_ci/nvegesna/nemo_rl_images/nemo-rl-nightly-5802754.sqsh"
IMAGE_COMMIT = "ae07eafe8035b5b2e84efa7234e70e7fd7e493c1"
SOURCE_COMMIT = "88424bdb7bfb526e541b4a7b07830bc31028e64f"
SOURCE_SHA = "1ae875a63786d687fcf46bd7fa7cb2f26b1031c3cf3c36e38114305c732820eb"
MEGATRON_SHA = "98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
PROTOCOL_SHA = "4f2a5e496030e836b3449697928f4f363d7d36e3cec668105b2cdcc701b9814c"
AUTH_SHA = "d29a823bf4d7b34e1a550112632bf0016b6c6a0511a093c26da8c2afa7567ff7"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def script(source: Path, megatron: Path, protocol: Path, authorization: Path) -> str:
    value = f'''set -euo pipefail
readonly PYTHON=/opt/nemo_rl_venv/bin/python
readonly RUN_REPO=/workspace/m4-llama3b-preflight-repo
readonly SOURCE=/workspace/m4-llama3b-source.tar.gz
readonly MEGATRON=/workspace/m4-llama3b-megatron.tar.gz
readonly PROTOCOL=/workspace/m4-llama3b-protocol.json
readonly AUTH=/workspace/m4-llama3b-local-authorization.json
readonly HF_CACHE=/workspace/m4-llama3b-preflight-hf-cache
readonly RESULT={{assets_dir}}/m4-llama3b-no-training-preflight-result.json
readonly HASHES={{assets_dir}}/m4-llama3b-no-training-preflight-hashes.sha256
test "${{NEMO_RL_COMMIT:-unknown}}" = "{IMAGE_COMMIT}"
test ! -e "$RUN_REPO"
mkdir -p "$RUN_REPO" "$HF_CACHE"
printf %s '{payload(source)}' | base64 -d > "$SOURCE"
printf %s '{payload(megatron)}' | base64 -d > "$MEGATRON"
printf %s '{payload(protocol)}' | base64 -d > "$PROTOCOL"
printf %s '{payload(authorization)}' | base64 -d > "$AUTH"
test "$(sha256sum "$SOURCE" | cut -d' ' -f1)" = "{SOURCE_SHA}"
test "$(sha256sum "$MEGATRON" | cut -d' ' -f1)" = "{MEGATRON_SHA}"
test "$(sha256sum "$PROTOCOL" | cut -d' ' -f1)" = "{PROTOCOL_SHA}"
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
print("M4_LLAMA3B_SAFE_SOURCE_PASS",len(names),len(symlinks))
PY
"$PYTHON" - "$MEGATRON" "$RUN_REPO" <<'PY'
import sys,tarfile
from pathlib import Path,PurePosixPath
archive,root=map(Path,sys.argv[1:]); dest=root/"3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty"
with tarfile.open(archive,"r:gz") as source:
 members=source.getmembers(); assert len(members)==726
 assert all(not PurePosixPath(m.name).is_absolute() and ".." not in PurePosixPath(m.name).parts for m in members)
 assert not any(m.issym() or m.islnk() for m in members)
 dest.mkdir(parents=True,exist_ok=True); source.extractall(dest)
assert (dest/"Megatron-LM/megatron/core/__init__.py").is_file()
print("M4_LLAMA3B_MEGATRON_PASS")
PY
cd "$RUN_REPO"
export PYTHONPATH="$RUN_REPO"
export HF_HOME="$HF_CACHE"
export CUDA_VISIBLE_DEVICES=0,1
"$PYTHON" - "$PROTOCOL" "$AUTH" <<'PY'
import json,sys
from pathlib import Path
protocol,auth=[json.loads(Path(p).read_bytes()) for p in sys.argv[1:]]
assert protocol["schema"]=="m4-llama3p2-3b-within-family-size-extension-v1"
assert protocol["status"]=="FROZEN_LOCAL_DESIGN_NO_LAUNCH_AUTHORITY"
assert protocol["model"]["extension"]=="meta-llama/Llama-3.2-3B-Instruct"
assert not protocol["no_training_preflight"]["authorized"]
assert not protocol["qualification"]["authorized"]
assert not protocol["acquisition"]["authorized"]
assert auth["source_commit"]=="{SOURCE_COMMIT}"
assert auth["source_archive_sha256"]=="{SOURCE_SHA}"
assert auth["protocol_sha256"]=="{PROTOCOL_SHA}"
assert auth["local_package_build_authorized"] is True
assert not any(auth[k] for k in ("eos_submission_authorized","model_weight_download_authorized","training_authorized","qualification_authorized","acquisition_authorized","automatic_retry","automatic_extension"))
if not auth["eos_submission_authorized"]:
 raise SystemExit("M4_LLAMA3B_LOCAL_CANDIDATE_NO_LAUNCH_AUTHORITY")
PY
"$PYTHON" - <<'PY'
from omegaconf import OmegaConf
from nemo_rl.algorithms.single_controller_utils.config import MasterConfig,validate_single_controller_config
from nemo_rl.utils.config import load_config,register_omegaconf_resolvers
configs=(
 "examples/configs/grpo_math_3B_megatron_single_controller_m4_llama3p2_3b_openmath_neutral_qualification_v1.yaml",
 "examples/configs/grpo_math_3B_megatron_single_controller_m4_llama3p2_3b_gsm8k_neutral_qualification_v1.yaml",
)
register_omegaconf_resolvers()
for path in configs:
 c=OmegaConf.to_container(load_config(path),resolve=True)
 assert c["grpo"]["max_num_steps"]==64
 assert c["policy"]["model_name"]=="meta-llama/Llama-3.2-3B-Instruct"
 assert c["policy"]["tokenizer"]["name"]=="meta-llama/Llama-3.2-3B-Instruct"
 assert c["cluster"]["num_nodes"]==1 and c["cluster"]["gpus_per_node"]==2
 assert c["async_rl"]["controlled_release_delay"]["arms"]==[{{"label":"neutral","delay_seconds":0.0,"mass":1}}]
 assert c["async_rl"]["gradient_opportunity_audit"]["enabled"] is False
 assert c["async_rl"]["lifecycle_derived_opportunity_audit"]["enabled"] is True
 validate_single_controller_config(MasterConfig(**c))
print("M4_LLAMA3B_CONFIG_RESOLUTION_PASS")
PY
"$PYTHON" - "$HF_CACHE" "$RESULT" <<'PY'
import json,sys
from pathlib import Path
from transformers import AutoConfig,AutoTokenizer
cache,result=map(Path,sys.argv[1:]); model="meta-llama/Llama-3.2-3B-Instruct"
config=AutoConfig.from_pretrained(model)
tokenizer=AutoTokenizer.from_pretrained(model)
for path in cache.rglob("*"):
 if path.is_file():
  assert not path.name.endswith((".safetensors",".bin",".pt",".ckpt")),path
value={{
 "schema":"m4-llama3p2-3b-no-training-preflight-result-v1",
 "source_commit":"{SOURCE_COMMIT}",
 "source_archive_sha256":"{SOURCE_SHA}",
 "protocol_sha256":"{PROTOCOL_SHA}",
 "model":model,
 "model_type":config.model_type,
 "tokenizer_class":tokenizer.__class__.__name__,
 "config_resolution_passed":True,
 "model_metadata_access_passed":True,
 "model_weights_downloaded":False,
 "trainer_constructed":False,
 "trainer_steps_started":0,
 "training_started":False,
 "qualification_started":False,
 "acquisition_started":False,
 "automatic_retry":False,
 "automatic_extension":False
}}
result.write_text(json.dumps(value,sort_keys=True,separators=(",",":"))+"\\n")
print("M4_LLAMA3B_NO_TRAINING_PREFLIGHT_GREEN")
PY
sha256sum "$SOURCE" "$MEGATRON" "$PROTOCOL" "$AUTH" "$RESULT" > "$HASHES"
'''
    token = "__JET_ASSETS_DIR__"
    value = value.replace("{assets_dir}", token)
    value = value.replace("{", "{{").replace("}", "}}")
    return value.replace(token, "{assets_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.name.endswith("-local-candidate.json"):
        raise RuntimeError("builder may emit only a non-launchable local candidate")
    for path, expected in (
        (args.source, SOURCE_SHA),
        (args.megatron, MEGATRON_SHA),
        (args.protocol, PROTOCOL_SHA),
        (args.authorization, AUTH_SHA),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen input moved: {path.name}")
    manifest = {
        "type": "basic",
        "format_version": 1,
        "maintainers": ["nvegesna"],
        "loggers": ["stdout"],
        "labels": {"target": "silicon"},
        "launchers": {"type:slurm": {"nodes": 1, "ntasks_per_node": 1}},
        "spec": {
            "name": "m4-llama3b-no-training-preflight-local-candidate-do-not-submit",
            "workspace": "/workspace",
            "nodes": 1,
            "time_limit": 900,
            "image_source": {"local_path": IMAGE},
            "script": script(args.source, args.megatron, args.protocol, args.authorization),
        },
    }
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(args.output, sha256(args.output))


if __name__ == "__main__":
    main()
