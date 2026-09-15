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

"""Build one credential-free FIFO or OARS actuation qualification manifest."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import tarfile
from pathlib import Path

PROTOCOL_SHA256 = "3bc49412a212419635e55b616e080708e17cf265c1926c1d93faebf072a87bfd"
AUTHORIZATION_SHA256 = "e8600a27f9391d44262ba4606120085b2c52ca6cb8d563e9da85d1f585d90596"
MEGATRON_SHA256 = "98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
ANALYZER_SHA256 = "19dbdad60bd1352eb2430b00cbe763663194f97a8bce8333af382a676c3e5d92"
IMAGE_PATH = (
    "/lustre/fsw/coreai_dlalgo_ci/nvegesna/nemo_rl_images/"
    "nemo-rl-nightly-5802754.sqsh"
)
IMAGE_COMMIT = "ae07eafe8035b5b2e84efa7234e70e7fd7e493c1"
ANALYZER_PATH = "tools/m4_oars_actuation_qualification.py"
ARM_CONFIG = {
    "fifo": (
        "observe",
        "examples/configs/grpo_math_1B_megatron_single_controller_"
        "m4_oars_qualification_fifo.yaml",
        "17ad965489aa9595d266295125880fe55edb4af303bd95aed66a61034c943c9b",
    ),
    "act": (
        "act",
        "examples/configs/grpo_math_1B_megatron_single_controller_"
        "m4_oars_qualification_act.yaml",
        "a1fe0b700955e53845794bcf3011fcb992a7f10b881b03d325c2b3385e502ca0",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARM_CONFIG), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mode, config_path, config_sha256 = ARM_CONFIG[args.arm]
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        raise RuntimeError("source commit must be a full lowercase SHA-1")
    for path, expected in (
        (args.megatron, MEGATRON_SHA256),
        (args.protocol, PROTOCOL_SHA256),
        (args.authorization, AUTHORIZATION_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen package input moved: {path.name}")
    authorization = json.loads(args.authorization.read_bytes())
    if (
        authorization["schema"]
        != "m4-oars-actuation-qualification-authorization-v1"
        or authorization["protocol_sha256"] != PROTOCOL_SHA256
        or authorization["source_commit"] != args.source_commit
        or authorization["required_launcher"] != "runllm.py --no_wait"
        or authorization["submission_attempt_limit"] != 2
        or authorization["submission_order"] != ["fifo", "act"]
        or authorization["trainer_steps_per_arm"] != 64
        or authorization["gpus_per_arm"] != 2
        or authorization["scheduler_time_limit_seconds"] != 14400
        or authorization["queue_deadline_override"] is not None
        or not authorization["eos_submission_authorized"]
        or not authorization["systems_qualification_authorized"]
        or not authorization["oars_actuation_authorized_for_qualification_only"]
        or authorization["scientific_outcome_acquisition_authorized"]
        or authorization["automatic_retry"]
        or authorization["automatic_extension"]
    ):
        raise RuntimeError("qualification authority differs")
    source_sha256 = sha256(args.source)
    with tarfile.open(args.source, "r:gz") as archive:
        names = set(archive.getnames())
    if config_path not in names or ANALYZER_PATH not in names:
        raise RuntimeError("source archive lacks qualification runtime files")

    name = f"m4-oars-qualification-{args.arm}"
    script = f'''set -euo pipefail
readonly PYTHON=/opt/nemo_rl_venv/bin/python
readonly RUN_REPO=/workspace/{name}-repo
readonly SOURCE=/workspace/{name}-source.tar.gz
readonly MEGATRON=/workspace/{name}-megatron.tar.gz
readonly PROTOCOL=/workspace/{name}-protocol.json
readonly AUTH=/workspace/{name}-authorization.json
readonly ASSETS={{assets_dir}}
readonly RUN_LOG=$ASSETS/{name}-run.log
readonly LIFECYCLE=$ASSETS/{name}-lifecycle.jsonl
readonly OPPORTUNITY=$ASSETS/{name}-opportunity.jsonl
readonly DUTY=$ASSETS/{name}-gradient-observer-duty.json
readonly OARS=$ASSETS/{name}-oars.jsonl
readonly RESULT=$ASSETS/{name}-result.json
readonly HASHES=$ASSETS/{name}-artifacts.sha256
readonly METRICS=$ASSETS/{name}-metrics
test "${{NEMO_RL_COMMIT:-unknown}}" = "{IMAGE_COMMIT}"
test ! -e "$RUN_REPO"
printf %s '{payload(args.source)}' | base64 -d > "$SOURCE"
printf %s '{payload(args.megatron)}' | base64 -d > "$MEGATRON"
printf %s '{payload(args.protocol)}' | base64 -d > "$PROTOCOL"
printf %s '{payload(args.authorization)}' | base64 -d > "$AUTH"
test "$(sha256sum "$SOURCE" | cut -d' ' -f1)" = "{source_sha256}"
test "$(sha256sum "$MEGATRON" | cut -d' ' -f1)" = "{MEGATRON_SHA256}"
test "$(sha256sum "$PROTOCOL" | cut -d' ' -f1)" = "{PROTOCOL_SHA256}"
test "$(sha256sum "$AUTH" | cut -d' ' -f1)" = "{AUTHORIZATION_SHA256}"
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
print("M4_OARS_SAFE_SOURCE_PASS",len(names),len(symlinks))
PY
"$PYTHON" - "$MEGATRON" "$RUN_REPO" <<'PY'
import sys,tarfile
from pathlib import Path,PurePosixPath
archive,root=map(Path,sys.argv[1:]); dest=root/"3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty"
with tarfile.open(archive,"r:gz") as source:
 members=source.getmembers(); names=[PurePosixPath(m.name) for m in members]
 assert len(members)==726 and len(names)==len(set(names))
 assert not any(n.is_absolute() or ".." in n.parts for n in names)
 dest.mkdir(parents=True,exist_ok=True); source.extractall(dest)
assert (dest/"Megatron-LM/megatron/core/__init__.py").is_file()
print("M4_OARS_MEGATRON_DEPENDENCY_PASS")
PY
cd "$RUN_REPO"
export PYTHONPATH="$RUN_REPO"
export CUDA_VISIBLE_DEVICES=0,1
mkdir -p "$METRICS"
"$PYTHON" - "$PROTOCOL" "$AUTH" <<'PY'
import hashlib,json,sys
from pathlib import Path
from omegaconf import OmegaConf
from nemo_rl.algorithms.single_controller_utils.config import MasterConfig,validate_single_controller_config
from nemo_rl.utils.config import load_config,register_omegaconf_resolvers
p_path,a_path=map(Path,sys.argv[1:]); p=json.loads(p_path.read_bytes()); a=json.loads(a_path.read_bytes())
assert hashlib.sha256(p_path.read_bytes()).hexdigest()=="{PROTOCOL_SHA256}"
assert hashlib.sha256(a_path.read_bytes()).hexdigest()=="{AUTHORIZATION_SHA256}"
assert p["analysis_state"]=="frozen_before_actuation_qualification"
assert a["source_commit"]=="{args.source_commit}" and a["pair_training_seed"]==20261421
config_path=Path("{config_path}"); analyzer_path=Path("{ANALYZER_PATH}")
assert hashlib.sha256(config_path.read_bytes()).hexdigest()=="{config_sha256}"
assert hashlib.sha256(analyzer_path.read_bytes()).hexdigest()=="{ANALYZER_SHA256}"
register_omegaconf_resolvers(); raw=OmegaConf.to_container(load_config(config_path),resolve=True)
config=MasterConfig(**raw); validate_single_controller_config(config)
assert config.grpo.max_num_steps==64 and config.grpo.num_prompts_per_step==4
assert config.grpo.seed==20261421 and config.cluster["num_nodes"]==1 and config.cluster["gpus_per_node"]==2
assert config.async_rl.sampler.name=="weight_fifo" and config.async_rl.sampler.selection_candidate_watermark==8
assert config.async_rl.opportunity_at_risk_shadow.enabled and config.async_rl.opportunity_at_risk_shadow.mode=="{mode}"
assert config.async_rl.opportunity_at_risk_shadow.service_budget_multiplier==1.02
assert not config.async_rl.lifecycle_derived_opportunity_audit.enabled
print("M4_OARS_QUALIFICATION_RUNTIME_AUTHORITY_AND_CONFIG_PASS")
PY
readonly START_NS=$("$PYTHON" -c 'import time; print(time.time_ns())')
set +e
"$PYTHON" examples/run_grpo_single_controller.py --config "{config_path}" \
  "logger.log_dir=$METRICS" \
  "async_rl.lifecycle_audit_path=$LIFECYCLE" \
  "async_rl.gradient_opportunity_audit.output_path=$OPPORTUNITY" \
  "async_rl.gradient_opportunity_audit.observer_duty_path=$DUTY" \
  "async_rl.opportunity_at_risk_shadow.output_path=$OARS" 2>&1 | tee "$RUN_LOG"
readonly TRAIN_RC=${{PIPESTATUS[0]}}
set -e
readonly END_NS=$("$PYTHON" -c 'import time; print(time.time_ns())')
if test "$TRAIN_RC" -eq 0; then
  set +e
  "$PYTHON" "{ANALYZER_PATH}" --mode "{mode}" --oars "$OARS" --lifecycle "$LIFECYCLE" --observer-duty "$DUTY" --source-commit "{args.source_commit}" --run-start-ns "$START_NS" --run-end-ns "$END_NS" --output "$RESULT"
  readonly GATE_RC=$?
  set -e
else
  printf '{{"mode":"{mode}","schema":"m4-oars-actuation-qualification-arm-result-v1","scientific_outcome_acquisition":false,"status":"TRAINING_FAILURE","training_exit_code":%s,"training_quality_analyzed":false}}\n' "$TRAIN_RC" > "$RESULT"
  readonly GATE_RC=$TRAIN_RC
fi
find "$ASSETS" -maxdepth 1 -type f ! -name "$(basename "$HASHES")" -print0 | sort -z | xargs -0 sha256sum > "$HASHES"
exit "$GATE_RC"
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
            "name": name,
            "workspace": "/workspace",
            "nodes": 1,
            "time_limit": 14400,
            "image_source": {"local_path": IMAGE_PATH},
            "script": script,
        },
    }
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(args.output, sha256(args.output), source_sha256)


if __name__ == "__main__":
    main()
