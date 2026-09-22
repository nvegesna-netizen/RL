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

"""Build the authorized credential-free controlled-frontier qualification."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import tarfile
from pathlib import Path

SOURCE_COMMIT = "b4f78633e987dc2c29dccb2e02af0b8e650331f6"
PROTOCOL_SHA256 = "8a900d218866347c4ce5ebd41c85b954badcaf5021a956209bd61fc8fb1fe5d5"
AUTHORIZATION_SHA256 = (
    "efeaba0bc4a8783fefb0f00ff37bbb38ed2b97e50fcf3f967936217a74ecb260"
)
MEGATRON_SHA256 = "98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
CONFIG_SHA256 = "939218f0217fbef784b77ce5c978ff6a8c9cb1fffcabf2ecb2a468892a8a16da"
ANALYZER_SHA256 = "caff44951b2bcb269bba94d78a6ef5e8f5de52232b0e149590f4f16e3641634d"
LOCAL_VERIFICATION_SHA256 = (
    "30dfdee78d5d70ac74719ddafdb28991f5c3e90f35ae9f5d1d02164126d5329b"
)
TRANSFERQUEUE_COMMIT = "b266d39a15aae114730de36cf8317b6285436f7f"
IMAGE_PATH = (
    "/lustre/fsw/coreai_dlalgo_ci/nvegesna/nemo_rl_images/nemo-rl-nightly-5802754.sqsh"
)
IMAGE_COMMIT = "ae07eafe8035b5b2e84efa7234e70e7fd7e493c1"
CONFIG_PATH = (
    "examples/configs/grpo_math_1B_megatron_single_controller_"
    "m4_oars_v2_controlled_frontier.yaml"
)
ANALYZER_PATH = "tools/m4_oars_v2_controlled_frontier_qualification.py"
LOCAL_VERIFICATION_PATH = (
    "reports/auto_research/2026-09-22-m4-oars-v2-controlled-contention/"
    "local_verification.json"
)
ANALYZER_TEST_PATH = (
    "reports/auto_research/2026-09-22-m4-oars-v2-offline/"
    "test_live_shadow_qualification.py"
)
NAME = "m4-oars-v2-controlled-frontier-qualification"


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload(path: Path) -> str:
    """Return one base64-encoded credential-free input."""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    """Validate frozen inputs and write one credential-free JET manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path, expected in (
        (args.megatron, MEGATRON_SHA256),
        (args.protocol, PROTOCOL_SHA256),
        (args.authorization, AUTHORIZATION_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen input moved: {path.name}")
    authorization = json.loads(args.authorization.read_bytes())
    if (
        authorization["schema"]
        != "m4-oars-v2-controlled-frontier-qualification-authorization-v1"
        or authorization["source_commit"] != SOURCE_COMMIT
        or authorization["protocol_sha256"] != PROTOCOL_SHA256
        or authorization["required_launcher"] != "runllm.py --no_wait"
        or authorization["submission_attempt_limit"] != 1
        or authorization["trainer_steps"] != 64
        or authorization["gpus"] != 2
        or authorization["scheduler_time_limit_seconds"] != 14400
        or authorization["queue_deadline_override"] is not None
        or authorization["selection_candidate_watermark"] != 8
        or authorization["selection_cardinality"] != 4
        or not authorization["eos_submission_authorized"]
        or not authorization["controlled_fifo_training_authorized"]
        or not authorization["oars_v2_shadow_observation_authorized"]
        or authorization["oars_v2_actuation_authorized"]
        or authorization["training_quality_analysis_authorized"]
        or authorization["scientific_outcome_acquisition_authorized"]
        or authorization["automatic_retry"]
        or authorization["automatic_extension"]
    ):
        raise RuntimeError("controlled-frontier authority differs")
    source_sha256 = sha256(args.source)
    with tarfile.open(args.source, "r:gz") as archive:
        names = set(archive.getnames())
    required = {
        CONFIG_PATH,
        ANALYZER_PATH,
        LOCAL_VERIFICATION_PATH,
        ANALYZER_TEST_PATH,
        "uv.lock",
        "nemo_rl/algorithms/async_utils/opportunity_at_risk_v2.py",
        "nemo_rl/algorithms/async_utils/staleness_sampler.py",
        "nemo_rl/algorithms/single_controller.py",
        "nemo_rl/algorithms/single_controller_utils/config.py",
        "nemo_rl/data_plane/adapters/transfer_queue.py",
        "tests/unit/single_controller/test_opportunity_at_risk_v2.py",
        "tests/unit/single_controller/test_single_controller.py",
    }
    if not required.issubset(names):
        raise RuntimeError("source archive lacks required runtime or test files")

    script = f'''set -euo pipefail
readonly PYTHON=/opt/nemo_rl_venv/bin/python
readonly RUN_REPO=/workspace/{NAME}-repo
readonly SOURCE=/workspace/{NAME}-source.tar.gz
readonly MEGATRON=/workspace/{NAME}-megatron.tar.gz
readonly PROTOCOL=/workspace/{NAME}-protocol.json
readonly AUTH=/workspace/{NAME}-authorization.json
readonly ASSETS={{assets_dir}}
readonly RUN_LOG=$ASSETS/{NAME}-run.log
readonly LIFECYCLE=$ASSETS/{NAME}-lifecycle.jsonl
readonly OPPORTUNITY=$ASSETS/{NAME}-opportunity.jsonl
readonly DUTY=$ASSETS/{NAME}-observer-duty.json
readonly OARS=$ASSETS/{NAME}-oars-v2.jsonl
readonly RESULT=$ASSETS/{NAME}-result.json
readonly HASHES=$ASSETS/{NAME}-artifacts.sha256
readonly METRICS=$ASSETS/{NAME}-metrics
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
print("M4_OARS_V2_CONTROLLED_SAFE_SOURCE_PASS",len(names),len(symlinks))
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
print("M4_OARS_V2_CONTROLLED_MEGATRON_PASS")
PY
cd "$RUN_REPO"
export PYTHONPATH="$RUN_REPO"
export CUDA_VISIBLE_DEVICES=0,1
mkdir -p "$METRICS"
"$PYTHON" - "$PROTOCOL" "$AUTH" <<'PY'
import hashlib,json,sys,tomllib
from importlib.metadata import distribution
from pathlib import Path
from omegaconf import OmegaConf
from nemo_rl.algorithms.async_utils.opportunity_at_risk_v2 import OpportunityAtRiskV2ShadowRecorder
from nemo_rl.algorithms.single_controller_utils.config import MasterConfig,validate_single_controller_config
from nemo_rl.utils.config import load_config,register_omegaconf_resolvers
p_path,a_path=map(Path,sys.argv[1:]); p=json.loads(p_path.read_bytes()); a=json.loads(a_path.read_bytes())
assert hashlib.sha256(p_path.read_bytes()).hexdigest()=="{PROTOCOL_SHA256}"
assert hashlib.sha256(a_path.read_bytes()).hexdigest()=="{AUTHORIZATION_SHA256}"
assert p["status"]=="FROZEN_BEFORE_IMPLEMENTATION_COMMIT"
assert a["source_commit"]=="{SOURCE_COMMIT}"
config_path=Path("{CONFIG_PATH}"); analyzer_path=Path("{ANALYZER_PATH}"); local_path=Path("{LOCAL_VERIFICATION_PATH}")
assert hashlib.sha256(config_path.read_bytes()).hexdigest()=="{CONFIG_SHA256}"
assert hashlib.sha256(analyzer_path.read_bytes()).hexdigest()=="{ANALYZER_SHA256}"
assert hashlib.sha256(local_path.read_bytes()).hexdigest()=="{LOCAL_VERIFICATION_SHA256}"
assert OpportunityAtRiskV2ShadowRecorder.SCHEMA_VERSION==1
locked=tomllib.loads(Path("uv.lock").read_text())
tq=[x for x in locked["package"] if x["name"].lower()=="transferqueue"]
assert len(tq)==1 and tq[0]["source"]["git"].endswith("#{TRANSFERQUEUE_COMMIT}")
direct=json.loads(distribution("TransferQueue").read_text("direct_url.json"))
assert direct["vcs_info"]["commit_id"]=="{TRANSFERQUEUE_COMMIT}"
register_omegaconf_resolvers(); raw=OmegaConf.to_container(load_config(config_path),resolve=True)
config=MasterConfig(**raw); validate_single_controller_config(config)
assert config.grpo.max_num_steps==64 and config.grpo.num_prompts_per_step==4
assert config.grpo.seed==20262211 and config.cluster["num_nodes"]==1 and config.cluster["gpus_per_node"]==2
assert config.data_plane["actor_runtime_env_mode"]=="inherit_baked_single_node"
assert config.async_rl.sampler.name=="weight_fifo" and config.async_rl.sampler.selection_candidate_watermark==8
assert config.async_rl.opportunity_at_risk_v2_shadow.enabled
assert config.async_rl.opportunity_at_risk_v2_shadow.candidate_window_policy=="controlled_frontier"
assert not config.async_rl.opportunity_at_risk_shadow.enabled
print("M4_OARS_V2_CONTROLLED_RUNTIME_AUTHORITY_CONFIG_AND_TQ_PASS")
PY
"$PYTHON" -m pytest -q \
  tests/unit/single_controller/test_opportunity_at_risk_v2.py \
  tests/unit/single_controller/test_single_controller.py \
  -k opportunity_at_risk_v2
"$PYTHON" -m pytest -q "{ANALYZER_TEST_PATH}"
printenv NEMO_RL_COMMIT >/dev/null
readonly START_NS=$("$PYTHON" -c 'import time; print(time.time_ns())')
set +e
"$PYTHON" examples/run_grpo_single_controller.py --config "{CONFIG_PATH}" \
  "logger.log_dir=$METRICS" \
  "async_rl.lifecycle_audit_path=$LIFECYCLE" \
  "async_rl.gradient_opportunity_audit.output_path=$OPPORTUNITY" \
  "async_rl.gradient_opportunity_audit.observer_duty_path=$DUTY" \
  "async_rl.opportunity_at_risk_v2_shadow.output_path=$OARS" 2>&1 | tee "$RUN_LOG"
readonly TRAIN_RC=${{PIPESTATUS[0]}}
set -e
readonly END_NS=$("$PYTHON" -c 'import time; print(time.time_ns())')
if test "$TRAIN_RC" -eq 0; then
  set +e
  "$PYTHON" "{ANALYZER_PATH}" --oars "$OARS" --lifecycle "$LIFECYCLE" --observer-duty "$DUTY" --source-commit "{SOURCE_COMMIT}" --run-start-ns "$START_NS" --run-end-ns "$END_NS" --output "$RESULT"
  readonly GATE_RC=$?
  set -e
else
  printf '{{"schema":"m4-oars-v2-controlled-frontier-qualification-result-v1","scientific_outcome_acquisition":false,"status":"TRAINING_FAILURE","training_exit_code":%s,"training_quality_analyzed":false}}\n' "$TRAIN_RC" > "$RESULT"
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
            "name": NAME,
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
