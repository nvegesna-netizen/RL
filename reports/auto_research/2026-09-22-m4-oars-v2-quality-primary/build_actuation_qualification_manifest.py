# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build one credential-free, outcome-excluded OARS-v2 qualification."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import tarfile
from pathlib import Path

SOURCE_COMMIT = "74ea47bcb63dea04b00a30bf13c31dfedc012d7b"
SOURCE_SHA256 = "db0c1010de8fb6236131b923679476c9cbb5f44c755dc0303eff9b4ef16642a9"
PROTOCOL_SHA256 = "e0e0b26cc1a592bf14c4b0f77d990662f6c3ff3e09bfb6a772a3efbf149b9d2d"
MEGATRON_SHA256 = "98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
CONFIG_SHA256 = "fac9bc021c504029aa378d855132afe00635432c575def744302e5276ccb7c8e"
ANALYZER_SHA256 = "93b286caaaec513f8c002ca3c596b343b8945fc14292393dde36afbf1273c783"
VERIFICATION_SHA256 = "2646ea8e4897e7c047f283fad068c533cd99da0d47ab855e87fb76cc244b2509"
TRANSFERQUEUE_COMMIT = "b266d39a15aae114730de36cf8317b6285436f7f"
IMAGE_PATH = (
    "/lustre/fsw/coreai_dlalgo_ci/nvegesna/nemo_rl_images/nemo-rl-nightly-5802754.sqsh"
)
IMAGE_COMMIT = "ae07eafe8035b5b2e84efa7234e70e7fd7e493c1"
CONFIG_PATH = (
    "examples/configs/grpo_math_1B_megatron_single_controller_"
    "m4_oars_v2_controlled_frontier.yaml"
)
ANALYZER_PATH = "tools/m4_oars_v2_actuation_qualification.py"
VERIFICATION_PATH = (
    "reports/auto_research/2026-09-22-m4-oars-v2-quality-primary/"
    "actuation_implementation_verification.json"
)
ANALYZER_TEST_PATH = (
    "reports/auto_research/2026-09-22-m4-oars-v2-quality-primary/"
    "test_actuation_qualification.py"
)
SCOPES = {
    "reward_variance_risk": (
        "reward-variance",
        "exactly_one_reward_variance_outcome_excluded_actuation_qualification",
        20262231,
    ),
    "absolute_m4_risk": (
        "absolute-m4",
        "exactly_one_absolute_m4_outcome_excluded_actuation_qualification",
        20262232,
    ),
}


def sha256(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload(path: Path) -> str:
    """Return one base64-encoded credential-free input."""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    """Validate locked inputs and write one JET manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer", choices=tuple(SCOPES), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    slug, scope, seed = SCOPES[args.scorer]
    name = f"m4-oars-v2-{slug}-actuation-qualification"
    authorization_sha256 = sha256(args.authorization)
    for path, expected in (
        (args.source, SOURCE_SHA256),
        (args.megatron, MEGATRON_SHA256),
        (args.protocol, PROTOCOL_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen input moved: {path.name}")
    authorization = json.loads(args.authorization.read_bytes())
    if (
        authorization["schema"] != "m4-oars-v2-actuation-qualification-authorization-v1"
        or authorization["scope"] != scope
        or authorization["source_commit"] != SOURCE_COMMIT
        or authorization["protocol_sha256"] != PROTOCOL_SHA256
        or authorization["actuation_scorer"] != args.scorer
        or authorization["training_seed"] != seed
        or authorization["required_launcher"] != "runllm.py --no_wait"
        or authorization["submission_attempt_limit"] != 1
        or authorization["trainer_steps"] != 64
        or authorization["nodes"] != 1
        or authorization["gpus"] != 2
        or authorization["scheduler_time_limit_seconds"] != 14400
        or authorization["queue_deadline_override"] is not None
        or authorization["selection_candidate_watermark"] != 8
        or authorization["selection_cardinality"] != 4
        or not authorization["eos_submission_authorized"]
        or not authorization["oars_v2_actuation_authorized"]
        or authorization["training_quality_analysis_authorized"]
        or authorization["scientific_outcome_acquisition_authorized"]
        or authorization["automatic_retry"]
        or authorization["automatic_extension"]
    ):
        raise RuntimeError("qualification authority differs")
    with tarfile.open(args.source, "r:gz") as archive:
        names = set(archive.getnames())
    required = {
        CONFIG_PATH,
        ANALYZER_PATH,
        VERIFICATION_PATH,
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
readonly RUN_REPO=/workspace/{name}-repo
readonly SOURCE=/workspace/{name}-source.tar.gz
readonly MEGATRON=/workspace/{name}-megatron.tar.gz
readonly PROTOCOL=/workspace/{name}-protocol.json
readonly AUTH=/workspace/{name}-authorization.json
readonly ASSETS={{assets_dir}}
readonly RUN_LOG=$ASSETS/{name}-run.log
readonly LIFECYCLE=$ASSETS/{name}-lifecycle.jsonl
readonly OPPORTUNITY=$ASSETS/{name}-opportunity.jsonl
readonly DUTY=$ASSETS/{name}-observer-duty.json
readonly OARS=$ASSETS/{name}-oars-v2.jsonl
readonly RESULT=$ASSETS/{name}-result.json
readonly HASHES=$ASSETS/{name}-artifacts.sha256
readonly METRICS=$ASSETS/{name}-metrics
test "${{NEMO_RL_COMMIT:-unknown}}" = "{IMAGE_COMMIT}"
test ! -e "$RUN_REPO"
mkdir -p "$METRICS"
printf %s '{payload(args.source)}' | base64 -d > "$SOURCE"
printf %s '{payload(args.megatron)}' | base64 -d > "$MEGATRON"
printf %s '{payload(args.protocol)}' | base64 -d > "$PROTOCOL"
printf %s '{payload(args.authorization)}' | base64 -d > "$AUTH"
test "$(sha256sum "$SOURCE" | cut -d' ' -f1)" = "{SOURCE_SHA256}"
test "$(sha256sum "$MEGATRON" | cut -d' ' -f1)" = "{MEGATRON_SHA256}"
test "$(sha256sum "$PROTOCOL" | cut -d' ' -f1)" = "{PROTOCOL_SHA256}"
test "$(sha256sum "$AUTH" | cut -d' ' -f1)" = "{authorization_sha256}"
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
print("M4_OARS_V2_ACTUATION_SAFE_SOURCE_PASS",len(names),len(symlinks))
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
print("M4_OARS_V2_ACTUATION_MEGATRON_PASS")
PY
cd "$RUN_REPO"
export PYTHONPATH="$RUN_REPO"
export CUDA_VISIBLE_DEVICES=0,1
"$PYTHON" - "$PROTOCOL" "$AUTH" <<'PY'
import hashlib,json,sys,tomllib
from importlib.metadata import distribution
from pathlib import Path
from omegaconf import OmegaConf
from nemo_rl.algorithms.single_controller_utils.config import MasterConfig,validate_single_controller_config
from nemo_rl.utils.config import load_config,parse_hydra_overrides,register_omegaconf_resolvers
p_path,a_path=map(Path,sys.argv[1:]); p=json.loads(p_path.read_bytes()); a=json.loads(a_path.read_bytes())
assert hashlib.sha256(p_path.read_bytes()).hexdigest()=="{PROTOCOL_SHA256}"
assert hashlib.sha256(a_path.read_bytes()).hexdigest()=="{authorization_sha256}"
assert p["status"]=="FROZEN_BEFORE_ACTUATION_IMPLEMENTATION_COMMIT"
assert a["source_commit"]=="{SOURCE_COMMIT}" and a["actuation_scorer"]=="{args.scorer}"
config_path=Path("{CONFIG_PATH}"); analyzer_path=Path("{ANALYZER_PATH}"); verification_path=Path("{VERIFICATION_PATH}")
assert hashlib.sha256(config_path.read_bytes()).hexdigest()=="{CONFIG_SHA256}"
assert hashlib.sha256(analyzer_path.read_bytes()).hexdigest()=="{ANALYZER_SHA256}"
assert hashlib.sha256(verification_path.read_bytes()).hexdigest()=="{VERIFICATION_SHA256}"
register_omegaconf_resolvers()
overrides=["grpo.seed={seed}","async_rl.opportunity_at_risk_v2_shadow.mode=act","async_rl.opportunity_at_risk_v2_shadow.actuation_scorer={args.scorer}"]
raw=parse_hydra_overrides(load_config(config_path),overrides)
config=MasterConfig(**OmegaConf.to_container(raw,resolve=True)); validate_single_controller_config(config)
assert config.grpo.max_num_steps==64 and config.grpo.num_prompts_per_step==4 and config.grpo.seed=={seed}
assert config.cluster["num_nodes"]==1 and config.cluster["gpus_per_node"]==2
assert config.data_plane["actor_runtime_env_mode"]=="inherit_baked_single_node"
assert config.async_rl.sampler.name=="weight_fifo" and config.async_rl.sampler.selection_candidate_watermark==8
assert config.async_rl.opportunity_at_risk_v2_shadow.enabled
assert config.async_rl.opportunity_at_risk_v2_shadow.mode=="act"
assert config.async_rl.opportunity_at_risk_v2_shadow.actuation_scorer=="{args.scorer}"
assert config.async_rl.opportunity_at_risk_v2_shadow.candidate_window_policy=="controlled_frontier"
locked=tomllib.loads(Path("uv.lock").read_text()); tq=[x for x in locked["package"] if x["name"].lower()=="transferqueue"]
assert len(tq)==1 and tq[0]["source"]["git"].endswith("#{TRANSFERQUEUE_COMMIT}")
direct=json.loads(distribution("TransferQueue").read_text("direct_url.json"))
assert direct["vcs_info"]["commit_id"]=="{TRANSFERQUEUE_COMMIT}"
print("M4_OARS_V2_ACTUATION_RUNTIME_AUTHORITY_CONFIG_AND_TQ_PASS")
PY
"$PYTHON" -m pytest -q tests/unit/single_controller/test_opportunity_at_risk_v2.py
"$PYTHON" -m pytest -q tests/unit/single_controller/test_single_controller.py -k opportunity_at_risk_v2
"$PYTHON" -m pytest -q "{ANALYZER_TEST_PATH}"
readonly START_NS=$("$PYTHON" -c 'import time; print(time.time_ns())')
set +e
"$PYTHON" examples/run_grpo_single_controller.py --config "{CONFIG_PATH}" \
  "grpo.seed={seed}" \
  "logger.log_dir=$METRICS" \
  "async_rl.lifecycle_audit_path=$LIFECYCLE" \
  "async_rl.gradient_opportunity_audit.output_path=$OPPORTUNITY" \
  "async_rl.gradient_opportunity_audit.observer_duty_path=$DUTY" \
  "async_rl.opportunity_at_risk_v2_shadow.mode=act" \
  "async_rl.opportunity_at_risk_v2_shadow.actuation_scorer={args.scorer}" \
  "async_rl.opportunity_at_risk_v2_shadow.output_path=$OARS" 2>&1 | tee "$RUN_LOG"
readonly TRAIN_RC=${{PIPESTATUS[0]}}
set -e
readonly END_NS=$("$PYTHON" -c 'import time; print(time.time_ns())')
if test "$TRAIN_RC" -eq 0; then
  set +e
  "$PYTHON" "{ANALYZER_PATH}" --scorer "{args.scorer}" --oars "$OARS" --lifecycle "$LIFECYCLE" --observer-duty "$DUTY" --source-commit "{SOURCE_COMMIT}" --run-start-ns "$START_NS" --run-end-ns "$END_NS" --output "$RESULT"
  readonly GATE_RC=$?
  set -e
else
  printf '{{"schema":"m4-oars-v2-actuation-qualification-result-v1","scientific_outcome_acquisition":false,"status":"TRAINING_FAILURE","training_exit_code":%s,"training_quality_analyzed":false}}\n' "$TRAIN_RC" > "$RESULT"
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
    print(args.output, sha256(args.output), authorization_sha256)


if __name__ == "__main__":
    main()
