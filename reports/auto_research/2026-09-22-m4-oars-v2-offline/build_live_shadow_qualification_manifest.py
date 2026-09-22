# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Build the one authorized credential-free OARS-v2 live-shadow manifest."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import tarfile
from pathlib import Path

SOURCE_COMMIT = "1284a3f20e7b948c5c6a1c83adf29e6da9626f6e"
PROTOCOL_SHA256 = "463fb719668b5ba53be7ceb7d2df455de1e28d1f354c48494e0e8161c482df7f"
AUTHORIZATION_SHA256 = (
    "e3a470a1afcc3f1959398a5ca9d8f52aa2cf5c8771f5f6e0a1fc4dc23ea6015e"
)
MEGATRON_SHA256 = "98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
CONFIG_SHA256 = "6e13d071f816858e75bf5ef218f1da920c7e14834431c806c8f591073b7a2e5c"
ANALYZER_SHA256 = "522b0b933f67a31fd7de5d67f9f8ec2a32c9ea2500bdab57c91dcb53666e53ba"
TRANSFERQUEUE_COMMIT = "b266d39a15aae114730de36cf8317b6285436f7f"
IMAGE_PATH = (
    "/lustre/fsw/coreai_dlalgo_ci/nvegesna/nemo_rl_images/nemo-rl-nightly-5802754.sqsh"
)
IMAGE_COMMIT = "ae07eafe8035b5b2e84efa7234e70e7fd7e493c1"
CONFIG_PATH = "examples/configs/grpo_math_1B_megatron_single_controller_m4_oars_v2_live_shadow.yaml"
ANALYZER_PATH = "tools/m4_oars_v2_live_shadow_qualification.py"
NAME = "m4-oars-v2-live-shadow-qualification"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
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
        != "m4-oars-v2-live-shadow-qualification-authorization-v1"
        or authorization["source_commit"] != SOURCE_COMMIT
        or authorization["protocol_sha256"] != PROTOCOL_SHA256
        or authorization["required_launcher"] != "runllm.py --no_wait"
        or authorization["submission_attempt_limit"] != 1
        or authorization["trainer_steps"] != 64
        or authorization["gpus"] != 2
        or authorization["scheduler_time_limit_seconds"] != 14400
        or authorization["queue_deadline_override"] is not None
        or authorization["selection_candidate_watermark"] is not None
        or not authorization["eos_submission_authorized"]
        or not authorization["fifo_controlled_training_authorized"]
        or not authorization["oars_v2_shadow_observation_authorized"]
        or authorization["oars_v2_actuation_authorized"]
        or authorization["training_quality_analysis_authorized"]
        or authorization["scientific_outcome_acquisition_authorized"]
        or authorization["automatic_retry"]
        or authorization["automatic_extension"]
    ):
        raise RuntimeError("live-shadow authority differs")
    source_sha256 = sha256(args.source)
    with tarfile.open(args.source, "r:gz") as archive:
        names = set(archive.getnames())
    required = {
        CONFIG_PATH,
        ANALYZER_PATH,
        "uv.lock",
        "nemo_rl/algorithms/async_utils/opportunity_at_risk_v2.py",
        "nemo_rl/algorithms/async_utils/replay_buffer.py",
        "nemo_rl/algorithms/single_controller.py",
        "nemo_rl/algorithms/single_controller_utils/config.py",
        "nemo_rl/data_plane/adapters/transfer_queue.py",
    }
    if not required.issubset(names):
        raise RuntimeError("source archive lacks required runtime files")
    if not re.fullmatch(r"[0-9a-f]{40}", SOURCE_COMMIT):
        raise RuntimeError("invalid source commit")

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
print("M4_OARS_V2_SAFE_SOURCE_PASS",len(names),len(symlinks))
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
print("M4_OARS_V2_MEGATRON_PASS")
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
assert p["status"]=="FROZEN_BEFORE_EOS_PACKAGING"
assert a["source_commit"]=="{SOURCE_COMMIT}"
config_path=Path("{CONFIG_PATH}"); analyzer_path=Path("{ANALYZER_PATH}")
assert hashlib.sha256(config_path.read_bytes()).hexdigest()=="{CONFIG_SHA256}"
assert hashlib.sha256(analyzer_path.read_bytes()).hexdigest()=="{ANALYZER_SHA256}"
assert OpportunityAtRiskV2ShadowRecorder.SCHEMA_VERSION==1
locked=tomllib.loads(Path("uv.lock").read_text())
tq=[x for x in locked["package"] if x["name"].lower()=="transferqueue"]
assert len(tq)==1 and tq[0]["source"]["git"].endswith("#{TRANSFERQUEUE_COMMIT}")
direct=json.loads(distribution("TransferQueue").read_text("direct_url.json"))
assert direct["vcs_info"]["commit_id"]=="{TRANSFERQUEUE_COMMIT}"
register_omegaconf_resolvers(); raw=OmegaConf.to_container(load_config(config_path),resolve=True)
config=MasterConfig(**raw); validate_single_controller_config(config)
assert config.grpo.max_num_steps==64 and config.grpo.num_prompts_per_step==4
assert config.grpo.seed==20262201 and config.cluster["num_nodes"]==1 and config.cluster["gpus_per_node"]==2
assert config.data_plane["actor_runtime_env_mode"]=="inherit_baked_single_node"
assert config.async_rl.sampler.name=="weight_fifo" and config.async_rl.sampler.selection_candidate_watermark is None
assert config.async_rl.opportunity_at_risk_v2_shadow.enabled
assert not config.async_rl.opportunity_at_risk_shadow.enabled
print("M4_OARS_V2_RUNTIME_AUTHORITY_CONFIG_AND_TQ_PASS")
PY
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
  printf '{{"schema":"m4-oars-v2-live-shadow-qualification-result-v1","scientific_outcome_acquisition":false,"status":"TRAINING_FAILURE","training_exit_code":%s,"training_quality_analyzed":false}}\n' "$TRAIN_RC" > "$RESULT"
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
