#!/usr/bin/env python3
"""Build the authorized credential-free repaired OARS contention preflight."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import tarfile
from pathlib import Path


PROTOCOL_SHA256 = "06df90b6d411428cb959b1ee2f8b3d4174ca5284f1841dbcdd88fcd53f2178a1"
AUTHORIZATION_SHA256 = (
    "9b150d35ff37469e39241d02bef3c611b9e1f656faf67f5daf4ba64c80d9bcd8"
)
CONFIG_SHA256 = "becbd749e516d7af2cf08b24475e13fa544bcdc9a6505bb67735f4f25cd401c9"
ANALYZER_SHA256 = "c3b2fd40baff10c7d77cd589a433c2ccb64f81b63c76ab257322fd6f75fd548e"
MEGATRON_SHA256 = "98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
IMAGE_PATH = (
    "/lustre/fsw/coreai_dlalgo_ci/nvegesna/nemo_rl_images/nemo-rl-nightly-5802754.sqsh"
)
IMAGE_COMMIT = "ae07eafe8035b5b2e84efa7234e70e7fd7e493c1"
CONFIG_PATH = "examples/configs/grpo_math_1B_megatron_single_controller_m4_oars_shadow_contention_preflight.yaml"
ANALYZER_PATH = "tools/m4_oars_shadow_contention_preflight.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        raise RuntimeError("source commit must be a full lowercase SHA-1")
    for path, expected in (
        (args.megatron, MEGATRON_SHA256),
        (args.protocol, PROTOCOL_SHA256),
        (args.authorization, AUTHORIZATION_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen package input moved: {path.name}")
    source_sha = sha256(args.source)
    with tarfile.open(args.source, "r:gz") as archive:
        names = set(archive.getnames())
    if CONFIG_PATH not in names or ANALYZER_PATH not in names:
        raise RuntimeError("source archive lacks frozen preflight runtime files")

    authorization = json.loads(args.authorization.read_bytes())
    required_true = (
        "eos_submission_authorized",
        "fifo_controlled_training_authorized",
        "oars_shadow_observation_authorized",
    )
    if (
        authorization["schema"]
        != "m4-oars-fifo-controlled-shadow-contention-preflight-authorization-v1"
        or authorization["protocol_sha256"] != PROTOCOL_SHA256
        or authorization["required_launcher"] != "runllm.py --no_wait"
        or authorization["submission_attempt_limit"] != 1
        or authorization["trainer_steps"] != 64
        or authorization["gpus"] != 2
        or authorization["scheduler_time_limit_seconds"] != 14400
        or authorization["queue_deadline_override"] is not None
        or authorization["selection_candidate_watermark"] != 8
        or authorization["selection_cardinality"] != 4
        or not all(authorization[key] for key in required_true)
        or authorization["oars_actuation_authorized"]
        or authorization["scientific_outcome_acquisition_authorized"]
        or authorization["automatic_retry"]
        or authorization["automatic_extension"]
    ):
        raise RuntimeError("preflight authority differs")

    script = f'''set -euo pipefail
readonly PYTHON=/opt/nemo_rl_venv/bin/python
readonly RUN_REPO=/workspace/m4-oars-shadow-contention-preflight-repo
readonly SOURCE=/workspace/m4-oars-shadow-contention-preflight-source.tar.gz
readonly MEGATRON=/workspace/m4-oars-shadow-contention-preflight-megatron.tar.gz
readonly PROTOCOL=/workspace/m4-oars-shadow-contention-preflight-protocol.json
readonly AUTH=/workspace/m4-oars-shadow-contention-preflight-authorization.json
readonly ASSETS={{assets_dir}}
readonly RUN_LOG=$ASSETS/m4-oars-shadow-contention-preflight-run.log
readonly LIFECYCLE=$ASSETS/m4-oars-shadow-contention-preflight-lifecycle.jsonl
readonly OPPORTUNITY=$ASSETS/m4-oars-shadow-contention-preflight-opportunity.jsonl
readonly DUTY=$ASSETS/m4-oars-shadow-contention-preflight-gradient-observer-duty.json
readonly OARS=$ASSETS/m4-oars-shadow-contention-preflight-oars.jsonl
readonly RESULT=$ASSETS/m4-oars-shadow-contention-preflight-result.json
readonly HASHES=$ASSETS/m4-oars-shadow-contention-preflight-artifacts.sha256
readonly METRICS=$ASSETS/m4-oars-shadow-contention-preflight-metrics
test "${{NEMO_RL_COMMIT:-unknown}}" = "{IMAGE_COMMIT}"
test ! -e "$RUN_REPO"
printf %s '{payload(args.source)}' | base64 -d > "$SOURCE"
printf %s '{payload(args.megatron)}' | base64 -d > "$MEGATRON"
printf %s '{payload(args.protocol)}' | base64 -d > "$PROTOCOL"
printf %s '{payload(args.authorization)}' | base64 -d > "$AUTH"
test "$(sha256sum "$SOURCE" | cut -d' ' -f1)" = "{source_sha}"
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
assert p["status"]=="FROZEN_AUTHORIZED_ONE_SHOT_NO_ACTUATION"
assert a["protocol_sha256"]=="{PROTOCOL_SHA256}" and a["submission_attempt_limit"]==1
assert a["eos_submission_authorized"] and a["fifo_controlled_training_authorized"] and a["oars_shadow_observation_authorized"]
assert not a["oars_actuation_authorized"] and not a["scientific_outcome_acquisition_authorized"]
assert not a["automatic_retry"] and not a["automatic_extension"]
config_path=Path("{CONFIG_PATH}"); analyzer_path=Path("{ANALYZER_PATH}")
assert hashlib.sha256(config_path.read_bytes()).hexdigest()=="{CONFIG_SHA256}"
assert hashlib.sha256(analyzer_path.read_bytes()).hexdigest()=="{ANALYZER_SHA256}"
register_omegaconf_resolvers(); raw=OmegaConf.to_container(load_config(config_path),resolve=True)
config=MasterConfig(**raw); validate_single_controller_config(config)
assert config.grpo.max_num_steps==64 and config.grpo.num_prompts_per_step==4
assert config.cluster["num_nodes"]==1 and config.cluster["gpus_per_node"]==2
assert config.async_rl.sampler.name=="weight_fifo"
assert config.async_rl.sampler.selection_candidate_watermark==8
assert config.async_rl.opportunity_at_risk_shadow.enabled
assert config.async_rl.lifecycle_derived_opportunity_audit.enabled is False
print("M4_OARS_RUNTIME_AUTHORITY_AND_CONFIG_PASS")
PY
readonly START_NS=$("$PYTHON" -c 'import time; print(time.time_ns())')
"$PYTHON" examples/run_grpo_single_controller.py --config "{CONFIG_PATH}" \
  "logger.log_dir=$METRICS" \
  "async_rl.lifecycle_audit_path=$LIFECYCLE" \
  "async_rl.gradient_opportunity_audit.output_path=$OPPORTUNITY" \
  "async_rl.gradient_opportunity_audit.observer_duty_path=$DUTY" \
  "async_rl.opportunity_at_risk_shadow.output_path=$OARS" 2>&1 | tee "$RUN_LOG"
readonly END_NS=$("$PYTHON" -c 'import time; print(time.time_ns())')
set +e
"$PYTHON" "{ANALYZER_PATH}" --oars "$OARS" --lifecycle "$LIFECYCLE" --observer-duty "$DUTY" --source-commit "{args.source_commit}" --run-start-ns "$START_NS" --run-end-ns "$END_NS" --output "$RESULT"
readonly ANALYZER_RC=$?
set -e
sha256sum "$RUN_LOG" "$LIFECYCLE" "$OPPORTUNITY" "$DUTY" "$OARS" "$RESULT" > "$HASHES"
exit "$ANALYZER_RC"
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
            "name": "m4-oars-fifo-controlled-shadow-contention-preflight",
            "workspace": "/workspace",
            "nodes": 1,
            "time_limit": 14400,
            "image_source": {"local_path": IMAGE_PATH},
            "script": script,
        },
    }
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(args.output, sha256(args.output), source_sha)


if __name__ == "__main__":
    main()
