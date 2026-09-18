# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Build the single authorized enacted-OARS liveness qualification."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import tarfile
from pathlib import Path

PROTOCOL_SHA256 = "3bc49412a212419635e55b616e080708e17cf265c1926c1d93faebf072a87bfd"
AUTHORIZATION_SHA256 = (
    "6a07e6f7a1375537bd3634d2d16e39e8f886d24749253b6918f6332e04bd04f8"
)
AMENDMENT_SHA256 = "1b8c10c157d3136bc1f02171920e0afc3ef25d64ea8030036ca64c0b2c32b7a0"
MEGATRON_SHA256 = "98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
ANALYZER_SHA256 = "53f07de5206bdb11392c5363c39d675b60443c969294cf6a6e61188a7a62fbcd"
CONFIG_SHA256 = "9cd92f17e96193f584f206747132c7ced70dd2e1a9eda44f53d4af752b1967fc"
VALIDATION_SHA256 = "7407a3da260f18578b4ff4038814bc933caa81140aa71f8e6211033a0dd99a86"
TRANSFERQUEUE_COMMIT = "b266d39a15aae114730de36cf8317b6285436f7f"
IMAGE_PATH = (
    "/lustre/fsw/coreai_dlalgo_ci/nvegesna/nemo_rl_images/nemo-rl-nightly-5802754.sqsh"
)
IMAGE_COMMIT = "ae07eafe8035b5b2e84efa7234e70e7fd7e493c1"
CONFIG_PATH = (
    "examples/configs/grpo_math_1B_megatron_single_controller_"
    "m4_oars_qualification_act.yaml"
)
ANALYZER_PATH = "tools/m4_oars_actuation_qualification.py"
VALIDATION_PATH = (
    "reports/auto_research/2026-09-15-m4-oars-randomized/"
    "qualification_liveness_repair_validation.json"
)
NAME = "m4-oars-qualification-act-liveness-repair"


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload(path: Path) -> str:
    """Return a base64-encoded file payload."""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    """Validate frozen inputs and write one credential-free JET manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        raise RuntimeError("source commit must be a full lowercase SHA-1")
    for path, expected in (
        (args.megatron, MEGATRON_SHA256),
        (args.protocol, PROTOCOL_SHA256),
        (args.authorization, AUTHORIZATION_SHA256),
        (args.amendment, AMENDMENT_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen liveness input moved: {path.name}")
    authorization = json.loads(args.authorization.read_bytes())
    amendment = json.loads(args.amendment.read_bytes())
    if (
        authorization["schema"]
        != "m4-oars-actuation-qualification-liveness-repair-authorization-v1"
        or authorization["source_commit"] != args.source_commit
        or authorization["protocol_sha256"] != PROTOCOL_SHA256
        or authorization["required_launcher"] != "runllm.py --no_wait"
        or authorization["repair_attempt_limit"] != 1
        or authorization["trainer_steps"] != 64
        or authorization["gpus"] != 2
        or authorization["scheduler_time_limit_seconds"] != 14400
        or authorization["queue_deadline_override"] is not None
        or not authorization["eos_submission_authorized"]
        or not authorization["systems_qualification_authorized"]
        or not authorization["oars_actuation_authorized_for_qualification_only"]
        or authorization["scientific_outcome_acquisition_authorized"]
        or authorization["automatic_retry"]
        or authorization["automatic_extension"]
    ):
        raise RuntimeError("liveness repair authority differs")
    if (
        amendment["schema"]
        != "m4-oars-actuation-qualification-liveness-repair-amendment-v1"
        or amendment["status"] != "FROZEN_BEFORE_LIVENESS_REPAIR_SUBMISSION"
        or amendment["repair_provenance"]["repair_source_commit"] != args.source_commit
        or amendment["repair_provenance"]["offline_validation_sha256"]
        != VALIDATION_SHA256
        or amendment["allowed_changes"]["qualification_ledger_schema"] != 3
        or amendment["interpretation"]["scientific_outcome_acquisition"]
    ):
        raise RuntimeError("liveness repair amendment differs")
    source_sha256 = sha256(args.source)
    with tarfile.open(args.source, "r:gz") as archive:
        names = set(archive.getnames())
    required_source = {
        CONFIG_PATH,
        ANALYZER_PATH,
        VALIDATION_PATH,
        "uv.lock",
        "nemo_rl/algorithms/async_utils/opportunity_at_risk.py",
        "nemo_rl/algorithms/async_utils/staleness_sampler.py",
        "nemo_rl/data_plane/adapters/transfer_queue.py",
    }
    if not required_source.issubset(names):
        raise RuntimeError("liveness source archive lacks runtime files")

    script = f'''set -euo pipefail
readonly PYTHON=/opt/nemo_rl_venv/bin/python
readonly RUN_REPO=/workspace/{NAME}-repo
readonly SOURCE=/workspace/{NAME}-source.tar.gz
readonly MEGATRON=/workspace/{NAME}-megatron.tar.gz
readonly PROTOCOL=/workspace/{NAME}-protocol.json
readonly AUTH=/workspace/{NAME}-authorization.json
readonly AMENDMENT=/workspace/{NAME}-amendment.json
readonly ASSETS={{assets_dir}}
readonly RUN_LOG=$ASSETS/{NAME}-run.log
readonly LIFECYCLE=$ASSETS/{NAME}-lifecycle.jsonl
readonly OPPORTUNITY=$ASSETS/{NAME}-opportunity.jsonl
readonly DUTY=$ASSETS/{NAME}-gradient-observer-duty.json
readonly OARS=$ASSETS/{NAME}-oars.jsonl
readonly RESULT=$ASSETS/{NAME}-result.json
readonly HASHES=$ASSETS/{NAME}-artifacts.sha256
readonly METRICS=$ASSETS/{NAME}-metrics
test "${{NEMO_RL_COMMIT:-unknown}}" = "{IMAGE_COMMIT}"
test ! -e "$RUN_REPO"
printf %s '{payload(args.source)}' | base64 -d > "$SOURCE"
printf %s '{payload(args.megatron)}' | base64 -d > "$MEGATRON"
printf %s '{payload(args.protocol)}' | base64 -d > "$PROTOCOL"
printf %s '{payload(args.authorization)}' | base64 -d > "$AUTH"
printf %s '{payload(args.amendment)}' | base64 -d > "$AMENDMENT"
test "$(sha256sum "$SOURCE" | cut -d' ' -f1)" = "{source_sha256}"
test "$(sha256sum "$MEGATRON" | cut -d' ' -f1)" = "{MEGATRON_SHA256}"
test "$(sha256sum "$PROTOCOL" | cut -d' ' -f1)" = "{PROTOCOL_SHA256}"
test "$(sha256sum "$AUTH" | cut -d' ' -f1)" = "{AUTHORIZATION_SHA256}"
test "$(sha256sum "$AMENDMENT" | cut -d' ' -f1)" = "{AMENDMENT_SHA256}"
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
print("M4_OARS_LIVENESS_SAFE_SOURCE_PASS",len(names),len(symlinks))
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
print("M4_OARS_LIVENESS_MEGATRON_PASS")
PY
cd "$RUN_REPO"
export PYTHONPATH="$RUN_REPO"
export CUDA_VISIBLE_DEVICES=0,1
mkdir -p "$METRICS"
"$PYTHON" - "$PROTOCOL" "$AUTH" "$AMENDMENT" <<'PY'
import hashlib,json,sys,tomllib
from importlib.metadata import distribution
from pathlib import Path
from omegaconf import OmegaConf
from nemo_rl.algorithms.async_utils.opportunity_at_risk import OpportunityAtRiskShadowRecorder
from nemo_rl.algorithms.single_controller_utils.config import MasterConfig,validate_single_controller_config
from nemo_rl.utils.config import load_config,register_omegaconf_resolvers
p_path,a_path,m_path=map(Path,sys.argv[1:]); p=json.loads(p_path.read_bytes()); a=json.loads(a_path.read_bytes()); m=json.loads(m_path.read_bytes())
assert hashlib.sha256(p_path.read_bytes()).hexdigest()=="{PROTOCOL_SHA256}"
assert hashlib.sha256(a_path.read_bytes()).hexdigest()=="{AUTHORIZATION_SHA256}"
assert hashlib.sha256(m_path.read_bytes()).hexdigest()=="{AMENDMENT_SHA256}"
assert p["analysis_state"]=="frozen_before_actuation_qualification"
assert a["source_commit"]=="{args.source_commit}" and a["pair_training_seed"]==20261421
assert m["status"]=="FROZEN_BEFORE_LIVENESS_REPAIR_SUBMISSION"
config_path=Path("{CONFIG_PATH}"); analyzer_path=Path("{ANALYZER_PATH}"); validation_path=Path("{VALIDATION_PATH}")
assert hashlib.sha256(config_path.read_bytes()).hexdigest()=="{CONFIG_SHA256}"
assert hashlib.sha256(analyzer_path.read_bytes()).hexdigest()=="{ANALYZER_SHA256}"
assert hashlib.sha256(validation_path.read_bytes()).hexdigest()=="{VALIDATION_SHA256}"
assert OpportunityAtRiskShadowRecorder.SCHEMA_VERSION==3
locked=tomllib.loads(Path("uv.lock").read_text())
tq_lock=[x for x in locked["package"] if x["name"].lower()=="transferqueue"]
assert len(tq_lock)==1 and tq_lock[0]["source"]["git"].endswith("#{TRANSFERQUEUE_COMMIT}")
direct=json.loads(distribution("TransferQueue").read_text("direct_url.json"))
assert direct["vcs_info"]["commit_id"]=="{TRANSFERQUEUE_COMMIT}"
register_omegaconf_resolvers(); raw=OmegaConf.to_container(load_config(config_path),resolve=True)
config=MasterConfig(**raw); validate_single_controller_config(config)
assert config.grpo.max_num_steps==64 and config.grpo.num_prompts_per_step==4
assert config.grpo.seed==20261421 and config.cluster["num_nodes"]==1 and config.cluster["gpus_per_node"]==2
assert config.data_plane["actor_runtime_env_mode"]=="inherit_baked_single_node"
assert config.async_rl.sampler.name=="weight_fifo" and config.async_rl.sampler.selection_candidate_watermark==8
assert config.async_rl.opportunity_at_risk_shadow.enabled and config.async_rl.opportunity_at_risk_shadow.mode=="act"
assert config.async_rl.opportunity_at_risk_shadow.service_budget_multiplier==1.02
assert not config.async_rl.lifecycle_derived_opportunity_audit.enabled
print("M4_OARS_LIVENESS_RUNTIME_AUTHORITY_CONFIG_AND_TQ_PASS")
PY
readonly START_NS=$("$PYTHON" -c 'import time; print(time.time_ns())')
set +e
"$PYTHON" examples/run_grpo_single_controller.py --config "{CONFIG_PATH}" \
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
  "$PYTHON" "{ANALYZER_PATH}" --mode act --oars "$OARS" --lifecycle "$LIFECYCLE" --observer-duty "$DUTY" --source-commit "{args.source_commit}" --run-start-ns "$START_NS" --run-end-ns "$END_NS" --output "$RESULT"
  readonly GATE_RC=$?
  set -e
else
  printf '{{"mode":"act","schema":"m4-oars-actuation-qualification-arm-result-v1","scientific_outcome_acquisition":false,"status":"TRAINING_FAILURE","training_exit_code":%s,"training_quality_analyzed":false}}\n' "$TRAIN_RC" > "$RESULT"
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
