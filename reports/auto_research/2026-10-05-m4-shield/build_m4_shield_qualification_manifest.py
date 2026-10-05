"""Build the credential-free, outcome-excluded M4-Shield qualification."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import tarfile
from pathlib import Path


SOURCE_COMMIT = "cdfa1c8e56394b7d8e3e299b6524a1a286243746"
SOURCE_SHA256 = "e9a70e436852f747db1812a2af275351643473cf6f4f20d0e2a4e0573ccc1111"
MEGATRON_SHA256 = "98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
PROTOCOL_SHA256 = "a3c8f84297c2ce60c5cd5f0f0eb0a0286555ced63bbe5b169af0e3b008aca216"
DESIGN_SHA256 = "d7b35097584b1b9d3be2f8651879b7196830766a99a4b3287f063f41ba744d84"
PARENT_PROTOCOL_SHA256 = (
    "1a4f9a5c57ee000cded9df3214930512d95f5b115f13e3a33070c55476c2a232"
)
OFFLINE_RESULT_SHA256 = (
    "2e6a41b7acd4d0de8e7ad60dfaf56b834291c937dbf9924b9be1ed4504f5e036"
)
CONFIG_SHA256 = "b3d2415244b1187f637f6f939fd8ffd8a3a6712026ec4d8581ff2e96f51ce7ea"
ANALYZER_SHA256 = "6fe084407c2ba24113bd33143e1905368085d3c2dd3ebdda17479683bd90cbb8"
RUNTIME_SHA256 = "5e4b9f172feda2525972cabf3013d080be98eda004b7050fc95f5b0ea764df0d"
CONTROLLER_TEST_SHA256 = (
    "dff5e8000017651d3b6b8ebfd6eadacb4cde90752ecf6bf69edcf3dd61823f99"
)
CONFIG_TEST_SHA256 = "511ee6ca99ff2eed243b5fc40e78854e209f6d8c6f123ae0bbdcf1841f07ee53"
ANALYZER_TEST_SHA256 = (
    "c7fb4a3a01abf0b0a7cf793365adf907ec04f4ad40927f7e0b0d106c32f200cb"
)
TRANSFERQUEUE_COMMIT = "b266d39a15aae114730de36cf8317b6285436f7f"
IMAGE_PATH = (
    "/lustre/fsw/coreai_dlalgo_ci/nvegesna/nemo_rl_images/nemo-rl-nightly-5802754.sqsh"
)
IMAGE_COMMIT = "ae07eafe8035b5b2e84efa7234e70e7fd7e493c1"
CONFIG_PATH = (
    "examples/configs/"
    "grpo_math_1B_megatron_single_controller_m4_shield_qualification.yaml"
)
ANALYZER_PATH = "tools/m4_shield_qualification.py"
RUNTIME_PATH = "nemo_rl/algorithms/async_utils/opportunity_at_risk_v2.py"
CONTROLLER_TEST_PATH = "tests/unit/single_controller/test_opportunity_at_risk_v2.py"
CONFIG_TEST_PATH = "tests/unit/single_controller/test_single_controller.py"
ANALYZER_TEST_PATH = (
    "reports/auto_research/2026-10-05-m4-shield/test_m4_shield_qualification.py"
)
DESIGN_PATH = "reports/auto_research/2026-10-05-m4-shield/m4_shield_live_design.json"
PARENT_PROTOCOL_PATH = (
    "reports/auto_research/2026-10-05-m4-shield/m4_shield_protocol.json"
)
OFFLINE_RESULT_PATH = (
    "reports/auto_research/2026-10-05-m4-shield/m4_shield_offline_result.json"
)
NAME = "m4-shield-outcome-excluded-qualification"


def sha256(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload(path: Path) -> str:
    """Return one file as base64 text."""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
        authorization.get("schema") != "m4-shield-live-submission-authorization-v1"
        or authorization.get("scope")
        != "exactly_one_outcome_excluded_m4_shield_live_qualification"
        or authorization.get("source_commit") != SOURCE_COMMIT
        or authorization.get("protocol_sha256") != PROTOCOL_SHA256
        or authorization.get("required_launcher") != "runllm.py --no_wait"
        or authorization.get("submission_attempt_limit") != 1
        or authorization.get("trainer_steps") != 64
        or authorization.get("training_seed") != 20261005
        or authorization.get("controlled_release_seed") != 20261007
        or authorization.get("assignment_domain") != "m4-shield-qualification-v1"
        or authorization.get("nodes") != 1
        or authorization.get("gpus") != 2
        or authorization.get("scheduler_time_limit_seconds") != 14400
        or authorization.get("queue_deadline_override") is not None
        or authorization.get("selection_candidate_watermark") != 8
        or authorization.get("selection_cardinality") != 4
        or authorization.get("base_proposer") != "reward_variance_risk"
        or authorization.get("actuation_scorer") != "m4_shield"
        or not authorization.get("eos_submission_authorized")
        or not authorization.get("m4_shield_actuation_authorized")
        or authorization.get("training_quality_analysis_authorized")
        or authorization.get("scientific_outcome_acquisition_authorized")
        or authorization.get("terminal_benchmark_authorized")
        or authorization.get("automatic_retry")
        or authorization.get("automatic_extension")
    ):
        raise RuntimeError("qualification authority differs")
    with tarfile.open(args.source, "r:gz") as archive:
        names = set(archive.getnames())
    required = {
        CONFIG_PATH,
        ANALYZER_PATH,
        RUNTIME_PATH,
        CONTROLLER_TEST_PATH,
        CONFIG_TEST_PATH,
        ANALYZER_TEST_PATH,
        DESIGN_PATH,
        PARENT_PROTOCOL_PATH,
        OFFLINE_RESULT_PATH,
        "nemo_rl/algorithms/single_controller.py",
        "nemo_rl/algorithms/single_controller_utils/config.py",
        "nemo_rl/algorithms/async_utils/staleness_sampler.py",
        "nemo_rl/data_plane/adapters/transfer_queue.py",
        "uv.lock",
    }
    if not required.issubset(names):
        raise RuntimeError("source archive lacks required runtime or evidence files")

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
print("M4_SHIELD_SAFE_SOURCE_PASS",len(names),len(symlinks))
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
print("M4_SHIELD_MEGATRON_PASS")
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
from nemo_rl.utils.config import load_config,register_omegaconf_resolvers
p_path,a_path=map(Path,sys.argv[1:]); p=json.loads(p_path.read_bytes()); a=json.loads(a_path.read_bytes())
assert hashlib.sha256(p_path.read_bytes()).hexdigest()=="{PROTOCOL_SHA256}"
assert hashlib.sha256(a_path.read_bytes()).hexdigest()=="{authorization_sha256}"
assert p["status"]=="FROZEN_AFTER_IMPLEMENTATION_BEFORE_SUBMISSION"
assert p["source_commit"]==a["source_commit"]=="{SOURCE_COMMIT}"
assert p["design_sha256"]=="{DESIGN_SHA256}"
assert p["parent_protocol_sha256"]=="{PARENT_PROTOCOL_SHA256}"
assert p["offline_result_sha256"]=="{OFFLINE_RESULT_SHA256}"
for path,expected in (("{DESIGN_PATH}","{DESIGN_SHA256}"),("{PARENT_PROTOCOL_PATH}","{PARENT_PROTOCOL_SHA256}"),("{OFFLINE_RESULT_PATH}","{OFFLINE_RESULT_SHA256}"),("{CONFIG_PATH}","{CONFIG_SHA256}"),("{ANALYZER_PATH}","{ANALYZER_SHA256}"),("{RUNTIME_PATH}","{RUNTIME_SHA256}"),("{CONTROLLER_TEST_PATH}","{CONTROLLER_TEST_SHA256}"),("{CONFIG_TEST_PATH}","{CONFIG_TEST_SHA256}"),("{ANALYZER_TEST_PATH}","{ANALYZER_TEST_SHA256}")):
 assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
assert json.loads(Path("{OFFLINE_RESULT_PATH}").read_bytes())["status"]=="PASS_M4_SHIELD_DEVELOPMENT_QUALIFICATION"
register_omegaconf_resolvers(); config=MasterConfig(**OmegaConf.to_container(load_config(Path("{CONFIG_PATH}")),resolve=True)); validate_single_controller_config(config)
assert config.grpo.max_num_steps==64 and config.grpo.seed==20261005
assert config.cluster["num_nodes"]==1 and config.cluster["gpus_per_node"]==2
assert config.async_rl.sampler.name=="weight_fifo" and config.async_rl.sampler.selection_candidate_watermark==8
assert config.async_rl.sampler.max_staleness_versions==1
v2=config.async_rl.opportunity_at_risk_v2_shadow
assert v2.enabled and v2.mode=="act" and v2.actuation_scorer=="m4_shield"
assert v2.candidate_window_policy=="controlled_frontier" and v2.minimum_service_multiplier==0.98 and v2.maximum_service_multiplier==1.02
assert config.async_rl.controlled_release_delay.seed==20261007
assert config.async_rl.controlled_release_delay.assignment_domain=="m4-shield-qualification-v1"
locked=tomllib.loads(Path("uv.lock").read_text()); tq=[x for x in locked["package"] if x["name"].lower()=="transferqueue"]
assert len(tq)==1 and tq[0]["source"]["git"].endswith("#{TRANSFERQUEUE_COMMIT}")
direct=json.loads(distribution("TransferQueue").read_text("direct_url.json"))
assert direct["vcs_info"]["commit_id"]=="{TRANSFERQUEUE_COMMIT}"
assert not p["outcome_exclusion"]["terminal_benchmark"] and not p["outcome_exclusion"]["training_quality_analysis"]
print("M4_SHIELD_AUTHORITY_CONFIG_AND_TQ_PASS")
PY
"$PYTHON" -m pytest -q {CONTROLLER_TEST_PATH}
"$PYTHON" -m pytest -q {CONFIG_TEST_PATH} -k opportunity_at_risk_v2
"$PYTHON" -m pytest -q {ANALYZER_TEST_PATH}
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
  grep -Fq "SC run complete: {{'train_steps': 64, 'trainer_version': 64" "$RUN_LOG"
  set +e
  "$PYTHON" "{ANALYZER_PATH}" --oars "$OARS" --lifecycle "$LIFECYCLE" --observer-duty "$DUTY" --source-commit "{SOURCE_COMMIT}" --protocol-sha256 "{PROTOCOL_SHA256}" --run-start-ns "$START_NS" --run-end-ns "$END_NS" --output "$RESULT"
  readonly GATE_RC=$?
  set -e
else
  printf '{{"schema":"m4-shield-qualification-result-v1","scientific_outcome_acquisition":false,"status":"TRAINING_FAILURE","training_exit_code":%s,"training_quality_analyzed":false}}\n' "$TRAIN_RC" > "$RESULT"
  readonly GATE_RC=$TRAIN_RC
fi
find "$ASSETS" -maxdepth 1 -type f ! -name "$(basename "$HASHES")" -print0 | sort -z | xargs -0 sha256sum > "$HASHES"
exit "$GATE_RC"
'''
    # JET renders scripts through an f-string pass. Escape every runtime brace
    # except the one intentional assets placeholder.
    script = script.replace("{", "{{").replace("}", "}}")
    script = script.replace("{{assets_dir}}", "{assets_dir}")
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
    print(args.output, sha256(args.output), authorization_sha256)


if __name__ == "__main__":
    main()
