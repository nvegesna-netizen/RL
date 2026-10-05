"""Build the credential-free, outcome-excluded M4-Rescue qualification."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import tarfile
from pathlib import Path

SOURCE_COMMIT = "ded043d092e18f236163184d9648201414c6c30a"
SOURCE_SHA256 = "d690537324127afcd0689481c749aa585d3b3226428adde9a8cf25ddeb842e24"
MEGATRON_SHA256 = "98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
PROTOCOL_SHA256 = "d5c66a822a0945ee09afa317977cd95c8c96be1df53fb31677f4a13d2409bff3"
PARENT_PROTOCOL_SHA256 = (
    "28cac36401f8bfa27eb6f3c9741a8c692a78c98c42fb9c6642a841a2a8bb8661"
)
OFFLINE_RESULT_SHA256 = (
    "81a06219f12860b0c0f0e2753fec7f2e146180f3751c9cc28fccdd56d740158e"
)
CONFIG_SHA256 = "86500bc81e2aedc5bc51adfa6261acfcc03b4621f740f5410c26e298dea101c0"
ANALYZER_SHA256 = "26d6150336b5ae440595f4756244ac934716a33015fbec57501b8c693e0587fc"
RUNTIME_SHA256 = "fb0524f2c1bbe1304145c225e39b49721a22d160c4f3251241752833190fdf98"
CONTROLLER_TEST_SHA256 = (
    "da2dc31cf3293d812e017cf98d69a55df22e2b95c973798c0db907751e7808d7"
)
ANALYZER_TEST_SHA256 = (
    "84f57219fbcb7403fd448ecc77830fc359d142d29157f79179719c8a36ff32b3"
)
TRANSFERQUEUE_COMMIT = "b266d39a15aae114730de36cf8317b6285436f7f"
IMAGE_PATH = (
    "/lustre/fsw/coreai_dlalgo_ci/nvegesna/nemo_rl_images/nemo-rl-nightly-5802754.sqsh"
)
IMAGE_COMMIT = "ae07eafe8035b5b2e84efa7234e70e7fd7e493c1"
CONFIG_PATH = (
    "examples/configs/"
    "grpo_math_1B_megatron_single_controller_m4_rescue_qualification.yaml"
)
ANALYZER_PATH = "tools/m4_rescue_qualification.py"
RUNTIME_PATH = "nemo_rl/algorithms/async_utils/opportunity_at_risk_v2.py"
CONTROLLER_TEST_PATH = "tests/unit/single_controller/test_opportunity_at_risk_v2.py"
CONFIG_TEST_PATH = "tests/unit/single_controller/test_single_controller.py"
ANALYZER_TEST_PATH = (
    "reports/auto_research/2026-10-05-m4-rescue/test_m4_rescue_qualification.py"
)
PARENT_PROTOCOL_PATH = (
    "reports/auto_research/2026-10-05-m4-rescue/m4_rescue_protocol.json"
)
OFFLINE_RESULT_PATH = (
    "reports/auto_research/2026-10-05-m4-rescue/m4_rescue_offline_result.json"
)
NAME = "m4-rescue-outcome-excluded-qualification"


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
        authorization.get("schema") != "m4-rescue-live-qualification-authorization-v1"
        or authorization.get("scope")
        != "exactly_one_outcome_excluded_m4_rescue_live_qualification"
        or authorization.get("source_commit") != SOURCE_COMMIT
        or authorization.get("protocol_sha256") != PROTOCOL_SHA256
        or authorization.get("required_launcher") != "runllm.py --no_wait"
        or authorization.get("submission_attempt_limit") != 1
        or authorization.get("trainer_steps") != 64
        or authorization.get("training_seed") != 20261005
        or authorization.get("controlled_release_seed") != 20261006
        or authorization.get("assignment_domain") != "m4-rescue-qualification-v1"
        or authorization.get("nodes") != 1
        or authorization.get("gpus") != 2
        or authorization.get("scheduler_time_limit_seconds") != 14400
        or authorization.get("queue_deadline_override") is not None
        or authorization.get("selection_candidate_watermark") != 8
        or authorization.get("selection_cardinality") != 4
        or authorization.get("actuation_scorer") != "m4_rescue"
        or not authorization.get("eos_submission_authorized")
        or not authorization.get("m4_rescue_actuation_authorized")
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
print("M4_RESCUE_SAFE_SOURCE_PASS",len(names),len(symlinks))
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
print("M4_RESCUE_MEGATRON_PASS")
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
assert p["status"]=="FROZEN_AFTER_OFFLINE_PASS_BEFORE_SUBMISSION"
assert p["source_commit"]==a["source_commit"]=="{SOURCE_COMMIT}"
assert p["parent_protocol_sha256"]=="{PARENT_PROTOCOL_SHA256}"
assert p["offline_result_sha256"]=="{OFFLINE_RESULT_SHA256}"
assert hashlib.sha256(Path("{PARENT_PROTOCOL_PATH}").read_bytes()).hexdigest()=="{PARENT_PROTOCOL_SHA256}"
offline=Path("{OFFLINE_RESULT_PATH}")
assert hashlib.sha256(offline.read_bytes()).hexdigest()=="{OFFLINE_RESULT_SHA256}"
assert json.loads(offline.read_bytes())["status"]=="PASS_M4_RESCUE_OFFLINE_QUALIFICATION"
for path,expected in (("{CONFIG_PATH}","{CONFIG_SHA256}"),("{ANALYZER_PATH}","{ANALYZER_SHA256}"),("{RUNTIME_PATH}","{RUNTIME_SHA256}"),("{CONTROLLER_TEST_PATH}","{CONTROLLER_TEST_SHA256}"),("{ANALYZER_TEST_PATH}","{ANALYZER_TEST_SHA256}")):
 assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
register_omegaconf_resolvers(); config=MasterConfig(**OmegaConf.to_container(load_config(Path("{CONFIG_PATH}")),resolve=True)); validate_single_controller_config(config)
assert config.grpo.max_num_steps==64 and config.grpo.seed==20261005
assert config.cluster["num_nodes"]==1 and config.cluster["gpus_per_node"]==2
assert config.async_rl.sampler.name=="weight_fifo" and config.async_rl.sampler.selection_candidate_watermark==8
assert config.async_rl.sampler.max_staleness_versions==1
v2=config.async_rl.opportunity_at_risk_v2_shadow
assert v2.enabled and v2.mode=="act" and v2.actuation_scorer=="m4_rescue"
assert v2.candidate_window_policy=="controlled_frontier" and v2.minimum_service_multiplier==0.98 and v2.maximum_service_multiplier==1.02
assert config.async_rl.controlled_release_delay.seed==20261006
assert config.async_rl.controlled_release_delay.assignment_domain=="m4-rescue-qualification-v1"
locked=tomllib.loads(Path("uv.lock").read_text()); tq=[x for x in locked["package"] if x["name"].lower()=="transferqueue"]
assert len(tq)==1 and tq[0]["source"]["git"].endswith("#{TRANSFERQUEUE_COMMIT}")
direct=json.loads(distribution("TransferQueue").read_text("direct_url.json"))
assert direct["vcs_info"]["commit_id"]=="{TRANSFERQUEUE_COMMIT}"
assert not p["outcome_exclusion"]["terminal_benchmark"] and not p["outcome_exclusion"]["training_quality_analysis"]
print("M4_RESCUE_AUTHORITY_CONFIG_AND_TQ_PASS")
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
  printf '{{"schema":"m4-rescue-qualification-result-v1","scientific_outcome_acquisition":false,"status":"TRAINING_FAILURE","training_exit_code":%s,"training_quality_analyzed":false}}\n' "$TRAIN_RC" > "$RESULT"
  readonly GATE_RC=$TRAIN_RC
fi
find "$ASSETS" -maxdepth 1 -type f ! -name "$(basename "$HASHES")" -print0 | sort -z | xargs -0 sha256sum > "$HASHES"
exit "$GATE_RC"
'''
    # JET renders workload scripts through an f-string substitution pass. Preserve
    # its one intentional placeholder while escaping every brace that belongs to
    # Bash, embedded Python, log matching, or JSON emitted at runtime.
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
