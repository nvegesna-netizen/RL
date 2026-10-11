#!/usr/bin/env python3
"""Build the credential-free eligible-live v2 liveness diagnostic manifest."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import tarfile
from pathlib import Path


SOURCE_COMMIT = "f70361f2cc3568a2f43fd56bacbf440b08a6b977"
SOURCE_SHA256 = "649c5e28de8cbba98ba83e50259f3376f64039cf003cec43c5ce2bd19026d304"
MEGATRON_SHA256 = "98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
PROTOCOL_SHA256 = "59f99259d56d8bc993c59e3291e4ccf3f2ecf9832729e72e81222ee01f4dbc9c"
AUTHORIZATION_SHA256 = (
    "796a60d0ab52ef782b91a1390e5ae2502c12db0490ae41ee2c40b9793b2f6de0"
)
LOCK_SHA256 = "113758c56418359bb50ec367369d385a78206651ed7a6d37d9fab7fe40c0d6ce"
V1_TERMINAL_SHA256 = "cebc287b0edd71c313bf1b70a21e70ff9f8c7f844880c4ab0cf459d19f02b456"
CAUSAL_PROTOCOL_SHA256 = (
    "716ba0eb8f980ca5af0ef3d57e2415befc1fb1a315afeb01b9bc326a60341b48"
)
TRANSFERQUEUE_COMMIT = "b266d39a15aae114730de36cf8317b6285436f7f"
IMAGE_PATH = (
    "/lustre/fsw/coreai_dlalgo_ci/nvegesna/nemo_rl_images/nemo-rl-nightly-5802754.sqsh"
)
IMAGE_COMMIT = "ae07eafe8035b5b2e84efa7234e70e7fd7e493c1"
CONFIG_PATH = (
    "examples/configs/"
    "grpo_math_1B_megatron_single_controller_m4_eligible_live_capture.yaml"
)
REPORT_DIR = "reports/auto_research/2026-10-07-m4-causal-utility"
CAUSAL_PROTOCOL_PATH = f"{REPORT_DIR}/eligible_live_decision_causal_protocol.json"
ANALYZER_PATH = f"{REPORT_DIR}/analyze_eligible_live_decision_capture.py"
FILE_HASHES = {
    CONFIG_PATH: "9e39ff918e10096400fdce8e3d9129c37bddce690c5f196d6b855dc8a01e7d10",
    CAUSAL_PROTOCOL_PATH: CAUSAL_PROTOCOL_SHA256,
    ANALYZER_PATH: "bd6734b03f9be962377373ce2f019e5243e07796ae5297af0fe233b3e16bd6ee",
    f"{REPORT_DIR}/eligible_live_v2_liveness_diagnostic_protocol.json": PROTOCOL_SHA256,
    "nemo_rl/algorithms/async_utils/eligible_live_decision_capsule.py": "d647cf0113cc4caefa244e61f26622f1453ee569b60a30105b17ac486c0d8913",
    "nemo_rl/algorithms/async_utils/opportunity_at_risk_v2.py": "d6221b7f9766d07f9e1a4f1a0886af9c5625f3e4a6eaf0ddc283078c6d982c3b",
    "nemo_rl/algorithms/async_utils/rollout_lifecycle.py": "4d62574c1695dbad310e5fd80c2ee65d7f1efeab22fb283b1e99fd11a0ef22ac",
    "nemo_rl/algorithms/single_controller.py": "f5ff5e7cf0485b876c4047b8894dfb5402873df7bc59f081ffc6016390abb93c",
    "nemo_rl/algorithms/single_controller_utils/config.py": "bec32196c6208f5dbeb4729b9ade070519be6fa85fc4417463aad87949dc9501",
    "tests/unit/single_controller/test_eligible_live_decision_capsule.py": "39d44b35ab5ad1a23bd85bfdd8830bdfa72cc5ae605d219cd579f793ebe7ffc0",
    "tests/unit/single_controller/test_opportunity_at_risk_v2.py": "e5de5f257460c04de4d157ab755c2407d6ab93e04f91f9caf34407b7ef541515",
    "tests/unit/single_controller/test_single_controller.py": "da13cce6a9c61c7c23406209409eb58f3995346e8ffe67dd4b5894c3ef7df294",
    "uv.lock": "bc69ceddc9f35d2c120235c96afac2e6c7b76827e044efda3e61f908be72c4cc",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoded(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--v1-terminal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path, expected in (
        (args.source, SOURCE_SHA256),
        (args.megatron, MEGATRON_SHA256),
        (args.protocol, PROTOCOL_SHA256),
        (args.authorization, AUTHORIZATION_SHA256),
        (args.execution_lock, LOCK_SHA256),
        (args.v1_terminal, V1_TERMINAL_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen diagnostic input moved: {path}")

    protocol = json.loads(args.protocol.read_bytes())
    authorization = json.loads(args.authorization.read_bytes())
    lock = json.loads(args.execution_lock.read_bytes())
    v1_terminal = json.loads(args.v1_terminal.read_bytes())
    if (
        protocol["status"] != "FROZEN_BEFORE_DIAGNOSTIC_EXECUTION"
        or protocol["diagnostic_identity"]["seed"] != 20261102
        or protocol["instrument"]["progress_timeout_seconds"] != 1200
        or protocol["execution"]["workload_time_limit_seconds"] != 14400
        or protocol["execution"]["client_queue_deadline"] is not None
        or authorization["status"] != "AUTHORIZED_BEFORE_DIAGNOSTIC_EXECUTION"
        or authorization["authorized"]["diagnostic_eos_submission_attempts"] != 1
        or lock["source_commit"] != SOURCE_COMMIT
        or lock["source_archive_sha256"] != SOURCE_SHA256
        or lock["execution"]["submission_attempt_limit"] != 1
        or lock["execution"]["client_queue_deadline"] is not None
        or lock["execution"]["automatic_retry"]
        or lock["outcome_boundary"]["v2_acquisition_started"]
        or lock["outcome_boundary"]["paired_replay_started"]
        or lock["outcome_boundary"]["post_update_loss_computed"]
        or v1_terminal["status"] != "TERMINAL_GATE_FAILED_CLOSED"
        or v1_terminal["paired_replay_opened"]
        or v1_terminal["post_update_outcomes_opened"]
    ):
        raise RuntimeError("eligible-live v2 diagnostic boundary differs")

    with tarfile.open(args.source, "r:gz") as archive:
        source_names = set(archive.getnames())
    if not set(FILE_HASHES).issubset(source_names):
        raise RuntimeError("source archive lacks required diagnostic files")

    template = r"""set -euo pipefail
readonly PYTHON=/opt/nemo_rl_venv/bin/python
readonly RUN_REPO=/workspace/m4-eligible-live-v2-diagnostic-repo
readonly SOURCE=/workspace/m4-eligible-live-v2-diagnostic-source.tar.gz
readonly MEGATRON=/workspace/m4-eligible-live-v2-diagnostic-megatron.tar.gz
readonly PROTOCOL=/workspace/m4-eligible-live-v2-diagnostic-protocol.json
readonly AUTHORIZATION=/workspace/m4-eligible-live-v2-diagnostic-authorization.json
readonly EXECUTION_LOCK=/workspace/m4-eligible-live-v2-diagnostic-lock.json
readonly V1_TERMINAL=/workspace/m4-eligible-live-v1-terminal.json
readonly IDENTITY=d01
readonly SEED=20261102
readonly ASSIGNMENT_DOMAIN=m4-eligible-live-v2-liveness-diagnostic-v1
readonly ASSETS={assets_dir}
readonly RUN_LOG=$ASSETS/m4-eligible-live-v2-$IDENTITY-run.log
readonly TEST_LOG=$ASSETS/m4-eligible-live-v2-$IDENTITY-tests.log
readonly LIFECYCLE=$ASSETS/m4-eligible-live-v2-$IDENTITY-lifecycle.jsonl
readonly OPPORTUNITY=$ASSETS/m4-eligible-live-v2-$IDENTITY-opportunity.jsonl
readonly DUTY=$ASSETS/m4-eligible-live-v2-$IDENTITY-observer-duty.json
readonly OARS=$ASSETS/m4-eligible-live-v2-$IDENTITY-oars-v2.jsonl
readonly LIVENESS=$ASSETS/m4-eligible-live-v2-$IDENTITY-liveness.json
readonly CAPSULE=$ASSETS/m4-eligible-live-v2-$IDENTITY-capsule
readonly CAPTURE_RESULT=$ASSETS/m4-eligible-live-v2-$IDENTITY-capture-result.json
readonly RESULT=$ASSETS/m4-eligible-live-v2-$IDENTITY-diagnostic-result.json
readonly HASHES=$ASSETS/m4-eligible-live-v2-$IDENTITY-artifacts.sha256
readonly METRICS=$ASSETS/m4-eligible-live-v2-$IDENTITY-metrics
test "${NEMO_RL_COMMIT:-unknown}" = "@IMAGE_COMMIT@"
test ! -e "$RUN_REPO"
printf %s '@SOURCE_PAYLOAD@' | base64 -d > "$SOURCE"
printf %s '@MEGATRON_PAYLOAD@' | base64 -d > "$MEGATRON"
printf %s '@PROTOCOL_PAYLOAD@' | base64 -d > "$PROTOCOL"
printf %s '@AUTHORIZATION_PAYLOAD@' | base64 -d > "$AUTHORIZATION"
printf %s '@LOCK_PAYLOAD@' | base64 -d > "$EXECUTION_LOCK"
printf %s '@V1_TERMINAL_PAYLOAD@' | base64 -d > "$V1_TERMINAL"
test "$(sha256sum "$SOURCE" | cut -d' ' -f1)" = "@SOURCE_SHA256@"
test "$(sha256sum "$MEGATRON" | cut -d' ' -f1)" = "@MEGATRON_SHA256@"
test "$(sha256sum "$PROTOCOL" | cut -d' ' -f1)" = "@PROTOCOL_SHA256@"
test "$(sha256sum "$AUTHORIZATION" | cut -d' ' -f1)" = "@AUTHORIZATION_SHA256@"
test "$(sha256sum "$EXECUTION_LOCK" | cut -d' ' -f1)" = "@LOCK_SHA256@"
test "$(sha256sum "$V1_TERMINAL" | cut -d' ' -f1)" = "@V1_TERMINAL_SHA256@"
"$PYTHON" - "$SOURCE" "$RUN_REPO" <<'PY'
import sys,tarfile
from pathlib import Path,PurePosixPath
archive,output=map(Path,sys.argv[1:])
with tarfile.open(archive,"r:gz") as source:
    members=source.getmembers(); names=[PurePosixPath(m.name) for m in members]
    assert len(names)==len(set(names))
    assert not any(n.is_absolute() or ".." in n.parts for n in names)
    symlinks={n for n,m in zip(names,members,strict=True) if m.issym()}
    assert not any(any(parent in symlinks for parent in n.parents) for n in names)
    source.extractall(output)
print("M4_ELIGIBLE_LIVE_V2_DIAGNOSTIC_SAFE_SOURCE_PASS",len(names),len(symlinks))
PY
"$PYTHON" - "$MEGATRON" "$RUN_REPO" <<'PY'
import sys,tarfile
from pathlib import Path,PurePosixPath
archive,root=map(Path,sys.argv[1:])
dest=root/"3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty"
with tarfile.open(archive,"r:gz") as source:
    members=source.getmembers(); names=[PurePosixPath(m.name) for m in members]
    assert len(members)==726 and len(names)==len(set(names))
    assert all(str(n)=="Megatron-LM" or str(n).startswith("Megatron-LM/") for n in names)
    assert not any(n.is_absolute() or ".." in n.parts for n in names)
    symlinks={n for n,m in zip(names,members,strict=True) if m.issym()}
    assert not any(any(parent in symlinks for parent in n.parents) for n in names)
    dest.mkdir(parents=True,exist_ok=True); source.extractall(dest)
assert (dest/"Megatron-LM/megatron/core/__init__.py").is_file()
print("M4_ELIGIBLE_LIVE_V2_DIAGNOSTIC_MEGATRON_PASS")
PY
cd "$RUN_REPO"
export PYTHONPATH="$RUN_REPO"
export CUDA_VISIBLE_DEVICES=0,1
mkdir -p "$METRICS"
"$PYTHON" - "$PROTOCOL" "$AUTHORIZATION" "$EXECUTION_LOCK" "$V1_TERMINAL" <<'PY'
import ast,hashlib,json,sys
from importlib.metadata import distribution
from pathlib import Path
from omegaconf import OmegaConf
from nemo_rl.algorithms.single_controller_utils.config import MasterConfig,validate_single_controller_config
from nemo_rl.utils.config import load_config,register_omegaconf_resolvers
p_path,a_path,l_path,v1_path=map(Path,sys.argv[1:])
p=json.loads(p_path.read_bytes()); auth=json.loads(a_path.read_bytes())
lock=json.loads(l_path.read_bytes()); v1=json.loads(v1_path.read_bytes())
assert hashlib.sha256(p_path.read_bytes()).hexdigest()=="@PROTOCOL_SHA256@"
assert hashlib.sha256(a_path.read_bytes()).hexdigest()=="@AUTHORIZATION_SHA256@"
assert hashlib.sha256(l_path.read_bytes()).hexdigest()=="@LOCK_SHA256@"
assert hashlib.sha256(v1_path.read_bytes()).hexdigest()=="@V1_TERMINAL_SHA256@"
assert p["diagnostic_identity"]["seed"]==20261102
assert p["instrument"]["progress_timeout_seconds"]==1200
assert auth["authorized"]["diagnostic_eos_submission_attempts"]==1
assert lock["source_commit"]=="@SOURCE_COMMIT@"
assert v1["status"]=="TERMINAL_GATE_FAILED_CLOSED"
assert not v1["paired_replay_opened"] and not v1["post_update_outcomes_opened"]
file_hashes=@FILE_HASHES_JSON@
for name,expected in file_hashes.items():
    actual=hashlib.sha256(Path(name).read_bytes()).hexdigest(); assert actual==expected,(name,actual,expected)
source=Path("nemo_rl/algorithms/single_controller.py").read_text(); tree=ast.parse(source)
diagnostic_names={"_eligible_live_liveness_state","_task_liveness_state","_eligible_live_progress_watchdog"}
diagnostic_nodes=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in diagnostic_names]
assert {n.name for n in diagnostic_nodes}==diagnostic_names
diagnostic_source="\n".join(ast.get_source_segment(source,n) or "" for n in diagnostic_nodes)
for forbidden in ("heldout_loss","post_update_loss","base_arm_outcome","shield_arm_outcome","causal_effect"):
    assert forbidden not in diagnostic_source
locked=Path("uv.lock").read_text(); assert "@TRANSFERQUEUE_COMMIT@" in locked
direct=json.loads(distribution("TransferQueue").read_text("direct_url.json"))
assert direct["vcs_info"]["commit_id"]=="@TRANSFERQUEUE_COMMIT@"
register_omegaconf_resolvers()
raw=OmegaConf.to_container(load_config(Path("@CONFIG_PATH@")),resolve=True)
raw["grpo"]["seed"]=20261102
raw["async_rl"]["controlled_release_delay"]["seed"]=20261102
raw["async_rl"]["controlled_release_delay"]["assignment_domain"]="m4-eligible-live-v2-liveness-diagnostic-v1"
capture=raw["async_rl"]["eligible_live_decision_capture"]
capture["trajectory_identity"]="d01"; capture["trajectory_seed"]=20261102
capture["progress_timeout_seconds"]=1200
capture["liveness_diagnostic_path"]="/tmp/m4-eligible-live-v2-liveness.json"
config=MasterConfig(**raw); validate_single_controller_config(config)
assert config.grpo.max_num_steps==64
assert config.async_rl.eligible_live_decision_capture.progress_timeout_seconds==1200
print("M4_ELIGIBLE_LIVE_V2_DIAGNOSTIC_RUNTIME_AUTHORITY_PASS")
PY
set +e
{
  "$PYTHON" -m pytest -q tests/unit/single_controller/test_eligible_live_decision_capsule.py
  "$PYTHON" -m pytest -q tests/unit/single_controller/test_opportunity_at_risk_v2.py -k 'm4_shield or liveness_diagnostic_state'
  "$PYTHON" -m pytest -q tests/unit/single_controller/test_single_controller.py -k 'eligible_live'
} 2>&1 | tee "$TEST_LOG"
readonly TEST_RC=${PIPESTATUS[0]}
set -e
test "$TEST_RC" -eq 0
set +e
"$PYTHON" examples/run_grpo_single_controller.py --config "@CONFIG_PATH@" \
  "grpo.seed=$SEED" \
  "async_rl.controlled_release_delay.seed=$SEED" \
  "async_rl.controlled_release_delay.assignment_domain=$ASSIGNMENT_DOMAIN" \
  "async_rl.eligible_live_decision_capture.trajectory_identity=$IDENTITY" \
  "async_rl.eligible_live_decision_capture.trajectory_seed=$SEED" \
  "async_rl.eligible_live_decision_capture.progress_timeout_seconds=1200" \
  "async_rl.eligible_live_decision_capture.liveness_diagnostic_path=$LIVENESS" \
  "logger.log_dir=$METRICS" \
  "async_rl.lifecycle_audit_path=$LIFECYCLE" \
  "async_rl.gradient_opportunity_audit.output_path=$OPPORTUNITY" \
  "async_rl.gradient_opportunity_audit.observer_duty_path=$DUTY" \
  "async_rl.opportunity_at_risk_v2_shadow.output_path=$OARS" \
  "async_rl.eligible_live_decision_capture.output_dir=$CAPSULE" 2>&1 | tee "$RUN_LOG"
readonly RUN_RC=${PIPESTATUS[0]}
set -e
if test "$RUN_RC" -eq 0 && test -f "$CAPSULE/manifest.json"; then
  "$PYTHON" "@ANALYZER_PATH@" --capsule-dir "$CAPSULE" \
    --protocol "@CAUSAL_PROTOCOL_PATH@" --output "$CAPTURE_RESULT"
fi
set +e
"$PYTHON" - "$PROTOCOL" "$RUN_RC" "$LIVENESS" "$LIFECYCLE" "$OPPORTUNITY" "$DUTY" "$OARS" "$CAPTURE_RESULT" "$RUN_LOG" "$RESULT" <<'PY'
import json,sys
from pathlib import Path
protocol_path=Path(sys.argv[1]); run_rc=int(sys.argv[2]); liveness=Path(sys.argv[3])
lifecycle,opportunity,duty,oars=map(Path,sys.argv[4:8])
capture_result,run_log,result_path=map(Path,sys.argv[8:11])
protocol=json.loads(protocol_path.read_bytes())
required_ledgers=(lifecycle,opportunity,duty,oars)
result={
    "schema":"m4-shield-eligible-live-v2-liveness-diagnostic-result-v1",
    "status":"FAIL_DIAGNOSTIC_INSTRUMENT",
    "identity":"d01",
    "seed":20261102,
    "assignment_domain":"m4-eligible-live-v2-liveness-diagnostic-v1",
    "run_exit_code":run_rc,
    "v2_acquisition_started":False,
    "paired_replay_started":False,
    "post_update_outcomes_opened":False,
    "causal_effect_estimated":False,
}
if liveness.is_file():
    diagnostic=json.loads(liveness.read_bytes())
    keys=set()
    def visit(value):
        if isinstance(value,dict):
            keys.update(value); [visit(item) for item in value.values()]
        elif isinstance(value,list):
            [visit(item) for item in value]
    visit(diagnostic)
    forbidden={"heldout_loss","post_update_loss","base_arm_outcome","shield_arm_outcome","causal_effect"}
    assert not (keys&forbidden)
    assert run_rc!=0
    assert diagnostic["schema"]=="eligible-live-progress-timeout-v1"
    assert diagnostic["trajectory_identity"]=="d01"
    assert diagnostic["trajectory_seed"]==20261102
    assert diagnostic["timeout_seconds"]==1200
    assert diagnostic["seconds_without_learner_progress"]>=1200
    assert diagnostic["tasks"]["rollout_pump"]["stack"]
    assert diagnostic["tasks"]["train_pump"]["stack"]
    assert all(path.is_file() and path.stat().st_size>0 for path in required_ledgers)
    result["status"]="PASS_DIAGNOSTIC_TIMEOUT_CAPTURED"
    result["diagnostic_phase"]=diagnostic["controller_state"]["eligible_live_progress_phase"]
    result["last_completed_update"]=diagnostic["controller_state"]["train_steps"]
elif capture_result.is_file() and run_rc==0:
    capture=json.loads(capture_result.read_bytes())
    assert capture["status"]=="PASS_OUTCOME_EXCLUDED_ELIGIBLE_CAPTURE"
    assert capture["trajectory_identity"]=="d01" and capture["trajectory_seed"]==20261102
    assert capture["post_update_outcomes_opened"] is False
    result["status"]="PASS_NO_STALL_REPRODUCED_CAPTURE_EXCLUDED"
    result["learner_version"]=capture["learner_version"]
elif "eligible live decision capture ended without a capsule" in run_log.read_text():
    rows=[json.loads(line) for line in lifecycle.read_text().splitlines() if line]
    versions=[row for row in rows if row["stage"]=="learner_version_advanced"]
    assert len(versions)==64 and max(row["learner_weight_version"] for row in versions)==64
    assert all(path.is_file() and path.stat().st_size>0 for path in required_ledgers)
    result["status"]="PASS_NO_STALL_REPRODUCED_NO_ELIGIBILITY"
    result["last_completed_update"]=64
result_path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
assert result["status"] in protocol["terminal_classification"]
print(result["status"])
PY
readonly GATE_RC=$?
set -e
find "$ASSETS" -type f ! -name "$(basename "$HASHES")" -print0 | sort -z | xargs -0 sha256sum > "$HASHES"
exit "$GATE_RC"
"""

    substitutions = {
        "@SOURCE_PAYLOAD@": encoded(args.source),
        "@MEGATRON_PAYLOAD@": encoded(args.megatron),
        "@PROTOCOL_PAYLOAD@": encoded(args.protocol),
        "@AUTHORIZATION_PAYLOAD@": encoded(args.authorization),
        "@LOCK_PAYLOAD@": encoded(args.execution_lock),
        "@V1_TERMINAL_PAYLOAD@": encoded(args.v1_terminal),
        "@IMAGE_COMMIT@": IMAGE_COMMIT,
        "@SOURCE_COMMIT@": SOURCE_COMMIT,
        "@SOURCE_SHA256@": SOURCE_SHA256,
        "@MEGATRON_SHA256@": MEGATRON_SHA256,
        "@PROTOCOL_SHA256@": PROTOCOL_SHA256,
        "@AUTHORIZATION_SHA256@": AUTHORIZATION_SHA256,
        "@LOCK_SHA256@": LOCK_SHA256,
        "@V1_TERMINAL_SHA256@": V1_TERMINAL_SHA256,
        "@FILE_HASHES_JSON@": json.dumps(FILE_HASHES, sort_keys=True),
        "@TRANSFERQUEUE_COMMIT@": TRANSFERQUEUE_COMMIT,
        "@CONFIG_PATH@": CONFIG_PATH,
        "@CAUSAL_PROTOCOL_PATH@": CAUSAL_PROTOCOL_PATH,
        "@ANALYZER_PATH@": ANALYZER_PATH,
    }
    script = template
    for token, value in substitutions.items():
        script = script.replace(token, value)
    if "@" in script:
        raise RuntimeError("unsubstituted diagnostic manifest token")
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
            "name": "m4-eligible-live-v2-liveness-diagnostic-d01",
            "workspace": "/workspace",
            "nodes": 1,
            "time_limit": 14400,
            "image_source": {"local_path": IMAGE_PATH},
            "script": script,
        },
    }
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(args.output, sha256(args.output))


if __name__ == "__main__":
    main()
