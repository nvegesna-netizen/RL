#!/usr/bin/env python3
"""Build six credential-free eligible-live v2 liveness manifests."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import tarfile
from pathlib import Path


SOURCE_COMMIT = "11143be695f3a59b934a6522d89f1dc6a6a27ce6"
SOURCE_SHA256 = "ff8cb7f09b64b6ff5d781c12c52d566c8c8821ebab559d569e322ec2fabe691f"
MEGATRON_SHA256 = "98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
PROTOCOL_SHA256 = "eabfcb0fae3f70206dc57182f7c25661a893d77418f8b8ea05daef6730003479"
AUTHORIZATION_SHA256 = "aee963ca3dd20951bfba9d71056496e870120b3a52bb68979b272d0d33cb177a"
LOCK_SHA256 = "12cf416f41fb814a5a6f9c038c83ce88fbbc87146fc1963ea51b7a49f8e65a2b"
V1_TERMINAL_SHA256 = "cebc287b0edd71c313bf1b70a21e70ff9f8c7f844880c4ab0cf459d19f02b456"
DIAGNOSTIC_TERMINAL_SHA256 = (
    "e2e9ea887d066902f0b64f0cb9b841ef45ffc502c36ccde48626fe6f0c36642c"
)
TRANSFERQUEUE_COMMIT = "b266d39a15aae114730de36cf8317b6285436f7f"
IMAGE_PATH = (
    "/lustre/fsw/coreai_dlalgo_ci/nvegesna/nemo_rl_images/"
    "nemo-rl-nightly-5802754.sqsh"
)
IMAGE_COMMIT = "ae07eafe8035b5b2e84efa7234e70e7fd7e493c1"
CONFIG_PATH = (
    "examples/configs/"
    "grpo_math_1B_megatron_single_controller_m4_eligible_live_capture.yaml"
)
REPORT_DIR = "reports/auto_research/2026-10-07-m4-causal-utility"
ASSIGNMENT_DOMAIN = "m4-eligible-live-v2-forced-no-eligibility-liveness-v1"
IDENTITIES = (
    ("lq01", 20261102),
    ("lq02", 20261103),
    ("lq03", 20261109),
    ("lq04", 20261110),
    ("lq05", 20261111),
    ("lq06", 20261119),
)
FILE_HASHES = {
    CONFIG_PATH: "94b6bb744beca304be20371135b9c0251d9e2b3d4b870b4be86d9c1c0be112f2",
    f"{REPORT_DIR}/eligible_live_v2_liveness_qualification_protocol.json": PROTOCOL_SHA256,
    f"{REPORT_DIR}/eligible_live_v2_liveness_qualification_authorization.json": AUTHORIZATION_SHA256,
    f"{REPORT_DIR}/eligible_live_acquisition_v1_terminal_authentication.json": V1_TERMINAL_SHA256,
    f"{REPORT_DIR}/eligible_live_v2_liveness_diagnostic_terminal_authentication.json": DIAGNOSTIC_TERMINAL_SHA256,
    "nemo_rl/algorithms/async_utils/opportunity_at_risk_v2.py": "d6221b7f9766d07f9e1a4f1a0886af9c5625f3e4a6eaf0ddc283078c6d982c3b",
    "nemo_rl/algorithms/single_controller.py": "97164bd7649830c9a5499f0f6825e257c3314a34e5b27488955213de3d02047f",
    "nemo_rl/algorithms/single_controller_utils/config.py": "e7f86e102ebb5094e65cd377ac531cf9c7152b6e43362065021721d16327dab4",
    "tests/unit/single_controller/test_opportunity_at_risk_v2.py": "e5de5f257460c04de4d157ab755c2407d6ab93e04f91f9caf34407b7ef541515",
    "tests/unit/single_controller/test_single_controller.py": "ca1879a47ee7d4a8aa2322d59fd137fbbd7fb1ff3fc94012639730b70a7e15ff",
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
    parser.add_argument("--diagnostic-terminal", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    expected_inputs = (
        (args.source, SOURCE_SHA256),
        (args.megatron, MEGATRON_SHA256),
        (args.protocol, PROTOCOL_SHA256),
        (args.authorization, AUTHORIZATION_SHA256),
        (args.execution_lock, LOCK_SHA256),
        (args.v1_terminal, V1_TERMINAL_SHA256),
        (args.diagnostic_terminal, DIAGNOSTIC_TERMINAL_SHA256),
    )
    for path, expected in expected_inputs:
        if sha256(path) != expected:
            raise RuntimeError(f"frozen qualification input moved: {path}")

    protocol = json.loads(args.protocol.read_bytes())
    authorization = json.loads(args.authorization.read_bytes())
    lock = json.loads(args.execution_lock.read_bytes())
    if (
        protocol["status"] != "FROZEN_BEFORE_IMPLEMENTATION_AND_EXECUTION"
        or tuple(
            (item["identity"], item["seed"])
            for item in protocol["qualification"]["identities"]
        )
        != IDENTITIES
        or protocol["joint_gate"]["required_passes"] != 6
        or authorization["status"]
        != "AUTHORIZED_BEFORE_IMPLEMENTATION_AND_EXECUTION"
        or authorization["authorized"]["qualification_eos_submission_attempts"]
        != 6
        or lock["source_commit"] != SOURCE_COMMIT
        or lock["source_archive_sha256"] != SOURCE_SHA256
        or lock["execution"]["submission_attempt_limit_total"] != 6
        or lock["execution"]["client_queue_deadline"] is not None
        or lock["execution"]["automatic_retry"]
        or lock["outcome_boundary"]["v2_acquisition_started"]
        or lock["outcome_boundary"]["paired_replay_started"]
    ):
        raise RuntimeError("eligible-live liveness qualification boundary differs")

    with tarfile.open(args.source, "r:gz") as archive:
        source_names = set(archive.getnames())
    if not set(FILE_HASHES).issubset(source_names):
        raise RuntimeError("source archive lacks required qualification files")

    common_payloads = {
        "@SOURCE_PAYLOAD@": encoded(args.source),
        "@MEGATRON_PAYLOAD@": encoded(args.megatron),
        "@PROTOCOL_PAYLOAD@": encoded(args.protocol),
        "@AUTHORIZATION_PAYLOAD@": encoded(args.authorization),
        "@LOCK_PAYLOAD@": encoded(args.execution_lock),
        "@V1_TERMINAL_PAYLOAD@": encoded(args.v1_terminal),
        "@DIAGNOSTIC_TERMINAL_PAYLOAD@": encoded(args.diagnostic_terminal),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for identity, seed in IDENTITIES:
        script = _runtime_template()
        substitutions = common_payloads | {
            "@IDENTITY@": identity,
            "@SEED@": str(seed),
            "@ASSIGNMENT_DOMAIN@": ASSIGNMENT_DOMAIN,
            "@IMAGE_COMMIT@": IMAGE_COMMIT,
            "@SOURCE_COMMIT@": SOURCE_COMMIT,
            "@SOURCE_SHA256@": SOURCE_SHA256,
            "@MEGATRON_SHA256@": MEGATRON_SHA256,
            "@PROTOCOL_SHA256@": PROTOCOL_SHA256,
            "@AUTHORIZATION_SHA256@": AUTHORIZATION_SHA256,
            "@LOCK_SHA256@": LOCK_SHA256,
            "@V1_TERMINAL_SHA256@": V1_TERMINAL_SHA256,
            "@DIAGNOSTIC_TERMINAL_SHA256@": DIAGNOSTIC_TERMINAL_SHA256,
            "@FILE_HASHES_JSON@": json.dumps(FILE_HASHES, sort_keys=True),
            "@TRANSFERQUEUE_COMMIT@": TRANSFERQUEUE_COMMIT,
            "@CONFIG_PATH@": CONFIG_PATH,
        }
        for token, value in substitutions.items():
            script = script.replace(token, value)
        if "@" in script:
            raise RuntimeError(f"unsubstituted token in {identity}")
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
                "name": f"m4-eligible-live-v2-liveness-qualification-{identity}",
                "workspace": "/workspace",
                "nodes": 1,
                "time_limit": 14400,
                "image_source": {"local_path": IMAGE_PATH},
                "script": script,
            },
        }
        output = args.output_dir / f"{identity}.json"
        output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(identity, seed, output, sha256(output))


def _runtime_template() -> str:
    return r'''set -euo pipefail
readonly PYTHON=/opt/nemo_rl_venv/bin/python
readonly RUN_REPO=/workspace/m4-eligible-live-v2-liveness-qualification-repo
readonly SOURCE=/workspace/m4-eligible-live-v2-liveness-source.tar.gz
readonly MEGATRON=/workspace/m4-eligible-live-v2-liveness-megatron.tar.gz
readonly PROTOCOL=/workspace/m4-eligible-live-v2-liveness-protocol.json
readonly AUTHORIZATION=/workspace/m4-eligible-live-v2-liveness-authorization.json
readonly EXECUTION_LOCK=/workspace/m4-eligible-live-v2-liveness-lock.json
readonly V1_TERMINAL=/workspace/m4-eligible-live-v1-terminal.json
readonly DIAGNOSTIC_TERMINAL=/workspace/m4-eligible-live-v2-diagnostic-terminal.json
readonly IDENTITY=@IDENTITY@
readonly SEED=@SEED@
readonly ASSIGNMENT_DOMAIN=@ASSIGNMENT_DOMAIN@
readonly ASSETS={assets_dir}
readonly RUN_LOG=$ASSETS/m4-eligible-live-v2-$IDENTITY-run.log
readonly TEST_LOG=$ASSETS/m4-eligible-live-v2-$IDENTITY-tests.log
readonly LIFECYCLE=$ASSETS/m4-eligible-live-v2-$IDENTITY-lifecycle.jsonl
readonly OPPORTUNITY=$ASSETS/m4-eligible-live-v2-$IDENTITY-opportunity.jsonl
readonly DUTY=$ASSETS/m4-eligible-live-v2-$IDENTITY-observer-duty.json
readonly OARS=$ASSETS/m4-eligible-live-v2-$IDENTITY-oars-v2.jsonl
readonly LIVENESS=$ASSETS/m4-eligible-live-v2-$IDENTITY-liveness.json
readonly CAPSULE=$ASSETS/m4-eligible-live-v2-$IDENTITY-forbidden-capsule
readonly RESULT=$ASSETS/m4-eligible-live-v2-$IDENTITY-qualification-result.json
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
printf %s '@DIAGNOSTIC_TERMINAL_PAYLOAD@' | base64 -d > "$DIAGNOSTIC_TERMINAL"
test "$(sha256sum "$SOURCE" | cut -d' ' -f1)" = "@SOURCE_SHA256@"
test "$(sha256sum "$MEGATRON" | cut -d' ' -f1)" = "@MEGATRON_SHA256@"
test "$(sha256sum "$PROTOCOL" | cut -d' ' -f1)" = "@PROTOCOL_SHA256@"
test "$(sha256sum "$AUTHORIZATION" | cut -d' ' -f1)" = "@AUTHORIZATION_SHA256@"
test "$(sha256sum "$EXECUTION_LOCK" | cut -d' ' -f1)" = "@LOCK_SHA256@"
test "$(sha256sum "$V1_TERMINAL" | cut -d' ' -f1)" = "@V1_TERMINAL_SHA256@"
test "$(sha256sum "$DIAGNOSTIC_TERMINAL" | cut -d' ' -f1)" = "@DIAGNOSTIC_TERMINAL_SHA256@"
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
print("M4_ELIGIBLE_LIVE_V2_LIVENESS_SAFE_SOURCE_PASS",len(names),len(symlinks))
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
print("M4_ELIGIBLE_LIVE_V2_LIVENESS_MEGATRON_PASS")
PY
cd "$RUN_REPO"
export PYTHONPATH="$RUN_REPO"
export CUDA_VISIBLE_DEVICES=0,1
mkdir -p "$METRICS"
"$PYTHON" - "$PROTOCOL" "$AUTHORIZATION" "$EXECUTION_LOCK" "$V1_TERMINAL" "$DIAGNOSTIC_TERMINAL" "$IDENTITY" "$SEED" <<'PY'
import ast,hashlib,json,sys
from importlib.metadata import distribution
from pathlib import Path
from omegaconf import OmegaConf
from nemo_rl.algorithms.single_controller_utils.config import MasterConfig,validate_single_controller_config
from nemo_rl.utils.config import load_config,register_omegaconf_resolvers
p_path,a_path,l_path,v1_path,d_path=map(Path,sys.argv[1:6])
identity=sys.argv[6]; seed=int(sys.argv[7])
p=json.loads(p_path.read_bytes()); auth=json.loads(a_path.read_bytes())
lock=json.loads(l_path.read_bytes()); v1=json.loads(v1_path.read_bytes())
diagnostic=json.loads(d_path.read_bytes())
assert hashlib.sha256(p_path.read_bytes()).hexdigest()=="@PROTOCOL_SHA256@"
assert hashlib.sha256(a_path.read_bytes()).hexdigest()=="@AUTHORIZATION_SHA256@"
assert hashlib.sha256(l_path.read_bytes()).hexdigest()=="@LOCK_SHA256@"
assert hashlib.sha256(v1_path.read_bytes()).hexdigest()=="@V1_TERMINAL_SHA256@"
assert hashlib.sha256(d_path.read_bytes()).hexdigest()=="@DIAGNOSTIC_TERMINAL_SHA256@"
bindings={(row["identity"],row["seed"]) for row in p["qualification"]["identities"]}
assert (identity,seed) in bindings and len(bindings)==6
assert p["joint_gate"]["required_passes"]==6
assert auth["authorized"]["qualification_eos_submission_attempts"]==6
assert lock["source_commit"]=="@SOURCE_COMMIT@"
assert lock["qualification"]["suppress_capture_for_liveness_qualification"]
assert v1["status"]=="TERMINAL_GATE_FAILED_CLOSED"
assert diagnostic["status"]=="TERMINAL_PASS_NO_STALL_REPRODUCED_CAPTURE_EXCLUDED"
assert not v1["paired_replay_opened"] and not v1["post_update_outcomes_opened"]
file_hashes=@FILE_HASHES_JSON@
for name,expected in file_hashes.items():
    actual=hashlib.sha256(Path(name).read_bytes()).hexdigest(); assert actual==expected,(name,actual,expected)
source=Path("nemo_rl/algorithms/single_controller.py").read_text(); tree=ast.parse(source)
function=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="_suppress_eligible_live_capture")
function_source=ast.get_source_segment(source,function) or ""
for forbidden in ("heldout_loss","post_update_loss","base_arm_outcome","shield_arm_outcome","causal_effect"):
    assert forbidden not in function_source
locked=Path("uv.lock").read_text(); assert "@TRANSFERQUEUE_COMMIT@" in locked
direct=json.loads(distribution("TransferQueue").read_text("direct_url.json"))
assert direct["vcs_info"]["commit_id"]=="@TRANSFERQUEUE_COMMIT@"
register_omegaconf_resolvers()
raw=OmegaConf.to_container(load_config(Path("@CONFIG_PATH@")),resolve=True)
raw["grpo"]["seed"]=seed
raw["async_rl"]["controlled_release_delay"]["seed"]=seed
raw["async_rl"]["controlled_release_delay"]["assignment_domain"]="@ASSIGNMENT_DOMAIN@"
capture=raw["async_rl"]["eligible_live_decision_capture"]
capture["trajectory_identity"]=identity; capture["trajectory_seed"]=seed
capture["progress_timeout_seconds"]=1200
capture["liveness_diagnostic_path"]="/tmp/m4-eligible-live-v2-liveness.json"
capture["suppress_capture_for_liveness_qualification"]=True
config=MasterConfig(**raw); validate_single_controller_config(config)
assert config.grpo.max_num_steps==64
assert config.async_rl.eligible_live_decision_capture.suppress_capture_for_liveness_qualification
assert not config.checkpointing["enabled"]
print("M4_ELIGIBLE_LIVE_V2_LIVENESS_RUNTIME_AUTHORITY_PASS",identity,seed)
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
  "async_rl.eligible_live_decision_capture.suppress_capture_for_liveness_qualification=true" \
  "logger.log_dir=$METRICS" \
  "async_rl.lifecycle_audit_path=$LIFECYCLE" \
  "async_rl.gradient_opportunity_audit.output_path=$OPPORTUNITY" \
  "async_rl.gradient_opportunity_audit.observer_duty_path=$DUTY" \
  "async_rl.opportunity_at_risk_v2_shadow.output_path=$OARS" \
  "async_rl.eligible_live_decision_capture.output_dir=$CAPSULE" 2>&1 | tee "$RUN_LOG"
readonly RUN_RC=${PIPESTATUS[0]}
set -e
set +e
"$PYTHON" - "$PROTOCOL" "$IDENTITY" "$SEED" "$RUN_RC" "$LIVENESS" "$LIFECYCLE" "$OPPORTUNITY" "$DUTY" "$OARS" "$CAPSULE" "$RUN_LOG" "$RESULT" <<'PY'
import ast,json,re,sys
from pathlib import Path
protocol_path=Path(sys.argv[1]); identity=sys.argv[2]; seed=int(sys.argv[3]); run_rc=int(sys.argv[4])
liveness,lifecycle,opportunity,duty,oars,capsule,run_log,result_path=map(Path,sys.argv[5:13])
protocol=json.loads(protocol_path.read_bytes())
result={
    "schema":"m4-shield-eligible-live-v2-liveness-qualification-result-v1",
    "status":"FAIL_LIVENESS_QUALIFICATION",
    "identity":identity,
    "seed":seed,
    "assignment_domain":"@ASSIGNMENT_DOMAIN@",
    "run_exit_code":run_rc,
    "v2_acquisition_started":False,
    "paired_replay_started":False,
    "post_update_heldout_outcomes_opened":False,
    "causal_effect_estimated":False,
}
required=(lifecycle,opportunity,duty,oars)
if run_rc==0 and not liveness.exists() and not capsule.exists() and all(p.is_file() and p.stat().st_size>0 for p in required):
    lifecycle_rows=[json.loads(line) for line in lifecycle.read_text().splitlines() if line]
    advances=[row for row in lifecycle_rows if row["stage"]=="learner_version_advanced"]
    versions=[row["learner_weight_version"] for row in advances]
    oars_rows=[json.loads(line) for line in oars.read_text().splitlines() if line]
    eligible=sum(
        row.get("skip_reason") is None
        and row.get("proposals",{}).get("m4_shield",{}).get("proposed_group_ids")
        != row.get("proposals",{}).get("reward_variance_risk",{}).get("proposed_group_ids")
        for row in oars_rows
    )
    matches=sum(row.get("proposal_matches_actual") is True for row in oars_rows)
    match=re.search(r"(?m)^SC run complete: (\{.*\})$",run_log.read_text())
    assert match is not None
    controller=ast.literal_eval(match.group(1))
    qualification=controller["eligible_live_liveness_qualification"]
    assert controller["train_steps"]==64 and controller["trainer_version"]==64
    assert qualification["status"]=="PASS_FULL_64_UPDATE_LIVENESS"
    assert qualification["suppressed_eligible_decisions"]==eligible
    assert qualification["capsule_written"] is False
    assert qualification["checkpoint_written"] is False
    assert qualification["post_update_outcomes_opened"] is False
    assert len(advances)==64 and versions==list(range(1,65))
    assert oars_rows and matches==len(oars_rows)
    result.update({
        "status":"PASS_FULL_64_UPDATE_LIVENESS",
        "train_steps":64,
        "trainer_version":64,
        "lifecycle_learner_advances":64,
        "oars_decisions":len(oars_rows),
        "oars_actions_matching_m4_shield":matches,
        "suppressed_eligible_decisions":eligible,
        "capsule_written":False,
        "checkpoint_written":False,
        "watchdog_timeout_record_present":False,
    })
elif liveness.is_file():
    diagnostic=json.loads(liveness.read_bytes())
    result["watchdog_timeout_record_present"]=True
    result["last_completed_update"]=diagnostic["controller_state"]["train_steps"]
    result["diagnostic_phase"]=diagnostic["controller_state"]["eligible_live_progress_phase"]
result_path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
assert result["status"]==protocol["per_identity_gate"]["pass_status"]
print(result["status"],identity,seed)
PY
readonly GATE_RC=$?
set -e
find "$ASSETS" -type f ! -name "$(basename "$HASHES")" -print0 | sort -z | xargs -0 sha256sum > "$HASHES"
exit "$GATE_RC"
'''


if __name__ == "__main__":
    main()
