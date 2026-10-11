#!/usr/bin/env python3
"""Build the 20 sealed outcome-excluded acquisition-capture manifests."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import tarfile
from pathlib import Path


SOURCE_COMMIT = "545860ed572d5018c106a6438156d7046cf9027d"
SOURCE_SHA256 = "4cafd18a9dc16b6921e3e39995633d3a32de46f935e4818e8a177104a3a89548"
MEGATRON_SHA256 = "98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
PROTOCOL_SHA256 = "716ba0eb8f980ca5af0ef3d57e2415befc1fb1a315afeb01b9bc326a60341b48"
RESTORE_GATE_SHA256 = "dbe064e4c294694daccc9be6c86622bf033d81ac915724b4034acff5385103aa"
PLAN_SHA256 = "49f51e2e1447febcf7777887911a72c60a34050ad170ab70a35e961cfedf3bae"
AUTHORIZATION_SHA256 = "3e82b9f069877006738cadaa88516df000c47b5880278b7349372d76645774f5"
LOCK_SHA256 = "9aebf27d7ab61a25b823525824b101e2a8cf9540e6529ae82bedbe876c24c94b"
TRANSFERQUEUE_COMMIT = "b266d39a15aae114730de36cf8317b6285436f7f"
IMAGE_PATH = (
    "/lustre/fsw/coreai_dlalgo_ci/nvegesna/nemo_rl_images/"
    "nemo-rl-nightly-5802754.sqsh"
)
IMAGE_COMMIT = "ae07eafe8035b5b2e84efa7234e70e7fd7e493c1"
ASSIGNMENT_DOMAIN = "m4-eligible-live-causal-acquisition-v1"
CONFIG_PATH = (
    "examples/configs/"
    "grpo_math_1B_megatron_single_controller_m4_eligible_live_capture.yaml"
)
ANALYZER_PATH = (
    "reports/auto_research/2026-10-07-m4-causal-utility/"
    "analyze_eligible_live_decision_capture.py"
)
FILE_HASHES = {
    CONFIG_PATH: "9e39ff918e10096400fdce8e3d9129c37bddce690c5f196d6b855dc8a01e7d10",
    ANALYZER_PATH: "bd6734b03f9be962377373ce2f019e5243e07796ae5297af0fe233b3e16bd6ee",
    "reports/auto_research/2026-10-07-m4-causal-utility/eligible_live_decision_causal_protocol.json": PROTOCOL_SHA256,
    "nemo_rl/algorithms/async_utils/eligible_live_decision_capsule.py": "d647cf0113cc4caefa244e61f26622f1453ee569b60a30105b17ac486c0d8913",
    "nemo_rl/algorithms/async_utils/opportunity_at_risk_v2.py": "1c450f3b720ccae9be0466496b0f4a89b19ed9cfb88f35e77d78aea9d67e8e80",
    "nemo_rl/algorithms/async_utils/rollout_lifecycle.py": "4d62574c1695dbad310e5fd80c2ee65d7f1efeab22fb283b1e99fd11a0ef22ac",
    "nemo_rl/algorithms/single_controller.py": "cd60e196d19febd67b2a5bc46f8d3db36131941ae8699a387cc83b14bb0e2bf0",
    "nemo_rl/algorithms/single_controller_utils/config.py": "37cf58a608d6dce41150325889940c38b64a8ef7cbf3d60b74033f64c271a108",
    "nemo_rl/distributed/virtual_cluster.py": "ed491307e7f457db3733875a26d35153f3c943a0bf8430b81e3b13e54b81b829",
    "tests/unit/single_controller/test_eligible_live_decision_capsule.py": "39d44b35ab5ad1a23bd85bfdd8830bdfa72cc5ae605d219cd579f793ebe7ffc0",
    "tests/unit/single_controller/test_opportunity_at_risk_v2.py": "d59e3efa75ce65a9102a948750661caa38b1fa10c59806568fd6e167b7b15d78",
    "tests/unit/single_controller/test_single_controller.py": "9244cbbb20ed19b3747ccca6add318f0720945cc69c75e7d74285602f543dae1",
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
    parser.add_argument("--restore-gate", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path, expected in (
        (args.source, SOURCE_SHA256),
        (args.megatron, MEGATRON_SHA256),
        (args.protocol, PROTOCOL_SHA256),
        (args.restore_gate, RESTORE_GATE_SHA256),
        (args.plan, PLAN_SHA256),
        (args.authorization, AUTHORIZATION_SHA256),
        (args.execution_lock, LOCK_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen input moved: {path}")

    protocol = json.loads(args.protocol.read_bytes())
    restore_gate = json.loads(args.restore_gate.read_bytes())
    plan = json.loads(args.plan.read_bytes())
    authorization = json.loads(args.authorization.read_bytes())
    lock = json.loads(args.execution_lock.read_bytes())
    trajectories = plan["trajectories"]
    if (
        protocol["schema"] != "m4-shield-eligible-live-causal-v1"
        or restore_gate["status"] != "PASS_RESTORE_AND_EVALUATION_GATE"
        or plan["status"] != "FROZEN_BEFORE_FIRST_ACQUISITION_TRAJECTORY"
        or len(trajectories) != 20
        or len({row["identity"] for row in trajectories}) != 20
        or len({row["seed"] for row in trajectories}) != 20
        or plan["capture_phase"]["minimum_authenticated_capsules_to_open_replay"]
        != 16
        or plan["capture_phase"]["replacement_seeds"] != 0
        or authorization["status"]
        != "AUTHORIZED_BEFORE_FIRST_ACQUISITION_TRAJECTORY"
        or authorization["authorized"]["fixed_outcome_excluded_capture_trajectories"]
        != 20
        or not authorization["authorized"]["parallel_capture_submission"]
        or lock["source_commit"] != SOURCE_COMMIT
        or lock["source_archive_sha256"] != SOURCE_SHA256
        or lock["capture_execution"]["submission_attempt_limit_per_identity"] != 1
        or lock["capture_execution"]["workload_time_limit_seconds"] != 14400
        or lock["capture_execution"]["client_queue_deadline"] is not None
        or lock["capture_execution"]["automatic_retry"]
        or lock["capture_execution"]["automatic_replacement"]
        or lock["capture_execution"]["automatic_extension"]
        or lock["outcome_boundary"]["base_or_shield_replay_started"]
        or lock["outcome_boundary"]["post_update_loss_computed"]
    ):
        raise RuntimeError("eligible-live acquisition boundary differs")

    with tarfile.open(args.source, "r:gz") as archive:
        source_names = set(archive.getnames())
    if not set(FILE_HASHES).issubset(source_names):
        raise RuntimeError("source archive lacks required acquisition files")

    template = r'''set -euo pipefail
readonly PYTHON=/opt/nemo_rl_venv/bin/python
readonly RUN_REPO=/workspace/m4-eligible-live-acquisition-repo
readonly SOURCE=/workspace/m4-eligible-live-acquisition-source.tar.gz
readonly MEGATRON=/workspace/m4-eligible-live-acquisition-megatron.tar.gz
readonly PROTOCOL=/workspace/m4-eligible-live-causal-protocol.json
readonly RESTORE_GATE=/workspace/m4-eligible-live-restore-gate.json
readonly PLAN=/workspace/m4-eligible-live-acquisition-plan.json
readonly AUTHORIZATION=/workspace/m4-eligible-live-acquisition-authorization.json
readonly EXECUTION_LOCK=/workspace/m4-eligible-live-acquisition-lock.json
readonly IDENTITY=@IDENTITY@
readonly SEED=@SEED@
readonly ASSIGNMENT_DOMAIN=@ASSIGNMENT_DOMAIN@
readonly ASSETS={assets_dir}
readonly RUN_LOG=$ASSETS/m4-eligible-live-$IDENTITY-run.log
readonly LIFECYCLE=$ASSETS/m4-eligible-live-$IDENTITY-lifecycle.jsonl
readonly OPPORTUNITY=$ASSETS/m4-eligible-live-$IDENTITY-opportunity.jsonl
readonly DUTY=$ASSETS/m4-eligible-live-$IDENTITY-observer-duty.json
readonly OARS=$ASSETS/m4-eligible-live-$IDENTITY-oars-v2.jsonl
readonly CAPSULE=$ASSETS/m4-eligible-live-$IDENTITY-capsule
readonly RESULT=$ASSETS/m4-eligible-live-$IDENTITY-capture-result.json
readonly HASHES=$ASSETS/m4-eligible-live-$IDENTITY-artifacts.sha256
readonly METRICS=$ASSETS/m4-eligible-live-$IDENTITY-metrics
test "${NEMO_RL_COMMIT:-unknown}" = "@IMAGE_COMMIT@"
test ! -e "$RUN_REPO"
printf %s '@SOURCE_PAYLOAD@' | base64 -d > "$SOURCE"
printf %s '@MEGATRON_PAYLOAD@' | base64 -d > "$MEGATRON"
printf %s '@PROTOCOL_PAYLOAD@' | base64 -d > "$PROTOCOL"
printf %s '@RESTORE_GATE_PAYLOAD@' | base64 -d > "$RESTORE_GATE"
printf %s '@PLAN_PAYLOAD@' | base64 -d > "$PLAN"
printf %s '@AUTHORIZATION_PAYLOAD@' | base64 -d > "$AUTHORIZATION"
printf %s '@LOCK_PAYLOAD@' | base64 -d > "$EXECUTION_LOCK"
test "$(sha256sum "$SOURCE" | cut -d' ' -f1)" = "@SOURCE_SHA256@"
test "$(sha256sum "$MEGATRON" | cut -d' ' -f1)" = "@MEGATRON_SHA256@"
test "$(sha256sum "$PROTOCOL" | cut -d' ' -f1)" = "@PROTOCOL_SHA256@"
test "$(sha256sum "$RESTORE_GATE" | cut -d' ' -f1)" = "@RESTORE_GATE_SHA256@"
test "$(sha256sum "$PLAN" | cut -d' ' -f1)" = "@PLAN_SHA256@"
test "$(sha256sum "$AUTHORIZATION" | cut -d' ' -f1)" = "@AUTHORIZATION_SHA256@"
test "$(sha256sum "$EXECUTION_LOCK" | cut -d' ' -f1)" = "@LOCK_SHA256@"
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
print("M4_ELIGIBLE_ACQUISITION_SAFE_SOURCE_PASS",len(names),len(symlinks))
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
print("M4_ELIGIBLE_ACQUISITION_MEGATRON_PASS")
PY
cd "$RUN_REPO"
export PYTHONPATH="$RUN_REPO"
export CUDA_VISIBLE_DEVICES=0,1
mkdir -p "$METRICS"
"$PYTHON" - "$PROTOCOL" "$RESTORE_GATE" "$PLAN" "$AUTHORIZATION" "$EXECUTION_LOCK" "$IDENTITY" "$SEED" <<'PY'
import ast,hashlib,json,sys
from importlib.metadata import distribution
from pathlib import Path
from omegaconf import OmegaConf
from nemo_rl.algorithms.single_controller_utils.config import MasterConfig,validate_single_controller_config
from nemo_rl.utils.config import load_config,register_omegaconf_resolvers
p_path,g_path,plan_path,a_path,l_path=map(Path,sys.argv[1:6]); identity=sys.argv[6]; seed=int(sys.argv[7])
p=json.loads(p_path.read_bytes()); gate=json.loads(g_path.read_bytes()); plan=json.loads(plan_path.read_bytes())
auth=json.loads(a_path.read_bytes()); lock=json.loads(l_path.read_bytes())
assert hashlib.sha256(p_path.read_bytes()).hexdigest()=="@PROTOCOL_SHA256@"
assert hashlib.sha256(g_path.read_bytes()).hexdigest()=="@RESTORE_GATE_SHA256@"
assert hashlib.sha256(plan_path.read_bytes()).hexdigest()=="@PLAN_SHA256@"
assert hashlib.sha256(a_path.read_bytes()).hexdigest()=="@AUTHORIZATION_SHA256@"
assert hashlib.sha256(l_path.read_bytes()).hexdigest()=="@LOCK_SHA256@"
assert p["schema"]=="m4-shield-eligible-live-causal-v1"
assert gate["status"]=="PASS_RESTORE_AND_EVALUATION_GATE"
row=next(row for row in plan["trajectories"] if row["identity"]==identity)
assert row["seed"]==seed and plan["assignment_domain"]=="@ASSIGNMENT_DOMAIN@"
assert auth["authorized"]["fixed_outcome_excluded_capture_trajectories"]==20
assert not lock["outcome_boundary"]["base_or_shield_replay_started"]
file_hashes=@FILE_HASHES_JSON@
for name,expected in file_hashes.items():
    actual=hashlib.sha256(Path(name).read_bytes()).hexdigest(); assert actual==expected,(name,actual,expected)
source=Path("nemo_rl/algorithms/async_utils/eligible_live_decision_capsule.py").read_text()
analyzer=Path("@ANALYZER_PATH@").read_text()
assert '"trajectory_identity": trajectory_identity' in source
assert '"trajectory_seed": trajectory_seed' in source
assert '"assignment_domain": assignment_domain' in source
assert "weights_only=True" in analyzer and "weights_only=False" not in analyzer
locked=Path("uv.lock").read_text(); assert "@TRANSFERQUEUE_COMMIT@" in locked
direct=json.loads(distribution("TransferQueue").read_text("direct_url.json"))
assert direct["vcs_info"]["commit_id"]=="@TRANSFERQUEUE_COMMIT@"
register_omegaconf_resolvers()
raw=OmegaConf.to_container(load_config(Path("@CONFIG_PATH@")),resolve=True)
raw["grpo"]["seed"]=seed
raw["async_rl"]["controlled_release_delay"]["seed"]=seed
raw["async_rl"]["controlled_release_delay"]["assignment_domain"]="@ASSIGNMENT_DOMAIN@"
raw["async_rl"]["eligible_live_decision_capture"]["trajectory_identity"]=identity
raw["async_rl"]["eligible_live_decision_capture"]["trajectory_seed"]=seed
config=MasterConfig(**raw); validate_single_controller_config(config)
capture=config.async_rl.eligible_live_decision_capture
assert config.policy["model_name"]=="meta-llama/Llama-3.2-1B-Instruct"
assert config.grpo.seed==seed and config.grpo.max_num_steps==64
assert capture.trajectory_identity==identity and capture.trajectory_seed==seed
assert config.async_rl.controlled_release_delay.assignment_domain=="@ASSIGNMENT_DOMAIN@"
tree=ast.parse(Path("nemo_rl/algorithms/single_controller.py").read_bytes())
capture_fn=next(n for n in ast.walk(tree) if isinstance(n,ast.AsyncFunctionDef) and n.name=="_capture_eligible_live_decision")
symbols={n.attr for n in ast.walk(capture_fn) if isinstance(n,ast.Attribute)}|{n.id for n in ast.walk(capture_fn) if isinstance(n,ast.Name)}
assert not ({"begin_train_step","finish_train_step","train_microbatches_from_meta","train_from_meta"}&symbols),symbols
assert {"trajectory_identity","trajectory_seed","assignment_domain"}.issubset(symbols)
print("M4_ELIGIBLE_ACQUISITION_RUNTIME_AUTHORITY_PASS",identity,seed)
PY
"$PYTHON" -m pytest -q tests/unit/single_controller/test_eligible_live_decision_capsule.py
"$PYTHON" -m pytest -q tests/unit/single_controller/test_opportunity_at_risk_v2.py -k 'm4_shield'
"$PYTHON" -m pytest -q tests/unit/single_controller/test_single_controller.py -k 'eligible_live_capture'
set +e
"$PYTHON" examples/run_grpo_single_controller.py --config "@CONFIG_PATH@" \
  "grpo.seed=$SEED" \
  "async_rl.controlled_release_delay.seed=$SEED" \
  "async_rl.controlled_release_delay.assignment_domain=$ASSIGNMENT_DOMAIN" \
  "async_rl.eligible_live_decision_capture.trajectory_identity=$IDENTITY" \
  "async_rl.eligible_live_decision_capture.trajectory_seed=$SEED" \
  "logger.log_dir=$METRICS" \
  "async_rl.lifecycle_audit_path=$LIFECYCLE" \
  "async_rl.gradient_opportunity_audit.output_path=$OPPORTUNITY" \
  "async_rl.gradient_opportunity_audit.observer_duty_path=$DUTY" \
  "async_rl.opportunity_at_risk_v2_shadow.output_path=$OARS" \
  "async_rl.eligible_live_decision_capture.output_dir=$CAPSULE" 2>&1 | tee "$RUN_LOG"
readonly RUN_RC=${PIPESTATUS[0]}
set -e
if test "$RUN_RC" -eq 0; then
  set +e
  "$PYTHON" "@ANALYZER_PATH@" --capsule-dir "$CAPSULE" --protocol "$PROTOCOL" --output "$RESULT"
  readonly GATE_RC=$?
  set -e
  "$PYTHON" - "$RESULT" "$IDENTITY" "$SEED" "$ASSIGNMENT_DOMAIN" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1]); result=json.loads(path.read_bytes())
assert result["status"]=="PASS_OUTCOME_EXCLUDED_ELIGIBLE_CAPTURE"
assert result["trajectory_identity"]==sys.argv[2]
assert result["trajectory_seed"]==int(sys.argv[3])
assert result["assignment_domain"]==sys.argv[4]
assert result["post_update_outcomes_opened"] is False
assert not any(key in result for key in ("heldout_loss","post_update_loss","causal_effect"))
PY
elif grep -q 'eligible live decision capture ended without a capsule' "$RUN_LOG"; then
  printf '{"schema":"m4-shield-eligible-live-acquisition-capture-result-v1","status":"NO_ELIGIBLE_DECISION","trajectory_identity":"%s","trajectory_seed":%s,"assignment_domain":"%s","maximum_completed_updates":64,"post_update_outcomes_opened":false,"paired_replay_started":false}\n' "$IDENTITY" "$SEED" "$ASSIGNMENT_DOMAIN" > "$RESULT"
  readonly GATE_RC=0
else
  printf '{"schema":"m4-shield-eligible-live-acquisition-capture-result-v1","status":"RUNTIME_FAILURE","trajectory_identity":"%s","trajectory_seed":%s,"run_exit_code":%s,"post_update_outcomes_opened":false,"paired_replay_started":false}\n' "$IDENTITY" "$SEED" "$RUN_RC" > "$RESULT"
  readonly GATE_RC=$RUN_RC
fi
find "$ASSETS" -type f ! -name "$(basename "$HASHES")" -print0 | sort -z | xargs -0 sha256sum > "$HASHES"
exit "$GATE_RC"
'''

    common = {
        "@SOURCE_PAYLOAD@": encoded(args.source),
        "@MEGATRON_PAYLOAD@": encoded(args.megatron),
        "@PROTOCOL_PAYLOAD@": encoded(args.protocol),
        "@RESTORE_GATE_PAYLOAD@": encoded(args.restore_gate),
        "@PLAN_PAYLOAD@": encoded(args.plan),
        "@AUTHORIZATION_PAYLOAD@": encoded(args.authorization),
        "@LOCK_PAYLOAD@": encoded(args.execution_lock),
        "@IMAGE_COMMIT@": IMAGE_COMMIT,
        "@SOURCE_SHA256@": SOURCE_SHA256,
        "@MEGATRON_SHA256@": MEGATRON_SHA256,
        "@PROTOCOL_SHA256@": PROTOCOL_SHA256,
        "@RESTORE_GATE_SHA256@": RESTORE_GATE_SHA256,
        "@PLAN_SHA256@": PLAN_SHA256,
        "@AUTHORIZATION_SHA256@": AUTHORIZATION_SHA256,
        "@LOCK_SHA256@": LOCK_SHA256,
        "@ASSIGNMENT_DOMAIN@": ASSIGNMENT_DOMAIN,
        "@CONFIG_PATH@": CONFIG_PATH,
        "@ANALYZER_PATH@": ANALYZER_PATH,
        "@FILE_HASHES_JSON@": json.dumps(FILE_HASHES, sort_keys=True),
        "@TRANSFERQUEUE_COMMIT@": TRANSFERQUEUE_COMMIT,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for row in trajectories:
        identity = row["identity"]
        script = template
        for token, value in common.items():
            script = script.replace(token, value)
        script = script.replace("@IDENTITY@", identity).replace(
            "@SEED@", str(row["seed"])
        )
        if "@" in script:
            raise RuntimeError(f"unsubstituted manifest token for {identity}")
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
                "name": f"m4-eligible-live-capture-{identity}",
                "workspace": "/workspace",
                "nodes": 1,
                "time_limit": 14400,
                "image_source": {"local_path": IMAGE_PATH},
                "script": script,
            },
        }
        output = args.output_dir / f"{identity}.json"
        output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(identity, row["seed"], sha256(output))


if __name__ == "__main__":
    main()
