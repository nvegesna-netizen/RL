#!/usr/bin/env python3
"""Build the credential-free eligible-live M4 capture qualification manifest."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import tarfile
from pathlib import Path

NAME = "m4-eligible-live-capture-qualification"
SOURCE_COMMIT = "61e283868d8d4588f4de24ee315b1c7f0e7ce0a8"
SOURCE_SHA256 = "f1d59b5553c66d6633da8a9aeccf9e3a634b611ce09d062dc2997fa0ef393334"
MEGATRON_SHA256 = "98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
PROTOCOL_SHA256 = "716ba0eb8f980ca5af0ef3d57e2415befc1fb1a315afeb01b9bc326a60341b48"
AUTHORIZATION_SHA256 = (
    "eae191a02f5c66123b9a6409d7c84f3e2eac45c2145ec4abc4e18ec75e85fcaf"
)
LOCK_SHA256 = "9a44b56183158ae390b519c7f45aed3d3dddc300c636348565191233d3f72392"
TRANSFERQUEUE_COMMIT = "b266d39a15aae114730de36cf8317b6285436f7f"
IMAGE_PATH = (
    "/lustre/fsw/coreai_dlalgo_ci/nvegesna/nemo_rl_images/nemo-rl-nightly-5802754.sqsh"
)
IMAGE_COMMIT = "ae07eafe8035b5b2e84efa7234e70e7fd7e493c1"
CONFIG_PATH = (
    "examples/configs/"
    "grpo_math_1B_megatron_single_controller_m4_eligible_live_capture.yaml"
)
ANALYZER_PATH = (
    "reports/auto_research/2026-10-07-m4-causal-utility/"
    "analyze_eligible_live_decision_capture.py"
)
PROTOCOL_PATH = (
    "reports/auto_research/2026-10-07-m4-causal-utility/"
    "eligible_live_decision_causal_protocol.json"
)
FILE_HASHES = {
    CONFIG_PATH: "acc329781a8b4b0ff49df76ee5f7188879ebf436b2de687e91c7973da67efed0",
    ANALYZER_PATH: "30029eebec9096550bb017f405ade2e500eb3911ed334ca3ff1af36813194633",
    PROTOCOL_PATH: PROTOCOL_SHA256,
    "nemo_rl/algorithms/async_utils/eligible_live_decision_capsule.py": "ff349f70284309ce9a6e17027aa289521beb36c07d0567b2993b0327dc45c1fb",
    "nemo_rl/algorithms/async_utils/opportunity_at_risk_v2.py": "1c450f3b720ccae9be0466496b0f4a89b19ed9cfb88f35e77d78aea9d67e8e80",
    "nemo_rl/algorithms/async_utils/rollout_lifecycle.py": "4d62574c1695dbad310e5fd80c2ee65d7f1efeab22fb283b1e99fd11a0ef22ac",
    "nemo_rl/algorithms/single_controller.py": "c9c5e639b9ada68942aaac76ce891981087f2880961ac32a4dd499ed18b29b59",
    "nemo_rl/algorithms/single_controller_utils/config.py": "72710053dddd3caf574f4dc6f2c79153cf474681aea2a56aa508ed389474d3d6",
    "tests/unit/single_controller/test_eligible_live_decision_capsule.py": "9b0f08add982b2211bdccbb479bbb34c6ad94459bfd51d22945d1e48203dd402",
    "tests/unit/single_controller/test_opportunity_at_risk_v2.py": "d59e3efa75ce65a9102a948750661caa38b1fa10c59806568fd6e167b7b15d78",
    "tests/unit/single_controller/test_single_controller.py": "165a393fda9be0f2e3bb8e4c8c3331061c82a275f5ee6533e0ea89547efe833e",
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path, expected in (
        (args.source, SOURCE_SHA256),
        (args.megatron, MEGATRON_SHA256),
        (args.protocol, PROTOCOL_SHA256),
        (args.authorization, AUTHORIZATION_SHA256),
        (args.execution_lock, LOCK_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen input moved: {path}")

    authorization = json.loads(args.authorization.read_bytes())
    lock = json.loads(args.execution_lock.read_bytes())
    if (
        authorization["status"] != "AUTHORIZED_BEFORE_OUTCOME_EXCLUDED_QUALIFICATION"
        or authorization["authorized"]["eos_submission_attempts"] != 1
        or not authorization["authorized"][
            "outcome_excluded_live_capture_qualification"
        ]
        or lock["status"] != "FROZEN_AFTER_IMPLEMENTATION_BEFORE_QUALIFICATION"
        or lock["source_commit"] != SOURCE_COMMIT
        or lock["source_archive_sha256"] != SOURCE_SHA256
        or lock["protocol_sha256"] != PROTOCOL_SHA256
        or lock["execution"]["launcher"] != "runllm.py --no_wait"
        or lock["execution"]["submission_attempt_limit"] != 1
        or lock["execution"]["workload_time_limit_seconds"] != 14400
        or lock["execution"]["client_queue_deadline"] is not None
        or lock["execution"]["automatic_retry"]
        or lock["qualification"]["captured_decision_training_update_authorized"]
        or lock["qualification"]["post_update_learning_outcomes_authorized"]
        or lock["qualification"]["terminal_evaluation_authorized"]
    ):
        raise RuntimeError("eligible live qualification authority differs")

    with tarfile.open(args.source, "r:gz") as archive:
        names = set(archive.getnames())
    required = set(FILE_HASHES)
    if not required.issubset(names):
        raise RuntimeError(f"source archive lacks: {sorted(required - names)}")

    script = r"""set -euo pipefail
readonly PYTHON=/opt/nemo_rl_venv/bin/python
readonly RUN_REPO=/workspace/m4-eligible-live-repo
readonly SOURCE=/workspace/m4-eligible-live-source.tar.gz
readonly MEGATRON=/workspace/m4-eligible-live-megatron.tar.gz
readonly PROTOCOL=/workspace/m4-eligible-live-protocol.json
readonly AUTHORIZATION=/workspace/m4-eligible-live-authorization.json
readonly EXECUTION_LOCK=/workspace/m4-eligible-live-execution-lock.json
readonly ASSETS={assets_dir}
readonly RUN_LOG=$ASSETS/m4-eligible-live-run.log
readonly LIFECYCLE=$ASSETS/m4-eligible-live-lifecycle.jsonl
readonly OPPORTUNITY=$ASSETS/m4-eligible-live-opportunity.jsonl
readonly DUTY=$ASSETS/m4-eligible-live-observer-duty.json
readonly OARS=$ASSETS/m4-eligible-live-oars-v2.jsonl
readonly CAPSULE=$ASSETS/m4-eligible-live-capsule
readonly RESULT=$ASSETS/m4-eligible-live-capture-result.json
readonly HASHES=$ASSETS/m4-eligible-live-artifacts.sha256
readonly METRICS=$ASSETS/m4-eligible-live-metrics
test "${{NEMO_RL_COMMIT:-unknown}}" = "@IMAGE_COMMIT@"
test ! -e "$RUN_REPO"
printf %s '@SOURCE_PAYLOAD@' | base64 -d > "$SOURCE"
printf %s '@MEGATRON_PAYLOAD@' | base64 -d > "$MEGATRON"
printf %s '@PROTOCOL_PAYLOAD@' | base64 -d > "$PROTOCOL"
printf %s '@AUTHORIZATION_PAYLOAD@' | base64 -d > "$AUTHORIZATION"
printf %s '@LOCK_PAYLOAD@' | base64 -d > "$EXECUTION_LOCK"
test "$(sha256sum "$SOURCE" | cut -d' ' -f1)" = "@SOURCE_SHA256@"
test "$(sha256sum "$MEGATRON" | cut -d' ' -f1)" = "@MEGATRON_SHA256@"
test "$(sha256sum "$PROTOCOL" | cut -d' ' -f1)" = "@PROTOCOL_SHA256@"
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
print("M4_ELIGIBLE_LIVE_SAFE_SOURCE_PASS",len(names),len(symlinks))
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
print("M4_ELIGIBLE_LIVE_MEGATRON_PASS")
PY
cd "$RUN_REPO"
export PYTHONPATH="$RUN_REPO"
export CUDA_VISIBLE_DEVICES=0,1
mkdir -p "$METRICS"
"$PYTHON" - "$PROTOCOL" "$AUTHORIZATION" "$EXECUTION_LOCK" <<'PY'
import ast,hashlib,json,sys
from importlib.metadata import distribution
from pathlib import Path
from omegaconf import OmegaConf
from nemo_rl.algorithms.single_controller_utils.config import MasterConfig,validate_single_controller_config
from nemo_rl.utils.config import load_config,register_omegaconf_resolvers
p_path,a_path,l_path=map(Path,sys.argv[1:])
p=json.loads(p_path.read_bytes()); auth=json.loads(a_path.read_bytes()); lock=json.loads(l_path.read_bytes())
assert hashlib.sha256(p_path.read_bytes()).hexdigest()=="@PROTOCOL_SHA256@"
assert hashlib.sha256(a_path.read_bytes()).hexdigest()=="@AUTHORIZATION_SHA256@"
assert hashlib.sha256(l_path.read_bytes()).hexdigest()=="@LOCK_SHA256@"
assert p["schema"]=="m4-shield-eligible-live-causal-v1"
assert not p["boundaries"]["qualification_opens_post_update_outcomes"]
assert auth["status"]=="AUTHORIZED_BEFORE_OUTCOME_EXCLUDED_QUALIFICATION"
assert lock["source_commit"]=="@SOURCE_COMMIT@"
file_hashes=@FILE_HASHES_JSON@
for name,expected in file_hashes.items():
    actual=hashlib.sha256(Path(name).read_bytes()).hexdigest()
    assert actual==expected,(name,actual,expected)
source=Path("nemo_rl/algorithms/async_utils/eligible_live_decision_capsule.py").read_text()
analyzer=Path("@ANALYZER_PATH@").read_text()
assert "pickle_protocol=2" in source and "pickle_protocol=4" not in source
assert "weights_only=True" in analyzer and "weights_only=False" not in analyzer
locked=Path("uv.lock").read_text(); assert "@TRANSFERQUEUE_COMMIT@" in locked
direct=json.loads(distribution("TransferQueue").read_text("direct_url.json"))
assert direct["vcs_info"]["commit_id"]=="@TRANSFERQUEUE_COMMIT@"
register_omegaconf_resolvers()
raw=OmegaConf.to_container(load_config(Path("@CONFIG_PATH@")),resolve=True)
config=MasterConfig(**raw); validate_single_controller_config(config)
capture=config.async_rl.eligible_live_decision_capture
shadow=config.async_rl.opportunity_at_risk_v2_shadow
assert config.policy["model_name"]=="meta-llama/Llama-3.2-1B-Instruct"
assert config.grpo.seed==20261031 and config.grpo.max_num_steps==64
assert config.grpo.num_prompts_per_step==4 and config.grpo.num_generations_per_prompt==8
assert config.policy["train_global_batch_size"]==32
assert config.cluster["num_nodes"]==1 and config.cluster["gpus_per_node"]==2
assert config.data_plane["actor_runtime_env_mode"]=="inherit_baked_single_node"
assert config.async_rl.sampler.name=="weight_fifo"
assert config.async_rl.sampler.max_staleness_versions==1
assert config.async_rl.sampler.selection_candidate_watermark==8
assert capture.enabled and capture.heldout_groups==4
assert shadow.enabled and shadow.mode=="act" and shadow.actuation_scorer=="m4_shield"
assert shadow.candidate_window_policy=="controlled_frontier"
assert shadow.minimum_service_multiplier==0.98 and shadow.maximum_service_multiplier==1.02
assert shadow.exact_search_max_candidates==16
assert not config.async_rl.terminal_policy_export.enabled
tree=ast.parse(Path("nemo_rl/algorithms/single_controller.py").read_bytes())
capture_fn=next(n for n in ast.walk(tree) if isinstance(n,ast.AsyncFunctionDef) and n.name=="_capture_eligible_live_decision")
symbols={n.attr for n in ast.walk(capture_fn) if isinstance(n,ast.Attribute)}|{n.id for n in ast.walk(capture_fn) if isinstance(n,ast.Name)}
assert not ({"begin_train_step","finish_train_step","train_microbatches_from_meta","train_from_meta"}&symbols),symbols
assert "save_checkpoint" in symbols and "write_eligible_live_decision_capsule" in symbols
print("M4_ELIGIBLE_LIVE_RUNTIME_AUTHORITY_PASS")
PY
"$PYTHON" -m pytest -q tests/unit/single_controller/test_eligible_live_decision_capsule.py
"$PYTHON" -m pytest -q tests/unit/single_controller/test_opportunity_at_risk_v2.py -k 'm4_shield'
"$PYTHON" -m pytest -q tests/unit/single_controller/test_single_controller.py -k 'eligible_live_capture'
set +e
"$PYTHON" examples/run_grpo_single_controller.py --config "@CONFIG_PATH@" \
  "logger.log_dir=$METRICS" \
  "async_rl.lifecycle_audit_path=$LIFECYCLE" \
  "async_rl.gradient_opportunity_audit.output_path=$OPPORTUNITY" \
  "async_rl.gradient_opportunity_audit.observer_duty_path=$DUTY" \
  "async_rl.opportunity_at_risk_v2_shadow.output_path=$OARS" \
  "async_rl.eligible_live_decision_capture.output_dir=$CAPSULE" 2>&1 | tee "$RUN_LOG"
readonly RUN_RC=${{PIPESTATUS[0]}}
set -e
if test "$RUN_RC" -eq 0; then
  set +e
  "$PYTHON" "@ANALYZER_PATH@" --capsule-dir "$CAPSULE" --protocol "$PROTOCOL" --output "$RESULT"
  readonly GATE_RC=$?
  set -e
elif grep -q 'eligible live decision capture ended without a capsule' "$RUN_LOG"; then
  printf '{"schema":"m4-shield-eligible-live-capture-qualification-result-v1","status":"NO_ELIGIBLE_DECISION","run_exit_code":%s,"post_update_outcomes_opened":false}\n' "$RUN_RC" > "$RESULT"
  readonly GATE_RC=1
else
  printf '{"schema":"m4-shield-eligible-live-capture-qualification-result-v1","status":"RUNTIME_FAILURE","run_exit_code":%s,"post_update_outcomes_opened":false}\n' "$RUN_RC" > "$RESULT"
  readonly GATE_RC=$RUN_RC
fi
find "$ASSETS" -type f ! -name "$(basename "$HASHES")" -print0 | sort -z | xargs -0 sha256sum > "$HASHES"
exit "$GATE_RC"
"""
    replacements = {
        "@SOURCE_PAYLOAD@": encoded(args.source),
        "@MEGATRON_PAYLOAD@": encoded(args.megatron),
        "@PROTOCOL_PAYLOAD@": encoded(args.protocol),
        "@AUTHORIZATION_PAYLOAD@": encoded(args.authorization),
        "@LOCK_PAYLOAD@": encoded(args.execution_lock),
        "@IMAGE_COMMIT@": IMAGE_COMMIT,
        "@SOURCE_COMMIT@": SOURCE_COMMIT,
        "@SOURCE_SHA256@": SOURCE_SHA256,
        "@MEGATRON_SHA256@": MEGATRON_SHA256,
        "@PROTOCOL_SHA256@": PROTOCOL_SHA256,
        "@AUTHORIZATION_SHA256@": AUTHORIZATION_SHA256,
        "@LOCK_SHA256@": LOCK_SHA256,
        "@CONFIG_PATH@": CONFIG_PATH,
        "@ANALYZER_PATH@": ANALYZER_PATH,
        "@FILE_HASHES_JSON@": json.dumps(FILE_HASHES, sort_keys=True),
        "@TRANSFERQUEUE_COMMIT@": TRANSFERQUEUE_COMMIT,
    }
    for token, value in replacements.items():
        script = script.replace(token, value)
    if "@" in script:
        raise RuntimeError("unsubstituted manifest token")
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
    print(args.output, sha256(args.output))


if __name__ == "__main__":
    main()
