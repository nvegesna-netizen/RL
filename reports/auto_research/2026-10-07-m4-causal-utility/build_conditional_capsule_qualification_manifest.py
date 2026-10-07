#!/usr/bin/env python3
"""Build the credential-free conditional-M4 capsule qualification manifest."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import tarfile
from pathlib import Path

NAME = "m4-conditional-capsule-metadata-repair"
SOURCE_COMMIT = "883cc01e33a6534d31c8614a0c8e60b5a18c4a23"
SOURCE_SHA256 = "986c8aeeff31721cf037fbc98a0aff36899229cc98c4ac79cc6dfe3f917c8a05"
MEGATRON_SHA256 = "98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
PROTOCOL_SHA256 = "373c2764ef964022da2a0cc934b3dcf6210f194428cbb3cee653a807be5a1373"
REPAIR_SHA256 = "eba90562d2c22aa8ef0a428cf5c2d4f8be71a02482398785433512ad1fc75aa7"
AUTHORIZATION_SHA256 = (
    "015f51d7509ab9b37c0d37e7703722ba469d2f3b509899d1e17889ca060b258a"
)
LOCK_SHA256 = "f15b25717ad9113c6d01ce708af105b3063f0eb5e7c18ba773260958323a04f2"
CONFIG_SHA256 = "f6413fb211b08bab48a169ff8f272b86993d99c6d95773ede7ac956987c2cb07"
ANALYZER_SHA256 = "de990789f817acb0b2c53e2ac27c0cdf99199fa3be210f8eec037435d9b26fde"
CAPSULE_MODULE_SHA256 = (
    "01f9f96511ebaa87f5529f7e88eebd9f6a2e53206de5d37932389b54e8b4dd78"
)
CONTROLLER_SHA256 = "1ee43bc7513e523880c7e09816463e55b395a82ca7a4be131597ca4c8791586b"
CONFIG_MODULE_SHA256 = (
    "4b251b1987518eab80f5ac1fac6dc97fbcfeac0146ebbc16e2ab3804b8520e82"
)
CAPSULE_TEST_SHA256 = "6746dae0bc14232492f9238d9982a217b6fe3147c3a72e28f568c3c4fa178617"
CONTROLLER_TEST_SHA256 = (
    "300d76d278a48ec808f6101a49807aec021f77597f4b0b65f0c24916bbfb47db"
)
UV_LOCK_SHA256 = "bc69ceddc9f35d2c120235c96afac2e6c7b76827e044efda3e61f908be72c4cc"
TRANSFERQUEUE_COMMIT = "b266d39a15aae114730de36cf8317b6285436f7f"
IMAGE_PATH = (
    "/lustre/fsw/coreai_dlalgo_ci/nvegesna/nemo_rl_images/nemo-rl-nightly-5802754.sqsh"
)
IMAGE_COMMIT = "ae07eafe8035b5b2e84efa7234e70e7fd7e493c1"
CONFIG_PATH = "examples/configs/grpo_math_1B_megatron_single_controller_m4_conditional_capsule.yaml"
ANALYZER_PATH = (
    "reports/auto_research/2026-10-07-m4-causal-utility/analyze_conditional_capsule.py"
)
PROTOCOL_PATH = "reports/auto_research/2026-10-07-m4-causal-utility/conditional_capsule_qualification_protocol.json"


def sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoded(path: Path) -> str:
    """Return a base64 payload suitable for a credential-free manifest."""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--repair-protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path, expected in (
        (args.source, SOURCE_SHA256),
        (args.megatron, MEGATRON_SHA256),
        (args.protocol, PROTOCOL_SHA256),
        (args.repair_protocol, REPAIR_SHA256),
        (args.authorization, AUTHORIZATION_SHA256),
        (args.execution_lock, LOCK_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen input moved: {path}")
    repair = json.loads(args.repair_protocol.read_bytes())
    authorization = json.loads(args.authorization.read_bytes())
    lock = json.loads(args.execution_lock.read_bytes())
    execution = lock["execution"]
    qualification = lock["qualification"]
    if (
        repair["schema"] != "conditional-m4-capsule-metadata-repair-protocol-v1"
        or repair["status"] != "FROZEN_AFTER_FAILURE_BEFORE_REPLACEMENT"
        or repair["scientific_attempt_completed"]
        or authorization["schema"]
        != "conditional-m4-capsule-metadata-replacement-authorization-v1"
        or authorization["repair_source_commit"] != SOURCE_COMMIT
        or not authorization["replacement"]["eos_submission_authorized"]
        or authorization["replacement"]["submission_attempt_limit"] != 1
        or authorization["replacement"]["automatic_duplicate_submission"]
        or lock["schema"] != "conditional-m4-capsule-metadata-replacement-lock-v1"
        or lock["source_commit"] != SOURCE_COMMIT
        or lock["source_archive_sha256"] != SOURCE_SHA256
        or lock["original_protocol_sha256"] != PROTOCOL_SHA256
        or lock["repair_protocol_sha256"] != REPAIR_SHA256
        or lock["authorization_sha256"] != AUTHORIZATION_SHA256
        or execution["required_launcher"] != "runllm.py --no_wait"
        or execution["submission_attempt_limit"] != 1
        or execution["automatic_duplicate_submission"]
        or execution["automatic_extension"]
        or execution["nodes"] != 1
        or execution["gpus"] != 2
        or execution["scheduler_time_limit_seconds"] != 14400
        or execution["queue_deadline_override"] is not None
        or qualification["training_updates_authorized"]
        or qualification["learning_outcomes_authorized"]
        or qualification["terminal_evaluation_authorized"]
    ):
        raise RuntimeError("conditional capsule authority differs")
    with tarfile.open(args.source, "r:gz") as archive:
        names = set(archive.getnames())
    required = {
        CONFIG_PATH,
        ANALYZER_PATH,
        PROTOCOL_PATH,
        "nemo_rl/algorithms/async_utils/conditional_m4_capsule.py",
        "nemo_rl/algorithms/single_controller.py",
        "nemo_rl/algorithms/single_controller_utils/config.py",
        "tests/unit/single_controller/test_conditional_m4_capsule.py",
        "tests/unit/single_controller/test_single_controller.py",
        "uv.lock",
    }
    if not required.issubset(names):
        raise RuntimeError(f"source archive lacks: {sorted(required - names)}")

    script = r"""set -euo pipefail
readonly PYTHON=/opt/nemo_rl_venv/bin/python
readonly RUN_REPO=/workspace/m4-conditional-capsule-repo
readonly SOURCE=/workspace/m4-conditional-capsule-source.tar.gz
readonly MEGATRON=/workspace/m4-conditional-capsule-megatron.tar.gz
readonly PROTOCOL=/workspace/m4-conditional-capsule-protocol.json
readonly REPAIR=/workspace/m4-conditional-capsule-metadata-repair-protocol.json
readonly AUTHORIZATION=/workspace/m4-conditional-capsule-metadata-replacement-authorization.json
readonly EXECUTION_LOCK=/workspace/m4-conditional-capsule-execution-lock.json
readonly ASSETS={assets_dir}
readonly RUN_LOG=$ASSETS/m4-conditional-capsule-run.log
readonly LIFECYCLE=$ASSETS/m4-conditional-capsule-lifecycle.jsonl
readonly OPPORTUNITY=$ASSETS/m4-conditional-capsule-opportunity.jsonl
readonly DUTY=$ASSETS/m4-conditional-capsule-observer-duty.json
readonly CAPSULE=$ASSETS/m4-conditional-capsule
readonly RESULT=$ASSETS/m4-conditional-capsule-result.json
readonly HASHES=$ASSETS/m4-conditional-capsule-artifacts.sha256
readonly METRICS=$ASSETS/m4-conditional-capsule-metrics
test "${NEMO_RL_COMMIT:-unknown}" = "@IMAGE_COMMIT@"
test ! -e "$RUN_REPO"
printf %s '@SOURCE_PAYLOAD@' | base64 -d > "$SOURCE"
printf %s '@MEGATRON_PAYLOAD@' | base64 -d > "$MEGATRON"
printf %s '@PROTOCOL_PAYLOAD@' | base64 -d > "$PROTOCOL"
printf %s '@REPAIR_PAYLOAD@' | base64 -d > "$REPAIR"
printf %s '@AUTHORIZATION_PAYLOAD@' | base64 -d > "$AUTHORIZATION"
printf %s '@LOCK_PAYLOAD@' | base64 -d > "$EXECUTION_LOCK"
test "$(sha256sum "$SOURCE" | cut -d' ' -f1)" = "@SOURCE_SHA256@"
test "$(sha256sum "$MEGATRON" | cut -d' ' -f1)" = "@MEGATRON_SHA256@"
test "$(sha256sum "$PROTOCOL" | cut -d' ' -f1)" = "@PROTOCOL_SHA256@"
test "$(sha256sum "$REPAIR" | cut -d' ' -f1)" = "@REPAIR_SHA256@"
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
print("M4_CONDITIONAL_CAPSULE_SAFE_SOURCE_PASS",len(names),len(symlinks))
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
print("M4_CONDITIONAL_CAPSULE_MEGATRON_PASS")
PY
cd "$RUN_REPO"
export PYTHONPATH="$RUN_REPO"
export CUDA_VISIBLE_DEVICES=0,1
mkdir -p "$METRICS"
"$PYTHON" - "$PROTOCOL" "$REPAIR" "$AUTHORIZATION" "$EXECUTION_LOCK" <<'PY'
import ast,hashlib,json,sys
from importlib.metadata import distribution
from pathlib import Path
from omegaconf import OmegaConf
from nemo_rl.algorithms.single_controller_utils.config import MasterConfig,validate_single_controller_config
from nemo_rl.utils.config import load_config,register_omegaconf_resolvers
p_path,r_path,a_path,l_path=map(Path,sys.argv[1:]); p=json.loads(p_path.read_bytes()); repair=json.loads(r_path.read_bytes()); auth=json.loads(a_path.read_bytes()); lock=json.loads(l_path.read_bytes())
assert hashlib.sha256(p_path.read_bytes()).hexdigest()=="@PROTOCOL_SHA256@"
assert hashlib.sha256(r_path.read_bytes()).hexdigest()=="@REPAIR_SHA256@"
assert hashlib.sha256(a_path.read_bytes()).hexdigest()=="@AUTHORIZATION_SHA256@"
assert hashlib.sha256(l_path.read_bytes()).hexdigest()=="@LOCK_SHA256@"
assert p["status"]=="FROZEN_BEFORE_LIVE_QUALIFICATION"
assert repair["status"]=="FROZEN_AFTER_FAILURE_BEFORE_REPLACEMENT" and not repair["scientific_attempt_completed"]
assert auth["status"]=="AUTHORIZED_AFTER_AUTHENTICATED_RUNTIME_FAILURE"
assert auth["replacement"]["eos_submission_authorized"] and auth["replacement"]["submission_attempt_limit"]==1
assert lock["status"]=="FROZEN_AFTER_REPAIR_BEFORE_REPLACEMENT"
assert lock["source_commit"]=="@SOURCE_COMMIT@"
file_hashes={
 "@CONFIG_PATH@":"@CONFIG_SHA256@",
 "@ANALYZER_PATH@":"@ANALYZER_SHA256@",
 "nemo_rl/algorithms/async_utils/conditional_m4_capsule.py":"@CAPSULE_MODULE_SHA256@",
 "nemo_rl/algorithms/single_controller.py":"@CONTROLLER_SHA256@",
 "nemo_rl/algorithms/single_controller_utils/config.py":"@CONFIG_MODULE_SHA256@",
 "tests/unit/single_controller/test_conditional_m4_capsule.py":"@CAPSULE_TEST_SHA256@",
 "tests/unit/single_controller/test_single_controller.py":"@CONTROLLER_TEST_SHA256@",
 "uv.lock":"@UV_LOCK_SHA256@",
}
for name,expected in file_hashes.items():
    actual=hashlib.sha256(Path(name).read_bytes()).hexdigest(); assert actual==expected,(name,actual,expected)
capsule_source=Path("nemo_rl/algorithms/async_utils/conditional_m4_capsule.py").read_text()
assert "pickle_protocol=2" in capsule_source and "pickle_protocol=4" not in capsule_source
controller_source=Path("nemo_rl/algorithms/single_controller.py").read_text()
controller_test_source=Path("tests/unit/single_controller/test_single_controller.py").read_text()
assert "or async_config.conditional_m4_capsule.enabled" in controller_source
assert "if _requires_opportunity_reward_moments(self._async_cfg):" in controller_source
assert "test_conditional_capsule_requests_reward_moment_metadata" in controller_test_source
analyzer_source=Path("@ANALYZER_PATH@").read_text()
assert "weights_only=True" in analyzer_source and "weights_only=False" not in analyzer_source
locked=Path("uv.lock").read_text()
assert "@TRANSFERQUEUE_COMMIT@" in locked
direct=json.loads(distribution("TransferQueue").read_text("direct_url.json"))
assert direct["vcs_info"]["commit_id"]=="@TRANSFERQUEUE_COMMIT@"
register_omegaconf_resolvers(); raw=OmegaConf.to_container(load_config(Path("@CONFIG_PATH@")),resolve=True)
config=MasterConfig(**raw); validate_single_controller_config(config)
capsule=config.async_rl.conditional_m4_capsule
assert config.policy["model_name"]=="meta-llama/Llama-3.2-1B-Instruct"
assert config.grpo.seed==20261027 and config.grpo.max_num_steps==1
assert config.grpo.num_prompts_per_step==1 and config.grpo.num_generations_per_prompt==8
assert config.policy["train_global_batch_size"]==8
assert config.cluster["num_nodes"]==1 and config.cluster["gpus_per_node"]==2
assert config.data_plane["actor_runtime_env_mode"]=="inherit_baked_single_node"
assert config.async_rl.sampler.name=="windowed" and config.async_rl.sampler.max_staleness_versions==0
assert capsule.enabled and capsule.candidate_groups==8 and capsule.batch_groups==4
assert capsule.heldout_groups==4 and capsule.max_frontiers==16
assert capsule.minimum_relative_predicted_gain==0.1
assert capsule.minimum_service_multiplier==0.98 and capsule.maximum_service_multiplier==1.02
assert config.async_rl.gradient_opportunity_audit.enabled
assert not config.async_rl.gradient_utility_audit.enabled
assert not config.async_rl.terminal_policy_export.enabled
tree=ast.parse(Path("nemo_rl/algorithms/single_controller.py").read_bytes())
pump=next(n for n in ast.walk(tree) if isinstance(n,ast.AsyncFunctionDef) and n.name=="_conditional_m4_capsule_pump")
called={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for n in ast.walk(pump) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Attribute,ast.Name))}
assert not ({"begin_train_step","finish_train_step","train_microbatches"}&called),called
print("M4_CONDITIONAL_CAPSULE_RUNTIME_AUTHORITY_PASS")
PY
"$PYTHON" -m pytest -q tests/unit/single_controller/test_conditional_m4_capsule.py
"$PYTHON" -m pytest -q tests/unit/single_controller/test_single_controller.py -k conditional_capsule
set +e
"$PYTHON" examples/run_grpo_single_controller.py --config "@CONFIG_PATH@" \
  "logger.log_dir=$METRICS" \
  "async_rl.lifecycle_audit_path=$LIFECYCLE" \
  "async_rl.gradient_opportunity_audit.output_path=$OPPORTUNITY" \
  "async_rl.gradient_opportunity_audit.observer_duty_path=$DUTY" \
  "async_rl.conditional_m4_capsule.output_dir=$CAPSULE" 2>&1 | tee "$RUN_LOG"
readonly RUN_RC=${PIPESTATUS[0]}
set -e
if test "$RUN_RC" -eq 0; then
  set +e
  "$PYTHON" "@ANALYZER_PATH@" --capsule-dir "$CAPSULE" --protocol "$PROTOCOL" --output "$RESULT"
  readonly GATE_RC=$?
  set -e
else
  printf '{"schema":"conditional-m4-capsule-qualification-result-v1","status":"RUNTIME_FAILURE","run_exit_code":%s,"learning_outcomes_opened":false,"optimizer_steps":0,"scheduler_steps":0}\n' "$RUN_RC" > "$RESULT"
  readonly GATE_RC=$RUN_RC
fi
find "$ASSETS" -type f ! -name "$(basename "$HASHES")" -print0 | sort -z | xargs -0 sha256sum > "$HASHES"
exit "$GATE_RC"
"""
    replacements = {
        "@SOURCE_PAYLOAD@": encoded(args.source),
        "@MEGATRON_PAYLOAD@": encoded(args.megatron),
        "@PROTOCOL_PAYLOAD@": encoded(args.protocol),
        "@REPAIR_PAYLOAD@": encoded(args.repair_protocol),
        "@AUTHORIZATION_PAYLOAD@": encoded(args.authorization),
        "@LOCK_PAYLOAD@": encoded(args.execution_lock),
        "@IMAGE_COMMIT@": IMAGE_COMMIT,
        "@SOURCE_COMMIT@": SOURCE_COMMIT,
        "@SOURCE_SHA256@": SOURCE_SHA256,
        "@MEGATRON_SHA256@": MEGATRON_SHA256,
        "@PROTOCOL_SHA256@": PROTOCOL_SHA256,
        "@REPAIR_SHA256@": REPAIR_SHA256,
        "@AUTHORIZATION_SHA256@": AUTHORIZATION_SHA256,
        "@LOCK_SHA256@": LOCK_SHA256,
        "@CONFIG_PATH@": CONFIG_PATH,
        "@CONFIG_SHA256@": CONFIG_SHA256,
        "@ANALYZER_PATH@": ANALYZER_PATH,
        "@ANALYZER_SHA256@": ANALYZER_SHA256,
        "@CAPSULE_MODULE_SHA256@": CAPSULE_MODULE_SHA256,
        "@CONTROLLER_SHA256@": CONTROLLER_SHA256,
        "@CONFIG_MODULE_SHA256@": CONFIG_MODULE_SHA256,
        "@CAPSULE_TEST_SHA256@": CAPSULE_TEST_SHA256,
        "@CONTROLLER_TEST_SHA256@": CONTROLLER_TEST_SHA256,
        "@UV_LOCK_SHA256@": UV_LOCK_SHA256,
        "@TRANSFERQUEUE_COMMIT@": TRANSFERQUEUE_COMMIT,
    }
    for token, value in replacements.items():
        script = script.replace(token, value)
    if "@" in script:
        raise RuntimeError("unsubstituted manifest token")
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
    print(args.output, sha256(args.output))


if __name__ == "__main__":
    main()
