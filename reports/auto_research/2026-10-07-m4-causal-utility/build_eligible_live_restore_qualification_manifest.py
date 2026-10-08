#!/usr/bin/env python3
"""Build the credential-free eligible-live restore qualification manifest."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import tarfile
from pathlib import Path

NAME = "m4-eligible-live-restore-qualification"
SOURCE_COMMIT = "ad7de4ed7df0d049c54374cd99815323eb2fbe6f"
SOURCE_SHA256 = "277565d81685121fdecc6f3b79dbc0f544a941a88fb4415e1dbfe60a049fd0ff"
MEGATRON_SHA256 = "98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
PROTOCOL_SHA256 = "5377529e30ffe6a27732e5ab9aaf87c09e70a6f8d2228a9ba7afc06617137129"
AUTHORIZATION_SHA256 = (
    "d16340f130c4d4d816b5d87d7f20c77a5ff372b020f30d4bd06f6940691c3cf7"
)
LOCK_SHA256 = "00e2e63c9b6c89258581631a7f23b4eefbfc03674c397c0b3e7e2862d4f5f249"
CAPSULE_MANIFEST_SHA256 = (
    "184004bfd24a47e7bd4f80ceee9f11f2891fdbae70ee60cbce5b8ca5ec4284c7"
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
REPORT_ROOT = "reports/auto_research/2026-10-07-m4-causal-utility"
RUNNER_PATH = f"{REPORT_ROOT}/run_eligible_live_restore_qualification.py"
ANALYZER_PATH = f"{REPORT_ROOT}/analyze_eligible_live_restore_qualification.py"
TEST_PATH = f"{REPORT_ROOT}/test_eligible_live_restore_qualification.py"
FILE_HASHES = {
    CONFIG_PATH: "acc329781a8b4b0ff49df76ee5f7188879ebf436b2de687e91c7973da67efed0",
    RUNNER_PATH: "86f0439db996968886226bde8cbba9e650419de0434a4904168a1885772e1dec",
    ANALYZER_PATH: "dda234566fb68e71a5c14560db14de9374f3ad7018204c3fab66cfbe53db0dc1",
    TEST_PATH: "8e9c812b21c6fa8db7533e13e9ebb1603314cfa6196e1190d9337f38e287e2c6",
    "nemo_rl/algorithms/async_utils/conditional_m4_capsule.py": "01f9f96511ebaa87f5529f7e88eebd9f6a2e53206de5d37932389b54e8b4dd78",
    "nemo_rl/models/policy/tq_policy.py": "6b2a5d2aa600d6e8defefc9a0044ebb0ffcaa6d5a31668597cec4d60ebb5eb4c",
    "nemo_rl/models/policy/workers/megatron_policy_worker.py": "160362a38ebe73eec5cfbc76d84355ea4c32f5c8c7aac8be8f4a7c13b461b7e9",
    "uv.lock": "bc69ceddc9f35d2c120235c96afac2e6c7b76827e044efda3e61f908be72c4cc",
}
CHECKPOINT_PATHS = (
    "policy/weights/iter_0000000/.metadata",
    "policy/weights/iter_0000000/__0_0.distcp",
    "policy/weights/iter_0000000/metadata.json",
    "policy/weights/iter_0000000/run_config.yaml",
    "policy/weights/iter_0000000/train_state.pt",
    "policy/weights/latest_checkpointed_iteration.txt",
    "policy/weights/latest_train_state.pt",
)
HELDOUT_GROUP_PATHS = tuple(f"groups/group-{index:02d}.pt" for index in range(8, 12))


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

    protocol = json.loads(args.protocol.read_bytes())
    authorization = json.loads(args.authorization.read_bytes())
    lock = json.loads(args.execution_lock.read_bytes())
    if (
        protocol["schema"] != "m4-shield-eligible-live-restore-qualification-v1"
        or protocol["source_capsule"]["manifest_sha256"] != CAPSULE_MANIFEST_SHA256
        or protocol["download_allowlist"]["heldout_group_indices"] != [8, 9, 10, 11]
        or not protocol["restore"]["optimizer_restore_requested"]
        or protocol["evaluation"]["optimizer_steps"] != 0
        or protocol["evaluation"]["scheduler_steps"] != 0
        or protocol["claim_boundary"]["causal_arm_outcomes_opened"]
        or protocol["claim_boundary"]["paired_acquisition_started"]
        or authorization["status"] != "AUTHORIZED_BEFORE_RESTORE_QUALIFICATION"
        or authorization["authorized"]["eos_submission_attempts"] != 1
        or authorization["not_authorized_by_this_record"]["training_update"] is not True
        or authorization["not_authorized_by_this_record"][
            "base_or_shield_outcome_opening"
        ]
        is not True
    ):
        raise RuntimeError("restore qualification protocol or authority differs")
    if (
        lock["status"] != "FROZEN_AFTER_IMPLEMENTATION_BEFORE_QUALIFICATION"
        or lock["source_commit"] != SOURCE_COMMIT
        or lock["source_archive_sha256"] != SOURCE_SHA256
        or lock["protocol_sha256"] != PROTOCOL_SHA256
        or lock["authorization_sha256"] != AUTHORIZATION_SHA256
        or lock["execution"]["launcher"] != "runllm.py --no_wait"
        or lock["execution"]["submission_attempt_limit"] != 1
        or lock["execution"]["workload_time_limit_seconds"] != 14400
        or lock["execution"]["client_queue_deadline"] is not None
        or lock["execution"]["automatic_retry"]
        or lock["qualification"]["training_updates"] != 0
        or lock["qualification"]["causal_arm_outcomes_opened"]
    ):
        raise RuntimeError("restore qualification execution lock differs")

    with tarfile.open(args.source, "r:gz") as archive:
        names = set(archive.getnames())
    if not set(FILE_HASHES).issubset(names):
        raise RuntimeError(f"source archive lacks: {sorted(set(FILE_HASHES) - names)}")

    download_lines = []
    for relative in CHECKPOINT_PATHS + HELDOUT_GROUP_PATHS:
        remote = f"$API/{relative}"
        local = (
            f"$CAPSULE/checkpoint/{relative}"
            if relative.startswith("policy/")
            else f"$CAPSULE/{relative}"
        )
        download_lines.append(f'download "{remote}" "{local}"')
    downloads = "\n".join(download_lines)

    script = r"""set -euo pipefail
readonly PYTHON=/opt/nemo_rl_venv/bin/python
readonly RUN_REPO=/workspace/m4-eligible-live-restore-repo
readonly SOURCE=/workspace/m4-eligible-live-restore-source.tar.gz
readonly MEGATRON=/workspace/m4-eligible-live-restore-megatron.tar.gz
readonly PROTOCOL=/workspace/m4-eligible-live-restore-protocol.json
readonly AUTHORIZATION=/workspace/m4-eligible-live-restore-authorization.json
readonly EXECUTION_LOCK=/workspace/m4-eligible-live-restore-execution-lock.json
readonly CAPSULE=/workspace/m4-eligible-live-restore-private-capsule
readonly API=https://gitlab-master.nvidia.com/api/v4/projects/70847/jobs/474952005/artifacts/workspace/assets/basic/m4-eligible-live-capture-qualification/m4-eligible-live-capsule
readonly ASSETS={assets_dir}
readonly RESTORE_A=$ASSETS/restore-a.json
readonly RESTORE_B=$ASSETS/restore-b.json
readonly RESULT=$ASSETS/restore-qualification-result.json
readonly LOG_A=$ASSETS/restore-a.log
readonly LOG_B=$ASSETS/restore-b.log
readonly HASHES=$ASSETS/m4-eligible-live-restore-artifacts.sha256
test "${NEMO_RL_COMMIT:-unknown}" = "@IMAGE_COMMIT@"
test -n "${CI_JOB_TOKEN:-}"
test ! -e "$RUN_REPO"
test ! -e "$CAPSULE"
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
print("M4_ELIGIBLE_LIVE_RESTORE_SAFE_SOURCE_PASS",len(names),len(symlinks))
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
print("M4_ELIGIBLE_LIVE_RESTORE_MEGATRON_PASS")
PY
cd "$RUN_REPO"
export PYTHONPATH="$RUN_REPO"
export CUDA_VISIBLE_DEVICES=0
"$PYTHON" - "$PROTOCOL" "$AUTHORIZATION" "$EXECUTION_LOCK" <<'PY'
import ast,hashlib,json,sys
from importlib.metadata import distribution
from pathlib import Path
p_path,a_path,l_path=map(Path,sys.argv[1:])
p=json.loads(p_path.read_bytes()); auth=json.loads(a_path.read_bytes()); lock=json.loads(l_path.read_bytes())
assert hashlib.sha256(p_path.read_bytes()).hexdigest()=="@PROTOCOL_SHA256@"
assert hashlib.sha256(a_path.read_bytes()).hexdigest()=="@AUTHORIZATION_SHA256@"
assert hashlib.sha256(l_path.read_bytes()).hexdigest()=="@LOCK_SHA256@"
assert p["source_capsule"]["manifest_sha256"]=="@CAPSULE_MANIFEST_SHA256@"
assert p["download_allowlist"]["heldout_group_indices"]==[8,9,10,11]
assert p["download_allowlist"]["frontier_group_indices_forbidden"]==list(range(8))
assert p["evaluation"]["eval_mode"] and p["evaluation"]["optimizer_steps"]==0
assert p["evaluation"]["scheduler_steps"]==0
assert not p["claim_boundary"]["causal_arm_outcomes_opened"]
assert not p["claim_boundary"]["paired_acquisition_started"]
assert auth["status"]=="AUTHORIZED_BEFORE_RESTORE_QUALIFICATION"
assert lock["source_commit"]=="@SOURCE_COMMIT@"
file_hashes=@FILE_HASHES_JSON@
for name,expected in file_hashes.items():
    actual=hashlib.sha256(Path(name).read_bytes()).hexdigest()
    assert actual==expected,(name,actual,expected)
runner=Path("@RUNNER_PATH@").read_text(); tree=ast.parse(runner)
names={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)}|{n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
assert "train_from_meta" in names
assert not ({"begin_train_step","finish_train_step","train_microbatches_from_meta"}&names)
assert "eval_mode=True" in runner and "eval_mode=False" not in runner
assert "weights_only=True" in runner and "weights_only=False" not in runner
locked=Path("uv.lock").read_text(); assert "@TRANSFERQUEUE_COMMIT@" in locked
direct=json.loads(distribution("TransferQueue").read_text("direct_url.json"))
assert direct["vcs_info"]["commit_id"]=="@TRANSFERQUEUE_COMMIT@"
print("M4_ELIGIBLE_LIVE_RESTORE_RUNTIME_AUTHORITY_PASS")
PY
"$PYTHON" -m pytest -q "@TEST_PATH@"
mkdir -p "$CAPSULE/groups" "$CAPSULE/checkpoint"
download() {
  local remote=$1
  local local_path=$2
  mkdir -p "$(dirname "$local_path")"
  curl --fail --silent --show-error --location --header "JOB-TOKEN: $CI_JOB_TOKEN" "$remote" --output "$local_path"
}
download "$API/manifest.json" "$CAPSULE/manifest.json"
test "$(sha256sum "$CAPSULE/manifest.json" | cut -d' ' -f1)" = "@CAPSULE_MANIFEST_SHA256@"
@DOWNLOADS@
"$PYTHON" "@RUNNER_PATH@" \
  --capsule-dir "$CAPSULE" \
  --config "@CONFIG_PATH@" \
  --protocol "$PROTOCOL" \
  --restore-label a \
  --ray-log-dir /workspace/m4-eligible-live-restore-ray-a \
  --output "$RESTORE_A" 2>&1 | tee "$LOG_A"
"$PYTHON" "@RUNNER_PATH@" \
  --capsule-dir "$CAPSULE" \
  --config "@CONFIG_PATH@" \
  --protocol "$PROTOCOL" \
  --restore-label b \
  --ray-log-dir /workspace/m4-eligible-live-restore-ray-b \
  --output "$RESTORE_B" 2>&1 | tee "$LOG_B"
"$PYTHON" "@ANALYZER_PATH@" \
  --protocol "$PROTOCOL" \
  --restore-a "$RESTORE_A" \
  --restore-b "$RESTORE_B" \
  --output "$RESULT"
test "$CAPSULE" = /workspace/m4-eligible-live-restore-private-capsule
rm -rf -- "$CAPSULE"
find "$ASSETS" -type f ! -name "$(basename "$HASHES")" -print0 | sort -z | xargs -0 sha256sum > "$HASHES"
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
        "@CAPSULE_MANIFEST_SHA256@": CAPSULE_MANIFEST_SHA256,
        "@CONFIG_PATH@": CONFIG_PATH,
        "@RUNNER_PATH@": RUNNER_PATH,
        "@ANALYZER_PATH@": ANALYZER_PATH,
        "@TEST_PATH@": TEST_PATH,
        "@FILE_HASHES_JSON@": json.dumps(FILE_HASHES, sort_keys=True),
        "@TRANSFERQUEUE_COMMIT@": TRANSFERQUEUE_COMMIT,
        "@DOWNLOADS@": downloads,
    }
    for token, value in replacements.items():
        script = script.replace(token, value)
    if "@" in script:
        raise RuntimeError("unsubstituted manifest token")
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
    print(args.output, sha256(args.output))


if __name__ == "__main__":
    main()
