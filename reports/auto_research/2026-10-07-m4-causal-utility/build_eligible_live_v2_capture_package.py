"""Prepare the frozen v2 cohort without changing qualified native runtime files."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile

from research.m4_eligible_replay.contracts import canonical_json, digest_file

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
SESSION = ROOT / "session/20260914_m4_opportunity_control"
PLAN_HASH = "5dbe1ec155a9c3e7d57363820625286290f6803c0bd2e47484847ed3980e141e"
PROJECTION_HASH = "ed3050a2b6af92eb8ce93dca18aa9e4dec2d69b89cc23a0c757d8270ee20d70e"
CORE_HASH = "98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
PROTOCOL_HASH = "716ba0eb8f980ca5af0ef3d57e2415befc1fb1a315afeb01b9bc326a60341b48"
CONFIG = "examples/configs/grpo_math_1B_megatron_single_controller_m4_eligible_live_capture.yaml"
ANALYZER = f"{REPORT.relative_to(ROOT)}/analyze_eligible_live_decision_capture.py"
NATIVE_FILES = (
    CONFIG,
    ANALYZER,
    "uv.lock",
    "nemo_rl/algorithms/async_utils/eligible_live_decision_capsule.py",
    "nemo_rl/algorithms/async_utils/opportunity_at_risk_v2.py",
    "nemo_rl/algorithms/async_utils/rollout_lifecycle.py",
    "nemo_rl/algorithms/single_controller.py",
    "nemo_rl/algorithms/single_controller_utils/config.py",
    "nemo_rl/distributed/virtual_cluster.py",
)
OVERLAY_FILES = (
    "research/m4_eligible_replay/__init__.py",
    "research/m4_eligible_replay/contracts.py",
    "research/m4_paired_replay/__init__.py",
    "research/m4_paired_replay/capture_completion.py",
    "research/m4_paired_replay/capture_finalize.py",
)


def put(path: Path, value: dict) -> None:
    with path.open("xb") as stream:
        stream.write(canonical_json(value))


def native_prefix() -> str:
    """Reuse only the historical safe extraction blocks, not its v1 authority."""
    tree = ast.parse(
        (REPORT / "build_eligible_live_acquisition_capture_manifests.py").read_bytes()
    )
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "template" for t in node.targets)
    ]
    if len(assignments) != 1:
        raise ValueError("reference extraction template differs")
    prefix = ast.literal_eval(assignments[0].value).split('cd "$RUN_REPO"\n', 1)[0]
    prefix = prefix.replace(
        "readonly CAPSULE=$ASSETS/m4-eligible-live-$IDENTITY-capsule",
        "readonly CAPSULE=$ASSETS/m4-eligible-live-capsule",
    ).replace(
        "readonly HASHES=$ASSETS/m4-eligible-live-$IDENTITY-artifacts.sha256",
        "readonly HASHES=$ASSETS/artifacts.sha256",
    )
    return prefix


RUNTIME = r"""cd "$RUN_REPO"
export PYTHONPATH="$RUN_REPO"
export CUDA_VISIBLE_DEVICES=0,1
readonly WATCHDOG=$ASSETS/m4-eligible-live-$IDENTITY-watchdog.json
readonly TEST_LOG=$ASSETS/m4-eligible-live-$IDENTITY-tests.log
mkdir -p "$METRICS"
"$PYTHON" - "$RUN_REPO" <<'PY'
import base64,hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); overlay=@OVERLAY_JSON@
for name,row in overlay.items():
    data=base64.b64decode(row["base64"],validate=True)
    assert hashlib.sha256(data).hexdigest()==row["sha256"]
    path=root/name
    # Overlay only new outcome-free artifact tools, never qualified native files.
    assert name.startswith("research/m4_")
    path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
for name,expected in @NATIVE_HASHES_JSON@.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
print("M4_V2_NATIVE_SOURCE_UNCHANGED")
PY
"$PYTHON" - "$PLAN" "$EXECUTION_LOCK" "$IDENTITY" "$SEED" <<'PY'
import json,sys
from importlib.metadata import distribution
from pathlib import Path
from omegaconf import OmegaConf
from nemo_rl.algorithms.single_controller_utils.config import MasterConfig,validate_single_controller_config
from nemo_rl.utils.config import load_config,register_omegaconf_resolvers
plan=json.loads(Path(sys.argv[1]).read_bytes()); lock=json.loads(Path(sys.argv[2]).read_bytes())
identity=sys.argv[3]; seed=int(sys.argv[4])
assert {"identity":identity,"seed":seed} in plan["trajectories"]
assert lock["plan_sha256"]=="@PLAN_SHA256@"
assert lock["cohort"]==plan["trajectories"]
assert not lock["paired_replay_started"]
direct=json.loads(distribution("TransferQueue").read_text("direct_url.json"))
assert direct["vcs_info"]["commit_id"]=="b266d39a15aae114730de36cf8317b6285436f7f"
register_omegaconf_resolvers()
raw=OmegaConf.to_container(load_config(Path("@CONFIG_PATH@")),resolve=True)
raw["grpo"]["seed"]=seed
raw["async_rl"]["controlled_release_delay"]["seed"]=seed
raw["async_rl"]["controlled_release_delay"]["assignment_domain"]="@ASSIGNMENT_DOMAIN@"
capture=raw["async_rl"]["eligible_live_decision_capture"]
capture.update(trajectory_identity=identity,trajectory_seed=seed,progress_timeout_seconds=1200,
    suppress_capture_for_liveness_qualification=False)
config=MasterConfig(**raw); validate_single_controller_config(config)
assert config.grpo.max_num_steps==64 and not config.checkpointing["enabled"]
assert config.policy["model_name"]=="meta-llama/Llama-3.2-1B-Instruct"
assert config.async_rl.sampler.max_staleness_versions==1
assert not config.async_rl.eligible_live_decision_capture.suppress_capture_for_liveness_qualification
print("M4_V2_NATIVE_RUNTIME_AUTHORITY_PASS",identity,seed)
PY
# Each test command fails independently; no grouped-shell status masking.
"$PYTHON" -m pytest -q tests/unit/single_controller/test_eligible_live_decision_capsule.py 2>&1 | tee "$TEST_LOG"
"$PYTHON" -m pytest -q tests/unit/single_controller/test_opportunity_at_risk_v2.py -k 'm4_shield or liveness_diagnostic_state' 2>&1 | tee -a "$TEST_LOG"
"$PYTHON" -m pytest -q tests/unit/single_controller/test_single_controller.py -k 'eligible_live' 2>&1 | tee -a "$TEST_LOG"
set +e
"$PYTHON" examples/run_grpo_single_controller.py --config "@CONFIG_PATH@" \
  "grpo.seed=$SEED" \
  "async_rl.controlled_release_delay.seed=$SEED" \
  "async_rl.controlled_release_delay.assignment_domain=$ASSIGNMENT_DOMAIN" \
  "async_rl.eligible_live_decision_capture.trajectory_identity=$IDENTITY" \
  "async_rl.eligible_live_decision_capture.trajectory_seed=$SEED" \
  "async_rl.eligible_live_decision_capture.progress_timeout_seconds=1200" \
  "async_rl.eligible_live_decision_capture.liveness_diagnostic_path=$WATCHDOG" \
  "async_rl.eligible_live_decision_capture.suppress_capture_for_liveness_qualification=false" \
  "logger.log_dir=$METRICS" \
  "async_rl.lifecycle_audit_path=$LIFECYCLE" \
  "async_rl.gradient_opportunity_audit.output_path=$OPPORTUNITY" \
  "async_rl.gradient_opportunity_audit.observer_duty_path=$DUTY" \
  "async_rl.opportunity_at_risk_v2_shadow.output_path=$OARS" \
  "async_rl.eligible_live_decision_capture.output_dir=$CAPSULE" 2>&1 | tee "$RUN_LOG"
readonly RUN_RC=${PIPESTATUS[0]}
set -e
"$PYTHON" -m research.m4_paired_replay.capture_finalize \
  --assets "$ASSETS" --identity "$IDENTITY" --seed "$SEED" \
  --run-exit-code "$RUN_RC" --protocol "$PROTOCOL" --analyzer "@ANALYZER_PATH@"
"""


def build(output: Path) -> dict:
    """Freeze all inputs before producing any launchable manifests."""
    plan_path = REPORT / "eligible_live_acquisition_v2_plan.json"
    if digest_file(plan_path) != PLAN_HASH:
        raise ValueError("frozen scientific plan changed")
    plan = json.loads(plan_path.read_bytes())
    subprocess.run(
        [
            sys.executable,
            str(REPORT / "validate_eligible_live_acquisition_v2_design.py"),
        ],
        check=True,
        capture_output=True,
        cwd=ROOT,
    )
    engineering = json.loads(
        (REPORT / "paired_engineering_terminal_result.json").read_bytes()
    )
    if (
        engineering["status"] != "PASS_SYNTHETIC_ENGINEERING_QUALIFICATION"
        or not engineering["qualification_passed"]
    ):
        raise ValueError("engineering qualification not passed")
    source = (
        SESSION
        / "paired-engineering-generator-failure/source-runtime-projection.tar.gz"
    )
    core = SESSION / "m4-oars-shadow-preflight-megatron.tar.gz"
    if digest_file(source) != PROJECTION_HASH or digest_file(core) != CORE_HASH:
        raise ValueError("qualified transport archives changed")
    original = (
        SESSION / "m4-eligible-live-v2-liveness-qualification-source-11143be69.tar.gz"
    )
    if digest_file(original) != plan["source"]["qualified_runtime_archive_sha256"]:
        raise ValueError("original qualified source archive changed")
    spec = importlib.util.spec_from_file_location(
        "source_projection", REPORT / "build_paired_source_projection.py"
    )
    projection_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(projection_module)
    projection_proof = projection_module.validate(original, source)
    hashes = {}
    with tarfile.open(original) as full, tarfile.open(source) as projected:
        for name in NATIVE_FILES:
            before, after = (
                full.extractfile(name).read(),
                projected.extractfile(name).read(),
            )
            if before != after:
                raise ValueError(f"native runtime source changed: {name}")
            hashes[name] = hashlib.sha256(after).hexdigest()
    overlay = {
        name: {
            "sha256": digest_file(ROOT / name),
            "base64": base64.b64encode((ROOT / name).read_bytes()).decode(),
        }
        for name in OVERLAY_FILES
    }
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    output.mkdir(parents=True, exist_ok=False)
    authorization = {
        "schema": "m4-v2-fixed-capture-authorization-v1",
        "authorized_by": "human_user",
        "scope": "Previously approved frozen twenty-trajectory capture-only sequence; latest instruction: do it",
        "plan_sha256": PLAN_HASH,
        "attempts_per_identity": 1,
        "identities": plan["trajectories"],
        "automatic_retry": False,
        "paired_replay_authorized_by_this_record": False,
    }
    put(output / "authorization.json", authorization)
    lock = {
        "schema": "m4-v2-fixed-capture-execution-lock-v1",
        "plan_sha256": PLAN_HASH,
        "execution_source_commit": source_commit,
        "qualified_native_source_commit": plan["source"]["qualified_runtime_commit"],
        "source_projection_sha256": PROJECTION_HASH,
        "source_projection_validation": projection_proof,
        "megatron_sha256": CORE_HASH,
        "native_file_sha256": hashes,
        "artifact_tool_sha256": {name: row["sha256"] for name, row in overlay.items()},
        "cohort": plan["trajectories"],
        "execution": plan["execution"],
        "custom_config_sha256": digest_file(
            REPORT / "eligible_live_v2_capture_eos_custom_config.yaml"
        ),
        "authorization_sha256": digest_file(output / "authorization.json"),
        "engineering_result_sha256": digest_file(
            REPORT / "paired_engineering_terminal_result.json"
        ),
        "builder_sha256": digest_file(Path(__file__)),
        "extraction_reference_sha256": digest_file(
            REPORT / "build_eligible_live_acquisition_capture_manifests.py"
        ),
        "downstream_log_collector_enabled": False,
        "paired_replay_started": False,
        "post_update_outcomes_opened": False,
    }
    put(output / "execution-lock.json", lock)
    payload_paths = {
        "SOURCE": source,
        "MEGATRON": core,
        "PROTOCOL": REPORT / "eligible_live_decision_causal_protocol.json",
        "RESTORE_GATE": REPORT
        / "eligible_live_restore_qualification_terminal_result.json",
        "PLAN": plan_path,
        "AUTHORIZATION": output / "authorization.json",
        "LOCK": output / "execution-lock.json",
    }
    substitutions = {
        "@IMAGE_COMMIT@": "ae07eafe8035b5b2e84efa7234e70e7fd7e493c1",
        "@ASSIGNMENT_DOMAIN@": plan["assignment_domain"],
        "@CONFIG_PATH@": CONFIG,
        "@ANALYZER_PATH@": ANALYZER,
        "@OVERLAY_JSON@": repr(overlay),
        "@NATIVE_HASHES_JSON@": repr(hashes),
    }
    for key, path in payload_paths.items():
        substitutions[f"@{key}_PAYLOAD@"] = base64.b64encode(path.read_bytes()).decode()
        substitutions[f"@{key}_SHA256@"] = digest_file(path)
    candidates = []
    for row in plan["trajectories"]:
        script = native_prefix() + RUNTIME
        for token, value in substitutions.items():
            script = script.replace(token, value)
        script = script.replace("@IDENTITY@", row["identity"]).replace(
            "@SEED@", str(row["seed"])
        )
        if "@" in script:
            raise ValueError("unsubstituted runtime token")
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)
        script = (
            script.replace("{", "{{")
            .replace("}", "}}")
            .replace("{{assets_dir}}", "{assets_dir}")
        )
        value = {
            "type": "basic",
            "format_version": 1,
            "maintainers": ["nvegesna"],
            "loggers": ["stdout"],
            "labels": {"target": "silicon"},
            "launchers": {"type:slurm": {"nodes": 1, "ntasks_per_node": 1}},
            "spec": {
                "name": f"m4-eligible-live-acquisition-{row['identity']}",
                "workspace": "/workspace",
                "nodes": 1,
                "time_limit": 14400,
                "image_source": {
                    "local_path": "/lustre/fsw/coreai_dlalgo_ci/nvegesna/nemo_rl_images/nemo-rl-nightly-5802754.sqsh"
                },
                "script": script,
            },
        }
        path = output / f"{row['identity']}.json"
        put(path, value)
        candidates.append(
            row | {"manifest": str(path.relative_to(ROOT)), "sha256": digest_file(path)}
        )
    receipt = {
        "schema": "m4-v2-fixed-capture-package-v1",
        "status": "BUILT_NOT_VERIFIED_NOT_LAUNCH_READY",
        "plan_sha256": PLAN_HASH,
        "execution_lock_sha256": digest_file(output / "execution-lock.json"),
        "candidates": candidates,
        "native_file_sha256": hashes,
        "artifact_tool_sha256": lock["artifact_tool_sha256"],
        "new_eos_launches": 0,
    }
    put(output / "package.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build(args.output.resolve())
    print(
        json.dumps(
            {"status": receipt["status"], "candidates": len(receipt["candidates"])}
        )
    )


if __name__ == "__main__":
    main()
