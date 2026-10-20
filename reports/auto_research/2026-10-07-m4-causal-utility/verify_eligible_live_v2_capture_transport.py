"""Verify capture manifests, native extraction and real JET collector suppression."""

from __future__ import annotations

import argparse
import ast
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from jetclient import JETWorkloadManifest
from jet.scripts.plugins.core.jobs import Job, JobStep, Pipeline
from jet.scripts.plugins.core.logs import LogsPluginConfig, on_core_jobs
from jet.scripts.plugins.executors.gitlab.gitlab import (
    GitlabExecutorConfig,
    on_gitlab_pipeline_create_steps,
    on_gitlab_pipeline_create_names,
)
from jet.utils.auth_context import AuthContext
from jet.utils.file_builders import YamlFileBuilder

from research.m4_eligible_replay.contracts import canonical_json, digest_file
from research.m4_paired_replay.capture_authentication import eos_job_name
from research.m4_paired_replay.config_validation import (
    execute_capture_overrides,
    validate_native_constraints,
)

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent


def verify(package_dir: Path) -> dict:
    package = json.loads((package_dir / "package.json").read_bytes())
    lock = json.loads((package_dir / "execution-lock.json").read_bytes())
    if package["execution_lock_sha256"] != digest_file(
        package_dir / "execution-lock.json"
    ):
        raise ValueError("execution lock changed")
    config = LogsPluginConfig(enabled=False)
    pipeline = Pipeline(
        jobs=[
            Job(
                name="basic/m4-eligible-live-acquisition-v2a01 dgxh100_eos",
                launcher="dgxh100_eos",
                steps=[JobStep(name="dgxh100_eos")],
            )
        ]
    )
    on_core_jobs(pipeline, config, AuthContext.unauthenticated())
    if [step.name for step in next(iter(pipeline.jobs)).steps] != ["dgxh100_eos"]:
        raise ValueError("real JET hook created a collector")
    on_gitlab_pipeline_create_names("jet-ci", pipeline)
    executor = GitlabExecutorConfig(type="gitlab", default_tags=[], environments={})
    generated = YamlFileBuilder(data={})
    on_gitlab_pipeline_create_steps("jet-ci", executor, generated, pipeline, "prod")
    jobs = generated.data
    expected_name = eos_job_name("v2a01")
    if set(jobs) != {expected_name} or jobs[expected_name]["needs"] != [
        {"pipeline": "${JET_PIPELINE_ID}", "job": "generate"}
    ]:
        raise ValueError("capture topology contains unexpected dependency")
    if ".artifacts.logs" not in jobs[expected_name]["extends"]:
        raise ValueError("direct EOS workload artifacts disabled")
    row = package["candidates"][0]
    manifest = json.loads((ROOT / row["manifest"]).read_bytes())
    render = manifest["spec"]["script"].format(
        assets_dir=f"/jet/assets/basic/{manifest['spec']['name']}"
    )
    with tempfile.TemporaryDirectory(prefix="m4-v2-capture-clean-room-") as directory:
        temp = Path(directory)
        # Execute the actual immutable payload/extraction blocks, not GPU training.
        prefix = render.split('cd "$RUN_REPO"\n', 1)[0]
        prefix = prefix.replace("/opt/nemo_rl_venv/bin/python", sys.executable)
        prefix = prefix.replace("/workspace/", str(temp) + "/")
        prefix = prefix.replace(
            "${NEMO_RL_COMMIT:-unknown}", "ae07eafe8035b5b2e84efa7234e70e7fd7e493c1"
        )
        subprocess.run(
            ["bash"], input=prefix, check=True, capture_output=True, text=True
        )
        extracted = temp / "m4-eligible-live-acquisition-repo"
        for name, expected in lock["native_file_sha256"].items():
            if digest_file(extracted / name) != expected:
                raise ValueError(f"clean-room native source changed: {name}")
        # Syntax-check every embedded Python heredoc independently.

        blocks = re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", render, re.S)
        for block in blocks:
            ast.parse(block)
        if len(blocks) != (5 if package.get("preflight_only") else 4):
            raise ValueError("unexpected runtime heredoc count")
        overlay_block = blocks[2]
        subprocess.run(
            [sys.executable, "-c", overlay_block, str(extracted)],
            check=True,
            capture_output=True,
            text=True,
        )
        for name, expected in lock["artifact_tool_sha256"].items():
            if digest_file(extracted / name) != expected:
                raise ValueError("clean-room artifact tool changed")
        # Execute the complete native cross-section validator on the actual
        # packaged preflight assignments. Parsing alone missed the original bug.
        # Optional host-only Hydra dependencies are not needed for JET imports.
        from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
        from omegaconf import OmegaConf

        register_omegaconf_resolvers()
        config_path = (
            extracted
            / "examples/configs/grpo_math_1B_megatron_single_controller_m4_eligible_live_capture.yaml"
        )
        raw = OmegaConf.to_container(load_config(config_path), resolve=True)
        for candidate in package["candidates"]:
            candidate_manifest = json.loads((ROOT / candidate["manifest"]).read_bytes())
            asset_root = f"/jet/assets/basic/{candidate_manifest['spec']['name']}"
            runtime = candidate_manifest["spec"]["script"].format(assets_dir=asset_root)
            watchdog = (
                f"{asset_root}/m4-eligible-live-{candidate['identity']}-watchdog.json"
            )
            configured = execute_capture_overrides(
                runtime,
                raw,
                identity=candidate["identity"],
                seed=candidate["seed"],
                watchdog=watchdog,
                domain="m4-eligible-live-causal-acquisition-v2",
            )
            validate_native_constraints(extracted, configured)
        # Run real closed-file completion against synthetic boundary inputs only.
        sys.path.insert(0, str(ROOT))
        # Test-only fixture dependencies are unnecessary for JET schema imports.
        from research.m4_paired_replay.test_integration import synthetic_window

        _, files = synthetic_window(native_exception=True)
        assets = temp / "closed-assets"
        assets.mkdir()
        for name, data in files.items():
            (assets / f"m4-eligible-live-v2a01-{name}").write_bytes(data)
        command = [
            sys.executable,
            "-m",
            "research.m4_paired_replay.capture_finalize",
            "--assets",
            str(assets),
            "--identity",
            "v2a01",
            "--seed",
            "20261201",
            "--run-exit-code",
            "1",
            "--protocol",
            str(temp / "protocol"),
            "--analyzer",
            str(temp / "not-used"),
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = str(extracted)
        subprocess.run(
            command, cwd=extracted, env=env, check=True, capture_output=True, text=True
        )
        proof = json.loads(
            (assets / "m4-eligible-live-v2a01-progress.json").read_bytes()
        )
        if proof["run_exit_code"] != 1 or proof["eligible_divergence_count"] != 0:
            raise ValueError("extracted finalizer boundary changed")
    hashes = []
    for candidate in package["candidates"]:
        path = ROOT / candidate["manifest"]
        if digest_file(path) != candidate["sha256"]:
            raise ValueError("candidate manifest changed")
        value = json.loads(path.read_bytes())
        JETWorkloadManifest.model_validate(copy.deepcopy(value))
        if value["spec"]["time_limit"] != 14400:
            raise ValueError("allocation limit changed")
        script = value["spec"]["script"].format(
            assets_dir=f"/jet/assets/basic/{value['spec']['name']}"
        )
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)
        # Exactly three source/script copies are generated by JET; use a fully
        # uncompressed conservative bound, with 4MiB for other generated files.
        bound = 3 * len(canonical_json(value)) + (4 << 20)
        if bound >= (50 << 20):
            raise ValueError("generator artifact budget lacks sufficient margin")
        hashes.append(candidate["sha256"])
    return {
        "schema": "m4-v2-capture-transport-verification-v1",
        "status": "PASS_OFFLINE_TRANSPORT_AND_NATIVE_EXTRACTION",
        "package_sha256": digest_file(package_dir / "package.json"),
        "execution_lock_sha256": digest_file(package_dir / "execution-lock.json"),
        "manifest_count": len(hashes),
        "manifest_sha256": hashes,
        "native_files_verified_after_extraction": len(lock["native_file_sha256"]),
        "all_embedded_python_parsed": True,
        "all_rendered_shells_parsed": True,
        "actual_jet_log_hook_created_no_collector": True,
        "actual_jet_gitlab_emitter_only_generation_dependency": True,
        "direct_eos_artifact_inheritance_retained": True,
        "clean_room_native_no_event_finalizer_passed": True,
        "complete_native_cross_section_validator_executed": True,
        "native_configuration_probe_count": len(package["candidates"]),
        "generated_capture_job_name_matches_reader": True,
        "runtime_and_checkpoint_training_not_executed_locally": True,
        "jet_api_version": importlib.metadata.version("jet-api"),
        "large_checkpoint_downloads": 0,
        "new_eos_launches": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.package)
    with args.output.open("xb") as stream:
        stream.write(canonical_json(result))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
