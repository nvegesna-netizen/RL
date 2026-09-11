#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Clean-room validate the Bridge-packaging-repaired EOS preflight v4."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.machinery
import json
import re
import subprocess
import sys
import tarfile
import tempfile
from copy import deepcopy
from pathlib import Path, PurePosixPath

from jetclient import JETWorkloadManifest

MANIFEST_SHA256 = "fcd82a3e65da33c86f51923aa7236c8a2a2b7e4aeb233f3254f66f228f931222"
V3_MANIFEST_SHA256 = "77096eb6fe922905fda2b2560fef49212d439a7c69a79a6abf9a3665cd8222d0"
V3_AUTHORIZATION_SHA256 = (
    "8f542ee8c21a6ee11e8ad7eb37df5755f79ec7573b7caaa16db8b327db34f809"
)
V3_TERMINAL_RESULT_SHA256 = (
    "58ee78843d6967ad19d95ba59fe6532ab6f7de9d5b27c059d1883d92469f7e6f"
)
V4_AUTHORIZATION_SHA256 = (
    "76cc225ff4d09b06cfbf85f8d97c3056cda7b558a2c3e60e4a4d84f1545544a2"
)
SOURCE_SHA256 = "3e1cf6fb4fb287c120e7502beeedd6fb50534af3bc83b4becad610e76bb34966"
MEGATRON_DEPENDENCY_SHA256 = (
    "98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
)
BRIDGE_ARCHIVE_SHA256 = (
    "1429945d1d50045900e40314fa283fa5f66484e945aad4920da137d7ad2c9313"
)
BUILDER_SHA256 = "3ef3b93706d200d82ed4470af389ebd101d097a1d6307a88abbf6c55daebaf5f"
CUSTOM_SHA256 = "c13a73b87669569429c36fd0c60f09d2a6ecabfb4086063395924cc26566dc8d"
ADDITIONAL_SHA256 = "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356"


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_extract(
    archive: Path, output: Path, expected_count: int, expected_prefix: str
) -> tuple[int, int]:
    """Extract an archive after rejecting path and symlink-parent escapes."""
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        names = [PurePosixPath(member.name) for member in members]
        if len(members) != expected_count:
            raise RuntimeError(f"archive member count differs: {archive.name}")
        if len(names) != len(set(names)):
            raise RuntimeError(f"archive has duplicate paths: {archive.name}")
        if not all(
            str(name) == expected_prefix.rstrip("/")
            or str(name).startswith(expected_prefix)
            for name in names
        ):
            raise RuntimeError(f"archive prefix differs: {archive.name}")
        if any(name.is_absolute() or ".." in name.parts for name in names):
            raise RuntimeError(f"archive has an unsafe path: {archive.name}")
        symlinks = {
            name for name, member in zip(names, members, strict=True) if member.issym()
        }
        if any(any(parent in symlinks for parent in name.parents) for name in names):
            raise RuntimeError(f"archive descends through a symlink: {archive.name}")
        source.extractall(output)
    return len(members), len(symlinks)


def main() -> None:
    """Validate identity, dependency topology, authority, and no-training bounds."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--v3-manifest", type=Path, required=True)
    parser.add_argument("--v3-authorization", type=Path, required=True)
    parser.add_argument("--v3-terminal-result", type=Path, required=True)
    parser.add_argument("--v4-authorization", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--megatron-dependency", type=Path, required=True)
    parser.add_argument("--bridge-archive", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--custom-config", type=Path, required=True)
    parser.add_argument("--additional-variables", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path, expected in (
        (args.manifest, MANIFEST_SHA256),
        (args.v3_manifest, V3_MANIFEST_SHA256),
        (args.v3_authorization, V3_AUTHORIZATION_SHA256),
        (args.v3_terminal_result, V3_TERMINAL_RESULT_SHA256),
        (args.v4_authorization, V4_AUTHORIZATION_SHA256),
        (args.source, SOURCE_SHA256),
        (args.megatron_dependency, MEGATRON_DEPENDENCY_SHA256),
        (args.bridge_archive, BRIDGE_ARCHIVE_SHA256),
        (args.builder, BUILDER_SHA256),
        (args.custom_config, CUSTOM_SHA256),
        (args.additional_variables, ADDITIONAL_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen v4 preflight input moved: {path.name}")

    authorization = json.loads(args.v4_authorization.read_bytes())
    if (
        authorization["schema"]
        != "m4-downstream-quality-no-training-eos-preflight-v4-authorization-v1"
        or authorization["scope"]
        != "exactly_one_bridge_packaging_repaired_credential_free_downstream_quality_no_training_eos_preflight_v4"
        or authorization["predecessor_terminal_result_sha256"]
        != V3_TERMINAL_RESULT_SHA256
        or authorization["predecessor_manifest_sha256"] != V3_MANIFEST_SHA256
        or authorization["source_archive_sha256"] != SOURCE_SHA256
        or authorization["megatron_bridge_archive_sha256"] != BRIDGE_ARCHIVE_SHA256
        or authorization["megatron_dependency_sha256"] != MEGATRON_DEPENDENCY_SHA256
        or authorization["required_repair"]
        != "package_pinned_megatron_bridge_src_and_gate_actual_converter_import_before_model_access"
        or authorization["required_launcher"] != "runllm.py --no_wait"
        or authorization["submission_attempt_limit"] != 1
        or authorization["eos_submission_authorized"] is not True
        or authorization["model_weight_access_authorized"] is not True
        or authorization["optimizer_steps_allowed"] != 0
    ):
        raise RuntimeError("v4 authority differs")
    blocked = (
        "optimizer_initialization_authorized",
        "training_authorized",
        "qualification_authorized",
        "pilot_authorized",
        "scientific_acquisition_authorized",
        "automatic_retry",
        "automatic_extension",
    )
    if any(authorization[key] for key in blocked):
        raise RuntimeError("v4 authority is not fail-closed")

    custom = json.loads(args.custom_config.read_bytes())
    launcher = custom["launchers"]["dgxh100_eos"]
    if (
        custom["environments"]["retrier"]["enabled"] is not False
        or launcher["sbatch_additional_flags"]["time"] != "02:00:00"
        or launcher["srun_additional_flags"]["time"] != "02:00:00"
        or json.loads(args.additional_variables.read_bytes()) != {}
    ):
        raise RuntimeError("v4 scheduler or retry boundary differs")

    manifest = json.loads(args.manifest.read_bytes())
    JETWorkloadManifest.model_validate(deepcopy(manifest))
    spec = manifest["spec"]
    if (
        spec["name"] != "m4-downstream-quality-no-training-preflight-v4"
        or spec["nodes"] != 1
        or spec["time_limit"] != 7200
    ):
        raise RuntimeError("v4 manifest boundary differs")
    rendered = spec["script"].format(
        assets_dir="/tmp/m4-downstream-quality-preflight-v4"
    )
    subprocess.run(["bash", "-n"], input=rendered, text=True, check=True)
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", rendered, flags=re.DOTALL)
    if len(blocks) != 9:
        raise RuntimeError(f"embedded Python block count differs: {len(blocks)}")
    for index, block in enumerate(blocks):
        compile(block, f"<m4-downstream-quality-v4-{index}>", "exec")

    authority = subprocess.run(
        [sys.executable, "-c", blocks[0], str(args.v4_authorization)],
        cwd=args.repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if (
        authority.returncode != 0
        or "M4_DOWNSTREAM_QUALITY_EOS_PREFLIGHT_V4_AUTHORITY_PASS"
        not in authority.stdout
    ):
        raise RuntimeError("v4 dynamic authority did not pass")

    encoded = re.findall(r"printf %s '([A-Za-z0-9+/=]+)' \| base64 -d", rendered)
    embedded_hashes = {
        hashlib.sha256(base64.b64decode(value)).hexdigest() for value in encoded
    }
    required_payloads = {
        SOURCE_SHA256,
        V4_AUTHORIZATION_SHA256,
        MEGATRON_DEPENDENCY_SHA256,
        BRIDGE_ARCHIVE_SHA256,
        authorization["protocol_draft_sha256"],
        authorization["preflight_plan_sha256"],
    }
    if not required_payloads.issubset(embedded_hashes):
        raise RuntimeError("v4 embedded payload differs")
    if V3_AUTHORIZATION_SHA256 in embedded_hashes:
        raise RuntimeError("v4 contains predecessor authority")

    bridge_extract = rendered.index("M4_DOWNSTREAM_QUALITY_MEGATRON_DEPENDENCIES_PASS")
    pythonpath = rendered.index(
        'export PYTHONPATH="$RUN_REPO/3rdparty/Megatron-Bridge-workspace/'
        "Megatron-Bridge/src:$RUN_REPO/3rdparty/Megatron-Bridge-workspace/"
        'Megatron-Bridge/3rdparty/Megatron-LM:$RUN_REPO"'
    )
    bridge_import = rendered.index("M4_DOWNSTREAM_QUALITY_BRIDGE_IMPORT_PASS")
    converter_import = rendered.index("M4_DOWNSTREAM_QUALITY_CONVERTER_IMPORT_PASS")
    model_access = rendered.index("init_ray()")
    if (
        not bridge_extract
        < pythonpath
        < bridge_import
        < converter_import
        < model_access
    ):
        raise RuntimeError("v4 pre-model dependency gate ordering differs")
    if rendered.count("convert_megatron_to_hf.py --help") != 1:
        raise RuntimeError("actual converter import gate differs")

    config_block = blocks[5]
    register_text = "register_omegaconf_resolvers()"
    resolve_text = "OmegaConf.to_container(load_config(config_path),resolve=True)"
    if (
        config_block.count(register_text) != 1
        or config_block.count(resolve_text) != 1
        or config_block.index(register_text) > config_block.index(resolve_text)
    ):
        raise RuntimeError("v1 resolver repair moved")

    forbidden = (
        "finish_train_step(",
        "train_microbatches(",
        "begin_train_step(",
        "examples/run_grpo",
        "python runllm.py",
        "sbatch ",
        "srun ",
        "HF_TOKEN",
        "PRIVATE-TOKEN",
        "WANDB_API_KEY",
        "NVIDIA_API_KEY",
        "NGC_API_KEY",
    )
    if any(value in rendered for value in forbidden):
        raise RuntimeError("training, nested submission, or credential surface present")
    if "init_optimizer=False" not in rendered or '"optimizer_steps":0' not in rendered:
        raise RuntimeError("zero-optimizer boundary differs")

    with tempfile.TemporaryDirectory(prefix="m4-downstream-quality-v4-") as raw:
        cleanroom = Path(raw)
        source_count, source_symlinks = safe_extract(
            args.source, cleanroom / "source", 2261, ""
        )
        bridge_count, bridge_symlinks = safe_extract(
            args.bridge_archive, cleanroom / "deps", 1060, "Megatron-Bridge/"
        )
        megatron_count, megatron_symlinks = safe_extract(
            args.megatron_dependency,
            cleanroom / "deps/Megatron-Bridge/3rdparty",
            726,
            "Megatron-LM/",
        )
        bridge_src = cleanroom / "deps/Megatron-Bridge/src"
        megatron_src = cleanroom / "deps/Megatron-Bridge/3rdparty/Megatron-LM"
        namespace = importlib.machinery.PathFinder.find_spec(
            "megatron", [str(bridge_src), str(megatron_src)]
        )
        if namespace is None or namespace.submodule_search_locations is None:
            raise RuntimeError("assembled megatron namespace is absent")
        search = list(namespace.submodule_search_locations)
        bridge_spec = importlib.machinery.PathFinder.find_spec(
            "megatron.bridge", search
        )
        core_spec = importlib.machinery.PathFinder.find_spec("megatron.core", search)
        if bridge_spec is None or core_spec is None:
            raise RuntimeError("assembled Bridge/Megatron topology is incomplete")
        if "Megatron-Bridge/src" not in str(bridge_spec.origin):
            raise RuntimeError("Bridge resolves outside the packaged source")
        if "Megatron-LM/megatron/core" not in str(core_spec.origin):
            raise RuntimeError("Megatron Core resolves outside the packaged source")

        worker_text = (
            cleanroom / "source/nemo_rl/models/policy/workers/megatron_policy_worker.py"
        ).read_text()
        if (
            "forward_pre_hook_was_enabled = (" not in worker_text
            or worker_text.count("if forward_pre_hook_was_enabled:") != 2
        ):
            raise RuntimeError("v2 DDP-hook repair moved")

        tampered = dict(authorization)
        tampered["optimizer_initialization_authorized"] = True
        tampered_path = cleanroom / "tampered.json"
        tampered_path.write_text(json.dumps(tampered))
        negative = subprocess.run(
            [sys.executable, "-c", blocks[0], str(tampered_path)],
            cwd=args.repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if negative.returncode == 0 or "AssertionError" not in negative.stderr:
            raise RuntimeError("v4 expanded-authority negative control passed")

        rebuilt = cleanroom / "rebuilt.json"
        subprocess.run(
            [
                sys.executable,
                str(args.builder),
                "--v3-manifest",
                str(args.v3_manifest),
                "--v3-authorization",
                str(args.v3_authorization),
                "--v3-terminal-result",
                str(args.v3_terminal_result),
                "--v4-authorization",
                str(args.v4_authorization),
                "--megatron-dependency",
                str(args.megatron_dependency),
                "--bridge-archive",
                str(args.bridge_archive),
                "--output",
                str(rebuilt),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if rebuilt.read_bytes() != args.manifest.read_bytes():
            raise RuntimeError("deterministic v4 rebuild differs")

    result = {
        "schema": "m4-downstream-quality-eos-preflight-v4-package-validation-v1",
        "status": "PASS",
        "manifest_sha256": MANIFEST_SHA256,
        "authorization_sha256": V4_AUTHORIZATION_SHA256,
        "predecessor_terminal_result_sha256": V3_TERMINAL_RESULT_SHA256,
        "source_archive_sha256": SOURCE_SHA256,
        "source_archive_members": source_count,
        "source_archive_symlinks": source_symlinks,
        "megatron_bridge_commit": authorization["megatron_bridge_commit"],
        "megatron_bridge_archive_sha256": BRIDGE_ARCHIVE_SHA256,
        "megatron_bridge_archive_members": bridge_count,
        "megatron_bridge_archive_symlinks": bridge_symlinks,
        "megatron_lm_commit": authorization["megatron_lm_commit"],
        "megatron_dependency_sha256": MEGATRON_DEPENDENCY_SHA256,
        "megatron_dependency_members": megatron_count,
        "megatron_dependency_symlinks": megatron_symlinks,
        "builder_sha256": BUILDER_SHA256,
        "jet_manifest_schema_passed": True,
        "bash_syntax_passed": True,
        "embedded_python_blocks": len(blocks),
        "embedded_payload_hashes_passed": True,
        "dynamic_authority_passed": True,
        "expanded_authority_negative_control_passed": True,
        "credential_surface_absent": True,
        "resolver_repair_preserved": True,
        "ddp_hook_repair_preserved": True,
        "assembled_namespace_topology_passed": True,
        "actual_converter_pre_model_gate_present": True,
        "zero_training_surface_passed": True,
        "deterministic_rebuild_passed": True,
        "jet_retrier_enabled": False,
        "submission_attempt_limit": 1,
        "submitted": False,
        "optimizer_initialized": False,
        "optimizer_steps": 0,
        "training_started": False,
        "automatic_retry": False,
        "automatic_extension": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
