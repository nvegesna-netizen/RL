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
"""Clean-room validate the DDP-hook-repaired EOS preflight v3."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import sys
import tarfile
import tempfile
from copy import deepcopy
from pathlib import Path, PurePosixPath

from jetclient import JETWorkloadManifest

MANIFEST_SHA256 = "77096eb6fe922905fda2b2560fef49212d439a7c69a79a6abf9a3665cd8222d0"
V2_MANIFEST_SHA256 = "975ddde2d33f074fa1d39f28007c924576ff8b93a653728e1e496f20faedded4"
V2_AUTHORIZATION_SHA256 = (
    "a87b1286a8864fc7e1fe8b94a99a431dcb72aba558d05b353c8e57b8d44c9250"
)
V2_TERMINAL_RESULT_SHA256 = (
    "b38b7fc7646f48135f58d661be29932898d45d2d44b5c68124af019cd461f0a1"
)
V3_AUTHORIZATION_SHA256 = (
    "8f542ee8c21a6ee11e8ad7eb37df5755f79ec7573b7caaa16db8b327db34f809"
)
OLD_SOURCE_SHA256 = "80741c66b32a7951e4ebd620e66a8043858159e7a871c7fb87d038db068aa714"
NEW_SOURCE_SHA256 = "3e1cf6fb4fb287c120e7502beeedd6fb50534af3bc83b4becad610e76bb34966"
BUILDER_SHA256 = "43f2be565183ca9191f87e1303d1d8849fc40ed46a7c05d3a0cbd7c5202f3fad"
CUSTOM_SHA256 = "c13a73b87669569429c36fd0c60f09d2a6ecabfb4086063395924cc26566dc8d"
ADDITIONAL_SHA256 = "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356"


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_extract(archive: Path, output: Path) -> tuple[int, int]:
    """Extract an archive after rejecting path and symlink-parent escapes."""
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        names = [PurePosixPath(member.name) for member in members]
        if len(names) != len(set(names)):
            raise RuntimeError("source archive has duplicate paths")
        if any(name.is_absolute() or ".." in name.parts for name in names):
            raise RuntimeError("source archive has an unsafe path")
        symlinks = {
            name for name, member in zip(names, members, strict=True) if member.issym()
        }
        if any(any(parent in symlinks for parent in name.parents) for name in names):
            raise RuntimeError("source archive descends through a symlink")
        source.extractall(output)
    return len(members), len(symlinks)


def main() -> None:
    """Validate identity, repair, authority, and zero-training boundaries."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--v2-manifest", type=Path, required=True)
    parser.add_argument("--v2-authorization", type=Path, required=True)
    parser.add_argument("--v2-terminal-result", type=Path, required=True)
    parser.add_argument("--v3-authorization", type=Path, required=True)
    parser.add_argument("--old-source", type=Path, required=True)
    parser.add_argument("--new-source", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--custom-config", type=Path, required=True)
    parser.add_argument("--additional-variables", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path, expected in (
        (args.manifest, MANIFEST_SHA256),
        (args.v2_manifest, V2_MANIFEST_SHA256),
        (args.v2_authorization, V2_AUTHORIZATION_SHA256),
        (args.v2_terminal_result, V2_TERMINAL_RESULT_SHA256),
        (args.v3_authorization, V3_AUTHORIZATION_SHA256),
        (args.old_source, OLD_SOURCE_SHA256),
        (args.new_source, NEW_SOURCE_SHA256),
        (args.builder, BUILDER_SHA256),
        (args.custom_config, CUSTOM_SHA256),
        (args.additional_variables, ADDITIONAL_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen v3 preflight input moved: {path.name}")

    authorization = json.loads(args.v3_authorization.read_bytes())
    if (
        authorization["schema"]
        != "m4-downstream-quality-no-training-eos-preflight-v3-authorization-v1"
        or authorization["scope"]
        != "exactly_one_repaired_credential_free_downstream_quality_no_training_eos_preflight_v3"
        or authorization["user_authorization"] != "I authorize"
        or authorization["predecessor_terminal_result_sha256"]
        != V2_TERMINAL_RESULT_SHA256
        or authorization["source_archive_sha256"] != NEW_SOURCE_SHA256
        or authorization["required_repair"]
        != "guard_checkpoint_forward_pre_hook_by_actual_enabled_state"
        or authorization["required_launcher"] != "runllm.py --no_wait"
        or authorization["submission_attempt_limit"] != 1
        or authorization["eos_submission_authorized"] is not True
        or authorization["model_weight_access_authorized"] is not True
        or authorization["optimizer_steps_allowed"] != 0
    ):
        raise RuntimeError("v3 authority differs")
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
        raise RuntimeError("v3 authority is not fail-closed")

    custom = json.loads(args.custom_config.read_bytes())
    launcher = custom["launchers"]["dgxh100_eos"]
    if (
        custom["environments"]["retrier"]["enabled"] is not False
        or launcher["sbatch_additional_flags"]["time"] != "02:00:00"
        or launcher["srun_additional_flags"]["time"] != "02:00:00"
        or json.loads(args.additional_variables.read_bytes()) != {}
    ):
        raise RuntimeError("v3 scheduler or retry boundary differs")

    manifest = json.loads(args.manifest.read_bytes())
    JETWorkloadManifest.model_validate(deepcopy(manifest))
    spec = manifest["spec"]
    if (
        spec["name"] != "m4-downstream-quality-no-training-preflight-v3"
        or spec["nodes"] != 1
        or spec["time_limit"] != 7200
    ):
        raise RuntimeError("v3 manifest boundary differs")
    rendered = spec["script"].format(
        assets_dir="/tmp/m4-downstream-quality-preflight-v3"
    )
    subprocess.run(["bash", "-n"], input=rendered, text=True, check=True)
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", rendered, flags=re.DOTALL)
    if len(blocks) != 8:
        raise RuntimeError(f"embedded Python block count differs: {len(blocks)}")
    for index, block in enumerate(blocks):
        compile(block, f"<m4-downstream-quality-v3-{index}>", "exec")

    authority = subprocess.run(
        [sys.executable, "-c", blocks[0], str(args.v3_authorization)],
        cwd=args.repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if (
        authority.returncode != 0
        or "M4_DOWNSTREAM_QUALITY_EOS_PREFLIGHT_V3_AUTHORITY_PASS"
        not in authority.stdout
    ):
        raise RuntimeError("v3 dynamic authority did not pass")

    config_block = blocks[4]
    register_text = "register_omegaconf_resolvers()"
    resolve_text = "OmegaConf.to_container(load_config(config_path),resolve=True)"
    if (
        config_block.count(register_text) != 1
        or config_block.count(resolve_text) != 1
        or config_block.index(register_text) > config_block.index(resolve_text)
    ):
        raise RuntimeError("v1 resolver repair moved")

    encoded = re.findall(r"printf %s '([A-Za-z0-9+/=]+)' \| base64 -d", rendered)
    embedded_hashes = {
        hashlib.sha256(base64.b64decode(value)).hexdigest() for value in encoded
    }
    required_payloads = {
        NEW_SOURCE_SHA256,
        V3_AUTHORIZATION_SHA256,
        authorization["megatron_dependency_sha256"],
        authorization["protocol_draft_sha256"],
        authorization["preflight_plan_sha256"],
    }
    if not required_payloads.issubset(embedded_hashes):
        raise RuntimeError("v3 embedded payload differs")
    if (
        OLD_SOURCE_SHA256 in embedded_hashes
        or V2_AUTHORIZATION_SHA256 in embedded_hashes
    ):
        raise RuntimeError("v3 contains a predecessor payload")

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

    with tempfile.TemporaryDirectory(prefix="m4-downstream-quality-v3-") as raw:
        cleanroom = Path(raw)
        member_count, symlink_count = safe_extract(
            args.new_source, cleanroom / "source"
        )
        worker_text = (
            cleanroom / "source/nemo_rl/models/policy/workers/megatron_policy_worker.py"
        ).read_text()
        test_text = (
            cleanroom / "source/tests/unit/models/policy/test_megatron_worker.py"
        ).read_text()
        if (
            "forward_pre_hook_was_enabled = (" not in worker_text
            or worker_text.count("if forward_pre_hook_was_enabled:") != 2
            or "test_megatron_save_checkpoint_only_toggles_enabled_forward_pre_hook"
            not in test_text
        ):
            raise RuntimeError("DDP-hook source repair is absent or differs")

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
            raise RuntimeError("v3 expanded-authority negative control passed")

        rebuilt = cleanroom / "rebuilt.json"
        subprocess.run(
            [
                sys.executable,
                str(args.builder),
                "--v2-manifest",
                str(args.v2_manifest),
                "--v2-authorization",
                str(args.v2_authorization),
                "--v2-terminal-result",
                str(args.v2_terminal_result),
                "--v3-authorization",
                str(args.v3_authorization),
                "--old-source",
                str(args.old_source),
                "--new-source",
                str(args.new_source),
                "--output",
                str(rebuilt),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if rebuilt.read_bytes() != args.manifest.read_bytes():
            raise RuntimeError("deterministic v3 rebuild differs")

    result = {
        "schema": "m4-downstream-quality-eos-preflight-v3-package-validation-v1",
        "status": "PASS",
        "manifest_sha256": MANIFEST_SHA256,
        "authorization_sha256": V3_AUTHORIZATION_SHA256,
        "predecessor_terminal_result_sha256": V2_TERMINAL_RESULT_SHA256,
        "source_archive_sha256": NEW_SOURCE_SHA256,
        "source_archive_members": member_count,
        "source_archive_symlinks": symlink_count,
        "builder_sha256": BUILDER_SHA256,
        "jet_manifest_schema_passed": True,
        "bash_syntax_passed": True,
        "embedded_python_blocks": len(blocks),
        "embedded_payload_hashes_passed": True,
        "dynamic_authority_passed": True,
        "expanded_authority_negative_control_passed": True,
        "credential_surface_absent": True,
        "resolver_repair_preserved": True,
        "ddp_hook_repair_present": True,
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
