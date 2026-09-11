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
"""Validate the resolver-repaired downstream-quality EOS preflight."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

from jetclient import JETWorkloadManifest

MANIFEST_SHA256 = "975ddde2d33f074fa1d39f28007c924576ff8b93a653728e1e496f20faedded4"
PREDECESSOR_MANIFEST_SHA256 = (
    "13dfd195a2ea2a5f6b9426b854531a294a51f31214d6e195df5d6cec5b5e5103"
)
PREDECESSOR_AUTHORIZATION_SHA256 = (
    "a2cd3cb97818d530b4dfb72312eb4ee266322c534c7a51e89f49006e04a26ceb"
)
V2_AUTHORIZATION_SHA256 = (
    "a87b1286a8864fc7e1fe8b94a99a431dcb72aba558d05b353c8e57b8d44c9250"
)
TERMINAL_RESULT_SHA256 = (
    "551383a7d5243821a84a3d3765f9222c568d03ea2e5e6a3f7732f4e44cfb7878"
)
BUILDER_SHA256 = "5c33c8afb6162f5cd711c5f1663aca92cefe99c4dad83701c3e1511533d2820e"
SOURCE_SHA256 = "80741c66b32a7951e4ebd620e66a8043858159e7a871c7fb87d038db068aa714"
CUSTOM_SHA256 = "c13a73b87669569429c36fd0c60f09d2a6ecabfb4086063395924cc26566dc8d"
ADDITIONAL_SHA256 = "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356"


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Validate the new identity, exact repair, and zero-training boundary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predecessor-manifest", type=Path, required=True)
    parser.add_argument("--predecessor-authorization", type=Path, required=True)
    parser.add_argument("--v2-authorization", type=Path, required=True)
    parser.add_argument("--terminal-result", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--custom-config", type=Path, required=True)
    parser.add_argument("--additional-variables", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path, expected in (
        (args.manifest, MANIFEST_SHA256),
        (args.predecessor_manifest, PREDECESSOR_MANIFEST_SHA256),
        (args.predecessor_authorization, PREDECESSOR_AUTHORIZATION_SHA256),
        (args.v2_authorization, V2_AUTHORIZATION_SHA256),
        (args.terminal_result, TERMINAL_RESULT_SHA256),
        (args.source, SOURCE_SHA256),
        (args.builder, BUILDER_SHA256),
        (args.custom_config, CUSTOM_SHA256),
        (args.additional_variables, ADDITIONAL_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen v2 preflight input moved: {path.name}")

    authorization = json.loads(args.v2_authorization.read_bytes())
    if (
        authorization["scope"]
        != "exactly_one_repaired_credential_free_downstream_quality_no_training_eos_preflight"
        or authorization["required_repair"]
        != "register_omegaconf_resolvers_before_load_and_resolve"
        or authorization["required_launcher"] != "runllm.py --no_wait"
        or authorization["submission_attempt_limit"] != 1
        or authorization["eos_submission_authorized"] is not True
        or authorization["model_weight_access_authorized"] is not True
        or authorization["optimizer_steps_allowed"] != 0
    ):
        raise RuntimeError("v2 authority differs")
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
        raise RuntimeError("v2 authority is not fail-closed")

    custom = json.loads(args.custom_config.read_bytes())
    launcher = custom["launchers"]["dgxh100_eos"]
    if (
        custom["environments"]["retrier"]["enabled"] is not False
        or launcher["sbatch_additional_flags"]["time"] != "02:00:00"
        or launcher["srun_additional_flags"]["time"] != "02:00:00"
        or json.loads(args.additional_variables.read_bytes()) != {}
    ):
        raise RuntimeError("v2 scheduler or retry boundary differs")

    manifest = json.loads(args.manifest.read_bytes())
    JETWorkloadManifest.model_validate(deepcopy(manifest))
    spec = manifest["spec"]
    if (
        spec["name"] != "m4-downstream-quality-no-training-preflight-v2"
        or spec["nodes"] != 1
        or spec["time_limit"] != 7200
    ):
        raise RuntimeError("v2 manifest boundary differs")
    rendered = spec["script"].format(
        assets_dir="/tmp/m4-downstream-quality-preflight-v2"
    )
    subprocess.run(["bash", "-n"], input=rendered, text=True, check=True)
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", rendered, flags=re.DOTALL)
    if len(blocks) != 8:
        raise RuntimeError(f"embedded Python block count differs: {len(blocks)}")
    for index, block in enumerate(blocks):
        compile(block, f"<m4-downstream-quality-v2-{index}>", "exec")

    authority = subprocess.run(
        [sys.executable, "-c", blocks[0], str(args.v2_authorization)],
        cwd=args.repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if (
        authority.returncode != 0
        or "M4_DOWNSTREAM_QUALITY_EOS_PREFLIGHT_V2_AUTHORITY_PASS"
        not in authority.stdout
    ):
        raise RuntimeError("v2 dynamic authority did not pass")

    config_block = blocks[4]
    import_text = (
        "from nemo_rl.utils.config import "
        "load_config,register_omegaconf_resolvers"
    )
    register_text = "register_omegaconf_resolvers()"
    resolve_text = "OmegaConf.to_container(load_config(config_path),resolve=True)"
    if (
        config_block.count(import_text) != 1
        or config_block.count(register_text) != 1
        or config_block.count(resolve_text) != 1
        or not config_block.index(import_text)
        < config_block.index(register_text)
        < config_block.index(resolve_text)
    ):
        raise RuntimeError("resolver repair is absent, duplicated, or misordered")

    encoded = re.findall(r"printf %s '([A-Za-z0-9+/=]+)' \| base64 -d", rendered)
    if hashlib.sha256(base64.b64decode(encoded[-1])).hexdigest() != V2_AUTHORIZATION_SHA256:
        raise RuntimeError("embedded v2 authorization differs")
    forbidden = (
        "finish_train_step(",
        "train_microbatches(",
        "begin_train_step(",
        "examples/run_grpo",
        "python runllm.py",
        "sbatch ",
        "srun ",
    )
    if any(value in rendered for value in forbidden):
        raise RuntimeError("training or nested-submission surface is present")
    if "init_optimizer=False" not in rendered or '"optimizer_steps":0' not in rendered:
        raise RuntimeError("zero-optimizer boundary differs")

    with tempfile.TemporaryDirectory(prefix="m4-downstream-quality-v2-") as raw:
        cleanroom = Path(raw)
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
            raise RuntimeError("v2 expanded-authority negative control passed")
        rebuilt = cleanroom / "rebuilt.json"
        subprocess.run(
            [
                sys.executable,
                str(args.builder),
                "--predecessor-manifest",
                str(args.predecessor_manifest),
                "--predecessor-authorization",
                str(args.predecessor_authorization),
                "--v2-authorization",
                str(args.v2_authorization),
                "--output",
                str(rebuilt),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if rebuilt.read_bytes() != args.manifest.read_bytes():
            raise RuntimeError("deterministic v2 rebuild differs")

    result = {
        "schema": "m4-downstream-quality-eos-preflight-v2-package-validation-v1",
        "status": "PASS",
        "manifest_sha256": MANIFEST_SHA256,
        "authorization_sha256": V2_AUTHORIZATION_SHA256,
        "predecessor_terminal_result_sha256": TERMINAL_RESULT_SHA256,
        "builder_sha256": BUILDER_SHA256,
        "jet_manifest_schema_passed": True,
        "bash_syntax_passed": True,
        "embedded_python_blocks": len(blocks),
        "dynamic_authority_passed": True,
        "expanded_authority_negative_control_passed": True,
        "resolver_import_count": 1,
        "resolver_registration_count": 1,
        "resolver_registration_precedes_resolution": True,
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
