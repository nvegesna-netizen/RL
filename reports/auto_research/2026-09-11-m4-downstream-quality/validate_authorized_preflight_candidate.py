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
"""Clean-room validate the sole authorized downstream-quality preflight."""

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

MANIFEST_SHA256 = "13dfd195a2ea2a5f6b9426b854531a294a51f31214d6e195df5d6cec5b5e5103"
LOCAL_CANDIDATE_SHA256 = (
    "e7de1aa88e2f4a9c7de7e1b7f9f6a5b34382ea18544d9920dcb65d3d0b5b4853"
)
LOCAL_AUTHORIZATION_SHA256 = (
    "d34ec018f5d75711bcab3efcf4225f0d4a57b75fa338cfcd050b79b821d3c7c2"
)
EOS_AUTHORIZATION_SHA256 = (
    "a2cd3cb97818d530b4dfb72312eb4ee266322c534c7a51e89f49006e04a26ceb"
)
BUILDER_SHA256 = "4612185db9f1e5c507fa9fdfd02e25747c98ee52cd3567aff056a8bd11188c64"
CUSTOM_CONFIG_SHA256 = (
    "c13a73b87669569429c36fd0c60f09d2a6ecabfb4086063395924cc26566dc8d"
)
ADDITIONAL_VARIABLES_SHA256 = (
    "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356"
)
RUNLLM_SHA256 = "30532cd638d8394fa0e651daf4a43ec9424cbaa0949cd59c323191177318e616"
SOURCE_SHA256 = "80741c66b32a7951e4ebd620e66a8043858159e7a871c7fb87d038db068aa714"
MEGATRON_SHA256 = "98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
PROTOCOL_SHA256 = "5ec319043f6e3be90e8e294946eee3596b4c8ab445630de93ae103bbcabb7162"
PLAN_SHA256 = "b160c6610226bf2b2250cbc20500667776bfb1e62df0918994c4221a690918fa"


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Validate frozen inputs, authority, candidate syntax, and reproducibility."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--local-candidate", type=Path, required=True)
    parser.add_argument("--local-authorization", type=Path, required=True)
    parser.add_argument("--eos-authorization", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--custom-config", type=Path, required=True)
    parser.add_argument("--additional-variables", type=Path, required=True)
    parser.add_argument("--runllm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path, expected in (
        (args.manifest, MANIFEST_SHA256),
        (args.local_candidate, LOCAL_CANDIDATE_SHA256),
        (args.local_authorization, LOCAL_AUTHORIZATION_SHA256),
        (args.eos_authorization, EOS_AUTHORIZATION_SHA256),
        (args.builder, BUILDER_SHA256),
        (args.custom_config, CUSTOM_CONFIG_SHA256),
        (args.additional_variables, ADDITIONAL_VARIABLES_SHA256),
        (args.runllm, RUNLLM_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen EOS preflight input moved: {path.name}")

    authorization = json.loads(args.eos_authorization.read_bytes())
    if (
        authorization["scope"]
        != "exactly_one_credential_free_downstream_quality_no_training_eos_preflight"
        or authorization["user_authorization"] != "I authorize"
        or authorization["required_launcher"] != "runllm.py --no_wait"
        or authorization["submission_attempt_limit"] != 1
        or authorization["eos_submission_authorized"] is not True
        or authorization["model_weight_access_authorized"] is not True
        or authorization["optimizer_steps_allowed"] != 0
    ):
        raise RuntimeError("authorized scope differs")
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
        raise RuntimeError("EOS preflight authority is not fail-closed")

    custom = json.loads(args.custom_config.read_bytes())
    launcher = custom["launchers"]["dgxh100_eos"]
    if (
        custom["environments"]["retrier"]["enabled"] is not False
        or launcher["sbatch_additional_flags"]["time"] != "02:00:00"
        or launcher["srun_additional_flags"]["time"] != "02:00:00"
        or json.loads(args.additional_variables.read_bytes()) != {}
    ):
        raise RuntimeError("scheduler or retry boundary differs")

    manifest = json.loads(args.manifest.read_bytes())
    JETWorkloadManifest.model_validate(deepcopy(manifest))
    spec = manifest["spec"]
    if (
        spec["name"] != "m4-downstream-quality-no-training-preflight"
        or spec["nodes"] != 1
        or spec["time_limit"] != 7200
        or manifest["launchers"] != {"type:slurm": {"nodes": 1, "ntasks_per_node": 1}}
    ):
        raise RuntimeError("authorized manifest boundary differs")

    script = spec["script"]
    rendered = script.format(assets_dir="/tmp/m4-downstream-quality-preflight")
    subprocess.run(["bash", "-n"], input=rendered, text=True, check=True)
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", rendered, flags=re.DOTALL)
    if len(blocks) != 8:
        raise RuntimeError(f"embedded Python block count differs: {len(blocks)}")
    for index, block in enumerate(blocks):
        compile(block, f"<m4-downstream-quality-eos-preflight-{index}>", "exec")

    encoded = re.findall(r"printf %s '([A-Za-z0-9+/=]+)' \| base64 -d", rendered)
    payload_hashes = tuple(
        hashlib.sha256(base64.b64decode(value)).hexdigest() for value in encoded
    )
    if payload_hashes != (
        SOURCE_SHA256,
        MEGATRON_SHA256,
        PROTOCOL_SHA256,
        PLAN_SHA256,
        EOS_AUTHORIZATION_SHA256,
    ):
        raise RuntimeError("authorized embedded payloads differ")

    authority = subprocess.run(
        [sys.executable, "-c", blocks[0], str(args.eos_authorization)],
        cwd=args.repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if (
        authority.returncode != 0
        or "M4_DOWNSTREAM_QUALITY_EOS_PREFLIGHT_AUTHORITY_PASS"
        not in authority.stdout
    ):
        raise RuntimeError("authorized runtime authority did not pass")

    with tempfile.TemporaryDirectory(prefix="m4-downstream-quality-eos-") as raw:
        cleanroom = Path(raw)
        tampered = dict(authorization)
        tampered["training_authorized"] = True
        tampered_path = cleanroom / "tampered-authorization.json"
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
            raise RuntimeError("expanded-authority negative control passed")

        rebuilt = cleanroom / "rebuilt.json"
        subprocess.run(
            [
                sys.executable,
                str(args.builder),
                "--candidate",
                str(args.local_candidate),
                "--local-authorization",
                str(args.local_authorization),
                "--eos-authorization",
                str(args.eos_authorization),
                "--output",
                str(rebuilt),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if rebuilt.read_bytes() != args.manifest.read_bytes():
            raise RuntimeError("deterministic authorized rebuild differs")

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
    if (
        "init_optimizer=False" not in rendered
        or '"optimizer_steps":0' not in rendered
        or "convert_megatron_to_hf.py" not in rendered
        or '"candidate_evaluation_prompts"]==1024' not in rendered
    ):
        raise RuntimeError("zero-training conversion/evaluation contract differs")

    result = {
        "schema": "m4-downstream-quality-eos-preflight-package-validation-v1",
        "status": "PASS",
        "manifest_sha256": MANIFEST_SHA256,
        "authorization_sha256": EOS_AUTHORIZATION_SHA256,
        "builder_sha256": BUILDER_SHA256,
        "custom_config_sha256": CUSTOM_CONFIG_SHA256,
        "additional_variables_sha256": ADDITIONAL_VARIABLES_SHA256,
        "runllm_sha256": RUNLLM_SHA256,
        "jet_manifest_schema_passed": True,
        "bash_syntax_passed": True,
        "embedded_python_blocks": len(blocks),
        "embedded_payload_hashes_passed": True,
        "dynamic_authority_passed": True,
        "expanded_authority_negative_control_passed": True,
        "zero_training_surface_passed": True,
        "conversion_and_fixed_evaluation_contract_present": True,
        "deterministic_rebuild_passed": True,
        "scheduler_time_limit": "02:00:00",
        "jet_retrier_enabled": False,
        "submission_attempt_limit": 1,
        "submitted": False,
        "optimizer_initialized": False,
        "optimizer_steps": 0,
        "training_started": False,
        "qualification_started": False,
        "pilot_started": False,
        "scientific_acquisition_started": False,
        "automatic_retry": False,
        "automatic_extension": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
