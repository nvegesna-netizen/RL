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
"""Clean-room validate the fail-closed downstream-quality preflight candidate."""

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
from pathlib import Path, PurePosixPath

from build_local_preflight_candidate import (
    AUTHORIZATION_SHA256,
    IMAGE_COMMIT,
    IMAGE_PATH,
    MEGATRON_SHA256,
    PLAN_SHA256,
    PROTOCOL_SHA256,
    SOURCE_COMMIT,
    SOURCE_SHA256,
)

BUILDER_SHA256 = "88240f2df802d828dac8ee95ac6bab82bb8dd98bf1de0f2efc883ef8e7fa5e30"
CANDIDATE_SHA256 = "e7de1aa88e2f4a9c7de7e1b7f9f6a5b34382ea18544d9920dcb65d3d0b5b4853"


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_extract(archive: Path, output: Path) -> tuple[int, int]:
    """Extract a tar archive after rejecting unsafe paths and symlink parents."""
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        names = [PurePosixPath(member.name) for member in members]
        if len(names) != len(set(names)):
            raise RuntimeError("duplicate source archive member")
        if any(name.is_absolute() or ".." in name.parts for name in names):
            raise RuntimeError("unsafe source archive path")
        symlinks = {
            name for name, member in zip(names, members, strict=True) if member.issym()
        }
        if any(any(parent in symlinks for parent in name.parents) for name in names):
            raise RuntimeError("source archive traverses a symlink parent")
        source.extractall(output)
    return len(members), len(symlinks)


def main() -> None:
    """Validate hashes, syntax, authority ordering, and deterministic rebuild."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path, expected in (
        (args.source, SOURCE_SHA256),
        (args.megatron, MEGATRON_SHA256),
        (args.protocol, PROTOCOL_SHA256),
        (args.plan, PLAN_SHA256),
        (args.authorization, AUTHORIZATION_SHA256),
        (args.builder, BUILDER_SHA256),
        (args.candidate, CANDIDATE_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen clean-room input moved: {path.name}")

    manifest = json.loads(args.candidate.read_bytes())
    spec = manifest["spec"]
    if (
        manifest["format_version"] != 1
        or manifest["labels"] != {"target": "silicon"}
        or manifest["launchers"] != {"type:slurm": {"nodes": 1, "ntasks_per_node": 1}}
        or spec["name"] != "m4-downstream-quality-no-training-preflight-local"
        or spec["workspace"] != "/workspace"
        or spec["nodes"] != 1
        or spec["time_limit"] != 7200
        or spec["image_source"] != {"local_path": IMAGE_PATH}
    ):
        raise RuntimeError("JET manifest boundary differs")

    script = spec["script"]
    rendered = script.format(assets_dir="/tmp/m4-downstream-quality-preflight")
    subprocess.run(["bash", "-n"], input=rendered, text=True, check=True)
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", rendered, flags=re.DOTALL)
    if len(blocks) != 8:
        raise RuntimeError(f"embedded Python block count differs: {len(blocks)}")
    for index, block in enumerate(blocks):
        compile(block, f"<m4-downstream-quality-preflight-{index}>", "exec")

    encoded = re.findall(r"printf %s '([A-Za-z0-9+/=]+)' \| base64 -d", rendered)
    payload_hashes = tuple(
        hashlib.sha256(base64.b64decode(value)).hexdigest() for value in encoded
    )
    if payload_hashes != (
        SOURCE_SHA256,
        MEGATRON_SHA256,
        PROTOCOL_SHA256,
        PLAN_SHA256,
        AUTHORIZATION_SHA256,
    ):
        raise RuntimeError("embedded payload order or hash differs")

    guard = rendered.index("M4_DOWNSTREAM_QUALITY_LOCAL_CANDIDATE_NO_EOS_AUTHORITY")
    extraction = rendered.index("import sys,tarfile")
    model_import = rendered.index(
        "from nemo_rl.models.policy.tq_policy import TQPolicy"
    )
    if not guard < extraction < model_import:
        raise RuntimeError("no-authority guard does not precede execution")
    forbidden_training_calls = (
        "finish_train_step(",
        "train_microbatches(",
        "begin_train_step(",
        "examples/run_grpo",
    )
    if any(value in rendered for value in forbidden_training_calls):
        raise RuntimeError("preflight contains a training entrypoint or method")
    if "init_optimizer=False" not in rendered or '"optimizer_steps":0' not in rendered:
        raise RuntimeError("zero-optimizer boundary is absent")
    if (
        "convert_megatron_to_hf.py" not in rendered
        or "configure_generation_config" not in rendered
        or '"candidate_evaluation_prompts"]==1024' not in rendered
        or '"seed":20260911' not in rendered
        or '"temperature":0.0' not in rendered
    ):
        raise RuntimeError("conversion or fixed evaluation contract differs")

    local_guard = subprocess.run(
        [sys.executable, "-c", blocks[0], str(args.authorization)],
        cwd=args.repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if (
        local_guard.returncode == 0
        or "M4_DOWNSTREAM_QUALITY_LOCAL_PREFLIGHT_AUTHORITY_PASS"
        not in local_guard.stdout
        or "M4_DOWNSTREAM_QUALITY_LOCAL_CANDIDATE_NO_EOS_AUTHORITY"
        not in local_guard.stderr
    ):
        raise RuntimeError("local authority did not fail closed dynamically")

    with tempfile.TemporaryDirectory(prefix="m4-downstream-quality-cleanroom-") as raw:
        cleanroom = Path(raw)
        extracted = cleanroom / "source"
        extracted.mkdir()
        member_count, symlink_count = safe_extract(args.source, extracted)
        required = (
            "nemo_rl/utils/terminal_policy_export.py",
            "nemo_rl/algorithms/single_controller.py",
            "examples/converters/convert_megatron_to_hf.py",
            "examples/run_eval.py",
            "reports/auto_research/2026-09-11-m4-downstream-quality/protocol_draft.json",
        )
        if any(not (extracted / path).is_file() for path in required):
            raise RuntimeError("source archive lacks a required runtime file")

        tampered = json.loads(args.authorization.read_bytes())
        tampered["eos_submission_authorized"] = True
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
            raise RuntimeError("mixed local/EOS authority negative control passed")

        rebuilt = cleanroom / "rebuilt.json"
        subprocess.run(
            [
                sys.executable,
                str(args.builder),
                "--template",
                str(args.template),
                "--source",
                str(args.source),
                "--megatron",
                str(args.megatron),
                "--protocol",
                str(args.protocol),
                "--plan",
                str(args.plan),
                "--authorization",
                str(args.authorization),
                "--output",
                str(rebuilt),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if rebuilt.read_bytes() != args.candidate.read_bytes():
            raise RuntimeError("deterministic candidate rebuild differs")

    result = {
        "schema": "m4-downstream-quality-local-preflight-package-validation-v1",
        "status": "PASS",
        "source_commit": SOURCE_COMMIT,
        "source_archive_sha256": SOURCE_SHA256,
        "megatron_dependency_sha256": MEGATRON_SHA256,
        "protocol_draft_sha256": PROTOCOL_SHA256,
        "preflight_plan_sha256": PLAN_SHA256,
        "authorization_sha256": AUTHORIZATION_SHA256,
        "builder_sha256": BUILDER_SHA256,
        "candidate_sha256": CANDIDATE_SHA256,
        "image_commit": IMAGE_COMMIT,
        "manifest_structure_passed": True,
        "bash_syntax_passed": True,
        "embedded_python_blocks": len(blocks),
        "embedded_payload_hashes_passed": True,
        "source_archive_member_count": member_count,
        "source_archive_symlink_count": symlink_count,
        "safe_source_extraction_passed": True,
        "dynamic_no_eos_guard_passed": True,
        "mixed_authority_negative_control_passed": True,
        "zero_training_call_static_gate_passed": True,
        "conversion_and_fixed_evaluation_contract_present": True,
        "deterministic_rebuild_passed": True,
        "eos_launch_attempted": False,
        "model_weight_accessed": False,
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
