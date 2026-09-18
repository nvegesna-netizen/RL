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
"""Clean-room validate the Llama no-training evaluation preflight package."""

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
from pathlib import Path, PurePosixPath

from build_evaluation_preflight_manifest import (
    AUTHORIZATION_SHA256,
    BRIDGE_SHA256,
    CONFIG_PATH,
    CONFIG_SHA256,
    IMAGE_PATH,
    MEGATRON_SHA256,
    PROTOCOL_SHA256,
    SOURCE_COMMIT,
    SOURCE_SHA256,
)

MANIFEST_SHA256 = "7f69a44d4c2e20930097e7dc086c2e3727dd1b76f204cc1f79ca714aa86f8718"


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_extract(
    archive_path: Path,
    output: Path,
    *,
    expected_members: int | None,
    prefix: str,
) -> tuple[int, int]:
    """Validate paths and extract a frozen tar archive."""
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        names = [PurePosixPath(member.name) for member in members]
        if expected_members is not None and len(members) != expected_members:
            raise RuntimeError(f"archive member count differs: {archive_path}")
        if len(names) != len(set(names)):
            raise RuntimeError(f"duplicate archive member: {archive_path}")
        if any(name.is_absolute() or ".." in name.parts for name in names):
            raise RuntimeError(f"unsafe archive member: {archive_path}")
        if prefix and not all(
            str(name) == prefix.rstrip("/") or str(name).startswith(prefix)
            for name in names
        ):
            raise RuntimeError(f"archive prefix differs: {archive_path}")
        symlinks = {
            name for name, member in zip(names, members, strict=True) if member.issym()
        }
        if any(any(parent in symlinks for parent in name.parents) for name in names):
            raise RuntimeError(f"archive traverses symlink parent: {archive_path}")
        output.mkdir(parents=True, exist_ok=True)
        archive.extractall(output)
    return len(members), len(symlinks)


def main() -> None:
    """Validate provenance, syntax, authority, and deterministic rebuild."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected_hashes = (
        (args.source, SOURCE_SHA256),
        (args.bridge, BRIDGE_SHA256),
        (args.megatron, MEGATRON_SHA256),
        (args.protocol, PROTOCOL_SHA256),
        (args.authorization, AUTHORIZATION_SHA256),
        (args.manifest, MANIFEST_SHA256),
    )
    for path, expected in expected_hashes:
        if sha256(path) != expected:
            raise RuntimeError(f"frozen package input moved: {path.name}")

    protocol = json.loads(args.protocol.read_bytes())
    authorization = json.loads(args.authorization.read_bytes())
    blocked = (
        "optimizer_initialization_authorized",
        "training_authorized",
        "qualification_authorized",
        "pilot_authorized",
        "scientific_acquisition_authorized",
        "automatic_retry",
        "automatic_replacement",
        "automatic_extension",
    )
    if (
        protocol["schema"] != "m4-oars-llama-evaluation-preflight-protocol-v1"
        or protocol["status"] != "FROZEN_BEFORE_EOS_SUBMISSION"
        or protocol["runtime"]["source_commit"] != SOURCE_COMMIT
        or protocol["evaluation"]["expected_prompt_count"] != 1319
        or protocol["model"]["optimizer_steps"] != 0
        or protocol["execution"]["queue_deadline_override"] is not None
        or authorization["protocol_sha256"] != PROTOCOL_SHA256
        or authorization["required_launcher"] != "runllm.py --no_wait"
        or authorization["submission_attempt_limit"] != 1
        or authorization["optimizer_steps_allowed"] != 0
        or any(authorization[key] for key in blocked)
    ):
        raise RuntimeError("protocol or authority boundary differs")

    manifest = json.loads(args.manifest.read_bytes())
    # The validator runs in the JET launcher environment for its authoritative
    # workload-manifest schema rather than duplicating that schema locally.
    from jetclient import JETWorkloadManifest

    JETWorkloadManifest.model_validate(json.loads(args.manifest.read_bytes()))
    spec = manifest["spec"]
    if (
        manifest["type"] != "basic"
        or manifest["format_version"] != 1
        or manifest["labels"] != {"target": "silicon"}
        or manifest["launchers"] != {"type:slurm": {"nodes": 1, "ntasks_per_node": 1}}
        or spec["name"] != "m4-oars-llama-evaluation-no-training-preflight"
        or spec["workspace"] != "/workspace"
        or spec["nodes"] != 1
        or spec["time_limit"] != 14400
        or spec["image_source"] != {"local_path": IMAGE_PATH}
    ):
        raise RuntimeError("JET manifest boundary differs")

    rendered = spec["script"].format(assets_dir="/tmp/m4-oars-eval-preflight")
    subprocess.run(["bash", "-n"], input=rendered, text=True, check=True)
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", rendered, flags=re.DOTALL)
    if len(blocks) != 8:
        raise RuntimeError(f"embedded Python block count differs: {len(blocks)}")
    for index, block in enumerate(blocks):
        compile(block, f"<m4-oars-evaluation-preflight-{index}>", "exec")

    encoded = re.findall(r"printf %s '([A-Za-z0-9+/=]+)' \| base64 -d", rendered)
    observed_payload_hashes = [
        hashlib.sha256(base64.b64decode(value)).hexdigest() for value in encoded
    ]
    if observed_payload_hashes != [
        SOURCE_SHA256,
        BRIDGE_SHA256,
        MEGATRON_SHA256,
        PROTOCOL_SHA256,
        AUTHORIZATION_SHA256,
    ]:
        raise RuntimeError("embedded payload hashes differ")

    forbidden = (
        "examples/run_grpo",
        "finish_train_step(",
        "begin_train_step(",
        "train_microbatches(",
        "python runllm.py",
        "sbatch ",
        "srun ",
        "ci_job_token",
        "private-token",
        "hf_token",
        "wandb_api_key",
        "nvidia_api_key",
        "ngc_api_key",
    )
    lowered = rendered.lower()
    if any(token in lowered for token in forbidden):
        raise RuntimeError("training, nested launch, or credential surface present")
    required = (
        "init_optimizer=False",
        "train_steps=0,trainer_version=0",
        'raw["grpo"]["max_num_steps"]=0',
        '"dataset_name":"gsm8k"',
        '"split":"test"',
        "len(dataset)==1319",
        '"temperature":0.0',
        '"scientific_endpoint":False',
        '"automatic_retry":False',
        '"automatic_replacement":False',
        '"automatic_extension":False',
        'env VIRTUAL_ENV="$MCORE_VENV" UV_PROJECT_ENVIRONMENT="$MCORE_VENV" "$MCORE_PYTHON" examples/converters/convert_megatron_to_hf.py --config',
    )
    if not all(token in rendered for token in required):
        raise RuntimeError("no-training runtime or result contract differs")
    if (
        "sbatch_additional_flags" in lowered
        or "--deadline" in lowered
        or "deadline=" in lowered
    ):
        raise RuntimeError("queue deadline surface present")

    with tempfile.TemporaryDirectory(prefix="m4-oars-eval-preflight-") as raw:
        root = Path(raw)
        source_count, source_links = safe_extract(
            args.source,
            root / "source",
            expected_members=None,
            prefix="",
        )
        bridge_count, bridge_links = safe_extract(
            args.bridge,
            root / "deps",
            expected_members=1060,
            prefix="Megatron-Bridge/",
        )
        megatron_count, megatron_links = safe_extract(
            args.megatron,
            root / "deps/Megatron-Bridge/3rdparty",
            expected_members=726,
            prefix="Megatron-LM/",
        )
        source_root = root / "source"
        if sha256(source_root / CONFIG_PATH) != CONFIG_SHA256:
            raise RuntimeError("anchor Llama config moved")
        worker = (
            source_root / "nemo_rl/models/policy/workers/megatron_policy_worker.py"
        ).read_text()
        if (
            "forward_pre_hook_was_enabled = (" not in worker
            or worker.count("if forward_pre_hook_was_enabled:") != 2
        ):
            raise RuntimeError("inference-only checkpoint-hook repair moved")
        bridge_src = root / "deps/Megatron-Bridge/src"
        megatron_src = root / "deps/Megatron-Bridge/3rdparty/Megatron-LM"
        namespace = importlib.machinery.PathFinder.find_spec(
            "megatron", [str(bridge_src), str(megatron_src)]
        )
        if namespace is None or namespace.submodule_search_locations is None:
            raise RuntimeError("assembled Megatron namespace is absent")
        search = list(namespace.submodule_search_locations)
        if (
            importlib.machinery.PathFinder.find_spec("megatron.bridge", search) is None
            or importlib.machinery.PathFinder.find_spec("megatron.core", search) is None
        ):
            raise RuntimeError("assembled Bridge/Megatron topology is incomplete")

        authorization_path = root / "authorization.json"
        authorization_path.write_bytes(args.authorization.read_bytes())
        protocol_path = root / "protocol.json"
        protocol_path.write_bytes(args.protocol.read_bytes())
        positive = subprocess.run(
            [sys.executable, "-", str(protocol_path), str(authorization_path)],
            input=blocks[0],
            text=True,
            capture_output=True,
            check=False,
        )
        if (
            positive.returncode != 0
            or "M4_OARS_LLAMA_EVALUATION_PREFLIGHT_AUTHORITY_PASS"
            not in positive.stdout
        ):
            raise RuntimeError("authorized authority block did not pass")
        tampered = dict(authorization)
        tampered["training_authorized"] = True
        tampered_path = root / "tampered-authorization.json"
        tampered_path.write_text(json.dumps(tampered))
        negative = subprocess.run(
            [sys.executable, "-", str(protocol_path), str(tampered_path)],
            input=blocks[0],
            text=True,
            capture_output=True,
            check=False,
        )
        if negative.returncode == 0:
            raise RuntimeError("expanded-authority negative control passed")

        rebuilt = root / "rebuilt.json"
        subprocess.run(
            [
                sys.executable,
                str(args.builder),
                "--source",
                str(args.source),
                "--bridge",
                str(args.bridge),
                "--megatron",
                str(args.megatron),
                "--protocol",
                str(args.protocol),
                "--authorization",
                str(args.authorization),
                "--output",
                str(rebuilt),
            ],
            cwd=args.repo,
            check=True,
            capture_output=True,
            text=True,
        )
        if rebuilt.read_bytes() != args.manifest.read_bytes():
            raise RuntimeError("deterministic manifest rebuild differs")

    result = {
        "schema": "m4-oars-llama-evaluation-preflight-package-validation-v1",
        "status": "PASS_UNSUBMITTED",
        "manifest_sha256": MANIFEST_SHA256,
        "manifest_bytes": args.manifest.stat().st_size,
        "builder_sha256": sha256(args.builder),
        "protocol_sha256": PROTOCOL_SHA256,
        "authorization_sha256": AUTHORIZATION_SHA256,
        "source_commit": SOURCE_COMMIT,
        "source_archive_sha256": SOURCE_SHA256,
        "source_archive_members": source_count,
        "source_archive_symlinks": source_links,
        "bridge_archive_sha256": BRIDGE_SHA256,
        "bridge_archive_members": bridge_count,
        "bridge_archive_symlinks": bridge_links,
        "megatron_archive_sha256": MEGATRON_SHA256,
        "megatron_archive_members": megatron_count,
        "megatron_archive_symlinks": megatron_links,
        "jet_manifest_schema_passed": True,
        "bash_and_python_syntax_passed": True,
        "embedded_python_blocks": len(blocks),
        "embedded_payload_hashes_passed": True,
        "dynamic_authority_passed": True,
        "expanded_authority_negative_control_passed": True,
        "credential_surface_absent": True,
        "inference_only_checkpoint_repair_present": True,
        "assembled_megatron_namespace_passed": True,
        "mcore_converter_gate_present": True,
        "full_gsm8k_contract_present": True,
        "deterministic_rebuild_passed": True,
        "workload_time_limit_seconds": 14400,
        "queue_deadline_override": None,
        "submission_attempt_limit": 1,
        "submitted": False,
        "optimizer_initialized": False,
        "optimizer_steps": 0,
        "trainer_steps": 0,
        "training_started": False,
        "qualification_started": False,
        "scientific_acquisition_started": False,
        "automatic_retry": False,
        "automatic_replacement": False,
        "automatic_extension": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
