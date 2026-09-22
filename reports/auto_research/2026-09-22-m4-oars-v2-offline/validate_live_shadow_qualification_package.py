# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Clean-room validation for the credential-free OARS-v2 shadow package."""

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

from build_live_shadow_qualification_manifest import (
    ANALYZER_PATH,
    ANALYZER_SHA256,
    AUTHORIZATION_SHA256,
    CONFIG_PATH,
    CONFIG_SHA256,
    IMAGE_PATH,
    MEGATRON_SHA256,
    NAME,
    PROTOCOL_SHA256,
    SOURCE_COMMIT,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_extract(source_path: Path, output: Path) -> tuple[int, int]:
    with tarfile.open(source_path, "r:gz") as source:
        members = source.getmembers()
        names = [PurePosixPath(member.name) for member in members]
        if len(names) != len(set(names)):
            raise RuntimeError("duplicate source member")
        if any(name.is_absolute() or ".." in name.parts for name in names):
            raise RuntimeError("unsafe source path")
        symlinks = {
            name for name, member in zip(names, members, strict=True) if member.issym()
        }
        if any(any(parent in symlinks for parent in name.parents) for name in names):
            raise RuntimeError("source traverses symlink parent")
        source.extractall(output, filter="fully_trusted")
    return len(members), len(symlinks)


def validate_manifest(manifest_path: Path, source_sha256: str) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_bytes())
    spec = manifest["spec"]
    if (
        manifest["format_version"] != 1
        or manifest["labels"] != {"target": "silicon"}
        or manifest["launchers"] != {"type:slurm": {"nodes": 1, "ntasks_per_node": 1}}
        or spec["name"] != NAME
        or spec["workspace"] != "/workspace"
        or spec["nodes"] != 1
        or spec["time_limit"] != 14400
        or spec["image_source"] != {"local_path": IMAGE_PATH}
    ):
        raise RuntimeError("manifest scheduler boundary differs")
    rendered = spec["script"].format(assets_dir=f"/tmp/{NAME}")
    subprocess.run(["bash", "-n"], input=rendered, text=True, check=True)
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", rendered, flags=re.DOTALL)
    if len(blocks) != 3:
        raise RuntimeError("embedded Python block count differs")
    for index, block in enumerate(blocks):
        compile(block, f"<m4-oars-v2-live-shadow-{index}>", "exec")
    encoded = re.findall(r"printf %s '([A-Za-z0-9+/=]+)' \| base64 -d", rendered)
    payload_hashes = [
        hashlib.sha256(base64.b64decode(value)).hexdigest() for value in encoded
    ]
    if payload_hashes != [
        source_sha256,
        MEGATRON_SHA256,
        PROTOCOL_SHA256,
        AUTHORIZATION_SHA256,
    ]:
        raise RuntimeError("embedded payload order or hash differs")
    lowered = rendered.lower()
    forbidden = (
        "runllm.py",
        "sbatch ",
        "srun ",
        "ci_job_token",
        "private-token",
        "authorization:",
        "hf_token",
        "nvidia_api_key",
        "queue_deadline",
    )
    if any(token in lowered for token in forbidden):
        raise RuntimeError("manifest contains launch or credential surface")
    required = (
        "M4_OARS_V2_RUNTIME_AUTHORITY_CONFIG_AND_TQ_PASS",
        'actor_runtime_env_mode"]=="inherit_baked_single_node"',
        "selection_candidate_watermark is None",
        "readonly TRAIN_RC=${PIPESTATUS[0]}",
        'find "$ASSETS"',
        'exit "$GATE_RC"',
    )
    if not all(token in rendered for token in required):
        raise RuntimeError("runtime or terminal gate missing")
    return {
        "manifest_bytes": manifest_path.stat().st_size,
        "manifest_sha256": sha256(manifest_path),
        "name": NAME,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path, expected in (
        (args.megatron, MEGATRON_SHA256),
        (args.protocol, PROTOCOL_SHA256),
        (args.authorization, AUTHORIZATION_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen clean-room input moved: {path.name}")
    source_sha256 = sha256(args.source)
    manifest_result = validate_manifest(args.manifest, source_sha256)
    with tempfile.TemporaryDirectory(prefix="m4-oars-v2-live-shadow-") as raw:
        root = Path(raw)
        source_root = root / "source"
        source_root.mkdir()
        member_count, symlink_count = safe_extract(args.source, source_root)
        if sha256(source_root / CONFIG_PATH) != CONFIG_SHA256:
            raise RuntimeError("qualification config moved")
        if sha256(source_root / ANALYZER_PATH) != ANALYZER_SHA256:
            raise RuntimeError("qualification analyzer moved")
        protocol_in_source = source_root / (
            "reports/auto_research/2026-09-22-m4-oars-v2-offline/"
            "live_shadow_qualification_protocol.json"
        )
        if sha256(protocol_in_source) != PROTOCOL_SHA256:
            raise RuntimeError("protocol in source archive moved")
        rebuilt = root / "rebuilt.json"
        subprocess.run(
            [
                sys.executable,
                str(args.builder),
                "--source",
                str(args.source),
                "--megatron",
                str(args.megatron),
                "--protocol",
                str(args.protocol),
                "--authorization",
                str(args.authorization),
                "--output",
                str(rebuilt),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if rebuilt.read_bytes() != args.manifest.read_bytes():
            raise RuntimeError("deterministic manifest rebuild differs")
    result = {
        "schema": "m4-oars-v2-live-shadow-package-validation-v1",
        "status": "PASS_AUTHORIZED_UNSUBMITTED",
        "source_commit": SOURCE_COMMIT,
        "source_archive_sha256": source_sha256,
        "source_archive_member_count": member_count,
        "source_archive_symlink_count": symlink_count,
        "protocol_sha256": PROTOCOL_SHA256,
        "authorization_sha256": AUTHORIZATION_SHA256,
        "megatron_sha256": MEGATRON_SHA256,
        "manifest": manifest_result,
        "bash_syntax_passed": True,
        "embedded_python_syntax_passed": True,
        "credential_free_static_gate_passed": True,
        "deterministic_rebuild_passed": True,
        "runtime_config_and_analyzer_hashes_passed": True,
        "manifest_time_limit_seconds": 14400,
        "queue_deadline_override": None,
        "submission_attempt_limit": 1,
        "acting_policy": "eager_weight_fifo",
        "candidate_watermark": None,
        "scientific_outcome_acquisition": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
