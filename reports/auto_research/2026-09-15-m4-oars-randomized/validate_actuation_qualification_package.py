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

"""Clean-room validation for the two-arm OARS qualification package."""

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

from build_actuation_qualification_manifest import (
    ANALYZER_PATH,
    ANALYZER_SHA256,
    ARM_CONFIG,
    AUTHORIZATION_SHA256,
    IMAGE_PATH,
    MEGATRON_SHA256,
    PROTOCOL_SHA256,
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
            name
            for name, member in zip(names, members, strict=True)
            if member.issym()
        }
        if any(any(parent in symlinks for parent in name.parents) for name in names):
            raise RuntimeError("source traverses symlink parent")
        source.extractall(output)
    return len(members), len(symlinks)


def validate_manifest(
    *,
    arm: str,
    manifest_path: Path,
    source_sha256: str,
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_bytes())
    spec = manifest["spec"]
    mode, _, _ = ARM_CONFIG[arm]
    if (
        manifest["format_version"] != 1
        or manifest["labels"] != {"target": "silicon"}
        or manifest["launchers"]
        != {"type:slurm": {"nodes": 1, "ntasks_per_node": 1}}
        or spec["name"] != f"m4-oars-qualification-{arm}"
        or spec["workspace"] != "/workspace"
        or spec["nodes"] != 1
        or spec["time_limit"] != 14400
        or spec["image_source"] != {"local_path": IMAGE_PATH}
    ):
        raise RuntimeError(f"{arm} manifest boundary differs")
    rendered = spec["script"].format(assets_dir=f"/tmp/m4-oars-{arm}")
    subprocess.run(["bash", "-n"], input=rendered, text=True, check=True)
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", rendered, flags=re.DOTALL)
    if len(blocks) != 3:
        raise RuntimeError(f"{arm} embedded Python block count differs")
    for index, block in enumerate(blocks):
        compile(block, f"<m4-oars-{arm}-{index}>", "exec")
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
        raise RuntimeError(f"{arm} embedded payload order or hash differs")
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
        raise RuntimeError(f"{arm} contains a forbidden launch or credential surface")
    if rendered.count("examples/run_grpo_single_controller.py") != 1:
        raise RuntimeError(f"{arm} training entrypoint count differs")
    required = (
        f'--mode "{mode}"',
        "selection_candidate_watermark==8",
        "readonly TRAIN_RC=${PIPESTATUS[0]}",
        'find "$ASSETS"',
        'exit "$GATE_RC"',
    )
    if not all(value in rendered for value in required):
        raise RuntimeError(f"{arm} runtime or terminal gate missing")
    if rendered.index('find "$ASSETS"') > rendered.index('exit "$GATE_RC"'):
        raise RuntimeError(f"{arm} hashes would not precede exit")
    return {
        "arm": arm,
        "manifest_bytes": manifest_path.stat().st_size,
        "manifest_sha256": sha256(manifest_path),
        "mode": mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--fifo", type=Path, required=True)
    parser.add_argument("--act", type=Path, required=True)
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
    validations = [
        validate_manifest(arm="fifo", manifest_path=args.fifo, source_sha256=source_sha256),
        validate_manifest(arm="act", manifest_path=args.act, source_sha256=source_sha256),
    ]
    with tempfile.TemporaryDirectory(prefix="m4-oars-qualification-") as raw:
        root = Path(raw)
        source_root = root / "source"
        source_root.mkdir()
        member_count, symlink_count = safe_extract(args.source, source_root)
        for _, config_path, config_sha256 in ARM_CONFIG.values():
            if sha256(source_root / config_path) != config_sha256:
                raise RuntimeError(f"source config moved: {config_path}")
        if sha256(source_root / ANALYZER_PATH) != ANALYZER_SHA256:
            raise RuntimeError("source analyzer moved")
        for arm, candidate in (("fifo", args.fifo), ("act", args.act)):
            rebuilt = root / f"rebuilt-{arm}.json"
            subprocess.run(
                [
                    sys.executable,
                    str(args.builder),
                    "--arm",
                    arm,
                    "--source",
                    str(args.source),
                    "--source-commit",
                    args.source_commit,
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
            if rebuilt.read_bytes() != candidate.read_bytes():
                raise RuntimeError(f"{arm} deterministic rebuild differs")
    result = {
        "authorization_sha256": AUTHORIZATION_SHA256,
        "bash_syntax_passed": True,
        "credential_free_static_gate_passed": True,
        "deterministic_rebuild_passed": True,
        "embedded_python_syntax_passed": True,
        "manifest_time_limit_seconds": 14400,
        "manifests": validations,
        "megatron_sha256": MEGATRON_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "queue_deadline_override": None,
        "schema": "m4-oars-actuation-qualification-package-validation-v1",
        "scientific_outcome_acquisition": False,
        "source_archive_member_count": member_count,
        "source_archive_sha256": source_sha256,
        "source_archive_symlink_count": symlink_count,
        "source_commit": args.source_commit,
        "status": "PASS_AUTHORIZED_UNSUBMITTED",
        "submission_attempt_limit": 2,
        "terminal_artifact_hash_on_failure_passed": True,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
