#!/usr/bin/env python3
"""Clean-room validation for the conditional-M4 capsule qualification package."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from build_conditional_capsule_qualification_manifest import (
    ANALYZER_PATH,
    ANALYZER_SHA256,
    CAPSULE_MODULE_SHA256,
    CONFIG_PATH,
    CONFIG_SHA256,
    CONTROLLER_SHA256,
    IMAGE_PATH,
    LOCK_SHA256,
    MEGATRON_SHA256,
    NAME,
    PROTOCOL_SHA256,
    SOURCE_COMMIT,
    SOURCE_SHA256,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_extract(archive_path: Path, output: Path) -> tuple[int, int]:
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        names = [PurePosixPath(member.name) for member in members]
        if len(names) != len(set(names)):
            raise RuntimeError("duplicate archive member")
        if any(name.is_absolute() or ".." in name.parts for name in names):
            raise RuntimeError("unsafe archive path")
        symlinks = {
            name for name, member in zip(names, members, strict=True) if member.issym()
        }
        if any(any(parent in symlinks for parent in name.parents) for name in names):
            raise RuntimeError("archive path traverses a symlink")
        archive.extractall(output)
    return len(members), len(symlinks)


def validate_no_update_ast(controller: Path) -> None:
    tree = ast.parse(controller.read_bytes(), filename=str(controller))
    pump = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "_conditional_m4_capsule_pump"
    )
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(pump)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    forbidden = {"begin_train_step", "finish_train_step", "train_microbatches"}
    if calls & forbidden:
        raise RuntimeError(f"capsule pump contains training calls: {calls & forbidden}")
    required = {
        "_take_version_zero_groups",
        "_prepare_capsule_group",
        "select_conditional_m4_contrast",
        "_clear_capsule_metas",
    }
    if not required.issubset(calls):
        raise RuntimeError(f"capsule pump lacks required calls: {required - calls}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected_inputs = (
        (args.source, SOURCE_SHA256),
        (args.megatron, MEGATRON_SHA256),
        (args.protocol, PROTOCOL_SHA256),
        (args.execution_lock, LOCK_SHA256),
    )
    for path, expected in expected_inputs:
        if sha256(path) != expected:
            raise RuntimeError(f"frozen validation input moved: {path}")
    manifest = json.loads(args.manifest.read_bytes())
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
        raise RuntimeError("manifest execution boundary differs")
    if json.dumps(manifest).lower().count("queue_deadline"):
        raise RuntimeError("manifest introduces a client-side queue deadline")
    if spec["script"].count("{assets_dir}") != 1:
        raise RuntimeError("JET artifact substitution count differs")
    rendered = spec["script"].format(assets_dir=f"/tmp/{NAME}")
    subprocess.run(["bash", "-n"], input=rendered, text=True, check=True)
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", rendered, flags=re.DOTALL)
    if len(blocks) != 3:
        raise RuntimeError(f"embedded Python block count differs: {len(blocks)}")
    for index, block in enumerate(blocks):
        compile(block, f"<conditional-capsule-{index}>", "exec")
    payloads = re.findall(r"printf %s '([A-Za-z0-9+/=]+)' \| base64 -d", rendered)
    payload_hashes = [
        hashlib.sha256(base64.b64decode(payload)).hexdigest() for payload in payloads
    ]
    if payload_hashes != [
        SOURCE_SHA256,
        MEGATRON_SHA256,
        PROTOCOL_SHA256,
        LOCK_SHA256,
    ]:
        raise RuntimeError("embedded payload order or content differs")
    lowered = rendered.lower()
    credentials = (
        "ci_job_token",
        "private-token",
        "authorization:",
        "hf_token",
        "nvidia_api_key",
    )
    if any(token in lowered for token in credentials):
        raise RuntimeError("manifest contains credential material")
    required_surface = (
        "M4_CONDITIONAL_CAPSULE_RUNTIME_AUTHORITY_PASS",
        "tests/unit/single_controller/test_conditional_m4_capsule.py",
        "-k conditional_capsule",
        "readonly RUN_RC=${PIPESTATUS[0]}",
        f'"$PYTHON" "{ANALYZER_PATH}"',
        'find "$ASSETS" -type f',
        'exit "$GATE_RC"',
    )
    if not all(value in rendered for value in required_surface):
        raise RuntimeError("qualification or terminal artifact gate is incomplete")
    if rendered.count("examples/run_grpo_single_controller.py") != 1:
        raise RuntimeError("qualification run count differs from one")

    with tempfile.TemporaryDirectory(prefix="m4-conditional-capsule-") as raw:
        root = Path(raw)
        source_root = root / "source"
        source_root.mkdir()
        members, symlinks = safe_extract(args.source, source_root)
        expected_files = {
            CONFIG_PATH: CONFIG_SHA256,
            ANALYZER_PATH: ANALYZER_SHA256,
            "nemo_rl/algorithms/async_utils/conditional_m4_capsule.py": CAPSULE_MODULE_SHA256,
            "nemo_rl/algorithms/single_controller.py": CONTROLLER_SHA256,
        }
        for name, expected in expected_files.items():
            if sha256(source_root / name) != expected:
                raise RuntimeError(f"clean-room source differs: {name}")
        validate_no_update_ast(source_root / "nemo_rl/algorithms/single_controller.py")
        rebuilt = root / "manifest.json"
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
                "--execution-lock",
                str(args.execution_lock),
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
        "schema": "conditional-m4-capsule-package-validation-v1",
        "status": "PASS_AUTHORIZED_UNSUBMITTED",
        "source_commit": SOURCE_COMMIT,
        "source_archive_sha256": SOURCE_SHA256,
        "source_archive_members": members,
        "source_archive_symlinks": symlinks,
        "protocol_sha256": PROTOCOL_SHA256,
        "execution_lock_sha256": LOCK_SHA256,
        "manifest_sha256": sha256(args.manifest),
        "manifest_time_limit_seconds": 14400,
        "queue_deadline_override": None,
        "submission_attempt_limit": 1,
        "credential_free_static_gate_passed": True,
        "bash_syntax_passed": True,
        "embedded_python_syntax_passed": True,
        "deterministic_rebuild_passed": True,
        "no_update_ast_gate_passed": True,
        "full_runtime_tests_required_before_collection": True,
        "scientific_outcome_acquisition": False,
        "learning_outcomes_opened": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
