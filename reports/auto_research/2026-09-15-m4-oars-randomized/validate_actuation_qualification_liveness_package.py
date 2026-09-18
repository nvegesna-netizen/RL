# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Clean-room validation for the enacted-OARS liveness qualification."""

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
from types import SimpleNamespace

from build_actuation_qualification_liveness_manifest import (
    AMENDMENT_SHA256,
    ANALYZER_PATH,
    ANALYZER_SHA256,
    AUTHORIZATION_SHA256,
    CONFIG_PATH,
    CONFIG_SHA256,
    IMAGE_PATH,
    MEGATRON_SHA256,
    NAME,
    PROTOCOL_SHA256,
    TRANSFERQUEUE_COMMIT,
    VALIDATION_PATH,
    VALIDATION_SHA256,
)

EXPECTED_CHANGED_PATHS = {
    "nemo_rl/algorithms/async_utils/opportunity_at_risk.py",
    "nemo_rl/algorithms/async_utils/rollout_lifecycle.py",
    "nemo_rl/algorithms/async_utils/staleness_sampler.py",
    VALIDATION_PATH,
    "reports/auto_research/2026-09-15-m4-oars-randomized/test_actuation_qualification.py",
    "tests/unit/single_controller/test_opportunity_at_risk.py",
    "tests/unit/tools/test_opportunity_ledger_join.py",
    ANALYZER_PATH,
    "tools/opportunity_ledger_join.py",
}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_extract(source_path: Path, output: Path) -> tuple[int, int]:
    """Extract a tar archive after rejecting traversal and link-parent risks."""
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


def changed_paths(new_root: Path, old_root: Path) -> set[str]:
    """Return exact regular-file content differences between two source trees."""
    new_files = {
        path.relative_to(new_root).as_posix(): path
        for path in new_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    old_files = {
        path.relative_to(old_root).as_posix(): path
        for path in old_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    changed = set(new_files) ^ set(old_files)
    for name in set(new_files).intersection(old_files):
        if new_files[name].read_bytes() != old_files[name].read_bytes():
            changed.add(name)
    return changed


def validate_manifest(manifest_path: Path, source_sha256: str) -> dict[str, object]:
    """Validate scheduler boundary and embedded credential-free payloads."""
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
        raise RuntimeError("liveness manifest boundary differs")
    rendered = spec["script"].format(assets_dir=f"/tmp/{NAME}")
    subprocess.run(["bash", "-n"], input=rendered, text=True, check=True)
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", rendered, flags=re.DOTALL)
    if len(blocks) != 3:
        raise RuntimeError("embedded Python block count differs")
    for index, block in enumerate(blocks):
        compile(block, f"<m4-oars-liveness-{index}>", "exec")
    encoded = re.findall(r"printf %s '([A-Za-z0-9+/=]+)' \| base64 -d", rendered)
    payload_hashes = [
        hashlib.sha256(base64.b64decode(value)).hexdigest() for value in encoded
    ]
    if payload_hashes != [
        source_sha256,
        MEGATRON_SHA256,
        PROTOCOL_SHA256,
        AUTHORIZATION_SHA256,
        AMENDMENT_SHA256,
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
        "M4_OARS_LIVENESS_RUNTIME_AUTHORITY_CONFIG_AND_TQ_PASS",
        'actor_runtime_env_mode"]=="inherit_baked_single_node"',
        "OpportunityAtRiskShadowRecorder.SCHEMA_VERSION==3",
        TRANSFERQUEUE_COMMIT,
        "readonly TRAIN_RC=${PIPESTATUS[0]}",
        'find "$ASSETS"',
        'exit "$GATE_RC"',
    )
    if not all(value in rendered for value in required):
        raise RuntimeError("runtime or terminal gate missing")
    if rendered.index('find "$ASSETS"') > rendered.index('exit "$GATE_RC"'):
        raise RuntimeError("artifact hashes would not precede exit")
    return {
        "manifest_bytes": manifest_path.stat().st_size,
        "manifest_sha256": sha256(manifest_path),
        "mode": "act",
        "name": NAME,
    }


def validate_single_node_guard(adapter_path: Path) -> None:
    """Execute the exact guarded inherited-dependency function."""
    tree = ast.parse(adapter_path.read_bytes(), filename=str(adapter_path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_configure_tq_actor_runtime_env"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    calls: list[str] = []
    namespace = {
        "ray": SimpleNamespace(nodes=lambda: [{"Alive": True}]),
        "_patch_tq_actor_runtime_env": lambda: calls.append("patched"),
    }
    exec(compile(module, str(adapter_path), "exec"), namespace)
    configure = namespace["_configure_tq_actor_runtime_env"]
    configure({"actor_runtime_env_mode": "pip"})
    if calls != ["patched"]:
        raise RuntimeError("pip compatibility mode no longer installs runtime env")
    calls.clear()
    configure({"actor_runtime_env_mode": "inherit_baked_single_node"})
    if calls:
        raise RuntimeError("single-node inheritance installs a runtime env")
    for count in (0, 2):
        namespace["ray"].nodes = lambda count=count: [
            {"Alive": True} for _ in range(count)
        ]
        try:
            configure({"actor_runtime_env_mode": "inherit_baked_single_node"})
        except RuntimeError as error:
            expected = (
                "inherit_baked_single_node requires exactly one live Ray node; "
                f"found {count}"
            )
            if str(error) != expected:
                raise RuntimeError("single-node guard error differs") from error
        else:
            raise RuntimeError("single-node inheritance accepted unsafe node count")


def main() -> None:
    """Validate and deterministically reconstruct the liveness package."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--predecessor-source", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path, expected in (
        (args.megatron, MEGATRON_SHA256),
        (args.protocol, PROTOCOL_SHA256),
        (args.authorization, AUTHORIZATION_SHA256),
        (args.amendment, AMENDMENT_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen clean-room input moved: {path.name}")
    source_sha256 = sha256(args.source)
    manifest_validation = validate_manifest(args.manifest, source_sha256)
    with tempfile.TemporaryDirectory(prefix="m4-oars-liveness-") as raw:
        root = Path(raw)
        source_root = root / "source"
        predecessor_root = root / "predecessor"
        source_root.mkdir()
        predecessor_root.mkdir()
        member_count, symlink_count = safe_extract(args.source, source_root)
        safe_extract(args.predecessor_source, predecessor_root)
        actual_changed_paths = changed_paths(source_root, predecessor_root)
        if actual_changed_paths != EXPECTED_CHANGED_PATHS:
            raise RuntimeError(
                f"liveness source scope differs: {sorted(actual_changed_paths)}"
            )
        if sha256(source_root / CONFIG_PATH) != CONFIG_SHA256:
            raise RuntimeError("qualification config moved")
        if sha256(source_root / ANALYZER_PATH) != ANALYZER_SHA256:
            raise RuntimeError("qualification analyzer moved")
        if sha256(source_root / VALIDATION_PATH) != VALIDATION_SHA256:
            raise RuntimeError("offline validation moved")
        if (source_root / CONFIG_PATH).read_bytes() != (
            predecessor_root / CONFIG_PATH
        ).read_bytes():
            raise RuntimeError("qualification config differs from failed run")
        protocol_path = (
            "reports/auto_research/2026-09-15-m4-oars-randomized/"
            "randomized_actuation_protocol.json"
        )
        if (source_root / protocol_path).read_bytes() != (
            predecessor_root / protocol_path
        ).read_bytes():
            raise RuntimeError("frozen scientific protocol changed")
        validation = json.loads((source_root / VALIDATION_PATH).read_bytes())
        if (
            validation["status"] != "PASS_OFFLINE_REPAIR_GATES"
            or validation["tests"]["sampler_and_oars"]["stress_updates"] != 64
            or validation["tests"]["sampler_and_oars"][
                "maximum_buffered_groups_in_64_update_stress"
            ]
            != 12
        ):
            raise RuntimeError("offline liveness gate differs")
        validate_single_node_guard(
            source_root / "nemo_rl/data_plane/adapters/transfer_queue.py"
        )
        rebuilt = root / "rebuilt.json"
        subprocess.run(
            [
                sys.executable,
                str(args.builder),
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
                "--amendment",
                str(args.amendment),
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
        "amendment_sha256": AMENDMENT_SHA256,
        "authorization_sha256": AUTHORIZATION_SHA256,
        "bash_syntax_passed": True,
        "credential_free_static_gate_passed": True,
        "deterministic_rebuild_passed": True,
        "embedded_python_syntax_passed": True,
        "exact_source_change_scope_passed": True,
        "manifest": manifest_validation,
        "manifest_time_limit_seconds": 14400,
        "megatron_sha256": MEGATRON_SHA256,
        "offline_64_update_liveness_gate_passed": True,
        "predecessor_config_and_protocol_preserved": True,
        "protocol_sha256": PROTOCOL_SHA256,
        "queue_deadline_override": None,
        "repair_attempt_limit": 1,
        "schema": "m4-oars-actuation-qualification-liveness-package-validation-v1",
        "scientific_outcome_acquisition": False,
        "single_node_inheritance_guard_executed": True,
        "source_archive_member_count": member_count,
        "source_archive_sha256": source_sha256,
        "source_archive_symlink_count": symlink_count,
        "source_commit": args.source_commit,
        "status": "PASS_AUTHORIZED_UNSUBMITTED",
        "terminal_artifact_hash_on_failure_passed": True,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
