#!/usr/bin/env python3
"""Clean-room validate all 32 locally nonlaunchable acquisition candidates."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tarfile
import tempfile
from copy import deepcopy
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
BUILDER = ROOT / "build_trained_paired_candidates.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_builder():
    spec = importlib.util.spec_from_file_location("trained_package_builder", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scan_archive_credentials(path: Path) -> int:
    patterns = (
        re.compile(rb"glpat-[A-Za-z0-9_-]{12,}"),
        re.compile(rb"hf_[A-Za-z0-9]{20,}"),
        re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    )
    scanned = 0
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or member.size > 8_000_000:
                continue
            stream = archive.extractfile(member)
            assert stream is not None
            data = stream.read()
            scanned += 1
            if any(pattern.search(data) for pattern in patterns):
                raise RuntimeError(f"credential-like value in {path.name}:{member.name}")
    return scanned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    builder = load_builder()
    expected_inputs = (
        (args.source, builder.SOURCE_SHA),
        (args.megatron, builder.MEGATRON_SHA),
        (args.bridge, builder.BRIDGE_SHA),
        (args.protocol, builder.PROTOCOL_SHA),
        (args.run_manifest, builder.RUN_MANIFEST_SHA),
        (args.authorization, builder.AUTHORIZATION_SHA),
    )
    for path, expected in expected_inputs:
        if sha256(path) != expected:
            raise RuntimeError(f"clean-room input moved: {path.name}")
    authority = json.loads(args.authorization.read_bytes())
    if any(
        authority[key]
        for key in (
            "model_weight_access_authorized",
            "optimizer_initialization_authorized",
            "training_authorized",
            "eos_submission_authorized",
            "qualification_authorized",
            "scientific_acquisition_authorized",
            "automatic_retry",
            "automatic_replacement",
            "automatic_extension",
        )
    ):
        raise RuntimeError("local package authority is executable")

    base = json.loads(args.base_manifest.read_bytes())
    run_manifest = json.loads(args.run_manifest.read_bytes())
    runs = run_manifest["runs"]
    if len(runs) != 32 or len({(r["block_id"], r["regime"]) for r in runs}) != 32:
        raise RuntimeError("run registry differs")
    with tarfile.open(args.source, "r:gz") as archive:
        members = archive.getmembers()
        names = [PurePosixPath(member.name) for member in members]
        symlinks = {
            name for name, member in zip(names, members) if member.issym()
        }
        if (
            len(names) != len(set(names))
            or any(name.is_absolute() or ".." in name.parts for name in names)
            or any(any(parent in symlinks for parent in name.parents) for name in names)
        ):
            raise RuntimeError("source archive is not clean-room safe")
    credential_scanned_files = sum(
        scan_archive_credentials(path)
        for path in (args.source, args.megatron, args.bridge)
    )

    tampered_authority = deepcopy(authority)
    tampered_authority["training_authorized"] = True

    candidate_hashes: dict[str, str] = {}
    candidate_bytes: dict[str, int] = {}
    expected_payload_prefix = (
        builder.SOURCE_SHA,
        builder.MEGATRON_SHA,
        builder.BRIDGE_SHA,
        builder.PROTOCOL_SHA,
        builder.RUN_MANIFEST_SHA,
        builder.AUTHORIZATION_SHA,
    )
    with tempfile.TemporaryDirectory(prefix="m4-downstream-trained-package-") as raw:
        temp = Path(raw)
        for run in runs:
            identity = f"{run['block_id']}_{run['regime']}"
            stem = f"m4-downstream-quality-{run['block_id']}-{run['regime'].replace('_', '-')}-local-candidate.yaml"
            path = args.candidate_dir / stem
            manifest = json.loads(path.read_bytes())
            spec = manifest["spec"]
            expected_name = stem.removesuffix(".yaml")
            if (
                manifest["format_version"] != base["format_version"]
                or manifest["type"] != base["type"]
                or manifest["labels"] != base["labels"]
                or manifest["launchers"] != base["launchers"]
                or manifest["maintainers"] != base["maintainers"]
                or spec["name"] != expected_name
                or spec["nodes"] != 1
                or spec["time_limit"] != 14400
                or spec["image_source"] != base["spec"]["image_source"]
                or spec["workspace"] != base["spec"]["workspace"]
            ):
                raise RuntimeError(f"JET manifest boundary differs: {identity}")
            rendered = spec["script"].format(assets_dir=f"/tmp/{identity}")
            subprocess.run(["bash", "-n"], input=rendered, text=True, check=True)
            blocks = re.findall(r"<<'PY'\n(.*?)\nPY", rendered, flags=re.DOTALL)
            if len(blocks) != 7:
                raise RuntimeError(f"embedded Python block count differs: {identity}")
            for index, block in enumerate(blocks):
                compile(block, f"<{identity}-{index}>", "exec")
            encoded = re.findall(
                r"printf %s '([A-Za-z0-9+/=]+)' \| base64 -d", rendered
            )
            payload_hashes = tuple(
                hashlib.sha256(base64.b64decode(value)).hexdigest()
                for value in encoded
            )
            if payload_hashes[:6] != expected_payload_prefix or payload_hashes[6] != run["config_sha256"]:
                raise RuntimeError(f"embedded payload identity differs: {identity}")
            guarded = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    blocks[0],
                    str(args.protocol),
                    str(args.run_manifest),
                    str(args.authorization),
                ],
                cwd=args.repo,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            marker = f"M4_DOWNSTREAM_QUALITY_{identity.upper()}_PACKAGE_EVIDENCE_PASS"
            if (
                guarded.returncode == 0
                or marker not in guarded.stdout
                or "M4_DOWNSTREAM_QUALITY_LOCAL_PACKAGE_NO_EXECUTION_AUTHORITY" not in guarded.stderr
            ):
                raise RuntimeError(f"candidate did not fail closed: {identity}")
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as tampered:
                json.dump(tampered_authority, tampered)
                tampered.flush()
                rejected = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        blocks[0],
                        str(args.protocol),
                        str(args.run_manifest),
                        tampered.name,
                    ],
                    cwd=args.repo,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
            if rejected.returncode == 0 or marker in rejected.stdout:
                raise RuntimeError(f"mutated authority was not rejected: {identity}")
            guard_index = rendered.index("M4_DOWNSTREAM_QUALITY_LOCAL_PACKAGE_NO_EXECUTION_AUTHORITY")
            extraction_index = rendered.index('mkdir -p "$RUN_REPO"')
            config_index = rendered.index("from omegaconf import OmegaConf")
            training_index = rendered.index("examples/run_grpo_single_controller.py")
            if not guard_index < extraction_index < config_index < training_index:
                raise RuntimeError(f"execution guard ordering differs: {identity}")
            if re.search(r"(?mi)^\s*(?:\S*/)?(?:runllm\.py|sbatch|srun)\b", rendered):
                raise RuntimeError(f"nested launcher found: {identity}")
            if re.search(r"(?i)(?:glpat-|hf_[A-Za-z0-9]{20,}|api[_-]?key\s*[=:])", rendered):
                raise RuntimeError(f"credential-like value found: {identity}")
            rebuilt = temp / stem
            subprocess.run(
                [
                    sys.executable,
                    str(BUILDER),
                    "--identity",
                    identity,
                    "--base-manifest",
                    str(args.base_manifest),
                    "--source",
                    str(args.source),
                    "--megatron",
                    str(args.megatron),
                    "--bridge",
                    str(args.bridge),
                    "--protocol",
                    str(args.protocol),
                    "--run-manifest",
                    str(args.run_manifest),
                    "--authorization",
                    str(args.authorization),
                    "--config-dir",
                    str(args.config_dir),
                    "--output",
                    str(rebuilt),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if rebuilt.read_bytes() != path.read_bytes():
                raise RuntimeError(f"deterministic rebuild differs: {identity}")
            candidate_hashes[identity] = sha256(path)
            candidate_bytes[identity] = path.stat().st_size

    result = {
        "schema": "m4-downstream-quality-trained-paired-local-package-validation-v1",
        "status": "PASS_LOCAL_PACKAGE_UNLAUNCHABLE",
        "runtime_source_commit": builder.SOURCE_COMMIT,
        "source_archive_sha256": builder.SOURCE_SHA,
        "source_archive_member_count": len(members),
        "source_archive_symlink_count": len(symlinks),
        "megatron_archive_sha256": builder.MEGATRON_SHA,
        "bridge_archive_sha256": builder.BRIDGE_SHA,
        "protocol_sha256": builder.PROTOCOL_SHA,
        "run_manifest_sha256": builder.RUN_MANIFEST_SHA,
        "authorization_sha256": builder.AUTHORIZATION_SHA,
        "builder_sha256": sha256(BUILDER),
        "candidate_count": 32,
        "candidate_sha256": candidate_hashes,
        "candidate_bytes": candidate_bytes,
        "base_manifest_schema_equivalence_passed": True,
        "bash_syntax_passed": True,
        "embedded_python_syntax_passed": True,
        "payload_authentication_passed": True,
        "dynamic_no_execution_guard_passed": True,
        "guard_precedes_extraction_config_and_training": True,
        "credential_scan_passed": True,
        "credential_scanned_archive_files": credential_scanned_files,
        "mutated_authority_rejected": True,
        "nested_launch_surface_absent": True,
        "deterministic_rebuild": True,
        "model_weight_accessed": False,
        "optimizer_initialized": False,
        "training_started": False,
        "eos_submission_attempted": False,
        "scientific_acquisition_started": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
