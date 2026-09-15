#!/usr/bin/env python3
"""Clean-room validate the repaired one-shot OARS contention preflight package."""

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

PROTOCOL_SHA256 = "06df90b6d411428cb959b1ee2f8b3d4174ca5284f1841dbcdd88fcd53f2178a1"
AUTHORIZATION_SHA256 = (
    "9b150d35ff37469e39241d02bef3c611b9e1f656faf67f5daf4ba64c80d9bcd8"
)
CONFIG_SHA256 = "becbd749e516d7af2cf08b24475e13fa544bcdc9a6505bb67735f4f25cd401c9"
ANALYZER_SHA256 = "c3b2fd40baff10c7d77cd589a433c2ccb64f81b63c76ab257322fd6f75fd548e"
MEGATRON_SHA256 = "98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
IMAGE_PATH = (
    "/lustre/fsw/coreai_dlalgo_ci/nvegesna/nemo_rl_images/nemo-rl-nightly-5802754.sqsh"
)
CONFIG_PATH = "examples/configs/grpo_math_1B_megatron_single_controller_m4_oars_shadow_contention_preflight.yaml"
ANALYZER_PATH = "tools/m4_oars_shadow_contention_preflight.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_extract(archive: Path, output: Path) -> tuple[int, int]:
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        names = [PurePosixPath(member.name) for member in members]
        if len(names) != len(set(names)):
            raise RuntimeError("duplicate archive member")
        if any(name.is_absolute() or ".." in name.parts for name in names):
            raise RuntimeError("unsafe archive path")
        symlinks = {
            name for name, member in zip(names, members, strict=True) if member.issym()
        }
        if any(any(parent in symlinks for parent in name.parents) for name in names):
            raise RuntimeError("archive traverses symlink parent")
        source.extractall(output)
    return len(members), len(symlinks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path, expected in (
        (args.megatron, MEGATRON_SHA256),
        (args.protocol, PROTOCOL_SHA256),
        (args.authorization, AUTHORIZATION_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen clean-room input moved: {path.name}")
    source_sha = sha256(args.source)
    manifest = json.loads(args.candidate.read_bytes())
    spec = manifest["spec"]
    if (
        manifest["format_version"] != 1
        or manifest["labels"] != {"target": "silicon"}
        or manifest["launchers"] != {"type:slurm": {"nodes": 1, "ntasks_per_node": 1}}
        or spec["name"] != "m4-oars-fifo-controlled-shadow-contention-preflight"
        or spec["workspace"] != "/workspace"
        or spec["nodes"] != 1
        or spec["time_limit"] != 14400
        or spec["image_source"] != {"local_path": IMAGE_PATH}
    ):
        raise RuntimeError("JET manifest boundary differs")

    script = spec["script"]
    rendered = script.format(assets_dir="/tmp/m4-oars-shadow-contention-preflight")
    subprocess.run(["bash", "-n"], input=rendered, text=True, check=True)
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", rendered, flags=re.DOTALL)
    if len(blocks) != 3:
        raise RuntimeError(f"embedded Python block count differs: {len(blocks)}")
    for index, block in enumerate(blocks):
        compile(block, f"<m4-oars-shadow-contention-preflight-{index}>", "exec")

    encoded = re.findall(r"printf %s '([A-Za-z0-9+/=]+)' \| base64 -d", rendered)
    payload_hashes = tuple(
        hashlib.sha256(base64.b64decode(value)).hexdigest() for value in encoded
    )
    if payload_hashes != (
        source_sha,
        MEGATRON_SHA256,
        PROTOCOL_SHA256,
        AUTHORIZATION_SHA256,
    ):
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
    )
    if any(token in lowered for token in forbidden):
        raise RuntimeError("forbidden nested launcher or credential surface present")
    if rendered.count("examples/run_grpo_single_controller.py") != 1:
        raise RuntimeError("training entrypoint count differs")
    if "--time" in rendered or "queue_deadline" in rendered:
        raise RuntimeError("script changes scheduler time or queue deadline")
    required_surfaces = (
        "selection_candidate_watermark==8",
        'oars_actuation_authorized"]',
        "set +e",
        'exit "$ANALYZER_RC"',
        'sha256sum "$RUN_LOG"',
    )
    if not all(surface in rendered for surface in required_surfaces):
        raise RuntimeError("contention or terminal-artifact gate is absent")
    if rendered.index('sha256sum "$RUN_LOG"') > rendered.index('exit "$ANALYZER_RC"'):
        raise RuntimeError("artifact hashes would not precede analyzer exit")

    with tempfile.TemporaryDirectory(prefix="m4-oars-contention-cleanroom-") as raw:
        cleanroom = Path(raw)
        extracted = cleanroom / "source"
        extracted.mkdir()
        member_count, symlink_count = safe_extract(args.source, extracted)
        for relative, expected in (
            (CONFIG_PATH, CONFIG_SHA256),
            (ANALYZER_PATH, ANALYZER_SHA256),
        ):
            path = extracted / relative
            if not path.is_file() or sha256(path) != expected:
                raise RuntimeError(f"source archive runtime moved: {relative}")
        rebuilt = cleanroom / "rebuilt.json"
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
                "--output",
                str(rebuilt),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if rebuilt.read_bytes() != args.candidate.read_bytes():
            raise RuntimeError("deterministic package rebuild differs")

    result = {
        "schema": "m4-oars-shadow-contention-preflight-package-validation-v1",
        "status": "PASS_AUTHORIZED_UNSUBMITTED",
        "source_commit": args.source_commit,
        "source_archive_sha256": source_sha,
        "megatron_sha256": MEGATRON_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "authorization_sha256": AUTHORIZATION_SHA256,
        "builder_sha256": sha256(args.builder),
        "candidate_sha256": sha256(args.candidate),
        "candidate_bytes": args.candidate.stat().st_size,
        "source_archive_member_count": member_count,
        "source_archive_symlink_count": symlink_count,
        "bash_syntax_passed": True,
        "embedded_python_syntax_passed": True,
        "credential_free_static_gate_passed": True,
        "single_training_entrypoint_passed": True,
        "deterministic_rebuild_passed": True,
        "terminal_artifact_hash_on_gate_failure_passed": True,
        "manifest_time_limit_seconds": 14400,
        "queue_deadline_override": None,
        "selection_candidate_watermark": 8,
        "selection_cardinality": 4,
        "acting_sampler": "weight_fifo",
        "oars_actuated": False,
        "scientific_outcome_acquisition": False,
        "submission_attempted": False,
        "automatic_retry": False,
        "automatic_extension": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
