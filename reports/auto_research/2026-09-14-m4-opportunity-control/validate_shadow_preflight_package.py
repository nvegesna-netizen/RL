#!/usr/bin/env python3
"""Clean-room validate the one-shot credential-free OARS preflight package."""

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

PROTOCOL_SHA256 = "1482d985a17197a58af018bd4e043a1008b78b6dbe4d99671954dc8452705513"
AUTHORIZATION_SHA256 = (
    "855b5e8fe61b338aedf2e79eef8f098ff00417d6540033e21fc918de0742471e"
)
CONFIG_SHA256 = "200164a4f7beab123d687a4674ac6ba4d3a1f84299cb676e672b8df282b8acb4"
ANALYZER_SHA256 = "151a9b7e726934647bd3f40ac38723c433622003cc3af772ce21c5f5ec23ae85"
MEGATRON_SHA256 = "98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
IMAGE_PATH = (
    "/lustre/fsw/coreai_dlalgo_ci/nvegesna/nemo_rl_images/nemo-rl-nightly-5802754.sqsh"
)
CONFIG_PATH = "examples/configs/grpo_math_1B_megatron_single_controller_m4_oars_shadow_preflight.yaml"
ANALYZER_PATH = "tools/m4_oars_shadow_preflight.py"


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
        or spec["name"] != "m4-oars-fifo-controlled-shadow-preflight"
        or spec["workspace"] != "/workspace"
        or spec["nodes"] != 1
        or spec["time_limit"] != 14400
        or spec["image_source"] != {"local_path": IMAGE_PATH}
    ):
        raise RuntimeError("JET manifest boundary differs")

    script = spec["script"]
    rendered = script.format(assets_dir="/tmp/m4-oars-shadow-preflight")
    subprocess.run(["bash", "-n"], input=rendered, text=True, check=True)
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", rendered, flags=re.DOTALL)
    if len(blocks) != 3:
        raise RuntimeError(f"embedded Python block count differs: {len(blocks)}")
    for index, block in enumerate(blocks):
        compile(block, f"<m4-oars-shadow-preflight-{index}>", "exec")

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
    if "weight_fifo" not in rendered or 'oars_actuation_authorized"]' not in rendered:
        raise RuntimeError("FIFO/OARS policy boundary is absent")

    with tempfile.TemporaryDirectory(prefix="m4-oars-shadow-cleanroom-") as raw:
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
        "schema": "m4-oars-shadow-preflight-package-validation-v1",
        "status": "PASS_UNSUBMITTED",
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
        "manifest_time_limit_seconds": 14400,
        "queue_deadline_override": None,
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
