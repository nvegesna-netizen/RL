#!/usr/bin/env python3
"""Clean-room validation for 20 nonlaunchable OARS confirmatory candidates."""

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

from build_confirmatory_candidates import (
    ANALYSIS_PLAN_SHA256,
    AUTHORIZATION_SHA256,
    CONFIG_SHA256,
    FIFO_CONFIG,
    IMAGE_PATH,
    MEGATRON_SHA256,
    OARS_CONFIG,
    PROTOCOL_SHA256,
    QUALIFICATION_GATE_SHA256,
    RUN_MANIFEST_SHA256,
    SOURCE_COMMIT,
    SOURCE_SHA256,
)

EXPECTED_ORDER = [
    ("oars", "fifo"),
    ("oars", "fifo"),
    ("oars", "fifo"),
    ("fifo", "oars"),
    ("oars", "fifo"),
    ("fifo", "oars"),
    ("fifo", "oars"),
    ("oars", "fifo"),
    ("fifo", "oars"),
    ("oars", "fifo"),
]


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_design(
    protocol_path: Path,
    run_manifest_path: Path,
    analysis_plan_path: Path,
    qualification_path: Path,
    authorization_path: Path,
) -> list[dict[str, object]]:
    """Validate the preregistered order, pairing, gates, and no-run authority."""
    protocol = json.loads(protocol_path.read_bytes())
    manifest = json.loads(run_manifest_path.read_bytes())
    analysis = json.loads(analysis_plan_path.read_bytes())
    qualification = json.loads(qualification_path.read_bytes())
    authorization = json.loads(authorization_path.read_bytes())
    if (
        protocol["protocol_id"] != "m4-oars-randomized-actuation-v1"
        or protocol["confirmatory_design"]["fixed_complete_pairs"] != 10
        or qualification["qualification_gate"] != "PASS"
        or manifest["run_count"] != 20
        or manifest["protocol_sha256"] != PROTOCOL_SHA256
        or manifest["qualification_gate"]["sha256"] != QUALIFICATION_GATE_SHA256
        or manifest["runtime_source"]["commit"] != SOURCE_COMMIT
        or manifest["runtime_source"]["archive_sha256"] != SOURCE_SHA256
        or analysis["confidence_intervals"]["df"] != 9
        or analysis["confidence_intervals"]["t_critical"]
        != 2.2621571628540993
        or analysis["exact_randomization"]["enumeration_count"] != 1024
    ):
        raise RuntimeError("confirmatory design boundary differs")
    runs = manifest["runs"]
    if len({run["identity"] for run in runs}) != 20:
        raise RuntimeError("run identities are not unique")
    if [run["global_sequence"] for run in runs] != list(range(1, 21)):
        raise RuntimeError("global run sequence differs")
    for index, run in enumerate(runs):
        expected_predecessor = None if index == 0 else runs[index - 1]["identity"]
        if run["predecessor"] != expected_predecessor:
            raise RuntimeError(f"{run['identity']}: predecessor differs")
    for pair, order in enumerate(EXPECTED_ORDER, start=1):
        pair_runs = [run for run in runs if run["pair"] == pair]
        if (
            tuple(run["arm"] for run in pair_runs) != order
            or {run["training_seed"] for run in pair_runs} != {20261500 + pair}
            or {run["assignment_seed"] for run in pair_runs} != {20261600 + pair}
            or {run["assignment_domain"] for run in pair_runs}
            != {f"m4-oars-confirmatory-pair-{pair:02d}-v1"}
            or {run["mode"] for run in pair_runs if run["arm"] == "fifo"}
            != {"observe"}
            or {run["mode"] for run in pair_runs if run["arm"] == "oars"}
            != {"act"}
        ):
            raise RuntimeError(f"pair {pair}: registered pairing differs")
    blocked = (
        "model_weight_access_authorized",
        "optimizer_initialization_authorized",
        "training_authorized",
        "eos_submission_authorized",
        "scientific_acquisition_authorized",
        "automatic_retry",
        "automatic_replacement",
        "automatic_extension",
    )
    if (
        authorization["schema"]
        != "m4-oars-randomized-confirmatory-local-package-authorization-v1"
        or authorization["scope"]
        != "build_and_clean_room_validate_credential_free_20_run_confirmatory_candidates_only"
        or authorization["candidate_count"] != 20
        or authorization["run_manifest_sha256"] != RUN_MANIFEST_SHA256
        or authorization["analysis_plan_sha256"] != ANALYSIS_PLAN_SHA256
        or authorization["qualification_gate_sha256"]
        != QUALIFICATION_GATE_SHA256
        or any(authorization[key] for key in blocked)
    ):
        raise RuntimeError("local no-execution authority differs")
    return runs


def validate_source(source: Path, output: Path) -> dict[str, int]:
    """Safely extract and validate the exact qualified runtime source."""
    with tarfile.open(source, "r:gz") as archive:
        members = archive.getmembers()
        names = [PurePosixPath(member.name) for member in members]
        if len(names) != len(set(names)):
            raise RuntimeError("duplicate source member")
        if any(name.is_absolute() or ".." in name.parts for name in names):
            raise RuntimeError("unsafe source member")
        symlinks = {
            name for name, member in zip(names, members, strict=True)
            if member.issym()
        }
        if any(any(parent in symlinks for parent in name.parents) for name in names):
            raise RuntimeError("source traverses symlink parent")
        archive.extractall(output, filter="fully_trusted")
    for relative, expected in (
        (FIFO_CONFIG, CONFIG_SHA256["fifo"]),
        (OARS_CONFIG, CONFIG_SHA256["oars"]),
        (
            "tools/m4_oars_actuation_qualification.py",
            "53f07de5206bdb11392c5363c39d675b60443c969294cf6a6e61188a7a62fbcd",
        ),
        (
            "examples/converters/convert_megatron_to_hf.py",
            "e18aceb8de2c22d2ab0d9a1c8c6aa96b28a6ff5fcdba156832ec103f3f0c56a7",
        ),
        (
            "nemo_rl/data/datasets/response_datasets/gsm8k.py",
            "9cb636fb6e05d6f15cb55add43f8b755498b7240da32cf364fff201b90f7aa19",
        ),
        (
            "examples/prompts/gsm8k.txt",
            "6b2539399c7928b602462c2bd37423e3ffaa72d1c23279efe7bab4d7a070b40e",
        ),
    ):
        if sha256(output / relative) != expected:
            raise RuntimeError(f"qualified runtime file moved: {relative}")
    return {"member_count": len(members), "symlink_count": len(symlinks)}


def validate_candidate(
    path: Path,
    run: dict[str, object],
    payload_hashes: list[str],
) -> dict[str, object]:
    """Validate one manifest, embedded programs, payloads, and authority denial."""
    manifest = json.loads(path.read_bytes())
    spec = manifest["spec"]
    identity = str(run["identity"])
    if (
        manifest["format_version"] != 1
        or manifest["labels"] != {"target": "silicon"}
        or manifest["launchers"]
        != {"type:slurm": {"nodes": 1, "ntasks_per_node": 1}}
        or spec["name"] != f"m4-oars-confirmatory-{identity}-local-candidate"
        or spec["workspace"] != "/workspace"
        or spec["nodes"] != 1
        or spec["time_limit"] != 14400
        or spec["image_source"] != {"local_path": IMAGE_PATH}
    ):
        raise RuntimeError(f"{identity}: JET boundary differs")
    rendered = spec["script"].format(
        assets_dir=f"/tmp/m4-oars-confirmatory-{identity}"
    )
    subprocess.run(["bash", "-n"], input=rendered, text=True, check=True)
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", rendered, flags=re.DOTALL)
    if len(blocks) != 7:
        raise RuntimeError(f"{identity}: embedded Python block count differs")
    for index, block in enumerate(blocks):
        compile(block, f"<m4-oars-confirmatory-{identity}-{index}>", "exec")
    encoded = re.findall(
        r"printf %s '([A-Za-z0-9+/=]+)' \| base64 -d", rendered
    )
    observed_payload_hashes = [
        hashlib.sha256(base64.b64decode(value)).hexdigest() for value in encoded
    ]
    if observed_payload_hashes != payload_hashes:
        raise RuntimeError(f"{identity}: embedded payload hashes differ")
    decoded = [base64.b64decode(value) for value in encoded]
    with tempfile.TemporaryDirectory(
        prefix=f"m4-oars-confirmatory-authority-{identity}-"
    ) as raw:
        root = Path(raw)
        authority_paths = []
        for name, value in zip(
            ("protocol.json", "runs.json", "analysis.json", "gate.json", "auth.json"),
            decoded[2:],
            strict=True,
        ):
            authority_path = root / name
            authority_path.write_bytes(value)
            authority_paths.append(authority_path)
        denial = subprocess.run(
            [sys.executable, "-", *map(str, authority_paths)],
            input=blocks[0],
            text=True,
            capture_output=True,
            check=False,
        )
        if (
            denial.returncode == 0
            or "M4_OARS_CONFIRMATORY_LOCAL_PACKAGE_NO_EXECUTION_AUTHORITY"
            not in denial.stderr
        ):
            raise RuntimeError(f"{identity}: local authority did not deny execution")
    lowered = rendered.lower()
    forbidden = (
        "ci_job_token",
        "private-token",
        "authorization:",
        "hf_token",
        "nvidia_api_key",
        "queue_deadline",
        "runllm.py",
        "sbatch ",
        "srun ",
    )
    if any(token in lowered for token in forbidden):
        raise RuntimeError(f"{identity}: credential or launch surface present")
    required = (
        "M4_OARS_CONFIRMATORY_LOCAL_PACKAGE_NO_EXECUTION_AUTHORITY",
        "strict_global_sequence_one_run_at_a_time",
        "terminal_gsm8k_accuracy",
        "retained_l1_per_selected_group",
        "valid_actor_tokens_per_update",
        "PASS_ALL_20_AUTHENTICATED_RESULTS_UNOPENED",
    )
    # The sequence and completion-gate strings are carried by encoded inputs.
    required_plaintext = required[:1] + required[2:5]
    if not all(token in rendered for token in required_plaintext):
        raise RuntimeError(f"{identity}: runtime/result contract missing")
    denial_index = rendered.index(
        "M4_OARS_CONFIRMATORY_LOCAL_PACKAGE_NO_EXECUTION_AUTHORITY"
    )
    extraction_index = rendered.index('"$PYTHON" - "$SOURCE" "$RUN_REPO"')
    training_index = rendered.index(
        '"$PYTHON" examples/run_grpo_single_controller.py'
    )
    if not denial_index < extraction_index < training_index:
        raise RuntimeError(f"{identity}: authority denial is not fail-closed")
    return {
        "bytes": path.stat().st_size,
        "identity": identity,
        "sha256": sha256(path),
        "bash_syntax": True,
        "embedded_python_blocks": len(blocks),
        "execution_authority": False,
        "execution_denial_exercised": True,
    }


def main() -> None:
    """Deterministically rebuild and validate all 20 candidates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--analysis-plan", type=Path, required=True)
    parser.add_argument("--qualification-gate", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--candidates-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path, expected in (
        (args.source, SOURCE_SHA256),
        (args.megatron, MEGATRON_SHA256),
        (args.protocol, PROTOCOL_SHA256),
        (args.run_manifest, RUN_MANIFEST_SHA256),
        (args.analysis_plan, ANALYSIS_PLAN_SHA256),
        (args.qualification_gate, QUALIFICATION_GATE_SHA256),
        (args.authorization, AUTHORIZATION_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen clean-room input moved: {path.name}")
    runs = validate_design(
        args.protocol,
        args.run_manifest,
        args.analysis_plan,
        args.qualification_gate,
        args.authorization,
    )
    payload_hashes = [
        SOURCE_SHA256,
        MEGATRON_SHA256,
        PROTOCOL_SHA256,
        RUN_MANIFEST_SHA256,
        ANALYSIS_PLAN_SHA256,
        QUALIFICATION_GATE_SHA256,
        AUTHORIZATION_SHA256,
    ]
    records = []
    with tempfile.TemporaryDirectory(prefix="m4-oars-confirmatory-cleanroom-") as raw:
        root = Path(raw)
        source_validation = validate_source(args.source, root / "source")
        for run in runs:
            identity = str(run["identity"])
            candidate = args.candidates_dir / f"{identity}.json"
            rebuilt = root / f"{identity}.json"
            subprocess.run(
                [
                    sys.executable,
                    str(args.builder),
                    "--identity",
                    identity,
                    "--source",
                    str(args.source),
                    "--megatron",
                    str(args.megatron),
                    "--protocol",
                    str(args.protocol),
                    "--run-manifest",
                    str(args.run_manifest),
                    "--analysis-plan",
                    str(args.analysis_plan),
                    "--qualification-gate",
                    str(args.qualification_gate),
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
                raise RuntimeError(f"{identity}: deterministic rebuild differs")
            records.append(validate_candidate(candidate, run, payload_hashes))
    if len({record["sha256"] for record in records}) != 20:
        raise RuntimeError("candidate manifests are not all distinct")
    result = {
        "analysis_plan_sha256": ANALYSIS_PLAN_SHA256,
        "authorization_sha256": AUTHORIZATION_SHA256,
        "candidate_count": 20,
        "candidates": records,
        "qualification_gate_sha256": QUALIFICATION_GATE_SHA256,
        "run_manifest_sha256": RUN_MANIFEST_SHA256,
        "runtime_source_commit": SOURCE_COMMIT,
        "runtime_source_sha256": SOURCE_SHA256,
        "schema": "m4-oars-randomized-confirmatory-package-validation-v1",
        "source_validation": source_validation,
        "status": "PASS_CLEAN_ROOM_NONLAUNCHABLE",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
