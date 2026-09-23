#!/usr/bin/env python3
"""Clean-room validation for all 54 OARS-v2 quality-primary identities."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from build_quality_primary_candidate import (
    ANALYSIS_PLAN_SHA256,
    AUTHORIZATION_SHA256,
    CONFIG_PATH,
    CONFIG_SHA256,
    IMAGE_PATH,
    MEGATRON_SHA256,
    PROTOCOL_SHA256,
    QUALIFICATION_GATE_SHA256,
    RUN_MANIFEST_SHA256,
    SOURCE_COMMIT,
    SOURCE_SHA256,
    TEMPLATE_SHA256,
    validate_authorization,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exercise_authority(
    block: str,
    decoded: list[bytes],
    authorization: dict[str, object],
    identity: str,
) -> None:
    with tempfile.TemporaryDirectory(
        prefix=f"m4-oars-v2-quality-authority-{identity}-"
    ) as raw:
        root = Path(raw)
        paths = []
        for name, value in zip(
            ("protocol.json", "runs.json", "analysis.json", "gate.json", "auth.json"),
            decoded[2:],
            strict=True,
        ):
            path = root / name
            path.write_bytes(value)
            paths.append(path)
        positive = subprocess.run(
            [sys.executable, "-", *map(str, paths)],
            input=block,
            text=True,
            capture_output=True,
            check=False,
        )
        if positive.returncode != 0:
            raise RuntimeError(
                f"{identity}: positive authority failed: {positive.stderr[-500:]}"
            )
        denied = dict(authorization)
        denied["training_authorized"] = False
        denied_path = root / "denied-auth.json"
        denied_path.write_text(json.dumps(denied, sort_keys=True) + "\n")
        denied_hash = sha256(denied_path)
        if block.count(AUTHORIZATION_SHA256) != 1:
            raise RuntimeError(f"{identity}: authority hash assertion differs")
        negative = subprocess.run(
            [sys.executable, "-", *map(str, paths[:-1]), str(denied_path)],
            input=block.replace(AUTHORIZATION_SHA256, denied_hash),
            text=True,
            capture_output=True,
            check=False,
        )
        if negative.returncode == 0:
            raise RuntimeError(f"{identity}: training-authority denial failed")


def exercise_finalizer(rendered: str, identity: str) -> None:
    match = re.search(
        r"(finalize_artifacts\(\) \{\n.*?\n\})", rendered, flags=re.DOTALL
    )
    if match is None:
        raise RuntimeError(f"{identity}: finalizer missing")
    with tempfile.TemporaryDirectory(
        prefix=f"m4-oars-v2-quality-finalizer-{identity}-"
    ) as raw:
        assets = Path(raw)
        (assets / "evaluation").mkdir()
        (assets / "jet_assets" / "output_logs").mkdir(parents=True)
        (assets / "core.txt").write_text("core\n")
        (assets / "evaluation" / "raw.json").write_text("{}\n")
        (assets / "jet_assets" / "stable.txt").write_text("stable\n")
        (assets / "jet_assets" / "output_logs" / "live.log").write_text("live\n")
        subprocess.run(
            ["bash", "-s", "--", str(assets)],
            input=(
                'set -euo pipefail\nASSETS="$1"\n'
                + match.group(1)
                + "\nfinalize_artifacts\n"
            ),
            text=True,
            capture_output=True,
            check=True,
        )
        paths = {
            line.split(maxsplit=1)[1]
            for line in (assets / "artifacts.sha256").read_text().splitlines()
        }
        if paths != {
            "./core.txt",
            "./evaluation/raw.json",
            "./jet_assets/stable.txt",
        }:
            raise RuntimeError(f"{identity}: artifact ownership boundary differs")


def validate_candidate(
    path: Path,
    run: dict[str, object],
    authorization: dict[str, object],
    *,
    exercise_finalizer_now: bool,
) -> dict[str, object]:
    manifest = json.loads(path.read_bytes())
    identity = str(run["identity"])
    spec = manifest["spec"]
    if (
        manifest["format_version"] != 1
        or manifest["labels"] != {"target": "silicon"}
        or manifest["launchers"] != {"type:slurm": {"nodes": 1, "ntasks_per_node": 1}}
        or spec["name"] != f"m4-oars-v2-quality-primary-{identity}-acquisition"
        or spec["workspace"] != "/workspace"
        or spec["nodes"] != 1
        or spec["time_limit"] != 14400
        or "sbatch_additional_flags" in spec
        or spec["image_source"] != {"local_path": IMAGE_PATH}
    ):
        raise RuntimeError(f"{identity}: JET boundary differs")
    rendered = spec["script"].format(
        assets_dir=f"/tmp/m4-oars-v2-quality-primary-{identity}"
    )
    syntax = subprocess.run(
        ["bash", "-n"], input=rendered, text=True, capture_output=True, check=False
    )
    if syntax.returncode != 0:
        raise RuntimeError(f"{identity}: bash syntax failed: {syntax.stderr}")
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", rendered, flags=re.DOTALL)
    if len(blocks) != 7:
        raise RuntimeError(f"{identity}: embedded Python topology differs")
    for index, block in enumerate(blocks):
        compile(block, f"<m4-oars-v2-quality-{identity}-{index}>", "exec")
    encoded = re.findall(
        r"printf %s '([A-Za-z0-9+/=]+)' \| base64 -d", rendered
    )
    decoded = [base64.b64decode(value) for value in encoded]
    payload_hashes = [hashlib.sha256(value).hexdigest() for value in decoded]
    if payload_hashes != [
        SOURCE_SHA256,
        MEGATRON_SHA256,
        PROTOCOL_SHA256,
        RUN_MANIFEST_SHA256,
        ANALYSIS_PLAN_SHA256,
        QUALIFICATION_GATE_SHA256,
        AUTHORIZATION_SHA256,
    ]:
        raise RuntimeError(f"{identity}: embedded payload hashes differ")
    exercise_authority(blocks[0], decoded, authorization, identity)
    if exercise_finalizer_now:
        exercise_finalizer(rendered, identity)
    lowered = rendered.lower()
    forbidden = (
        "ci_job_token",
        "private-token",
        "authorization:",
        "hf_token",
        "nvidia_api_key",
        "queue_deadline",
        "sbatch ",
        "srun ",
        "opportunity_at_risk_shadow.",
    )
    if any(token in lowered for token in forbidden):
        raise RuntimeError(f"{identity}: credential or forbidden surface present")
    required = (
        'a["required_launcher"]=="runllm.py --no_wait"',
        "strict_global_sequence_one_run_at_a_time",
        CONFIG_PATH,
        CONFIG_SHA256,
        "terminal_gsm8k_accuracy",
        "retained_l1_per_selected_group",
        "valid_actor_tokens_per_update",
        "complete_outcome_embargo_until_all_54_authenticated",
        "async_rl.terminal_policy_export.enabled=true",
    )
    if not all(token in rendered for token in required):
        raise RuntimeError(f"{identity}: acquisition contract incomplete")
    scorer = run["actuation_scorer"]
    if (
        f'async_rl.opportunity_at_risk_v2_shadow.mode={run["mode"]}' not in rendered
        or (
            "async_rl.opportunity_at_risk_v2_shadow.actuation_scorer="
            + ("null" if scorer is None else str(scorer))
        )
        not in rendered
    ):
        raise RuntimeError(f"{identity}: registered scheduler configuration differs")
    analyzer = (
        "m4_oars_v2_controlled_frontier_qualification.py"
        if scorer is None
        else "m4_oars_v2_actuation_qualification.py"
    )
    if analyzer not in rendered:
        raise RuntimeError(f"{identity}: systems gate differs")
    return {
        "arm": run["arm"],
        "bash_syntax": True,
        "bytes": path.stat().st_size,
        "global_sequence": run["global_sequence"],
        "identity": identity,
        "mode": run["mode"],
        "predecessor": run["predecessor"],
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--analysis-plan", type=Path, required=True)
    parser.add_argument("--qualification-gate", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--first-candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path, expected in (
        (args.template, TEMPLATE_SHA256),
        (args.source, SOURCE_SHA256),
        (args.megatron, MEGATRON_SHA256),
        (args.protocol, PROTOCOL_SHA256),
        (args.run_manifest, RUN_MANIFEST_SHA256),
        (args.analysis_plan, ANALYSIS_PLAN_SHA256),
        (args.qualification_gate, QUALIFICATION_GATE_SHA256),
        (args.authorization, AUTHORIZATION_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen package input moved: {path.name}")
    validate_authorization(args.authorization)
    authorization = json.loads(args.authorization.read_bytes())
    manifest = json.loads(args.run_manifest.read_bytes())
    runs = manifest["runs"]
    if len(runs) != 54:
        raise RuntimeError("registered run count differs")
    records = []
    with tempfile.TemporaryDirectory(
        prefix="m4-oars-v2-quality-primary-cleanroom-"
    ) as raw:
        rebuilt = Path(raw) / "candidate.json"
        for index, run in enumerate(runs):
            command = [
                sys.executable,
                str(args.builder),
                "--identity",
                str(run["identity"]),
                "--template",
                str(args.template),
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
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            if index == 0 and rebuilt.read_bytes() != args.first_candidate.read_bytes():
                raise RuntimeError("first candidate deterministic rebuild differs")
            records.append(
                validate_candidate(
                    rebuilt,
                    run,
                    authorization,
                    exercise_finalizer_now=index == 0,
                )
            )
    if len({record["sha256"] for record in records}) != 54:
        raise RuntimeError("candidate manifests are not all distinct")
    result = {
        "authorization_sha256": AUTHORIZATION_SHA256,
        "candidate_count": 54,
        "candidates": records,
        "complete_outcome_embargo": True,
        "first_authorized_identity": records[0]["identity"],
        "first_authorized_identity_predecessor": records[0]["predecessor"],
        "runtime_source_commit": SOURCE_COMMIT,
        "schema": "m4-oars-v2-quality-primary-package-validation-v1",
        "sequence_policy": "strict_global_sequence_one_run_at_a_time",
        "status": "PASS_AUTHORIZED_UNSUBMITTED",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
