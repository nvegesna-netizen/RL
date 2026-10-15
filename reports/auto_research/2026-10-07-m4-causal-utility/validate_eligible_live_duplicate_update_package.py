"""Extract and test the exact submitted qualification package in a fresh directory."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from research.m4_eligible_replay.contracts import digest_file


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = (
    ROOT
    / "session/20260914_m4_opportunity_control/m4-eligible-live-duplicate-update-qualification-manifest.json"
)


def validate_repaired_shell(script: str) -> None:
    """Fail before extraction if generated shell bypasses repaired startup gates."""
    if "--ray-log-dir" in script or "/workspace/m4-duplicate-ray-" in script:
        raise ValueError("long explicit Ray root remains in submitted shell")
    layout = script.index("-m research.m4_eligible_replay.ray_startup >")
    preflight = script.index("-m research.m4_eligible_replay.preflight ")
    transport = script.index("-m research.m4_eligible_replay.transport ")
    if not layout < preflight < transport:
        raise ValueError("socket/import gates must precede checkpoint transport")
    if (
        "for replicate in a b; do" not in script
        or script.count("-m research.m4_eligible_replay.run_qualification ") != 1
    ):
        raise ValueError("fresh-process duplicate replay loop changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-replacement", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--full-environment", action="store_true")
    parser.add_argument("--ray-startup-repair", action="store_true")
    parser.add_argument("--state-diagnostics", action="store_true")
    args = parser.parse_args()
    if args.ray_startup_repair and not args.full_environment:
        raise ValueError("Ray startup repair requires full-environment validation")
    if args.state_diagnostics and not args.ray_startup_repair:
        raise ValueError("diagnostics requires repaired startup validation")
    manifest_path = (
        MANIFEST.with_name(
            "m4-eligible-live-duplicate-update-bootstrap-replacement-manifest.json"
        )
        if args.bootstrap_replacement
        else MANIFEST
    )
    manifest_path = args.manifest or manifest_path
    value = json.loads(manifest_path.read_bytes())
    spec = value["spec"]
    expected_name = "m4-eligible-live-duplicate-update-qualification" + (
        "-bootstrap-replacement" if args.bootstrap_replacement else ""
    )
    if args.full_environment:
        expected_name = (
            "m4-eligible-live-duplicate-update-qualification-full-environment"
        )
    if args.ray_startup_repair:
        expected_name = (
            "m4-eligible-live-duplicate-update-qualification-ray-startup-repair"
        )
    if args.state_diagnostics:
        expected_name = (
            "m4-eligible-live-duplicate-update-qualification-state-diagnostics"
        )
    if spec["name"] != expected_name or spec["time_limit"] != 14400:
        raise ValueError("workload identity/runtime changed")
    script = spec["script"].replace("{{", "{").replace("}}", "}")
    if args.ray_startup_repair:
        validate_repaired_shell(script)
    if (
        args.state_diagnostics
        and script.count('--state-diagnostics-dir "$ASSETS"') != 1
    ):
        raise ValueError("diagnostic persistence directory missing or duplicated")
    if args.full_environment:
        interpreter = "/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python"
        if f"readonly PYTHON={interpreter}" not in script:
            raise ValueError("proven MCORE interpreter not selected")
        if script.index("-m research.m4_eligible_replay.preflight") > script.index(
            "-m research.m4_eligible_replay.transport"
        ):
            raise ValueError("environment gate is after checkpoint retrieval")
    if (
        "readonly CAPSULE=/workspace/m4-duplicate-update-input-capsule" not in script
        or 'cp "$CAPSULE/transport-receipt.json" "$ASSETS/transport-receipt.json"'
        not in script
    ):
        raise ValueError("checkpoint transport boundary changed")
    if (
        "--job-token" not in script
        or "--replicate" not in script
        or "-maxdepth 1" not in script
    ):
        raise ValueError("required invocation or checksum boundary missing")
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    extract = script.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    with tempfile.TemporaryDirectory(prefix="m4-duplicate-clean-room-") as directory:
        root = Path(directory) / "repo"
        subprocess.run(
            [sys.executable, "-", str(root)], input=extract, text=True, check=True
        )
        env = {**os.environ, "PYTHONPATH": str(root)}
        tests = [
            sys.executable,
            "-m",
            "unittest",
            "research.m4_eligible_replay.test_contracts",
        ]
        if args.bootstrap_replacement or args.full_environment:
            tests.append("research.m4_eligible_replay.test_bootstrap")
        if args.full_environment:
            tests.append("research.m4_eligible_replay.test_preflight")
        if args.ray_startup_repair:
            tests.append("research.m4_eligible_replay.test_ray_startup")
            subprocess.run(
                [sys.executable, "-m", "research.m4_eligible_replay.ray_startup"],
                cwd=root,
                env=env,
                check=True,
            )
        if args.state_diagnostics:
            tests.append("research.m4_eligible_replay.test_state_diagnostics")
        test_run = subprocess.run(
            [
                sys.executable,
                "-c",
                """import json,sys,unittest
suite=unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:])
result=unittest.TextTestRunner(verbosity=2).run(suite)
print(json.dumps({'tests_run':result.testsRun,'skipped':len(result.skipped)}))
raise SystemExit(0 if result.wasSuccessful() else 1)
""",
                *tests[3:],
            ],
            cwd=root,
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        print(test_run.stdout)
        test_summary = json.loads(test_run.stdout.splitlines()[-1])
        subprocess.run(
            [sys.executable, "-m", "compileall", "-q", "research/m4_eligible_replay"],
            cwd=root,
            env=env,
            check=True,
        )
    print(
        json.dumps(
            {
                "status": "PASS_CLEAN_ROOM_STATIC_PACKAGE",
                "manifest_sha256": digest_file(manifest_path),
                "cpu_contract_tests": test_summary["tests_run"],
                "cpu_tests_skipped": test_summary["skipped"],
                "gpu_replay_not_run_locally": True,
                "new_eos_launches": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
