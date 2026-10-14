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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-replacement", action="store_true")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
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
    if spec["name"] != expected_name or spec["time_limit"] != 14400:
        raise ValueError("workload identity/runtime changed")
    script = spec["script"].replace("{{", "{").replace("}}", "}")
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
        if args.bootstrap_replacement:
            tests.append("research.m4_eligible_replay.test_bootstrap")
        subprocess.run(
            tests + ["-v"],
            cwd=root,
            env=env,
            check=True,
        )
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
                "cpu_contract_tests": 10 if args.bootstrap_replacement else 8,
                "gpu_replay_not_run_locally": True,
                "new_eos_launches": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
