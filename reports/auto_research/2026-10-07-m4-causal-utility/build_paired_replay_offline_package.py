"""Freeze an offline overlay and test it atop the exact accepted source package."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

from research.m4_eligible_replay.contracts import canonical_json, digest_file
from research.m4_eligible_replay.deterministic_backward import MARKER, REQUIRED_ENV


ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
BASE_HASH = "de2ee7b5c1b4437b1d6ed53b9f930a3bf96dca471f31aee780aea3844fab238e"


def build(*, output: Path) -> dict:
    design = subprocess.run(
        [
            sys.executable,
            str(REPORT / "validate_eligible_live_acquisition_v2_design.py"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    design_result = json.loads(design.stdout)
    base = (
        ROOT
        / "session/20260914_m4_opportunity_control/m4-eligible-live-minimal-observer-verified-manifest.json"
    )
    if digest_file(base) != BASE_HASH:
        raise ValueError("accepted source manifest changed")
    previous = json.loads(
        (REPORT / "eligible_live_minimal_observer_package_receipt.json").read_bytes()
    )
    for name, expected in previous["overlay_sha256"].items():
        if digest_file(ROOT / name) != expected:
            raise ValueError(f"accepted qualification source changed: {name}")
    names = [
        str(p.relative_to(ROOT))
        for p in sorted((ROOT / "research/m4_paired_replay").glob("*.py"))
    ]
    names += [
        str((REPORT / n).relative_to(ROOT))
        for n in (
            "eligible_live_acquisition_v2_plan.json",
            "eligible_live_decision_causal_protocol.json",
            "eligible_live_minimal_observer_terminal_result.json",
            "paired_replay_runtime_amendment.json",
        )
    ]
    overlay = {
        name: {
            "sha256": digest_file(ROOT / name),
            "base64": base64.b64encode((ROOT / name).read_bytes()).decode(),
        }
        for name in names
    }
    value = {
        "schema": "m4-offline-paired-overlay-v1",
        "base_manifest_sha256": BASE_HASH,
        "gpu_launch_enabled": False,
        "overlay": overlay,
    }
    with output.open("xb") as stream:
        stream.write(canonical_json(value))
    # Reload the exact frozen bytes rather than testing the working checkout.
    frozen = json.loads(output.read_bytes())
    script = (
        json.loads(base.read_bytes())["spec"]["script"]
        .replace("{{", "{")
        .replace("}}", "}")
    )
    extract = script.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    test_line = next(
        line for line in script.splitlines() if '"$PYTHON" -m unittest ' in line
    )
    modules = [
        word
        for word in shlex.split(test_line)
        if word.startswith("research.m4_eligible_replay.test_")
    ]
    modules.append("research.m4_paired_replay.test_package")
    with tempfile.TemporaryDirectory(prefix="m4-paired-clean-room-") as directory:
        root = Path(directory) / "repo"
        subprocess.run(
            [sys.executable, "-", str(root)],
            input=extract,
            text=True,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        for name, row in frozen["overlay"].items():
            data = base64.b64decode(row["base64"])
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("offline overlay failed authentication")
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            # Generated clean-room files are never used to edit the worktree.
            path.write_bytes(data)
        env = {**os.environ, "PYTHONPATH": str(root), MARKER: "1", **REQUIRED_ENV}
        code = """import json,sys,unittest
suite=unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:])
result=unittest.TextTestRunner(verbosity=1).run(suite)
print(json.dumps({'run':result.testsRun,'skipped':len(result.skipped),'failures':len(result.failures),'errors':len(result.errors)}))
raise SystemExit(0 if result.wasSuccessful() else 1)
"""
        result = subprocess.run(
            [sys.executable, "-c", code, *modules],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        summary = json.loads(result.stdout.splitlines()[-1])
        subprocess.run(
            [sys.executable, "-m", "compileall", "-q", "research/m4_paired_replay"],
            cwd=root,
            env=env,
            check=True,
        )
    return {
        "schema": "m4-offline-paired-package-receipt-v1",
        "status": "PASS_OFFLINE_PACKAGE_ONLY",
        "recorded_at": "2026-10-18",
        "base_manifest_sha256": BASE_HASH,
        "overlay_sha256": digest_file(output),
        "source_sha256": {name: row["sha256"] for name, row in overlay.items()},
        "clean_room_tests": summary,
        "qualified_source_overlay_unchanged": True,
        "frozen_design_validation": {
            "status": design_result["status"],
            "plan_sha256": design_result["plan_sha256"],
            "runtime_files_checked": design_result["qualified_runtime_files_checked"],
        },
        "synthetic_test_outcomes_only": True,
        "new_eos_launches": 0,
        "gpu_two_action_replay_qualified": False,
        "gpu_launch_ready": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(output=args.output), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
