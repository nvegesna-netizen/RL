"""Freeze all 32 admitted arms atop the exact CUDA-qualified engineering prefix."""

from __future__ import annotations

import argparse
import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

from research.m4_eligible_replay.contracts import canonical_json, digest_file
from research.m4_paired_replay.capture_authentication import verify_admission_bundle
from research.m4_paired_replay.contracts import ARMS, capture_gate

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
SESSION = ROOT / "session/20260914_m4_opportunity_control"
REFERENCE = SESSION / "m4-synthetic-paired-loss-config-repaired-ready-manifest.json"
REFERENCE_SHA = "d2dfd5d0e0d6b94028316974e0777aeb1a72f7058d1f9fd7762f462c13e58c74"
CHANGED_CONTROL_FILES = {
    "research/m4_paired_replay/capture_authentication.py",
    "research/m4_paired_replay/test_integration.py",
}


def put(path: Path, value: dict | list) -> None:
    with path.open("xb") as stream:
        stream.write(canonical_json(value))


def fixed_arms(gate: dict) -> list[dict]:
    if gate["valid_capsules"] != 16:
        raise ValueError(
            "this execution is locked to exactly sixteen admitted capsules"
        )
    return [
        {
            "identity": f"{row['identity']}-{'base' if arm == ARMS[0] else 'shield'}",
            "capture_identity": row["identity"],
            "seed": row["seed"],
            "arm": arm,
            "manifest_sha256": row["manifest_sha256"],
            "source_eos_job_id": row["eos_job_id"],
        }
        for row in gate["admitted"]
        for arm in ARMS
    ]


def build(output: Path) -> dict:
    if digest_file(REFERENCE) != REFERENCE_SHA:
        raise ValueError("qualified engineering manifest changed")
    protocol_path = REPORT / "paired_fresh_execution_protocol.json"
    protocol = json.loads(protocol_path.read_bytes())
    plan_path = REPORT / "eligible_live_acquisition_v2_plan.json"
    captures_path = (
        REPORT / "eligible_live_v2_capture_retry_r1_authenticated_classifications.json"
    )
    gate_path = REPORT / "eligible_live_v2_capture_retry_r1_capture_gate.json"
    if any(
        digest_file(path) != protocol[key]
        for path, key in [
            (plan_path, "scientific_plan_file_sha256"),
            (captures_path, "classifications_file_sha256"),
            (gate_path, "capture_gate_file_sha256"),
        ]
    ):
        raise ValueError("frozen capture admission moved")
    plan = json.loads(plan_path.read_bytes())
    captures = json.loads(captures_path.read_bytes())
    evidence = SESSION / "m4-v2-capture-retry-r1-terminal-authentication"
    verify_admission_bundle(plan, captures, evidence)
    gate = capture_gate(plan, captures)
    arms = fixed_arms(gate)
    if (
        len(arms) != 32
        or protocol["arm_invocations"] != 32
        or protocol["automatic_retry"]
        or protocol["private_key_uploaded"]
    ):
        raise ValueError("fixed paired execution boundary changed")
    engineering = json.loads(
        (REPORT / "paired_engineering_terminal_result.json").read_bytes()
    )
    if (
        not engineering["qualification_passed"]
        or engineering["manifest_sha256"] != REFERENCE_SHA
    ):
        raise ValueError("native paired engineering qualification required")
    base = json.loads(REFERENCE.read_bytes())
    old = base["spec"]["script"].format(assets_dir="{assets_dir}")
    prefix = old.split('"$PYTHON" -m unittest', 1)[0]
    if prefix.count("export M4_PAIRED_SYNTHETIC_ENGINEERING=1") != 1:
        raise ValueError("synthetic-mode boundary differs")
    prefix = prefix.replace(
        "export M4_PAIRED_SYNTHETIC_ENGINEERING=1",
        "unset M4_PAIRED_SYNTHETIC_ENGINEERING",
    )
    blocks = [b.split("\nPY\n", 1)[0] for b in old.split("<<'PY'\n")[1:]]
    assignment = next(
        n
        for n in ast.parse(blocks[1]).body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "overlay" for t in n.targets)
    )
    accepted_overlay = json.loads(ast.literal_eval(assignment.value.args[0]))
    for name, row in accepted_overlay.items():
        if (
            name not in CHANGED_CONTROL_FILES
            and digest_file(ROOT / name) != row["sha256"]
        ):
            raise ValueError(f"qualified replay source changed: {name}")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    lock = {
        "schema": "m4-fixed-fresh-paired-lock-v1",
        "status": "REVIEWED_FRESH_PAIRED_EXECUTION_READY",
        "execution_source_commit": source_commit,
        "plan_file_sha256": digest_file(plan_path),
        "captures_file_sha256": digest_file(captures_path),
        "capture_gate_file_sha256": digest_file(gate_path),
        "protocol_sha256": digest_file(protocol_path),
        "fixed_arms": arms,
        "qualification_manifest_sha256": REFERENCE_SHA,
        "engineering_result_sha256": digest_file(
            REPORT / "paired_engineering_terminal_result.json"
        ),
        "qualified_actor_modules": json.loads(
            (REPORT / "paired_integration_protocol.json").read_bytes()
        )["actor_modules"],
        "automatic_retry": False,
        "primary_outcomes_opened": False,
        "required_receipts_before_outcome_open": 32,
    }
    output.mkdir(parents=True, exist_ok=False)
    put(output / "execution-lock.json", lock)
    overlay_names = sorted(
        CHANGED_CONTROL_FILES
        | {
            "research/m4_paired_replay/fresh_arm_job.py",
            "research/m4_paired_replay/test_fresh_arm_job.py",
            str(Path(__file__).relative_to(ROOT)),
            str(gate_path.relative_to(ROOT)),
        }
    )
    overlay = {
        name: {
            "sha256": digest_file(ROOT / name),
            "base64": base64.b64encode((ROOT / name).read_bytes()).decode(),
        }
        for name in overlay_names
    }
    inputs = {
        "plan.json": plan_path.read_bytes(),
        "captures.json": captures_path.read_bytes(),
        "lock.json": (output / "execution-lock.json").read_bytes(),
    }
    inputs.update(
        {
            "evidence/" + p.name: p.read_bytes()
            for p in sorted(evidence.glob("v2a*.json"))
        }
    )
    encoded_inputs = {
        name: {
            "sha256": hashlib.sha256(data).hexdigest(),
            "base64": base64.b64encode(data).decode(),
        }
        for name, data in inputs.items()
    }
    extract = r""""$PYTHON" - "$RUN_REPO" <<'PY'
import base64,hashlib,sys
from pathlib import Path,PurePosixPath
root=Path(sys.argv[1]); overlay=@OVERLAY@; inputs=@INPUTS@
for name,row in overlay.items():
 data=base64.b64decode(row["base64"],validate=True);assert hashlib.sha256(data).hexdigest()==row["sha256"]
 path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
for name,row in inputs.items():
 relative=PurePosixPath(name);assert not relative.is_absolute() and ".." not in relative.parts
 data=base64.b64decode(row["base64"],validate=True);assert hashlib.sha256(data).hexdigest()==row["sha256"]
 path=root/"paired-inputs"/name;path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("xb") as stream:stream.write(data)
print("PASS_FIXED_SIXTEEN_PAIRED_INPUTS")
PY
""".replace("@OVERLAY@", repr(overlay)).replace("@INPUTS@", repr(encoded_inputs))
    test_line = next(
        s for s in old.splitlines() if s.startswith('"$PYTHON" -m unittest ')
    )
    test_line = "M4_PAIRED_SYNTHETIC_ENGINEERING=1 " + test_line.replace(
        " -v >", " research.m4_paired_replay.test_fresh_arm_job -v >"
    )
    script = (
        prefix
        + extract
        + '@CANDIDATE_INPUT@\ntest -n "${CI_JOB_TOKEN:-}"\n'
        + test_line
        + "\n"
    )
    script += r""""$PYTHON" - "$ASSETS" <<'PY'
import json,re,sys
from pathlib import Path
root=Path(sys.argv[1]);text=(root/"tests.log").read_text()
counts=re.findall(r"Ran (\d+) tests?",text)
assert counts==["213"],counts
assert re.search(r"^OK\s*$",text,re.M) and not re.search(r"skipped=|\bFAILED\b",text)
with (root/"qualification.json").open("x") as stream:json.dump({"status":"PASS_NATIVE_REPLAY_TESTS","run":213,"skipped":0,"primary_outcomes_opened":False},stream,sort_keys=True)
print("PASS_NATIVE_REPLAY_TESTS_BEFORE_CHECKPOINT_IO")
PY
test -z "${M4_PAIRED_SYNTHETIC_ENGINEERING:-}"
"""
    script += r"""set +e
"$PYTHON" -m research.m4_paired_replay.fresh_arm_job --inputs "$RUN_REPO/paired-inputs" --public "$ASSETS"
readonly ARM_RC=$?
set -e
"$PYTHON" - "$ASSETS" <<'PY'
import hashlib,sys
from pathlib import Path
root=Path(sys.argv[1]);names=["tests.log","qualification.json","completion.json","failure.json","transport-receipt.json","arm-receipt.json","encrypted-envelope.json","encrypted-receipt.json","failure-envelope.json","failure-envelope-receipt.json"]
with (root/"artifacts.sha256").open("x") as stream:
 for name in names:
  path=root/name
  if path.is_file():stream.write(hashlib.sha256(path.read_bytes()).hexdigest()+"  "+str(path)+"\n")
PY
exit "$ARM_RC"
"""
    candidates = []
    for i, row in enumerate(arms):
        payload = base64.b64encode(canonical_json(row)).decode()
        command = (
            '"$PYTHON" - "$RUN_REPO/paired-inputs/candidate.json" <<\'PY\'\nimport base64,sys\nfrom pathlib import Path\nwith Path(sys.argv[1]).open("xb") as stream:stream.write(base64.b64decode("'
            + payload
            + '",validate=True))\nPY\n'
        )
        if script.count("@CANDIDATE_INPUT@") != 1:
            raise ValueError("candidate input boundary ambiguous")
        rendered = script.replace("@CANDIDATE_INPUT@", command)
        if i == 0:
            subprocess.run(["bash", "-n"], input=rendered, text=True, check=True)
            for block in rendered.split("<<'PY'\n")[1:]:
                ast.parse(block.split("\nPY\n", 1)[0])
        manifest = {
            **base,
            "spec": {
                **base["spec"],
                "name": "m4-paired-causal-" + row["identity"],
                "time_limit": 14400,
                "script": rendered.replace("{", "{{")
                .replace("}", "}}")
                .replace("{{assets_dir}}", "{assets_dir}"),
            },
        }
        path = output / f"{row['identity']}.json"
        put(path, manifest)
        budget = 3 * len(gzip.compress(canonical_json(manifest), compresslevel=1)) + (
            4 << 20
        )
        if budget >= 50 << 20:
            raise ValueError("generator compressed-copy budget exceeds limit")
        candidates.append(
            row
            | {
                "manifest": str(path.relative_to(ROOT)),
                "sha256": digest_file(path),
                "generator_compressed_copy_bound": budget,
            }
        )
    package = {
        "schema": "m4-fixed-fresh-paired-package-v1",
        "execution_lock_sha256": digest_file(output / "execution-lock.json"),
        "source_commit": source_commit,
        "protocol_sha256": digest_file(protocol_path),
        "candidates": candidates,
        "qualified_overlay_sha256": {
            name: row["sha256"] for name, row in accepted_overlay.items()
        },
        "control_overlay_sha256": {
            name: row["sha256"] for name, row in overlay.items()
        },
        "scientific_updates": 32,
        "zero_update_gradient_processes": 64,
        "primary_outcomes_opened": False,
    }
    put(output / "package.json", package)
    return package


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.output.resolve())
    print(
        json.dumps(
            {"candidates": len(result["candidates"]), "primary_outcomes_opened": False}
        )
    )
