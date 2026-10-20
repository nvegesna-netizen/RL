"""Package one excluded real-buffer qualification followed by a gated stall probe."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile

from research.m4_eligible_replay.contracts import canonical_json, digest_file

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
SESSION = ROOT / "session/20260914_m4_opportunity_control"
NAME = "m4-v2-watchdog-stall-diagnostic-d01"
CONTROLLER = "nemo_rl/algorithms/single_controller.py"
TESTS = "tests/unit/single_controller/test_single_controller.py"
REFERENCE_HASH = "e025aa13450425af5b02e7fc00d1407a74c93e1a8b7ab8782f79ec2423443f4b"

QUALIFICATION = r""""$PYTHON" - "$ASSETS" <<'PY'
import hashlib,json,re,sys
from pathlib import Path
root=Path(sys.argv[1])
log=re.sub(r"\x1b\[[0-9;]*m","",(root/"m4-eligible-live-v2a09-tests.log").read_text())
counts=[int(n) for n in re.findall(r"(\d+) passed",log)]
assert counts==[2,7,17],counts
assert not re.search(r"\d+ (?:skipped|failed|errors?)",log)
record={"schema":"m4-real-buffer-watchdog-qualification-v1","status":"PASS_BEFORE_DIAGNOSTIC_TRAINING","native_test_pass_counts":counts,"skipped":0,"protocol_sha256":"@PROTOCOL_HASH@","scientific_acquisition":False}
with (root/"watchdog-qualification.json").open("x") as stream:json.dump(record,stream,sort_keys=True);stream.write("\n")
print("M4_REAL_BUFFER_WATCHDOG_QUALIFICATION_PASS")
PY
"""

FINALIZE = r""""$PYTHON" - "$ASSETS" "$RUN_RC" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); rc=int(sys.argv[2])
qualification=json.loads((root/"watchdog-qualification.json").read_bytes())
assert qualification["status"]=="PASS_BEFORE_DIAGNOSTIC_TRAINING"
watchdog=root/"m4-eligible-live-v2a09-watchdog.json"
capsule=root/"m4-eligible-live-capsule/manifest.json"
status="UNEXPECTED_RUNTIME_FAILURE"
if watchdog.exists():
 record=json.loads(watchdog.read_bytes())
 assert record["schema"]=="eligible-live-progress-timeout-v1"
 assert record["trajectory_identity"]=="v2a09" and record["trajectory_seed"]==20261209
 assert record["timeout_seconds"]==1200 and record["seconds_without_learner_progress"]>=1200
 assert "buffer_end_weight_version_counts" in record["controller_state"]
 assert record["tasks"]["rollout_pump"]["stack"] and record["tasks"]["train_pump"]["stack"]
 assert rc!=0
 assert "eligible live capture made no learner progress" in (root/"m4-eligible-live-v2a09-run.log").read_text()
 status="WATCHDOG_DIAGNOSTIC_COLLECTED"
elif rc==0 and capsule.exists():
 record=json.loads(capsule.read_bytes())
 assert record["post_update_outcomes_opened"] is False
 status="EXCLUDED_CAPTURE_REACHED_BEFORE_STALL"
elif rc==0:
 status="DIAGNOSTIC_RUN_COMPLETED_WITHOUT_TIMEOUT"
result={"schema":"m4-excluded-stall-diagnostic-result-v1","status":status,"native_run_exit_code":rc,"protocol_sha256":"@PROTOCOL_HASH@","qualification_passed":True,"watchdog_record_present":watchdog.exists(),"excluded_capsule_present":capsule.exists(),"scientific_acquisition":False,"scientific_admission_permitted":False,"paired_replays":0,"primary_outcomes_opened":False}
with (root/"stall-diagnostic-result.json").open("x") as stream:json.dump(result,stream,sort_keys=True);stream.write("\n")
# Hash closed owned files only, never mutable JET wrapper output or a checkpoint.
names=["watchdog-qualification.json","stall-diagnostic-result.json","m4-eligible-live-v2a09-tests.log","m4-eligible-live-v2a09-run.log","m4-eligible-live-v2a09-watchdog.json","m4-eligible-live-v2a09-lifecycle.jsonl","m4-eligible-live-v2a09-opportunity.jsonl","m4-eligible-live-v2a09-observer-duty.json","m4-eligible-live-v2a09-oars-v2.jsonl"]
with (root/"diagnostic-artifacts.sha256").open("x") as stream:
 for name in names:
  path=root/name
  if path.is_file():stream.write(hashlib.sha256(path.read_bytes()).hexdigest()+"  "+str(path)+"\n")
print(status)
assert status!="UNEXPECTED_RUNTIME_FAILURE",result
PY
"""


def assert_diagnostic_only(before: bytes, after: bytes) -> None:
    """Require every controller AST node except the snapshot method unchanged."""
    trees = [ast.parse(data) for data in (before, after)]
    for tree in trees:
        controller = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "SingleControllerActor"
        )
        controller.body = [
            n
            for n in controller.body
            if not isinstance(n, ast.FunctionDef)
            or n.name != "_eligible_live_liveness_state"
        ]
    if ast.dump(trees[0], include_attributes=False) != ast.dump(
        trees[1], include_attributes=False
    ):
        raise ValueError("controller change exceeds diagnostic-only scope")


def build(output: Path) -> dict:
    reference = SESSION / "m4-v2-capture-retry-r1-ef6dc7ebc/v2a09.json"
    if digest_file(reference) != REFERENCE_HASH:
        raise ValueError("failed-seed reference manifest changed")
    protocol_path = REPORT / "watchdog_stall_diagnostic_protocol.json"
    protocol = json.loads(protocol_path.read_bytes())
    if (
        protocol["maximum_launcher_invocations"] != 1
        or protocol["scientific_acquisition"]
        or protocol["paired_replays"] != 0
    ):
        raise ValueError("excluded diagnostic authority changed")
    source = (
        SESSION
        / "paired-engineering-generator-failure/source-runtime-projection.tar.gz"
    )
    if digest_file(source) != protocol["prior_source_projection_sha256"]:
        raise ValueError("qualified base archive moved")
    patch = {}
    with tarfile.open(source) as archive:
        for name in (CONTROLLER, TESTS):
            before = archive.extractfile(name).read()
            after = (ROOT / name).read_bytes()
            if name == CONTROLLER:
                assert_diagnostic_only(before, after)
            patch[name] = {
                "before_sha256": hashlib.sha256(before).hexdigest(),
                "after_sha256": hashlib.sha256(after).hexdigest(),
                "base64": base64.b64encode(after).decode(),
            }
    protocol_hash = digest_file(protocol_path)
    patch_script = (
        """"$PYTHON" - "$RUN_REPO" <<'PY'
import base64,hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); patch=@PATCH@
protocol=base64.b64decode("@PROTOCOL@",validate=True)
assert hashlib.sha256(protocol).hexdigest()=="@PROTOCOL_HASH@"
for name,row in patch.items():
 path=root/name; assert hashlib.sha256(path.read_bytes()).hexdigest()==row["before_sha256"],name
 data=base64.b64decode(row["base64"],validate=True)
 assert hashlib.sha256(data).hexdigest()==row["after_sha256"],name
 path.write_bytes(data)
print("M4_DIAGNOSTIC_ONLY_NATIVE_PATCH_APPLIED")
PY
""".replace("@PATCH@", repr(patch))
        .replace("@PROTOCOL@", base64.b64encode(protocol_path.read_bytes()).decode())
        .replace("@PROTOCOL_HASH@", protocol_hash)
    )
    value = json.loads(reference.read_bytes())
    script = value["spec"]["script"].format(assets_dir="{assets_dir}")
    authority = '"$PYTHON" - "$PLAN" "$EXECUTION_LOCK" "$IDENTITY" "$SEED" "$WATCHDOG"'
    if script.count(authority) != 1:
        raise ValueError("native preflight boundary differs")
    script = script.replace(authority, patch_script + authority)
    boundary = 'set +e\n"$PYTHON" examples/run_grpo_single_controller.py'
    if script.count(boundary) != 1:
        raise ValueError("diagnostic training boundary differs")
    script = script.replace(
        boundary, QUALIFICATION.replace("@PROTOCOL_HASH@", protocol_hash) + boundary
    )
    finalizer = '"$PYTHON" -m research.m4_paired_replay.capture_finalize'
    if script.count(finalizer) != 1:
        raise ValueError("original scientific finalizer boundary differs")
    script = script.split(finalizer, 1)[0] + FINALIZE.replace(
        "@PROTOCOL_HASH@", protocol_hash
    )
    script = script.replace(
        "M4_V2_NATIVE_SOURCE_UNCHANGED", "M4_DIAGNOSTIC_BASE_ARCHIVE_AUTHENTICATED"
    ).replace("M4_V2_NATIVE_RUNTIME_AUTHORITY_PASS", "M4_STALL_DIAGNOSTIC_CONFIG_PASS")
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    for block in re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", script, re.S):
        ast.parse(block)
    value["spec"]["name"] = NAME
    value["spec"]["script"] = (
        script.replace("{", "{{")
        .replace("}", "}}")
        .replace("{{assets_dir}}", "{assets_dir}")
    )
    with output.open("xb") as stream:
        stream.write(canonical_json(value))
    return {
        "schema": "m4-watchdog-stall-diagnostic-package-v1",
        "manifest_sha256": digest_file(output),
        "reference_manifest_sha256": REFERENCE_HASH,
        "protocol_sha256": protocol_hash,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "native_patch_sha256": {
            name: {k: v for k, v in row.items() if k != "base64"}
            for name, row in patch.items()
        },
        "only_controller_diagnostic_method_changed": True,
        "scientific_acquisition": False,
        "training_requires_native_qualification": True,
        "maximum_launches": 1,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve()), sort_keys=True))
