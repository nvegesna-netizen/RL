"""Build one corrected, no-training preflight from the failed frozen manifest."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import subprocess

from research.m4_eligible_replay.contracts import canonical_json, digest_file

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
REFERENCE_HASH = "13f4c719e92bee1d29bead8f1b1d816fc152ad45e38b29587a180ba3082f474b"
NAME = "m4-v2-wrapper-repair-preflight-pf01"


def build(output: Path) -> dict:
    reference = (
        ROOT
        / "session/20260914_m4_opportunity_control/m4-v2-capture-frozen-82708a4e4/v2a01.json"
    )
    if digest_file(reference) != REFERENCE_HASH:
        raise ValueError("frozen failed reference manifest changed")
    value = json.loads(reference.read_bytes())
    script = value["spec"]["script"].format(assets_dir="{assets_dir}")
    old_call = '"$PYTHON" - "$PLAN" "$EXECUTION_LOCK" "$IDENTITY" "$SEED" <<\'PY\''
    old_update = "capture.update(trajectory_identity=identity,trajectory_seed=seed,progress_timeout_seconds=1200,\n    suppress_capture_for_liveness_qualification=False)"
    if script.count(old_call) != 1 or script.count(old_update) != 1:
        raise ValueError("failed authority block differs from diagnosed source")
    script = script.replace(
        old_call, old_call.replace('"$SEED"', '"$SEED" "$WATCHDOG"')
    )
    script = script.replace(
        old_update,
        old_update.replace(
            "suppress_capture_for_liveness_qualification=False",
            "liveness_diagnostic_path=sys.argv[5],suppress_capture_for_liveness_qualification=False",
        ),
    )
    training_boundary = 'set +e\n"$PYTHON" examples/run_grpo_single_controller.py'
    if script.count(training_boundary) != 1:
        raise ValueError("native training entrypoint boundary ambiguous")
    script = script.split(training_boundary, 1)[0]
    protocol = REPORT / "eligible_live_v2_wrapper_repair_preflight_protocol.json"
    authority = json.loads(protocol.read_bytes())
    if (
        authority["maximum_launcher_invocations"] != 1
        or authority["gates"]["actual_training_entrypoint_invocations"] != 0
    ):
        raise ValueError("bounded preflight authority changed")
    payload = base64.b64encode(protocol.read_bytes()).decode()
    digest = digest_file(protocol)
    script += """"$PYTHON" - "$ASSETS" <<'PY'
import base64,hashlib,json,re,sys
from pathlib import Path
root=Path(sys.argv[1]); protocol=base64.b64decode("@PROTOCOL@",validate=True)
assert hashlib.sha256(protocol).hexdigest()=="@HASH@"
assert not (root/"m4-eligible-live-capsule").exists()
assert not (root/"m4-eligible-live-v2a01-run.log").exists()
assert not (root/"m4-eligible-live-v2a01-watchdog.json").exists()
test_log=re.sub(r"\\x1b\\[[0-9;]*m","",(root/"m4-eligible-live-v2a01-tests.log").read_text())
passed=[int(n) for n in re.findall(r"(\\d+) passed",test_log)]
assert len(passed)==3 and all(n>0 for n in passed),passed
assert not re.search(r"\\d+ skipped",test_log)
result={"schema":"m4-v2-wrapper-preflight-result-v1","status":"PASS_NATIVE_CONFIG_AND_UNIT_TEST_PREFLIGHT",
"protocol_sha256":"@HASH@","configuration_identity":"v2a01","configuration_seed":20261201,
"full_native_master_config_validated":True,"native_test_commands_succeeded":3,"native_tests_passed":sum(passed),"native_tests_skipped":0,
"training_entrypoint_invocations":0,"scientific_capsules":0,"paired_replays":0,"scientific_outcomes_opened":False}
target=root/"wrapper-preflight-result.json"
with target.open("xb") as stream:stream.write((json.dumps(result,sort_keys=True,separators=(",",":"))+"\\n").encode())
files=[root/"m4-eligible-live-v2a01-tests.log",target]
with (root/"artifacts.sha256").open("x") as stream:
 for path in files:stream.write(hashlib.sha256(path.read_bytes()).hexdigest()+"  "+str(path)+"\\n")
print(result["status"])
PY
""".replace("@PROTOCOL@", payload).replace("@HASH@", digest)
    if "examples/run_grpo_single_controller.py" in script:
        raise ValueError("preflight accidentally retains actual training entrypoint")
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    value["spec"]["name"] = NAME
    value["spec"]["script"] = (
        script.replace("{", "{{")
        .replace("}", "}}")
        .replace("{{assets_dir}}", "{assets_dir}")
    )
    with output.open("xb") as stream:
        stream.write(canonical_json(value))
    return {
        "schema": "m4-v2-wrapper-preflight-package-v1",
        "reference_manifest_sha256": REFERENCE_HASH,
        "manifest_sha256": digest_file(output),
        "protocol_sha256": digest,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "training_entrypoint_present": False,
        "allocated_seconds": 14400,
        "queue_deadline": None,
        "old_cohort_guards_reset": False,
        "launches": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
