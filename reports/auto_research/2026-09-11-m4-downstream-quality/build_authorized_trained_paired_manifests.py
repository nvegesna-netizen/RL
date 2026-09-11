#!/usr/bin/env python3
"""Bind one frozen downstream-quality candidate to 32-run acquisition authority."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path


LOCAL_AUTH_SHA = "7b5830d5347125ebcf73d6df6da86ad1b181ef634f230687a516d504cf0fdbc3"
LOCAL_VALIDATION_SHA = "2ac1ea287b7a2386c4a38806fcd22cb6f8d230bbca9e17a589a1ce99f7814c92"
AUTHORIZED_AUTH_SHA = "5a64ec577b2acd14f6ddb65e9249abf9718d8655a1f9ff76aea2eed85ae24e30"
PROTOCOL_SHA = "dbe7a4f7d1938ef43d24e536ccf9dd57110a01bab5e6cb511980218e472ef2c0"
RUN_MANIFEST_SHA = "a461e83c7dbccdefa1c5c779062e709f34bfc502639a3e7de663a717fefa93bf"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--local-authorization", type=Path, required=True)
    parser.add_argument("--local-validation", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path, expected in (
        (args.local_authorization, LOCAL_AUTH_SHA),
        (args.local_validation, LOCAL_VALIDATION_SHA),
        (args.authorization, AUTHORIZED_AUTH_SHA),
        (args.run_manifest, RUN_MANIFEST_SHA),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"authorization-bound input moved: {path.name}")
    validation = json.loads(args.local_validation.read_bytes())
    authorization = json.loads(args.authorization.read_bytes())
    run_manifest = json.loads(args.run_manifest.read_bytes())
    runs = {f"{run['block_id']}_{run['regime']}": run for run in run_manifest["runs"]}
    if args.identity not in runs or authorization["identities"] != list(runs):
        raise RuntimeError("authorized run identities differ")
    run = runs[args.identity]
    if sha256(args.candidate) != validation["candidate_sha256"][args.identity]:
        raise RuntimeError("frozen local candidate moved")
    required_true = (
        "local_authorization_bound_package_build_authorized",
        "model_weight_access_authorized",
        "optimizer_initialization_authorized",
        "training_authorized",
        "eos_submission_authorized",
        "scientific_acquisition_authorized",
        "submit_all_32_before_inspecting_any_outcome",
    )
    if (
        authorization["schema"]
        != "m4-downstream-quality-trained-paired-32-run-acquisition-authorization-v1"
        or authorization["protocol_sha256"] != PROTOCOL_SHA
        or authorization["run_manifest_sha256"] != RUN_MANIFEST_SHA
        or authorization["trainer_steps_each"] != 448
        or authorization["terminal_trainer_version_each"] != 448
        or authorization["submission_attempt_limit_each"] != 1
        or authorization["required_launcher"] != "runllm.py --no_wait"
        or authorization["wall_clock_cap_hours_each"] != 4.0
        or authorization["h100_gpu_hour_cap_each"] != 8.0
        or not all(authorization[key] for key in required_true)
        or authorization["qualification_authorized"]
        or authorization["outcome_inspection_before_all_32_terminal_and_authenticated"]
        or authorization["partial_result_analysis_authorized"]
        or authorization["automatic_retry"]
        or authorization["automatic_replacement"]
        or authorization["automatic_extension"]
    ):
        raise RuntimeError("32-run acquisition authority differs")

    manifest = json.loads(args.candidate.read_bytes())
    script = manifest["spec"]["script"]
    old_payload = base64.b64encode(args.local_authorization.read_bytes()).decode("ascii")
    new_payload = base64.b64encode(args.authorization.read_bytes()).decode("ascii")
    for old, new, label in (
        (old_payload, new_payload, "authority payload"),
        (LOCAL_AUTH_SHA, AUTHORIZED_AUTH_SHA, "authority hash"),
    ):
        if script.count(old) != 1:
            raise RuntimeError(f"local candidate {label} surface differs")
        script = script.replace(old, new)
    start_marker = '"$PYTHON" - "$PROTOCOL" "$RUN_MANIFEST" "$AUTH" <<\'PY\'\n'
    start = script.index(start_marker)
    end = script.index('\nPY\nmkdir -p "$RUN_REPO"', start) + 3
    tag = args.identity.upper()
    authority_block = f'''"$PYTHON" - "$PROTOCOL" "$RUN_MANIFEST" "$AUTH" <<'PY'
import hashlib,json,sys
from pathlib import Path
protocol_path,manifest_path,auth_path=map(Path,sys.argv[1:])
p=json.loads(protocol_path.read_bytes()); m=json.loads(manifest_path.read_bytes()); a=json.loads(auth_path.read_bytes())
assert hashlib.sha256(protocol_path.read_bytes()).hexdigest()=="{PROTOCOL_SHA}"
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()=="{RUN_MANIFEST_SHA}"
assert a["schema"]=="m4-downstream-quality-trained-paired-32-run-acquisition-authorization-v1"
assert a["scope"]=="exactly_32_preregistered_downstream_quality_scientific_acquisitions_one_attempt_each"
assert a["identities"]==[f"{{{{r['block_id']}}}}_{{{{r['regime']}}}}" for r in m["runs"]]
assert a["trainer_steps_each"]==448 and a["terminal_trainer_version_each"]==448
assert a["submission_attempt_limit_each"]==1 and a["required_launcher"]=="runllm.py --no_wait"
required=("model_weight_access_authorized","optimizer_initialization_authorized","training_authorized","eos_submission_authorized","scientific_acquisition_authorized","submit_all_32_before_inspecting_any_outcome")
assert all(a[k] for k in required)
assert not a["qualification_authorized"] and not a["outcome_inspection_before_all_32_terminal_and_authenticated"] and not a["partial_result_analysis_authorized"]
assert not a["automatic_retry"] and not a["automatic_replacement"] and not a["automatic_extension"]
r=next(r for r in m["runs"] if r["block_id"]=="{run['block_id']}" and r["regime"]=="{run['regime']}")
assert r["config_sha256"]=="{run['config_sha256']}" and r["training_seed"]=={run['training_seed']}
assert r["assignment_seed"]=={run['assignment_seed']} and r["assignment_domain"]=="{run['assignment_domain']}"
print("M4_DOWNSTREAM_QUALITY_{tag}_ACQUISITION_AUTHORITY_PASS")
PY'''
    script = script[:start] + authority_block + script[end:]
    if "M4_DOWNSTREAM_QUALITY_LOCAL_PACKAGE_NO_EXECUTION_AUTHORITY" in script:
        raise RuntimeError("local no-execution block remains")
    stem = args.identity.replace("_", "-")
    manifest["spec"]["name"] = f"m4-downstream-quality-{stem}-acquisition-authorized"
    manifest["spec"]["time_limit"] = 14400
    manifest["spec"]["script"] = script
    args.output.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    print(args.output, sha256(args.output))


if __name__ == "__main__":
    main()
