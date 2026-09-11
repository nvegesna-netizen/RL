#!/usr/bin/env python3
"""Build the evaluation-metadata-repaired no-training EOS preflight v8."""

from __future__ import annotations
import argparse, base64, hashlib, json
from pathlib import Path

V7M = "f1bcc8751ce969aef42531049d7fa1a1daea3a9b4ee723502632fac3b61226d5"
V7A = "c22f3bf90a0493b2c892805ccf0f2b67af9575f8ede24911dc6228566bc1d9c2"
V7T = "d388f31490977f929bf11b3b9ca49d0f5c1b183c0cedcdf6fa4a82caa78db004"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def one(s, a, b, label):
    if s.count(a) != 1:
        raise RuntimeError(f"expected one {label}")
    return s.replace(a, b)


def main():
    p = argparse.ArgumentParser()
    for n in (
        "v7_manifest",
        "v7_authorization",
        "v7_terminal_result",
        "v8_authorization",
        "output",
    ):
        p.add_argument("--" + n.replace("_", "-"), type=Path, required=True)
    a = p.parse_args()
    if (sha(a.v7_manifest), sha(a.v7_authorization), sha(a.v7_terminal_result)) != (
        V7M,
        V7A,
        V7T,
    ):
        raise RuntimeError("v7 input moved")
    auth = json.loads(a.v8_authorization.read_bytes())
    blocked = (
        "optimizer_initialization_authorized",
        "training_authorized",
        "qualification_authorized",
        "pilot_authorized",
        "scientific_acquisition_authorized",
        "automatic_retry",
        "automatic_extension",
    )
    if (
        auth["submission_attempt_limit"] != 1
        or not auth["eos_submission_authorized"]
        or auth["optimizer_steps_allowed"] != 0
        or any(auth[k] for k in blocked)
    ):
        raise RuntimeError("v8 authority differs")
    m = json.loads(a.v7_manifest.read_bytes())
    s = m["spec"]["script"]
    s = one(
        s,
        base64.b64encode(a.v7_authorization.read_bytes()).decode(),
        base64.b64encode(a.v8_authorization.read_bytes()).decode(),
        "authority payload",
    )
    s = one(s, V7A, sha(a.v8_authorization), "authority hash")
    start = s.index('"$PYTHON" - "$AUTH" <<\'PY\'\n')
    end = s.index("\nPY\nmkdir -p", start) + 3
    block = """"$PYTHON" - "$AUTH" <<'PY'
import json,sys
from pathlib import Path
a=json.loads(Path(sys.argv[1]).read_bytes())
assert a["schema"]=="m4-downstream-quality-no-training-eos-preflight-v8-authorization-v1"
assert a["scope"]=="exactly_one_evaluation_dataset_metadata_repaired_credential_free_downstream_quality_no_training_eos_preflight_v8"
assert a["user_authorization"]=="do the repair. I authorize launches on eos"
assert a["predecessor_terminal_result_sha256"]=="d388f31490977f929bf11b3b9ca49d0f5c1b183c0cedcdf6fa4a82caa78db004"
assert a["required_repair"]=="add_top_level_openmath_dataset_name_to_evaluation_only_master_config"
assert a["required_launcher"]=="runllm.py --no_wait" and a["submission_attempt_limit"]==1
assert a["eos_submission_authorized"] is True and a["model_weight_access_authorized"] is True and a["optimizer_steps_allowed"]==0
blocked=("optimizer_initialization_authorized","training_authorized","qualification_authorized","pilot_authorized","scientific_acquisition_authorized","automatic_retry","automatic_extension")
assert not any(a[k] for k in blocked)
print("M4_DOWNSTREAM_QUALITY_EOS_PREFLIGHT_V8_AUTHORITY_PASS")
PY"""
    s = s[:start] + block + s[end:]
    old = 'data={{"max_input_seq_length":2048,"shuffle":False'
    new = 'data={{"dataset_name":"OpenMathInstruct-2","max_input_seq_length":2048,"shuffle":False'
    s = one(s, old, new, "evaluation dataset metadata repair")
    m["spec"]["script"] = s
    m["spec"]["name"] = "m4-downstream-quality-no-training-preflight-v8"
    a.output.write_text(json.dumps(m, separators=(",", ":")) + "\n")
    print(sha(a.output))


if __name__ == "__main__":
    main()
