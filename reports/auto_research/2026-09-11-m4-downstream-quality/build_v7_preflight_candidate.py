#!/usr/bin/env python3
import argparse, base64, hashlib, json
from pathlib import Path

V6M = "eb6b1e3de678672e03ec1d386b300fea1650f44f39b27f7a3ce4c9f50136abb9"
V6A = "05bc0a364cf1ee35f3f453f4a49951c839d135e4ce997866bd86d8639b18e60a"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def one(s, a, b):
    assert s.count(a) == 1
    return s.replace(a, b)


def main():
    p = argparse.ArgumentParser()
    for n in ("v6_manifest", "v6_authorization", "v7_authorization", "output"):
        p.add_argument("--" + n.replace("_", "-"), type=Path, required=True)
    a = p.parse_args()
    assert sha(a.v6_manifest) == V6M and sha(a.v6_authorization) == V6A
    auth = json.loads(a.v7_authorization.read_bytes())
    assert (
        auth["submission_attempt_limit"] == 1
        and auth["eos_submission_authorized"]
        and auth["optimizer_steps_allowed"] == 0
    )
    assert not any(
        auth[k]
        for k in (
            "optimizer_initialization_authorized",
            "training_authorized",
            "qualification_authorized",
            "pilot_authorized",
            "scientific_acquisition_authorized",
            "automatic_retry",
            "automatic_extension",
        )
    )
    m = json.loads(a.v6_manifest.read_bytes())
    s = m["spec"]["script"]
    s = one(
        s,
        base64.b64encode(a.v6_authorization.read_bytes()).decode(),
        base64.b64encode(a.v7_authorization.read_bytes()).decode(),
    )
    s = one(s, V6A, sha(a.v7_authorization))
    start = s.index('"$PYTHON" - "$AUTH" <<\'PY\'\n')
    end = s.index("\nPY\nmkdir -p", start) + 3
    block = """"$PYTHON" - "$AUTH" <<'PY'
import json,sys
from pathlib import Path
a=json.loads(Path(sys.argv[1]).read_bytes())
assert a["schema"]=="m4-downstream-quality-no-training-eos-preflight-v7-authorization-v1"
assert a["scope"]=="exactly_one_resubmission_after_local_v6_upload_interruption"
assert a["user_authorization"]=="I authorizr"
assert a["predecessor_eos_pipeline_created"] is False
assert a["required_launcher"]=="runllm.py --no_wait" and a["submission_attempt_limit"]==1
assert a["eos_submission_authorized"] is True and a["model_weight_access_authorized"] is True and a["optimizer_steps_allowed"]==0
blocked=("optimizer_initialization_authorized","training_authorized","qualification_authorized","pilot_authorized","scientific_acquisition_authorized","automatic_retry","automatic_extension")
assert not any(a[k] for k in blocked)
print("M4_DOWNSTREAM_QUALITY_EOS_PREFLIGHT_V7_AUTHORITY_PASS")
PY"""
    m["spec"]["script"] = s[:start] + block + s[end:]
    m["spec"]["name"] = "m4-downstream-quality-no-training-preflight-v7"
    a.output.write_text(json.dumps(m, separators=(",", ":")) + "\n")
    print(sha(a.output))


if __name__ == "__main__":
    main()
