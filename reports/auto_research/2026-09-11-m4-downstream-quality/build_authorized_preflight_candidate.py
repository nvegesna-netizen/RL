#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transform the validated local candidate into one EOS-authorized preflight."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

LOCAL_CANDIDATE_SHA256 = (
    "e7de1aa88e2f4a9c7de7e1b7f9f6a5b34382ea18544d9920dcb65d3d0b5b4853"
)
LOCAL_AUTHORIZATION_SHA256 = (
    "d34ec018f5d75711bcab3efcf4225f0d4a57b75fa338cfcd050b79b821d3c7c2"
)
EOS_AUTHORIZATION_SHA256 = (
    "a2cd3cb97818d530b4dfb72312eb4ee266322c534c7a51e89f49006e04a26ceb"
)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(value: str, old: str, new: str, label: str) -> str:
    """Replace exactly one frozen value or fail closed."""
    if value.count(old) != 1:
        raise RuntimeError(f"expected exactly one {label}")
    return value.replace(old, new)


def main() -> None:
    """Build the sole EOS-authorized manifest as an authorization-only delta."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--local-authorization", type=Path, required=True)
    parser.add_argument("--eos-authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path, expected in (
        (args.candidate, LOCAL_CANDIDATE_SHA256),
        (args.local_authorization, LOCAL_AUTHORIZATION_SHA256),
        (args.eos_authorization, EOS_AUTHORIZATION_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"authorized transform input moved: {path.name}")

    authorization = json.loads(args.eos_authorization.read_bytes())
    if (
        authorization["schema"]
        != "m4-downstream-quality-no-training-eos-preflight-authorization-v1"
        or authorization["scope"]
        != "exactly_one_credential_free_downstream_quality_no_training_eos_preflight"
        or authorization["user_authorization"] != "I authorize"
        or authorization["required_launcher"] != "runllm.py --no_wait"
        or authorization["submission_attempt_limit"] != 1
        or authorization["eos_submission_authorized"] is not True
        or authorization["model_weight_access_authorized"] is not True
        or authorization["optimizer_steps_allowed"] != 0
    ):
        raise RuntimeError("EOS preflight authority differs")
    for key in (
        "optimizer_initialization_authorized",
        "training_authorized",
        "qualification_authorized",
        "pilot_authorized",
        "scientific_acquisition_authorized",
        "automatic_retry",
        "automatic_extension",
    ):
        if authorization[key] is not False:
            raise RuntimeError(f"EOS preflight authority is not fail-closed: {key}")

    manifest = json.loads(args.candidate.read_bytes())
    script = manifest["spec"]["script"]
    old_payload = base64.b64encode(args.local_authorization.read_bytes()).decode()
    new_payload = base64.b64encode(args.eos_authorization.read_bytes()).decode()
    script = replace_once(script, old_payload, new_payload, "authorization payload")
    script = replace_once(
        script,
        LOCAL_AUTHORIZATION_SHA256,
        EOS_AUTHORIZATION_SHA256,
        "authorization hash",
    )

    start = script.index('"$PYTHON" - "$AUTH" <<\'PY\'\n')
    end = script.index("\nPY\nmkdir -p", start) + len("\nPY")
    authority_block = '''"$PYTHON" - "$AUTH" <<'PY'
import json,sys
from pathlib import Path
a=json.loads(Path(sys.argv[1]).read_bytes())
assert a["schema"]=="m4-downstream-quality-no-training-eos-preflight-authorization-v1"
assert a["scope"]=="exactly_one_credential_free_downstream_quality_no_training_eos_preflight"
assert a["user_authorization"]=="I authorize"
assert a["source_commit"]=="1941728f8a636d82126dec8632277cd63172e3b7"
assert a["source_archive_sha256"]=="80741c66b32a7951e4ebd620e66a8043858159e7a871c7fb87d038db068aa714"
assert a["megatron_dependency_sha256"]=="98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
assert a["protocol_draft_sha256"]=="5ec319043f6e3be90e8e294946eee3596b4c8ab445630de93ae103bbcabb7162"
assert a["preflight_plan_sha256"]=="b160c6610226bf2b2250cbc20500667776bfb1e62df0918994c4221a690918fa"
assert a["local_candidate_sha256"]=="e7de1aa88e2f4a9c7de7e1b7f9f6a5b34382ea18544d9920dcb65d3d0b5b4853"
assert a["required_launcher"]=="runllm.py --no_wait"
assert a["submission_attempt_limit"]==1
assert a["eos_submission_authorized"] is True
assert a["model_weight_access_authorized"] is True
assert a["optimizer_steps_allowed"]==0
blocked=("optimizer_initialization_authorized","training_authorized","qualification_authorized","pilot_authorized","scientific_acquisition_authorized","automatic_retry","automatic_extension")
assert not any(a[key] for key in blocked)
print("M4_DOWNSTREAM_QUALITY_EOS_PREFLIGHT_AUTHORITY_PASS")
PY'''
    script = script[:start] + authority_block + script[end:]
    manifest["spec"]["script"] = script
    manifest["spec"]["name"] = "m4-downstream-quality-no-training-preflight"
    args.output.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    print(args.output, sha256(args.output))


if __name__ == "__main__":
    main()
