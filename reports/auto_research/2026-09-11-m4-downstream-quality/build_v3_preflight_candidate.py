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
"""Build the DDP-hook-repaired downstream-quality EOS preflight v3."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

V2_MANIFEST_SHA256 = "975ddde2d33f074fa1d39f28007c924576ff8b93a653728e1e496f20faedded4"
V2_AUTHORIZATION_SHA256 = (
    "a87b1286a8864fc7e1fe8b94a99a431dcb72aba558d05b353c8e57b8d44c9250"
)
V2_TERMINAL_RESULT_SHA256 = (
    "b38b7fc7646f48135f58d661be29932898d45d2d44b5c68124af019cd461f0a1"
)
V3_AUTHORIZATION_SHA256 = (
    "8f542ee8c21a6ee11e8ad7eb37df5755f79ec7573b7caaa16db8b327db34f809"
)
OLD_SOURCE_SHA256 = "80741c66b32a7951e4ebd620e66a8043858159e7a871c7fb87d038db068aa714"
NEW_SOURCE_SHA256 = "3e1cf6fb4fb287c120e7502beeedd6fb50534af3bc83b4becad610e76bb34966"
OLD_SOURCE_COMMIT = "1941728f8a636d82126dec8632277cd63172e3b7"
NEW_SOURCE_COMMIT = "573dddce4cfca5fd615079c8045af05dddf80d03"


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_exact(value: str, old: str, new: str, count: int, label: str) -> str:
    """Replace an exact number of frozen occurrences or fail closed."""
    if value.count(old) != count:
        raise RuntimeError(f"expected {count} occurrences of {label}")
    return value.replace(old, new)


def main() -> None:
    """Apply only the new source, authority, identity, and DDP-hook repair."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-manifest", type=Path, required=True)
    parser.add_argument("--v2-authorization", type=Path, required=True)
    parser.add_argument("--v2-terminal-result", type=Path, required=True)
    parser.add_argument("--v3-authorization", type=Path, required=True)
    parser.add_argument("--old-source", type=Path, required=True)
    parser.add_argument("--new-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path, expected in (
        (args.v2_manifest, V2_MANIFEST_SHA256),
        (args.v2_authorization, V2_AUTHORIZATION_SHA256),
        (args.v2_terminal_result, V2_TERMINAL_RESULT_SHA256),
        (args.v3_authorization, V3_AUTHORIZATION_SHA256),
        (args.old_source, OLD_SOURCE_SHA256),
        (args.new_source, NEW_SOURCE_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"v3 transform input moved: {path.name}")

    authorization = json.loads(args.v3_authorization.read_bytes())
    if (
        authorization["scope"]
        != "exactly_one_repaired_credential_free_downstream_quality_no_training_eos_preflight_v3"
        or authorization["required_repair"]
        != "guard_checkpoint_forward_pre_hook_by_actual_enabled_state"
        or authorization["required_launcher"] != "runllm.py --no_wait"
        or authorization["submission_attempt_limit"] != 1
        or authorization["eos_submission_authorized"] is not True
        or authorization["model_weight_access_authorized"] is not True
        or authorization["optimizer_steps_allowed"] != 0
    ):
        raise RuntimeError("v3 EOS preflight authority differs")
    blocked = (
        "optimizer_initialization_authorized",
        "training_authorized",
        "qualification_authorized",
        "pilot_authorized",
        "scientific_acquisition_authorized",
        "automatic_retry",
        "automatic_extension",
    )
    if any(authorization[key] for key in blocked):
        raise RuntimeError("v3 authority is not fail-closed")

    manifest = json.loads(args.v2_manifest.read_bytes())
    script = manifest["spec"]["script"]
    script = replace_exact(
        script,
        base64.b64encode(args.old_source.read_bytes()).decode(),
        base64.b64encode(args.new_source.read_bytes()).decode(),
        1,
        "source payload",
    )
    script = replace_exact(
        script,
        base64.b64encode(args.v2_authorization.read_bytes()).decode(),
        base64.b64encode(args.v3_authorization.read_bytes()).decode(),
        1,
        "authorization payload",
    )
    script = replace_exact(
        script, OLD_SOURCE_SHA256, NEW_SOURCE_SHA256, 1, "source hash"
    )
    script = replace_exact(
        script, OLD_SOURCE_COMMIT, NEW_SOURCE_COMMIT, 2, "source commit"
    )
    script = replace_exact(
        script,
        V2_AUTHORIZATION_SHA256,
        V3_AUTHORIZATION_SHA256,
        1,
        "authorization hash",
    )

    start = script.index('"$PYTHON" - "$AUTH" <<\'PY\'\n')
    end = script.index("\nPY\nmkdir -p", start) + len("\nPY")
    authority_block = """"$PYTHON" - "$AUTH" <<'PY'
import json,sys
from pathlib import Path
a=json.loads(Path(sys.argv[1]).read_bytes())
assert a["schema"]=="m4-downstream-quality-no-training-eos-preflight-v3-authorization-v1"
assert a["scope"]=="exactly_one_repaired_credential_free_downstream_quality_no_training_eos_preflight_v3"
assert a["user_authorization"]=="I authorize"
assert a["predecessor_terminal_result_sha256"]=="b38b7fc7646f48135f58d661be29932898d45d2d44b5c68124af019cd461f0a1"
assert a["predecessor_manifest_sha256"]=="975ddde2d33f074fa1d39f28007c924576ff8b93a653728e1e496f20faedded4"
assert a["source_commit"]=="573dddce4cfca5fd615079c8045af05dddf80d03"
assert a["source_archive_sha256"]=="3e1cf6fb4fb287c120e7502beeedd6fb50534af3bc83b4becad610e76bb34966"
assert a["required_repair"]=="guard_checkpoint_forward_pre_hook_by_actual_enabled_state"
assert a["required_launcher"]=="runllm.py --no_wait"
assert a["submission_attempt_limit"]==1
assert a["eos_submission_authorized"] is True
assert a["model_weight_access_authorized"] is True
assert a["optimizer_steps_allowed"]==0
blocked=("optimizer_initialization_authorized","training_authorized","qualification_authorized","pilot_authorized","scientific_acquisition_authorized","automatic_retry","automatic_extension")
assert not any(a[key] for key in blocked)
print("M4_DOWNSTREAM_QUALITY_EOS_PREFLIGHT_V3_AUTHORITY_PASS")
PY"""
    script = script[:start] + authority_block + script[end:]
    manifest["spec"]["script"] = script
    manifest["spec"]["name"] = "m4-downstream-quality-no-training-preflight-v3"
    args.output.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    print(args.output, sha256(args.output))


if __name__ == "__main__":
    main()
