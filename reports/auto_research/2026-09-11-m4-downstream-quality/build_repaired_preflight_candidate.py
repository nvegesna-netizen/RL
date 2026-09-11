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
"""Build the resolver-repaired downstream-quality EOS preflight."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

PREDECESSOR_MANIFEST_SHA256 = (
    "13dfd195a2ea2a5f6b9426b854531a294a51f31214d6e195df5d6cec5b5e5103"
)
PREDECESSOR_AUTHORIZATION_SHA256 = (
    "a2cd3cb97818d530b4dfb72312eb4ee266322c534c7a51e89f49006e04a26ceb"
)
V2_AUTHORIZATION_SHA256 = (
    "a87b1286a8864fc7e1fe8b94a99a431dcb72aba558d05b353c8e57b8d44c9250"
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
    """Apply only the new authority and resolver-registration repair."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--predecessor-manifest", type=Path, required=True)
    parser.add_argument("--predecessor-authorization", type=Path, required=True)
    parser.add_argument("--v2-authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path, expected in (
        (args.predecessor_manifest, PREDECESSOR_MANIFEST_SHA256),
        (args.predecessor_authorization, PREDECESSOR_AUTHORIZATION_SHA256),
        (args.v2_authorization, V2_AUTHORIZATION_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"repaired transform input moved: {path.name}")

    authorization = json.loads(args.v2_authorization.read_bytes())
    if (
        authorization["scope"]
        != "exactly_one_repaired_credential_free_downstream_quality_no_training_eos_preflight"
        or authorization["required_repair"]
        != "register_omegaconf_resolvers_before_load_and_resolve"
        or authorization["required_launcher"] != "runllm.py --no_wait"
        or authorization["submission_attempt_limit"] != 1
        or authorization["eos_submission_authorized"] is not True
        or authorization["model_weight_access_authorized"] is not True
        or authorization["optimizer_steps_allowed"] != 0
    ):
        raise RuntimeError("repaired EOS preflight authority differs")
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
            raise RuntimeError(f"v2 authority is not fail-closed: {key}")

    manifest = json.loads(args.predecessor_manifest.read_bytes())
    script = manifest["spec"]["script"]
    old_payload = base64.b64encode(args.predecessor_authorization.read_bytes()).decode()
    new_payload = base64.b64encode(args.v2_authorization.read_bytes()).decode()
    script = replace_once(script, old_payload, new_payload, "authorization payload")
    script = replace_once(
        script,
        PREDECESSOR_AUTHORIZATION_SHA256,
        V2_AUTHORIZATION_SHA256,
        "authorization hash",
    )

    start = script.index('"$PYTHON" - "$AUTH" <<\'PY\'\n')
    end = script.index("\nPY\nmkdir -p", start) + len("\nPY")
    authority_block = '''"$PYTHON" - "$AUTH" <<'PY'
import json,sys
from pathlib import Path
a=json.loads(Path(sys.argv[1]).read_bytes())
assert a["schema"]=="m4-downstream-quality-no-training-eos-preflight-v2-authorization-v1"
assert a["scope"]=="exactly_one_repaired_credential_free_downstream_quality_no_training_eos_preflight"
assert a["user_authorization"]=="do the repair. I authorize launches on eos"
assert a["predecessor_terminal_result_sha256"]=="551383a7d5243821a84a3d3765f9222c568d03ea2e5e6a3f7732f4e44cfb7878"
assert a["predecessor_manifest_sha256"]=="13dfd195a2ea2a5f6b9426b854531a294a51f31214d6e195df5d6cec5b5e5103"
assert a["source_commit"]=="1941728f8a636d82126dec8632277cd63172e3b7"
assert a["required_repair"]=="register_omegaconf_resolvers_before_load_and_resolve"
assert a["required_launcher"]=="runllm.py --no_wait"
assert a["submission_attempt_limit"]==1
assert a["eos_submission_authorized"] is True
assert a["model_weight_access_authorized"] is True
assert a["optimizer_steps_allowed"]==0
blocked=("optimizer_initialization_authorized","training_authorized","qualification_authorized","pilot_authorized","scientific_acquisition_authorized","automatic_retry","automatic_extension")
assert not any(a[key] for key in blocked)
print("M4_DOWNSTREAM_QUALITY_EOS_PREFLIGHT_V2_AUTHORITY_PASS")
PY'''
    script = script[:start] + authority_block + script[end:]
    script = replace_once(
        script,
        "from nemo_rl.utils.config import load_config",
        "from nemo_rl.utils.config import load_config,register_omegaconf_resolvers",
        "resolver import",
    )
    script = replace_once(
        script,
        'raw=OmegaConf.to_container(load_config(config_path),resolve=True)',
        'register_omegaconf_resolvers()\nraw=OmegaConf.to_container(load_config(config_path),resolve=True)',
        "resolver registration",
    )
    manifest["spec"]["script"] = script
    manifest["spec"]["name"] = "m4-downstream-quality-no-training-preflight-v2"
    args.output.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    print(args.output, sha256(args.output))


if __name__ == "__main__":
    main()
