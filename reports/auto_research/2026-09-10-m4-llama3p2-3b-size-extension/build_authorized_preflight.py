#!/usr/bin/env python3
"""Transform the validated local candidate into one authorized EOS preflight."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path


CANDIDATE_SHA = "30d88f08be15777db977964ae670808617e240545ccab486042d2112d1183334"
LOCAL_AUTH_SHA = "d29a823bf4d7b34e1a550112632bf0016b6c6a0511a093c26da8c2afa7567ff7"
EOS_AUTH_SHA = "6bc6a54a95f577b235d9c6b77c5bf52a11b8236a3bd042a2b1f087dc4fbb95de"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(value: str, old: str, new: str, label: str) -> str:
    if value.count(old) != 1:
        raise RuntimeError(f"expected exactly one {label}")
    return value.replace(old, new)


def authorization_block(authorization: dict[str, object]) -> str:
    literal = repr(authorization).replace("{", "{{").replace("}", "}}")
    return f'''assert auth=={literal}
print("M4_LLAMA3B_EOS_PREFLIGHT_AUTHORIZATION_PASS")
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--local-authorization", type=Path, required=True)
    parser.add_argument("--eos-authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix != ".yaml":
        raise RuntimeError("authorized manifest must use a .yaml output")
    for path, expected in (
        (args.candidate, CANDIDATE_SHA),
        (args.local_authorization, LOCAL_AUTH_SHA),
        (args.eos_authorization, EOS_AUTH_SHA),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"authorized transform input moved: {path.name}")
    authorization = json.loads(args.eos_authorization.read_bytes())
    if authorization != {
        "schema": "m4-llama3p2-3b-no-training-eos-preflight-authorization-v1",
        "scope": "exactly_one_credential_free_llama3p2_3b_no_training_eos_preflight",
        "user_authorization": "I authorize",
        "source_commit": "88424bdb7bfb526e541b4a7b07830bc31028e64f",
        "source_archive_sha256": "1ae875a63786d687fcf46bd7fa7cb2f26b1031c3cf3c36e38114305c732820eb",
        "protocol_sha256": "4f2a5e496030e836b3449697928f4f363d7d36e3cec668105b2cdcc701b9814c",
        "local_candidate_sha256": CANDIDATE_SHA,
        "required_launcher": "runllm.py --no_wait",
        "submission_attempt_limit": 1,
        "eos_submission_authorized": True,
        "model_metadata_and_tokenizer_access_authorized": True,
        "model_weight_download_authorized": False,
        "trainer_construction_authorized": False,
        "training_authorized": False,
        "qualification_authorized": False,
        "acquisition_authorized": False,
        "automatic_retry": False,
        "automatic_extension": False,
    }:
        raise RuntimeError("EOS preflight authorization differs")
    manifest = json.loads(args.candidate.read_bytes())
    script = manifest["spec"]["script"]
    old_payload = base64.b64encode(args.local_authorization.read_bytes()).decode()
    new_payload = base64.b64encode(args.eos_authorization.read_bytes()).decode()
    script = replace_once(script, old_payload, new_payload, "authorization payload")
    script = replace_once(script, LOCAL_AUTH_SHA, EOS_AUTH_SHA, "authorization hash")
    start = script.index('assert auth["source_commit"]')
    end_marker = ' raise SystemExit("M4_LLAMA3B_LOCAL_CANDIDATE_NO_LAUNCH_AUTHORITY")\n'
    end = script.index(end_marker, start) + len(end_marker)
    script = script[:start] + authorization_block(authorization) + script[end:]
    manifest["spec"]["name"] = "m4-llama3b-no-training-preflight"
    manifest["spec"]["script"] = script
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(args.output, sha256(args.output))


if __name__ == "__main__":
    main()
