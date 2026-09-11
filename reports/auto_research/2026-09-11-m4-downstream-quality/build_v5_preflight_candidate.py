#!/usr/bin/env python3
"""Build the mcore-interpreter-repaired downstream-quality EOS preflight v5."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

V4_MANIFEST_SHA256 = "fcd82a3e65da33c86f51923aa7236c8a2a2b7e4aeb233f3254f66f228f931222"
V4_AUTHORIZATION_SHA256 = (
    "76cc225ff4d09b06cfbf85f8d97c3056cda7b558a2c3e60e4a4d84f1545544a2"
)
V4_TERMINAL_RESULT_SHA256 = (
    "d6c3fe8bd309b3ab62f151f06e3b9a4aa3cdf0efa84f74a45c70cc9409c2a259"
)
V5_AUTHORIZATION_SHA256 = (
    "dd5f5197d5d84cd0cb2378c6ec3bc6e9862a8df1b40c82be1bcb2b185c3bab66"
)
MCORE_VENV = "/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_exact(value: str, old: str, new: str, count: int, label: str) -> str:
    if value.count(old) != count:
        raise RuntimeError(f"expected {count} occurrences of {label}")
    return value.replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4-manifest", type=Path, required=True)
    parser.add_argument("--v4-authorization", type=Path, required=True)
    parser.add_argument("--v4-terminal-result", type=Path, required=True)
    parser.add_argument("--v5-authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path, expected in (
        (args.v4_manifest, V4_MANIFEST_SHA256),
        (args.v4_authorization, V4_AUTHORIZATION_SHA256),
        (args.v4_terminal_result, V4_TERMINAL_RESULT_SHA256),
        (args.v5_authorization, V5_AUTHORIZATION_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"v5 transform input moved: {path.name}")

    authorization = json.loads(args.v5_authorization.read_bytes())
    if (
        authorization["scope"]
        != "exactly_one_mcore_interpreter_repaired_credential_free_downstream_quality_no_training_eos_preflight_v5"
        or authorization["required_repair"]
        != "use_prefetched_megatron_policy_worker_mcore_interpreter_for_converter_gate_and_conversion"
        or authorization["required_launcher"] != "runllm.py --no_wait"
        or authorization["submission_attempt_limit"] != 1
        or authorization["eos_submission_authorized"] is not True
        or authorization["model_weight_access_authorized"] is not True
        or authorization["optimizer_steps_allowed"] != 0
    ):
        raise RuntimeError("v5 authority differs")
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
        raise RuntimeError("v5 authority is not fail-closed")

    manifest = json.loads(args.v4_manifest.read_bytes())
    script = manifest["spec"]["script"]
    script = replace_exact(
        script,
        base64.b64encode(args.v4_authorization.read_bytes()).decode(),
        base64.b64encode(args.v5_authorization.read_bytes()).decode(),
        1,
        "authorization payload",
    )
    script = replace_exact(
        script,
        V4_AUTHORIZATION_SHA256,
        V5_AUTHORIZATION_SHA256,
        1,
        "authorization hash",
    )
    start = script.index('"$PYTHON" - "$AUTH" <<\'PY\'\n')
    end = script.index("\nPY\nmkdir -p", start) + len("\nPY")
    authority_block = """"$PYTHON" - "$AUTH" <<'PY'
import json,sys
from pathlib import Path
a=json.loads(Path(sys.argv[1]).read_bytes())
assert a["schema"]=="m4-downstream-quality-no-training-eos-preflight-v5-authorization-v1"
assert a["scope"]=="exactly_one_mcore_interpreter_repaired_credential_free_downstream_quality_no_training_eos_preflight_v5"
assert a["user_authorization"]=="I authorize"
assert a["predecessor_terminal_result_sha256"]=="d6c3fe8bd309b3ab62f151f06e3b9a4aa3cdf0efa84f74a45c70cc9409c2a259"
assert a["predecessor_manifest_sha256"]=="fcd82a3e65da33c86f51923aa7236c8a2a2b7e4aeb233f3254f66f228f931222"
assert a["required_repair"]=="use_prefetched_megatron_policy_worker_mcore_interpreter_for_converter_gate_and_conversion"
assert a["required_launcher"]=="runllm.py --no_wait"
assert a["submission_attempt_limit"]==1
assert a["eos_submission_authorized"] is True
assert a["model_weight_access_authorized"] is True
assert a["optimizer_steps_allowed"]==0
blocked=("optimizer_initialization_authorized","training_authorized","qualification_authorized","pilot_authorized","scientific_acquisition_authorized","automatic_retry","automatic_extension")
assert not any(a[key] for key in blocked)
print("M4_DOWNSTREAM_QUALITY_EOS_PREFLIGHT_V5_AUTHORITY_PASS")
PY"""
    script = script[:start] + authority_block + script[end:]

    old_gate_start = script.index('"$PYTHON" - "$RUN_REPO" <<\'PY\'\n')
    old_gate_end = script.index(
        "echo M4_DOWNSTREAM_QUALITY_CONVERTER_IMPORT_PASS\n", old_gate_start
    ) + len("echo M4_DOWNSTREAM_QUALITY_CONVERTER_IMPORT_PASS\n")
    new_gate = f"""readonly MCORE_VENV={MCORE_VENV}
readonly MCORE_PYTHON="$MCORE_VENV/bin/python"
test -x "$MCORE_PYTHON"
env VIRTUAL_ENV="$MCORE_VENV" UV_PROJECT_ENVIRONMENT="$MCORE_VENV" "$MCORE_PYTHON" - "$RUN_REPO" "$MCORE_VENV" <<'PY'
import sys
from pathlib import Path
root=Path(sys.argv[1]).resolve(); venv=Path(sys.argv[2]).resolve()
import megatron.bridge
import megatron.core
import transformer_engine.pytorch as te
from megatron.bridge import AutoBridge
bridge_path=Path(megatron.bridge.__file__).resolve()
core_path=Path(megatron.core.__file__).resolve()
te_path=Path(te.__file__).resolve()
assert bridge_path.is_relative_to(root)
assert core_path.is_relative_to(root)
assert te_path.is_relative_to(venv)
assert AutoBridge.__module__.startswith("megatron.bridge.")
print("M4_DOWNSTREAM_QUALITY_MCORE_CONVERTER_ENV_PASS",sys.executable,bridge_path,core_path,te_path)
PY
env VIRTUAL_ENV="$MCORE_VENV" UV_PROJECT_ENVIRONMENT="$MCORE_VENV" "$MCORE_PYTHON" examples/converters/convert_megatron_to_hf.py --help >/dev/null
echo M4_DOWNSTREAM_QUALITY_CONVERTER_IMPORT_PASS
"""
    script = script[:old_gate_start] + new_gate + script[old_gate_end:]
    script = replace_exact(
        script,
        '"$PYTHON" examples/converters/convert_megatron_to_hf.py --config',
        'env VIRTUAL_ENV="$MCORE_VENV" UV_PROJECT_ENVIRONMENT="$MCORE_VENV" "$MCORE_PYTHON" examples/converters/convert_megatron_to_hf.py --config',
        1,
        "post-export converter interpreter",
    )
    manifest["spec"]["script"] = script
    manifest["spec"]["name"] = "m4-downstream-quality-no-training-preflight-v5"
    args.output.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    print(args.output, sha256(args.output))


if __name__ == "__main__":
    main()
