#!/usr/bin/env python3
"""Build the symlink-aware Transformer Engine provenance EOS preflight v6."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

V5_MANIFEST_SHA256 = "bb2ff3571a27e37c872a21669124ec42b5166242f8c7d447727dd6e2790784f7"
V5_AUTHORIZATION_SHA256 = (
    "dd5f5197d5d84cd0cb2378c6ec3bc6e9862a8df1b40c82be1bcb2b185c3bab66"
)
V5_TERMINAL_RESULT_SHA256 = (
    "890c126863df145fd52ae2e560520d10abce483083e735e92dec7b3c0371dd80"
)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_exact(value: str, old: str, new: str, count: int, label: str) -> str:
    """Replace an exact number of occurrences or fail closed."""
    if value.count(old) != count:
        raise RuntimeError(f"expected {count} occurrences of {label}")
    return value.replace(old, new)


def main() -> None:
    """Transform the frozen v5 manifest into the narrowly repaired v6 manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--v5-manifest", type=Path, required=True)
    parser.add_argument("--v5-authorization", type=Path, required=True)
    parser.add_argument("--v5-terminal-result", type=Path, required=True)
    parser.add_argument("--v6-authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path, expected in (
        (args.v5_manifest, V5_MANIFEST_SHA256),
        (args.v5_authorization, V5_AUTHORIZATION_SHA256),
        (args.v5_terminal_result, V5_TERMINAL_RESULT_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"v6 transform input moved: {path.name}")

    authorization = json.loads(args.v6_authorization.read_bytes())
    if (
        authorization["scope"]
        != "exactly_one_symlink_aware_transformer_engine_provenance_repaired_credential_free_downstream_quality_no_training_eos_preflight_v6"
        or authorization["required_repair"]
        != "replace_resolved_transformer_engine_containment_with_symlink_aware_logical_provenance_and_pre_assertion_diagnostics"
        or authorization["required_launcher"] != "runllm.py --no_wait"
        or authorization["submission_attempt_limit"] != 1
        or authorization["eos_submission_authorized"] is not True
        or authorization["model_weight_access_authorized"] is not True
        or authorization["optimizer_steps_allowed"] != 0
    ):
        raise RuntimeError("v6 authority differs")
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
        raise RuntimeError("v6 authority is not fail-closed")

    manifest = json.loads(args.v5_manifest.read_bytes())
    script = manifest["spec"]["script"]
    script = replace_exact(
        script,
        base64.b64encode(args.v5_authorization.read_bytes()).decode(),
        base64.b64encode(args.v6_authorization.read_bytes()).decode(),
        1,
        "authorization payload",
    )
    script = replace_exact(
        script,
        V5_AUTHORIZATION_SHA256,
        sha256(args.v6_authorization),
        1,
        "authorization hash",
    )
    start = script.index('"$PYTHON" - "$AUTH" <<\'PY\'\n')
    end = script.index("\nPY\nmkdir -p", start) + len("\nPY")
    authority_block = """"$PYTHON" - "$AUTH" <<'PY'
import json,sys
from pathlib import Path
a=json.loads(Path(sys.argv[1]).read_bytes())
assert a["schema"]=="m4-downstream-quality-no-training-eos-preflight-v6-authorization-v1"
assert a["scope"]=="exactly_one_symlink_aware_transformer_engine_provenance_repaired_credential_free_downstream_quality_no_training_eos_preflight_v6"
assert a["user_authorization"]=="I authorize"
assert a["predecessor_terminal_result_sha256"]=="890c126863df145fd52ae2e560520d10abce483083e735e92dec7b3c0371dd80"
assert a["predecessor_manifest_sha256"]=="bb2ff3571a27e37c872a21669124ec42b5166242f8c7d447727dd6e2790784f7"
assert a["required_repair"]=="replace_resolved_transformer_engine_containment_with_symlink_aware_logical_provenance_and_pre_assertion_diagnostics"
assert a["required_launcher"]=="runllm.py --no_wait"
assert a["submission_attempt_limit"]==1
assert a["eos_submission_authorized"] is True
assert a["model_weight_access_authorized"] is True
assert a["optimizer_steps_allowed"]==0
blocked=("optimizer_initialization_authorized","training_authorized","qualification_authorized","pilot_authorized","scientific_acquisition_authorized","automatic_retry","automatic_extension")
assert not any(a[key] for key in blocked)
print("M4_DOWNSTREAM_QUALITY_EOS_PREFLIGHT_V6_AUTHORITY_PASS")
PY"""
    script = script[:start] + authority_block + script[end:]

    gate_start = script.index(
        'env VIRTUAL_ENV="$MCORE_VENV" UV_PROJECT_ENVIRONMENT="$MCORE_VENV" "$MCORE_PYTHON" - "$RUN_REPO" "$MCORE_VENV" <<\'PY\'\n'
    )
    gate_end = script.index('\nPY\nenv VIRTUAL_ENV="$MCORE_VENV"', gate_start) + len(
        "\nPY"
    )
    new_gate = """env VIRTUAL_ENV="$MCORE_VENV" UV_PROJECT_ENVIRONMENT="$MCORE_VENV" "$MCORE_PYTHON" - "$RUN_REPO" "$MCORE_VENV" <<'PY'
import sys
from pathlib import Path
root=Path(sys.argv[1]).resolve(); venv=Path(sys.argv[2]).absolute()
import megatron.bridge
import megatron.core
import transformer_engine.pytorch as te
from megatron.bridge import AutoBridge
bridge_path=Path(megatron.bridge.__file__).absolute()
core_path=Path(megatron.core.__file__).absolute()
te_raw_path=Path(te.__file__).absolute()
te_resolved_path=te_raw_path.resolve()
print("M4_DOWNSTREAM_QUALITY_MCORE_PATHS",sys.executable,bridge_path,core_path,te_raw_path,te_resolved_path,flush=True)
assert bridge_path.resolve().is_relative_to(root)
assert core_path.resolve().is_relative_to(root)
assert te_raw_path.is_relative_to(venv)
assert AutoBridge.__module__.startswith("megatron.bridge.")
print("M4_DOWNSTREAM_QUALITY_MCORE_CONVERTER_ENV_PASS",flush=True)
PY"""
    script = script[:gate_start] + new_gate + script[gate_end:]
    manifest["spec"]["script"] = script
    manifest["spec"]["name"] = "m4-downstream-quality-no-training-preflight-v6"
    args.output.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    print(args.output, sha256(args.output))


if __name__ == "__main__":
    main()
