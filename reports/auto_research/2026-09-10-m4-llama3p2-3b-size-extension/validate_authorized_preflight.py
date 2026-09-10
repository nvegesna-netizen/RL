#!/usr/bin/env python3
"""Validate the exact authorized one-shot Llama 3B EOS preflight package."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path

from build_authorized_preflight import CANDIDATE_SHA, EOS_AUTH_SHA, LOCAL_AUTH_SHA
from jetclient import JETWorkloadManifest


RUNLLM_SHA = "30532cd638d8394fa0e651daf4a43ec9424cbaa0949cd59c323191177318e616"
CUSTOM_SHA = "3d7fefba24bdc21786faf5937aeda44c735602fcca00dc8eb46a349ae6e27748"
ADDITIONAL_SHA = "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356"
PAYLOAD_SHAS = (
    "1ae875a63786d687fcf46bd7fa7cb2f26b1031c3cf3c36e38114305c732820eb",
    "98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2",
    "4f2a5e496030e836b3449697928f4f363d7d36e3cec668105b2cdcc701b9814c",
    EOS_AUTH_SHA,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--local-authorization", type=Path, required=True)
    parser.add_argument("--eos-authorization", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--custom-config", type=Path, required=True)
    parser.add_argument("--additional-variables", type=Path, required=True)
    parser.add_argument("--runllm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path, expected in (
        (args.candidate, CANDIDATE_SHA),
        (args.local_authorization, LOCAL_AUTH_SHA),
        (args.eos_authorization, EOS_AUTH_SHA),
        (args.custom_config, CUSTOM_SHA),
        (args.additional_variables, ADDITIONAL_SHA),
        (args.runllm, RUNLLM_SHA),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"authorized package input moved: {path.name}")
    manifest = json.loads(args.manifest.read_bytes())
    candidate = json.loads(args.candidate.read_bytes())
    JETWorkloadManifest.model_validate(deepcopy(manifest))
    if manifest["spec"]["name"] != "m4-llama3b-no-training-preflight":
        raise RuntimeError("authorized workload name differs")
    local_script = candidate["spec"]["script"]
    eos_script = manifest["spec"]["script"]
    normalized = deepcopy(manifest)
    normalized["spec"]["name"] = candidate["spec"]["name"]
    normalized["spec"]["script"] = local_script
    if normalized != candidate:
        raise RuntimeError("manifest differs beyond name and authorization script")
    rendered = eos_script.format(assets_dir="/tmp/m4-llama3b-authorized-preflight")
    subprocess.run(["bash", "-n"], input=rendered, text=True, check=True)
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", rendered, flags=re.DOTALL)
    if len(blocks) != 5:
        raise RuntimeError("embedded Python block count differs")
    for index, block in enumerate(blocks):
        compile(block, f"<authorized-preflight-{index}>", "exec")
    encoded = re.findall(r"printf %s '([A-Za-z0-9+/=]+)' \| base64 -d", rendered)
    if tuple(hashlib.sha256(base64.b64decode(item)).hexdigest() for item in encoded) != PAYLOAD_SHAS:
        raise RuntimeError("authorized embedded payloads differ")
    surface = "\n".join(
        line for line in rendered.splitlines() if not line.startswith("printf %s '")
    )
    for forbidden in (
        "examples/run_grpo",
        "trainer.fit",
        "AutoModel",
        "sbatch ",
        "srun ",
        "M4_LLAMA3B_LOCAL_CANDIDATE_NO_LAUNCH_AUTHORITY",
    ):
        if forbidden in surface:
            raise RuntimeError(f"forbidden execution surface present: {forbidden}")
    if surface.count("runllm.py --no_wait") != 1:
        raise RuntimeError("declarative launcher authorization count differs")
    if re.search(r"(?:python\S*|exec\S*)\s+[^\n]*runllm\.py", surface):
        raise RuntimeError("nested runllm invocation present")
    for required in (
        "M4_LLAMA3B_EOS_PREFLIGHT_AUTHORIZATION_PASS",
        "AutoConfig.from_pretrained",
        "AutoTokenizer.from_pretrained",
        '"model_weights_downloaded":False',
        '"trainer_steps_started":0',
    ):
        if required not in surface:
            raise RuntimeError(f"required preflight gate missing: {required}")
    custom = json.loads(args.custom_config.read_bytes())
    launcher = custom["launchers"]["dgxh100_eos"]
    if (
        custom["environments"]["retrier"]["enabled"] is not False
        or launcher["sbatch_additional_flags"]["time"] != "00:30:00"
        or launcher["srun_additional_flags"]["time"] != "00:30:00"
        or json.loads(args.additional_variables.read_bytes()) != {}
    ):
        raise RuntimeError("scheduler or retry surface differs")
    with tempfile.TemporaryDirectory(prefix="m4-llama3b-authorized-") as raw:
        rebuilt = Path(raw) / "rebuilt.yaml"
        subprocess.run(
            [
                "python3", str(args.builder),
                "--candidate", str(args.candidate),
                "--local-authorization", str(args.local_authorization),
                "--eos-authorization", str(args.eos_authorization),
                "--output", str(rebuilt),
            ],
            check=True, capture_output=True, text=True,
        )
        if rebuilt.read_bytes() != args.manifest.read_bytes():
            raise RuntimeError("authorized manifest rebuild differs")
    result = {
        "schema": "m4-llama3p2-3b-no-training-eos-preflight-package-validation-v1",
        "status": "PASS",
        "manifest_sha256": sha256(args.manifest),
        "authorization_sha256": EOS_AUTH_SHA,
        "candidate_sha256": CANDIDATE_SHA,
        "runllm_sha256": RUNLLM_SHA,
        "deterministic_rebuild": True,
        "jet_manifest_schema_green": True,
        "jet_retrier_enabled": False,
        "submission_attempt_limit": 1,
        "model_weight_loading_surface_present": False,
        "training_surface_present": False,
        "qualification_surface_present": False,
        "acquisition_surface_present": False,
        "launch_attempted": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
