#!/usr/bin/env python3
"""Clean-room validate the mcore-interpreter-repaired EOS preflight v5."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

from jetclient import JETWorkloadManifest

from validate_v4_preflight_candidate import safe_extract

MANIFEST_SHA256 = "bb2ff3571a27e37c872a21669124ec42b5166242f8c7d447727dd6e2790784f7"
V4_MANIFEST_SHA256 = "fcd82a3e65da33c86f51923aa7236c8a2a2b7e4aeb233f3254f66f228f931222"
V4_AUTHORIZATION_SHA256 = (
    "76cc225ff4d09b06cfbf85f8d97c3056cda7b558a2c3e60e4a4d84f1545544a2"
)
V4_TERMINAL_RESULT_SHA256 = (
    "d6c3fe8bd309b3ab62f151f06e3b9a4aa3cdf0efa84f74a45c70cc9409c2a259"
)
V4_VALIDATION_SHA256 = (
    "46ef4acdd9b6403f1fc593a906be8c61de6c6472e2ad524e47f2b71999add191"
)
V5_AUTHORIZATION_SHA256 = (
    "dd5f5197d5d84cd0cb2378c6ec3bc6e9862a8df1b40c82be1bcb2b185c3bab66"
)
SOURCE_SHA256 = "3e1cf6fb4fb287c120e7502beeedd6fb50534af3bc83b4becad610e76bb34966"
BRIDGE_SHA256 = "1429945d1d50045900e40314fa283fa5f66484e945aad4920da137d7ad2c9313"
MEGATRON_SHA256 = "98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
BUILDER_SHA256 = "ed386a9f8c3de0f61b8f8a8e4fd21a856b4c8941330505b72d4a21c2093fbdf9"
CUSTOM_SHA256 = "c13a73b87669569429c36fd0c60f09d2a6ecabfb4086063395924cc26566dc8d"
ADDITIONAL_SHA256 = "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356"
MCORE_VENV = "/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "repo",
        "manifest",
        "v4_manifest",
        "v4_authorization",
        "v4_terminal_result",
        "v4_validation",
        "v5_authorization",
        "source",
        "bridge_archive",
        "megatron_dependency",
        "builder",
        "custom_config",
        "additional_variables",
        "output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    args = parser.parse_args()
    for path, expected in (
        (args.manifest, MANIFEST_SHA256),
        (args.v4_manifest, V4_MANIFEST_SHA256),
        (args.v4_authorization, V4_AUTHORIZATION_SHA256),
        (args.v4_terminal_result, V4_TERMINAL_RESULT_SHA256),
        (args.v4_validation, V4_VALIDATION_SHA256),
        (args.v5_authorization, V5_AUTHORIZATION_SHA256),
        (args.source, SOURCE_SHA256),
        (args.bridge_archive, BRIDGE_SHA256),
        (args.megatron_dependency, MEGATRON_SHA256),
        (args.builder, BUILDER_SHA256),
        (args.custom_config, CUSTOM_SHA256),
        (args.additional_variables, ADDITIONAL_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen v5 input moved: {path.name}")

    authorization = json.loads(args.v5_authorization.read_bytes())
    if (
        authorization["schema"]
        != "m4-downstream-quality-no-training-eos-preflight-v5-authorization-v1"
        or authorization["scope"]
        != "exactly_one_mcore_interpreter_repaired_credential_free_downstream_quality_no_training_eos_preflight_v5"
        or authorization["predecessor_terminal_result_sha256"]
        != V4_TERMINAL_RESULT_SHA256
        or authorization["predecessor_manifest_sha256"] != V4_MANIFEST_SHA256
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

    custom = json.loads(args.custom_config.read_bytes())
    launcher = custom["launchers"]["dgxh100_eos"]
    if (
        custom["environments"]["retrier"]["enabled"] is not False
        or launcher["sbatch_additional_flags"]["time"] != "02:00:00"
        or launcher["srun_additional_flags"]["time"] != "02:00:00"
        or json.loads(args.additional_variables.read_bytes()) != {}
    ):
        raise RuntimeError("v5 scheduler/retry boundary differs")

    manifest = json.loads(args.manifest.read_bytes())
    JETWorkloadManifest.model_validate(deepcopy(manifest))
    spec = manifest["spec"]
    if (
        spec["name"] != "m4-downstream-quality-no-training-preflight-v5"
        or spec["nodes"] != 1
        or spec["time_limit"] != 7200
    ):
        raise RuntimeError("v5 manifest boundary differs")
    rendered = spec["script"].format(assets_dir="/tmp/m4-v5")
    subprocess.run(["bash", "-n"], input=rendered, text=True, check=True)
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", rendered, flags=re.DOTALL)
    if len(blocks) != 9:
        raise RuntimeError(f"embedded Python count differs: {len(blocks)}")
    for index, block in enumerate(blocks):
        compile(block, f"<m4-v5-{index}>", "exec")
    authority = subprocess.run(
        [sys.executable, "-c", blocks[0], str(args.v5_authorization)],
        text=True,
        capture_output=True,
        check=False,
    )
    if authority.returncode or "V5_AUTHORITY_PASS" not in authority.stdout:
        raise RuntimeError("v5 dynamic authority failed")

    embedded = {
        hashlib.sha256(base64.b64decode(value)).hexdigest()
        for value in re.findall(r"printf %s '([A-Za-z0-9+/=]+)' \| base64 -d", rendered)
    }
    if not {
        SOURCE_SHA256,
        BRIDGE_SHA256,
        MEGATRON_SHA256,
        V5_AUTHORIZATION_SHA256,
        authorization["protocol_draft_sha256"],
        authorization["preflight_plan_sha256"],
    }.issubset(embedded):
        raise RuntimeError("v5 payload set differs")
    if V4_AUTHORIZATION_SHA256 in embedded:
        raise RuntimeError("v5 retains predecessor authority")

    dependency = rendered.index("M4_DOWNSTREAM_QUALITY_MEGATRON_DEPENDENCIES_PASS")
    executable_gate = rendered.index('test -x "$MCORE_PYTHON"')
    env_gate = rendered.index("M4_DOWNSTREAM_QUALITY_MCORE_CONVERTER_ENV_PASS")
    converter_gate = rendered.index("M4_DOWNSTREAM_QUALITY_CONVERTER_IMPORT_PASS")
    model_access = rendered.index("init_ray()")
    if not dependency < executable_gate < env_gate < converter_gate < model_access:
        raise RuntimeError("v5 pre-model gate ordering differs")
    if rendered.count(MCORE_VENV) != 1:
        raise RuntimeError("mcore venv identity differs")
    if rendered.count('env VIRTUAL_ENV="$MCORE_VENV"') != 3:
        raise RuntimeError("mcore interpreter use count differs")
    if (
        "import transformer_engine.pytorch as te" not in blocks[3]
        or "assert te_path.is_relative_to(venv)" not in blocks[3]
        or rendered.count(
            '"$MCORE_PYTHON" examples/converters/convert_megatron_to_hf.py'
        )
        != 2
    ):
        raise RuntimeError("v5 converter environment repair differs")
    forbidden = (
        "finish_train_step(",
        "train_microbatches(",
        "begin_train_step(",
        "examples/run_grpo",
        "python runllm.py",
        "sbatch ",
        "srun ",
        "HF_TOKEN",
        "PRIVATE-TOKEN",
        "WANDB_API_KEY",
        "NVIDIA_API_KEY",
        "NGC_API_KEY",
    )
    if any(value in rendered for value in forbidden):
        raise RuntimeError("forbidden execution or credential surface present")
    if "init_optimizer=False" not in rendered or '"optimizer_steps":0' not in rendered:
        raise RuntimeError("zero-optimizer boundary differs")

    with tempfile.TemporaryDirectory(prefix="m4-v5-") as raw:
        root = Path(raw)
        source_count, source_links = safe_extract(
            args.source, root / "source", 2261, ""
        )
        bridge_count, bridge_links = safe_extract(
            args.bridge_archive, root / "deps", 1060, "Megatron-Bridge/"
        )
        megatron_count, megatron_links = safe_extract(
            args.megatron_dependency,
            root / "deps/Megatron-Bridge/3rdparty",
            726,
            "Megatron-LM/",
        )
        tampered = dict(authorization)
        tampered["training_authorized"] = True
        tampered_path = root / "tampered.json"
        tampered_path.write_text(json.dumps(tampered))
        negative = subprocess.run(
            [sys.executable, "-c", blocks[0], str(tampered_path)],
            text=True,
            capture_output=True,
            check=False,
        )
        if negative.returncode == 0:
            raise RuntimeError("expanded-authority negative control passed")
        rebuilt = root / "rebuilt.json"
        subprocess.run(
            [
                sys.executable,
                str(args.builder),
                "--v4-manifest",
                str(args.v4_manifest),
                "--v4-authorization",
                str(args.v4_authorization),
                "--v4-terminal-result",
                str(args.v4_terminal_result),
                "--v5-authorization",
                str(args.v5_authorization),
                "--output",
                str(rebuilt),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if rebuilt.read_bytes() != args.manifest.read_bytes():
            raise RuntimeError("deterministic v5 rebuild differs")

    result = {
        "schema": "m4-downstream-quality-eos-preflight-v5-package-validation-v1",
        "status": "PASS",
        "manifest_sha256": MANIFEST_SHA256,
        "authorization_sha256": V5_AUTHORIZATION_SHA256,
        "predecessor_terminal_result_sha256": V4_TERMINAL_RESULT_SHA256,
        "source_archive_members": source_count,
        "source_archive_symlinks": source_links,
        "bridge_archive_members": bridge_count,
        "bridge_archive_symlinks": bridge_links,
        "megatron_archive_members": megatron_count,
        "megatron_archive_symlinks": megatron_links,
        "builder_sha256": BUILDER_SHA256,
        "jet_manifest_schema_passed": True,
        "bash_and_python_syntax_passed": True,
        "embedded_payload_hashes_passed": True,
        "dynamic_authority_passed": True,
        "expanded_authority_negative_control_passed": True,
        "credential_surface_absent": True,
        "mcore_interpreter_fail_closed_gate_present": True,
        "mcore_dependency_provenance_gate_present": True,
        "both_converter_invocations_use_mcore": True,
        "deterministic_rebuild_passed": True,
        "jet_retrier_enabled": False,
        "submission_attempt_limit": 1,
        "optimizer_initialized": False,
        "optimizer_steps": 0,
        "training_started": False,
        "automatic_retry": False,
        "automatic_extension": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
