#!/usr/bin/env python3
"""Fail-closed validation for the deadline-suppressed OARS replacement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


AMENDMENT_SHA256 = "037420f15d914deb05e1a2ee4ccf30328e89683aee342d3246d73ac719efe41a"
AUTHORIZATION_SHA256 = (
    "b220d569fea7addf7d9c90266f98fb0e6b8a771b7086ba32f3ddd289f1fe5c3b"
)
FAILURE_RECEIPT_SHA256 = (
    "61ff6ce04314f1e4f320ca40375ce0370e222c855a3ee0c180c462600c757bb7"
)
MANIFEST_SHA256 = "82e6d89d6dbb86dc42c7fad3a7560545e9a27ed11c2d40c6dd4b8bda11404192"
CUSTOM_CONFIG_SHA256 = (
    "b5db8882c7baccd3bf8c0369d7264e80f7ea08c17482559f82875abc5207f42e"
)
ADDITIONAL_VARIABLES_SHA256 = (
    "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356"
)
RUNLLM_SHA256 = "30532cd638d8394fa0e651daf4a43ec9424cbaa0949cd59c323191177318e616"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--failure-receipt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--custom-config", type=Path, required=True)
    parser.add_argument("--additional-variables", type=Path, required=True)
    parser.add_argument("--runllm", type=Path, required=True)
    parser.add_argument("--jet-slurm-renderer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path, expected in (
        (args.amendment, AMENDMENT_SHA256),
        (args.authorization, AUTHORIZATION_SHA256),
        (args.failure_receipt, FAILURE_RECEIPT_SHA256),
        (args.manifest, MANIFEST_SHA256),
        (args.custom_config, CUSTOM_CONFIG_SHA256),
        (args.additional_variables, ADDITIONAL_VARIABLES_SHA256),
        (args.runllm, RUNLLM_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"replacement input moved: {path.name}")

    amendment = json.loads(args.amendment.read_bytes())
    authorization = json.loads(args.authorization.read_bytes())
    failure = json.loads(args.failure_receipt.read_bytes())
    manifest = json.loads(args.manifest.read_bytes())
    custom = json.loads(args.custom_config.read_bytes())
    additional = json.loads(args.additional_variables.read_bytes())
    expected_custom = {
        "launchers": {"dgxh100_eos": {"sbatch_additional_flags": {"deadline": False}}}
    }
    if custom != expected_custom or additional != {}:
        raise RuntimeError("replacement transport surface differs")
    if (
        amendment["status"] != "FROZEN_OPERATIONAL_REPLACEMENT"
        or amendment["failed_submission_receipt_sha256"] != FAILURE_RECEIPT_SHA256
        or amendment["only_change"]["new_value"] is not False
        or amendment["boundaries"]["scientific_design_changed"]
        or amendment["boundaries"]["runtime_limit_changed"]
        or amendment["boundaries"]["replacement_submission_attempt_limit"] != 1
    ):
        raise RuntimeError("operational amendment differs")
    required_true = (
        "eos_submission_authorized",
        "deadline_suppression_authorized",
        "fifo_controlled_training_authorized",
        "oars_shadow_observation_authorized",
    )
    blocked = (
        "oars_actuation_authorized",
        "scientific_outcome_acquisition_authorized",
        "manifest_change_authorized",
        "source_change_authorized",
        "runtime_limit_change_authorized",
        "automatic_retry",
        "automatic_extension",
    )
    if (
        authorization["amendment_sha256"] != AMENDMENT_SHA256
        or authorization["failed_submission_receipt_sha256"] != FAILURE_RECEIPT_SHA256
        or authorization["manifest_sha256"] != MANIFEST_SHA256
        or authorization["custom_config_sha256"] != CUSTOM_CONFIG_SHA256
        or authorization["required_launcher"] != "runllm.py --no_wait"
        or authorization["replacement_submission_attempt_limit"] != 1
        or not all(authorization[key] for key in required_true)
        or any(authorization[key] for key in blocked)
    ):
        raise RuntimeError("replacement authority differs")
    if (
        failure["status"] != "TERMINAL_INFRASTRUCTURE_FAILURE_ZERO_RUNTIME"
        or failure["terminal"]["slurm_runtime"] != "00:00:00"
        or failure["declared_workload_artifacts_present"]
        or failure["replacement_authorized"]
    ):
        raise RuntimeError("replacement is not rooted in the zero-runtime failure")

    spec = manifest["spec"]
    script = spec["script"].format(assets_dir="/tmp/m4-oars-shadow-preflight")
    if (
        spec["time_limit"] != 14400
        or spec["nodes"] != 1
        or manifest["launchers"] != {"type:slurm": {"nodes": 1, "ntasks_per_node": 1}}
        or "--time" in script
        or "deadline" in script.lower()
        or script.count("examples/run_grpo_single_controller.py") != 1
    ):
        raise RuntimeError("unchanged manifest or runtime boundary differs")

    renderer = args.jet_slurm_renderer.read_text(encoding="utf-8")
    if (
        "if v is False:" not in renderer
        or "Additional sbatch flag {k} is set as False and thus will be skipped"
        not in renderer
        or 'flag = f\'--{k.replace("_", "-").lower()}\'' not in renderer
    ):
        raise RuntimeError("local JET renderer lacks boolean-false suppression")

    result = {
        "schema": "m4-oars-shadow-preflight-replacement-validation-v1",
        "status": "PASS_AUTHORIZED_UNSUBMITTED",
        "amendment_sha256": AMENDMENT_SHA256,
        "authorization_sha256": AUTHORIZATION_SHA256,
        "failed_submission_receipt_sha256": FAILURE_RECEIPT_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "custom_config_sha256": CUSTOM_CONFIG_SHA256,
        "additional_variables_sha256": ADDITIONAL_VARIABLES_SHA256,
        "runllm_sha256": RUNLLM_SHA256,
        "jet_slurm_renderer_sha256": sha256(args.jet_slurm_renderer),
        "deadline_value_is_boolean_false": True,
        "deadline_false_suppression_supported": True,
        "manifest_unchanged": True,
        "source_unchanged": True,
        "manifest_time_limit_seconds": 14400,
        "scientific_design_changed": False,
        "replacement_submission_attempt_limit": 1,
        "submitted": False,
        "automatic_retry": False,
        "automatic_extension": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
