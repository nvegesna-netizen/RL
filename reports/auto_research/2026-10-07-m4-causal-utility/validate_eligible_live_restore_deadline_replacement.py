#!/usr/bin/env python3
"""Fail closed unless the restore replacement changes only queue-deadline handling."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


FAILURE_SHA256 = "c5982623775b63c4a6e4c3b3ca4e20cd87f4c2ddcd976b6146e23360c2bff5f9"
REPAIR_SHA256 = "1f483ebe08be4b735e76e2b5a2a5c0249b228109f4592ac3023ac435582f0d18"
AUTHORIZATION_SHA256 = "9274e657453d603962ffc2a3dc2ad6ad4ec8208bbbe92c10c05cdec743b27f50"
MANIFEST_SHA256 = "0d91bc0e113ff9f358837577c6b5f4a3094299bcd58b72c3b51086e82667df1c"
CUSTOM_CONFIG_SHA256 = "0b72764d2ff41c690bab0f6a28408917f3f04faa311b581c94c56cef9ab35168"
VARIABLES_SHA256 = "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356"
PROTOCOL_SHA256 = "5377529e30ffe6a27732e5ab9aaf87c09e70a6f8d2228a9ba7afc06617137129"
ORIGINAL_LAUNCH_RECEIPT_SHA256 = (
    "523bd3dd4fae1ce619acd552c099f9007de9f6c1fa178922c8a1810586647dae"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--custom-config", type=Path, required=True)
    parser.add_argument("--variables", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--original-launch-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected = {
        args.failure: FAILURE_SHA256,
        args.repair: REPAIR_SHA256,
        args.authorization: AUTHORIZATION_SHA256,
        args.manifest: MANIFEST_SHA256,
        args.custom_config: CUSTOM_CONFIG_SHA256,
        args.variables: VARIABLES_SHA256,
        args.protocol: PROTOCOL_SHA256,
        args.original_launch_receipt: ORIGINAL_LAUNCH_RECEIPT_SHA256,
    }
    checks = {
        "frozen_input_hashes": all(sha256(path) == digest for path, digest in expected.items())
    }
    failure = json.loads(args.failure.read_bytes())
    repair = json.loads(args.repair.read_bytes())
    authorization = json.loads(args.authorization.read_bytes())
    manifest = json.loads(args.manifest.read_bytes())
    custom = yaml.safe_load(args.custom_config.read_bytes())
    variables = yaml.safe_load(args.variables.read_bytes())
    original_launch = json.loads(args.original_launch_receipt.read_bytes())
    expected_custom = {
        "launchers": {
            "dgxh100_eos": {
                "account": "coreai_dlalgo_ci",
                "partition": "batch",
                "exclusive": True,
                "sbatch_additional_flags": {
                    "deadline": False,
                    "signal": "CONT@1860",
                },
            }
        }
    }
    checks.update(
        {
            "zero_runtime_deadline_failure": failure["status"]
            == "TERMINAL_INFRASTRUCTURE_FAILURE_ZERO_RUNTIME"
            and failure["terminal"]["slurm_state"] == "DEADLINE"
            and failure["terminal"]["slurm_runtime"] == "00:00:00"
            and not failure["classification"]["workload_process_started"],
            "repair_is_operational_only": repair["status"]
            == "FROZEN_OPERATIONAL_REPLACEMENT"
            and repair["only_change"]["new_value"] is False
            and not repair["boundaries"]["scientific_design_changed"]
            and not repair["boundaries"]["manifest_changed"]
            and not repair["boundaries"]["source_changed"]
            and not repair["boundaries"]["runtime_limit_changed"],
            "replacement_authorized_once": authorization["status"]
            == "AUTHORIZED_BEFORE_OPERATIONAL_REPLACEMENT"
            and authorization["authorized"]["replacement_eos_submission_attempts"] == 1
            and authorization["authorized"]["deadline_suppression"]
            and authorization["not_authorized_by_this_record"]["automatic_retry"]
            and authorization["not_authorized_by_this_record"]["automatic_extension"],
            "deadline_false_exact": custom == expected_custom,
            "variables_empty": variables == {},
            "manifest_unchanged": sha256(args.manifest) == MANIFEST_SHA256,
            "manifest_time_limit_14400": manifest["spec"]["time_limit"] == 14400,
            "manifest_has_no_deadline": "deadline" not in manifest["spec"]
            and "--deadline" not in manifest["spec"]["script"],
            "original_launch_bound": original_launch["upstream_pipeline_id"] == 72333181
            and original_launch["generated_child"]["eos_job_id"] == 475367921,
            "no_training_update": repair["preserved"]["no_training_update"],
            "no_causal_outcome_opening": repair["preserved"][
                "no_causal_arm_outcome_opening"
            ],
        }
    )
    passed = all(checks.values())
    result = {
        "schema": "m4-shield-eligible-live-restore-deadline-replacement-validation-v1",
        "status": "PASS_AUTHORIZED_UNSUBMITTED" if passed else "FAIL_REPLACEMENT_GATE",
        "failure_receipt_sha256": FAILURE_SHA256,
        "repair_protocol_sha256": REPAIR_SHA256,
        "authorization_sha256": AUTHORIZATION_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "custom_config_sha256": CUSTOM_CONFIG_SHA256,
        "variables_sha256": VARIABLES_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "original_launch_receipt_sha256": ORIGINAL_LAUNCH_RECEIPT_SHA256,
        "checks": checks,
        "manifest_unchanged": True,
        "source_unchanged": True,
        "workload_time_limit_seconds": 14400,
        "queue_deadline_suppressed": True,
        "scientific_design_changed": False,
        "replacement_submission_attempt_limit": 1,
        "submitted": False,
        "automatic_retry": False,
        "automatic_extension": False,
    }
    args.output.write_bytes(canonical(result))
    print(json.dumps({"output": str(args.output), "status": result["status"]}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
