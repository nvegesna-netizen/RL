#!/usr/bin/env python3
"""Fail closed unless the restore replacement changes only checkpoint URL roots."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml


FAILURE_SHA256 = "8d28f46d905c712e9f1eb327f5009de03f1642d380644010d2801ebc65eb85c3"
REPAIR_SHA256 = "9a635f2f3eb20d9abde91f6cfaf6055ae0fc3783648dae3a6b0df2d21035d148"
AUTHORIZATION_SHA256 = "eb5ea9a86639d2522193d445f8bd3755e35efdc47bd911d4ddafdb6c9cb394c2"
MANIFEST_SHA256 = "c4be912b90151660bc4e93a171d002aa5b1be03215bf125a555f50f6b47682ac"
CUSTOM_CONFIG_SHA256 = "0b72764d2ff41c690bab0f6a28408917f3f04faa311b581c94c56cef9ab35168"
VARIABLES_SHA256 = "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356"
PROTOCOL_SHA256 = "5377529e30ffe6a27732e5ab9aaf87c09e70a6f8d2228a9ba7afc06617137129"
PACKAGE_VALIDATION_SHA256 = "f0ce22ef06efb9100df1b59b7bb378adad33a2079a92c10301a3c2b623ebcb7c"
PRIOR_LAUNCH_RECEIPT_SHA256 = "10112b2f18156a4c1424be3b2c81f27d19f18d5c5d45658fdb893b4fb92e2d4e"


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
    parser.add_argument("--package-validation", type=Path, required=True)
    parser.add_argument("--prior-launch-receipt", type=Path, required=True)
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
        args.package_validation: PACKAGE_VALIDATION_SHA256,
        args.prior_launch_receipt: PRIOR_LAUNCH_RECEIPT_SHA256,
    }
    checks = {
        "frozen_input_hashes": all(sha256(path) == digest for path, digest in expected.items())
    }
    failure = json.loads(args.failure.read_bytes())
    repair = json.loads(args.repair.read_bytes())
    authorization = json.loads(args.authorization.read_bytes())
    manifest = json.loads(args.manifest.read_bytes())
    package_validation = json.loads(args.package_validation.read_bytes())
    prior_launch = json.loads(args.prior_launch_receipt.read_bytes())
    custom = yaml.safe_load(args.custom_config.read_bytes())
    variables = yaml.safe_load(args.variables.read_bytes())
    script = manifest["spec"]["script"].replace("{{", "{").replace("}}", "}")
    remote_downloads = re.findall(r'download "(\$API/[^"]+)"', script)
    checkpoint_downloads = [path for path in remote_downloads if "/policy/" in path]
    heldout_downloads = [path for path in remote_downloads if "/groups/" in path]
    local_checkpoint_destinations = re.findall(
        r'download "\$API/checkpoint/policy/[^"]+" "(\$CAPSULE/checkpoint/policy/[^"]+)"',
        script,
    )
    expected_custom = {
        "launchers": {
            "dgxh100_eos": {
                "account": "coreai_dlalgo_ci",
                "partition": "batch",
                "exclusive": True,
                "sbatch_additional_flags": {"deadline": False, "signal": "CONT@1860"},
            }
        }
    }
    checks.update(
        {
            "terminal_pre_restore_transport_failure": failure["status"]
            == "TERMINAL_PRE_RESTORE_TRANSPORT_FAILURE"
            and failure["classification"]["failure_class"]
            == "checkpoint_remote_prefix_omitted"
            and failure["runtime_evidence"]["http_status"] == 404
            and not failure["classification"]["checkpoint_restore_started"],
            "corrected_probe_authenticated": failure["runtime_evidence"][
                "corrected_path_probe_sha256"
            ]
            == failure["runtime_evidence"]["expected_metadata_sha256"]
            and failure["runtime_evidence"]["corrected_path_probe_bytes"] == 82311,
            "repair_is_transport_only": repair["status"]
            == "FROZEN_TRANSPORT_REPLACEMENT"
            and repair["only_change"]["old_template"] == "$API/policy/weights/..."
            and repair["only_change"]["new_template"]
            == "$API/checkpoint/policy/weights/..."
            and not repair["boundaries"]["scientific_design_changed"]
            and not repair["boundaries"]["runtime_source_changed"]
            and not repair["boundaries"]["runtime_limit_changed"],
            "replacement_authorized_once": authorization["status"]
            == "AUTHORIZED_BEFORE_TRANSPORT_REPLACEMENT"
            and authorization["authorized"]["replacement_eos_submission_attempts"] == 1
            and authorization["authorized"]["checkpoint_remote_prefix_repair"]
            and authorization["not_authorized_by_this_record"]["automatic_retry"]
            and authorization["not_authorized_by_this_record"]["automatic_extension"],
            "checkpoint_urls_have_manifest_root": len(checkpoint_downloads) == 7
            and all(path.startswith("$API/checkpoint/policy/") for path in checkpoint_downloads)
            and "$API/policy/" not in script,
            "heldout_urls_unchanged": len(heldout_downloads) == 4
            and all(path.startswith("$API/groups/") for path in heldout_downloads),
            "local_checkpoint_layout_unchanged": len(local_checkpoint_destinations) == 7,
            "package_validation_passed": package_validation["status"]
            == "PASS_AUTHORIZED_UNSUBMITTED"
            and all(package_validation["checks"].values())
            and package_validation["manifest_sha256"] == MANIFEST_SHA256,
            "deadline_false_exact": custom == expected_custom,
            "variables_empty": variables == {},
            "manifest_time_limit_14400": manifest["spec"]["time_limit"] == 14400,
            "manifest_has_no_deadline": "deadline" not in manifest["spec"]
            and "--deadline" not in script,
            "prior_failed_launch_bound": prior_launch["upstream_pipeline_id"] == 72379365
            and prior_launch["generated_child"]["eos_job_id"] == 475731787,
            "no_training_update": repair["preserved"]["no_training_update"],
            "no_causal_outcome_opening": repair["preserved"][
                "no_causal_arm_outcome_opening"
            ],
        }
    )
    passed = all(checks.values())
    result = {
        "schema": "m4-shield-eligible-live-restore-checkpoint-prefix-replacement-validation-v1",
        "status": "PASS_AUTHORIZED_UNSUBMITTED" if passed else "FAIL_REPLACEMENT_GATE",
        "failure_receipt_sha256": FAILURE_SHA256,
        "repair_protocol_sha256": REPAIR_SHA256,
        "authorization_sha256": AUTHORIZATION_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "custom_config_sha256": CUSTOM_CONFIG_SHA256,
        "variables_sha256": VARIABLES_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "package_validation_sha256": PACKAGE_VALIDATION_SHA256,
        "prior_launch_receipt_sha256": PRIOR_LAUNCH_RECEIPT_SHA256,
        "checks": checks,
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
