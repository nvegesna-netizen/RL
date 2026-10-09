#!/usr/bin/env python3
"""Fail closed unless the restore replacement only disables Ray dashboard startup."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

import yaml


FAILURE_SHA256 = "a8e6292eb0285a6b74d3cc49da7667cc26a26bd11f25e8c1be0ed7a094cd1671"
REPAIR_SHA256 = "b5525eb18f34011c2208b2fb5416ddfbcb61f46955fed24d73d3d7463e85b5fe"
AUTHORIZATION_SHA256 = "22f104aca3098b4a87ef43e09e7cc9ee5ac9e486f91034aa72e23e7a35c790a2"
LOCK_SHA256 = "0222de8f1280139b7a20b190cb4ac5ab03b4a5d94b51d7456b2e364a89e513f3"
SOURCE_SHA256 = "9885b5e448e71d0c4946a55e2a8d4afe533af6a2926f5fbbbf97e22dc4435246"
MANIFEST_SHA256 = "53330111d90f6ae984d7e442f21ed2914af0c9abb0c76b76ba8cb63c8b9df9a2"
CUSTOM_CONFIG_SHA256 = "0b72764d2ff41c690bab0f6a28408917f3f04faa311b581c94c56cef9ab35168"
VARIABLES_SHA256 = "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356"
PROTOCOL_SHA256 = "5377529e30ffe6a27732e5ab9aaf87c09e70a6f8d2228a9ba7afc06617137129"
PACKAGE_VALIDATION_SHA256 = "27d02d72d77f395576e995a64e887de25800859f98923283681a6d3512f7b1ac"
PRIOR_LAUNCH_RECEIPT_SHA256 = "ba0338476a87ec5871282044697785098b44d44cc4a5dd72ae9ff61bb8575b08"
SOURCE_COMMIT = "1a07dfd7ec9fdeffc86195c54e30b4aa600d41ab"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def dashboard_values(source: bytes) -> list[bool]:
    tree = ast.parse(source)
    init_ray = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "init_ray"
    )
    return [
        keyword.value.value
        for node in ast.walk(init_ray)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "init"
        and any(keyword.arg == "resources" for keyword in node.keywords)
        for keyword in node.keywords
        if keyword.arg == "include_dashboard"
        and isinstance(keyword.value, ast.Constant)
        and isinstance(keyword.value.value, bool)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
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
        args.execution_lock: LOCK_SHA256,
        args.source: SOURCE_SHA256,
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
    lock = json.loads(args.execution_lock.read_bytes())
    manifest = json.loads(args.manifest.read_bytes())
    package_validation = json.loads(args.package_validation.read_bytes())
    prior_launch = json.loads(args.prior_launch_receipt.read_bytes())
    custom = yaml.safe_load(args.custom_config.read_bytes())
    variables = yaml.safe_load(args.variables.read_bytes())
    with tarfile.open(args.source, "r:gz") as archive:
        member = archive.extractfile("nemo_rl/distributed/virtual_cluster.py")
        if member is None:
            raise RuntimeError("source archive lacks virtual_cluster.py")
        archived_virtual_cluster = member.read()
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
            "terminal_pre_restore_runtime_failure": failure["status"]
            == "TERMINAL_PRE_RESTORE_RUNTIME_INFRASTRUCTURE_FAILURE"
            and failure["failure"]["operation"] == "Ray local-cluster initialization"
            and failure["classification"]["failure_class"]
            == "optional_ray_dashboard_startup"
            and not failure["classification"]["checkpoint_restore_started"],
            "checkpoint_transport_repair_verified": failure["classification"][
                "checkpoint_prefix_repair_verified"
            ]
            and failure["completed_before_failure"]["checkpoint_files_downloaded"] == 7
            and failure["completed_before_failure"]["heldout_group_files_downloaded"] == 4
            and failure["completed_before_failure"][
                "source_manifest_and_downloaded_subset_authenticated"
            ],
            "repair_is_observability_only": repair["status"]
            == "FROZEN_OPERATIONAL_RUNTIME_REPLACEMENT"
            and repair["only_change"]["old_value"] == "include_dashboard=True"
            and repair["only_change"]["new_value"] == "include_dashboard=False"
            and not repair["boundaries"]["scientific_design_changed"]
            and not repair["boundaries"]["model_or_optimizer_logic_changed"]
            and not repair["boundaries"]["runtime_limit_changed"],
            "replacement_authorized_once": authorization["status"]
            == "AUTHORIZED_BEFORE_OPERATIONAL_RUNTIME_REPLACEMENT"
            and authorization["authorized"]["replacement_eos_submission_attempts"] == 1
            and authorization["authorized"]["disable_optional_ray_dashboard"]
            and authorization["not_authorized_by_this_record"]["automatic_retry"]
            and authorization["not_authorized_by_this_record"]["automatic_extension"],
            "source_commit_bound": lock["source_commit"] == SOURCE_COMMIT
            and lock["source_archive_sha256"] == SOURCE_SHA256,
            "archived_local_dashboard_disabled": dashboard_values(
                archived_virtual_cluster
            )
            == [False],
            "package_validation_passed": package_validation["status"]
            == "PASS_AUTHORIZED_UNSUBMITTED"
            and all(package_validation["checks"].values())
            and package_validation["manifest_sha256"] == MANIFEST_SHA256,
            "deadline_false_exact": custom == expected_custom,
            "variables_empty": variables == {},
            "manifest_time_limit_14400": manifest["spec"]["time_limit"] == 14400,
            "manifest_has_no_deadline": "deadline" not in manifest["spec"]
            and "--deadline" not in manifest["spec"]["script"],
            "prior_failed_launch_bound": prior_launch["upstream_pipeline_id"] == 72611180
            and prior_launch["generated_child"]["eos_job_id"] == 477609383,
            "no_training_update": repair["preserved"]["no_training_update"],
            "no_causal_outcome_opening": repair["preserved"][
                "no_causal_arm_outcome_opening"
            ],
        }
    )
    passed = all(checks.values())
    result = {
        "schema": "m4-shield-eligible-live-restore-ray-dashboard-replacement-validation-v1",
        "status": "PASS_AUTHORIZED_UNSUBMITTED" if passed else "FAIL_REPLACEMENT_GATE",
        "failure_receipt_sha256": FAILURE_SHA256,
        "repair_protocol_sha256": REPAIR_SHA256,
        "authorization_sha256": AUTHORIZATION_SHA256,
        "execution_lock_sha256": LOCK_SHA256,
        "source_archive_sha256": SOURCE_SHA256,
        "source_commit": SOURCE_COMMIT,
        "manifest_sha256": MANIFEST_SHA256,
        "custom_config_sha256": CUSTOM_CONFIG_SHA256,
        "variables_sha256": VARIABLES_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "package_validation_sha256": PACKAGE_VALIDATION_SHA256,
        "prior_launch_receipt_sha256": PRIOR_LAUNCH_RECEIPT_SHA256,
        "checks": checks,
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
