#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Transform one repaired nonlaunchable candidate into an authorized candidate."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path

from build_confirmatory_candidates import (
    ANALYSIS_PLAN_SHA256,
    AUTHORIZATION_SHA256 as LOCAL_AUTHORIZATION_SHA256,
    PROTOCOL_SHA256,
    QUALIFICATION_GATE_SHA256,
    RUN_MANIFEST_SHA256,
    SOURCE_COMMIT,
)

ACQUISITION_AUTHORIZATION_SHA256 = (
    "90e48c344e5056121a020e16e0b49db0203026a8e524c73bf8101069b154384a"
)
ACQUISITION_AMENDMENT_SHA256 = (
    "34818c509452507f0868b2223880fa48eed3c3d396dcff9f312d571abb29f8a9"
)
REPAIR_VALIDATION_SHA256 = (
    "b28cb2a8ced5eeddc7a05973d0bd388e96d2656b3d755f8b3346e63c4d0dde2d"
)
REPAIR_COMMIT = "4416ee32b144bb9345b333ed40a4d336d4a01eb7"

LOCAL_SCHEMA = "m4-oars-randomized-confirmatory-local-package-authorization-v1"
ACQUISITION_SCHEMA = "m4-oars-randomized-confirmatory-acquisition-authorization-v1"
LOCAL_SCOPE = (
    "build_and_clean_room_validate_credential_free_20_run_confirmatory_candidates_only"
)
ACQUISITION_SCOPE = (
    "execute_the_frozen_20_run_confirmatory_acquisition_in_strict_global_sequence"
)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload(path: Path) -> str:
    """Return a base64-encoded file payload."""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def replace_exact(text: str, old: str, new: str, count: int) -> str:
    """Apply a replacement only when its exact expected count is present."""
    observed = text.count(old)
    if observed != count:
        raise RuntimeError(
            f"candidate transformation count differs: expected {count}, got {observed}"
        )
    return text.replace(old, new)


def validate_authority(
    authorization_path: Path,
    amendment_path: Path,
    run_manifest_path: Path,
) -> list[dict[str, object]]:
    """Validate the positive authority and unchanged registered sequence."""
    authorization = json.loads(authorization_path.read_bytes())
    amendment = json.loads(amendment_path.read_bytes())
    run_manifest = json.loads(run_manifest_path.read_bytes())
    runs = run_manifest["runs"]
    identities = [str(run["identity"]) for run in runs]
    if (
        authorization["schema"] != ACQUISITION_SCHEMA
        or authorization["scope"] != ACQUISITION_SCOPE
        or authorization["user_authorization"]
        != "do the sequence. I authorize launches on eos."
        or authorization["protocol_sha256"] != PROTOCOL_SHA256
        or authorization["run_manifest_sha256"] != RUN_MANIFEST_SHA256
        or authorization["analysis_plan_sha256"] != ANALYSIS_PLAN_SHA256
        or authorization["qualification_gate_sha256"] != QUALIFICATION_GATE_SHA256
        or authorization["runtime_source_commit"] != SOURCE_COMMIT
        or authorization["candidate_count"] != 20
        or authorization["authorized_identities"] != identities
        or authorization["required_launcher"] != "runllm.py --no_wait"
        or authorization["sequence_policy"]
        != "strict_global_sequence_one_run_at_a_time"
        or authorization["submission_attempt_limit_per_identity"] != 1
        or authorization["total_submission_attempt_limit"] != 20
        or authorization["gpus_per_run"] != 2
        or authorization["scheduler_time_limit_seconds"] != 14400
        or authorization["queue_deadline_override"] is not None
        or not authorization["credential_free"]
        or not authorization["eos_submission_authorized"]
        or not authorization["model_weight_access_authorized"]
        or not authorization["optimizer_initialization_authorized"]
        or not authorization["training_authorized"]
        or not authorization["scientific_acquisition_authorized"]
        or not authorization["complete_outcome_embargo_until_all_20_authenticated"]
        or authorization["automatic_retry"]
        or authorization["automatic_replacement"]
        or authorization["automatic_extension"]
    ):
        raise RuntimeError("confirmatory acquisition authority differs")
    if (
        amendment["schema"]
        != "m4-oars-randomized-confirmatory-acquisition-amendment-v1"
        or amendment["status"] != "FROZEN_BEFORE_ACQUISITION_PACKAGE_BUILD"
        or amendment["repair_provenance"]["repair_commit"] != REPAIR_COMMIT
        or amendment["repair_provenance"]["repair_validation_sha256"]
        != REPAIR_VALIDATION_SHA256
        or amendment["execution"]["first_identity"] != identities[0]
        or amendment["execution"]["first_identity_predecessor"] is not None
        or amendment["execution"]["later_identity_gate"]
        != "authenticated_terminal_predecessor_required"
        or not amendment["execution"]["one_shot_guard_per_identity"]
        or not amendment["execution"]["strict_global_sequence_one_run_at_a_time"]
        or not amendment["execution"][
            "complete_outcome_embargo_until_all_20_authenticated"
        ]
        or amendment["execution"]["automatic_retry"]
        or amendment["execution"]["automatic_replacement"]
        or amendment["execution"]["automatic_extension"]
    ):
        raise RuntimeError("confirmatory acquisition amendment differs")
    return runs


def main() -> None:
    """Write one launchable candidate using only frozen operational changes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--repair-validation", type=Path, required=True)
    parser.add_argument("--local-authorization", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path, expected in (
        (args.repair_validation, REPAIR_VALIDATION_SHA256),
        (args.local_authorization, LOCAL_AUTHORIZATION_SHA256),
        (args.authorization, ACQUISITION_AUTHORIZATION_SHA256),
        (args.amendment, ACQUISITION_AMENDMENT_SHA256),
        (args.run_manifest, RUN_MANIFEST_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen acquisition input moved: {path.name}")
    runs = validate_authority(args.authorization, args.amendment, args.run_manifest)
    run = next((item for item in runs if item["identity"] == args.identity), None)
    if run is None:
        raise RuntimeError("identity is not registered")
    validation = json.loads(args.repair_validation.read_bytes())
    record = next(
        (
            item
            for item in validation["candidates"]
            if item["identity"] == args.identity
        ),
        None,
    )
    if record is None or sha256(args.candidate) != record["sha256"]:
        raise RuntimeError("repaired candidate is not authenticated")

    historical = json.loads(args.candidate.read_bytes())
    candidate = copy.deepcopy(historical)
    old_name = f"m4-oars-confirmatory-{args.identity}-local-candidate"
    new_name = f"m4-oars-confirmatory-{args.identity}-acquisition"
    if candidate["spec"]["name"] != old_name:
        raise RuntimeError("repaired candidate name differs")
    candidate["spec"]["name"] = new_name

    old_authority_check = (
        'assert a["runtime_source_archive_build_authorized"] and '
        'a["credential_free_eos_package_build_authorized"] and '
        'a["clean_room_validation_authorized"]'
    )
    new_authority_check = (
        f'assert a["protocol_sha256"]=="{PROTOCOL_SHA256}" and '
        f'a["run_manifest_sha256"]=="{RUN_MANIFEST_SHA256}" and '
        f'a["analysis_plan_sha256"]=="{ANALYSIS_PLAN_SHA256}" and '
        f'a["qualification_gate_sha256"]=="{QUALIFICATION_GATE_SHA256}"'
    )
    old_block = (
        'blocked=("model_weight_access_authorized","optimizer_initialization_'
        'authorized","training_authorized","eos_submission_authorized",'
        '"scientific_acquisition_authorized","automatic_retry","automatic_'
        'replacement","automatic_extension")\n'
        "assert not any(a[key] for key in blocked)"
    )
    new_block = (
        'required=("model_weight_access_authorized","optimizer_initialization_'
        'authorized","training_authorized","eos_submission_authorized",'
        '"scientific_acquisition_authorized")\n'
        "assert all(a[key] for key in required)\n"
        'forbidden=("automatic_retry","automatic_replacement",'
        '"automatic_extension")\n'
        "assert not any(a[key] for key in forbidden)\n"
        'assert a["required_launcher"]=="runllm.py --no_wait" and '
        'a["sequence_policy"]=="strict_global_sequence_one_run_at_a_time"\n'
        'assert a["submission_attempt_limit_per_identity"]==1 and '
        'a["total_submission_attempt_limit"]==20\n'
        'assert a["complete_outcome_embargo_until_all_20_authenticated"]'
    )
    old_run_line = (
        f'run=next(item for item in r["runs"] if item["identity"]=="{args.identity}")'
    )
    new_run_line = (
        old_run_line
        + '\nassert a["authorized_identities"]=='
        + json.dumps([item["identity"] for item in runs], separators=(",", ":"))
        + f'\nassert a["authorized_identities"][run["global_sequence"]-1]=='
        f'"{args.identity}"'
    )
    replacements = [
        (payload(args.local_authorization), payload(args.authorization), 1),
        (LOCAL_AUTHORIZATION_SHA256, ACQUISITION_AUTHORIZATION_SHA256, 2),
        (LOCAL_SCHEMA, ACQUISITION_SCHEMA, 1),
        (LOCAL_SCOPE, ACQUISITION_SCOPE, 1),
        (old_authority_check, new_authority_check, 1),
        (old_block, new_block, 1),
        (old_run_line, new_run_line, 1),
        (
            "M4_OARS_CONFIRMATORY_LOCAL_PACKAGE_NO_EXECUTION_AUTHORITY",
            "M4_OARS_CONFIRMATORY_ACQUISITION_AUTHORITY_MISSING",
            1,
        ),
    ]
    script = str(candidate["spec"]["script"])
    for old, new, count in replacements:
        script = replace_exact(script, old, new, count)
    candidate["spec"]["script"] = script

    restored = copy.deepcopy(candidate)
    restored["spec"]["name"] = old_name
    restored_script = str(restored["spec"]["script"])
    for old, new, count in reversed(replacements):
        restored_script = replace_exact(restored_script, new, old, count)
    restored["spec"]["script"] = restored_script
    if restored != historical:
        raise RuntimeError("acquisition transformation exceeds frozen boundary")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(candidate, separators=(",", ":")) + "\n")
    print(args.output, sha256(args.output))


if __name__ == "__main__":
    main()
