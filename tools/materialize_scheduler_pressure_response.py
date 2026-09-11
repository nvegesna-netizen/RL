# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Materialize one confirmed fresh scheduler pressure-response pool."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Final

from tools import materialize_structured_generation_latency_pool as base


DESIGN_ID: Final[str] = "structured_scheduler_pressure_response_v1"
PROTOCOL_SHA256: Final[str] = (
    "c35c06797483c83bd1a29f8cf856447387729e0e612480026eb84320127576cc"
)
AMENDMENT_SHA256: Final[str] = (
    "bcd0977439deb1589ee236893000c9e8c2b6867bd86eec0dc240cc1321bf2488"
)
AMENDMENT_CONFIRMATION_SHA256: Final[str] = (
    "892ccdeccc2e91a11fe1a9f692949c7cc9fddcd63b784ae7b43d3191e6a64197"
)
REPLACEMENT_AMENDMENT_SHA256: Final[str] = (
    "ede4dc56f5138e2b04f46cba7592198e2f06a7a8c1e87286a7b08103947d9013"
)
REPLACEMENT_CONFIRMATION_SHA256: Final[str] = (
    "2047291bb0d0074a0db0fe1d7cf2af4ee0ab4e17fff3d478f41a81f40843e192"
)
ORIGINAL_REPLICATION_SEEDS: Final[dict[int, tuple[int, int]]] = {
    49001: (2026091001, 69001),
    49002: (2026091002, 69002),
    49003: (2026091003, 69003),
}
REPLICATION_SEEDS: Final[dict[int, tuple[int, int]]] = {
    **ORIGINAL_REPLICATION_SEEDS,
    49004: (2026091004, 69004),
    49005: (2026091005, 69005),
}


class SchedulerPressureResponseMaterializationError(ValueError):
    """The confirmed protocol or requested replication is inconsistent."""


def _load_materialization_authorization(path: Path, *, order_seed: int) -> bytes:
    raw = path.read_bytes()
    try:
        record = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SchedulerPressureResponseMaterializationError(
            "materialization authorization is invalid JSON"
        ) from error
    if order_seed in {49004, 49005}:
        authorization = (
            record.get("authorization") if isinstance(record, dict) else None
        )
        if order_seed == 49005:
            if (
                base._sha_bytes(raw) != REPLACEMENT_CONFIRMATION_SHA256
                or not isinstance(record, dict)
                or set(record)
                != {
                    "schema_version",
                    "status",
                    "confirmed_amendment_candidate_sha256",
                    "confirmed_on",
                    "confirmation_source",
                    "source_plan_id",
                    "source_plan_file_sha256",
                    "triggering_failure_audit_sha256",
                    "authorization",
                    "required_sequence",
                }
                or record.get("schema_version") != 1
                or record.get("status")
                != "confirmed_for_bounded_runtime_repair_validation_and_replacement_pool_materialization"
                or record.get("confirmed_amendment_candidate_sha256")
                != REPLACEMENT_AMENDMENT_SHA256
                or not isinstance(authorization, dict)
                or authorization
                != {
                    "bounded_runtime_repair": True,
                    "local_validation": True,
                    "pinned_image_no_rollout_validation": True,
                    "fresh_pool_49005_materialization_and_validation_after_pinned_image_pass": True,
                    "scheduler_arm_49005": False,
                    "counterfactual_replay": False,
                    "learner_training": False,
                    "population_claim": False,
                }
            ):
                raise SchedulerPressureResponseMaterializationError(
                    "49005 replacement authorization contract mismatch"
                )
            return raw
        if (
            base._sha_bytes(raw) != AMENDMENT_CONFIRMATION_SHA256
            or not isinstance(record, dict)
            or set(record)
            != {
                "schema_version",
                "analysis_status",
                "confirmed_amendment_candidate_sha256",
                "authorization",
            }
            or record.get("schema_version") != 1
            or record.get("analysis_status")
            != "confirmed_prospective_scheduler_pressure_response_concurrency_amendment"
            or record.get("confirmed_amendment_candidate_sha256") != AMENDMENT_SHA256
            or not isinstance(authorization, dict)
            or authorization.get("materialize_and_validate_pool_49004_without_rollout")
            is not True
            or any(
                authorization.get(field) is not False
                for field in (
                    "scheduler_arm_49002",
                    "scheduler_arm_49003",
                    "scheduler_arm_49004",
                    "counterfactual_replay",
                    "learner_training",
                    "population_claim",
                )
            )
        ):
            raise SchedulerPressureResponseMaterializationError(
                "49004 amendment authorization contract mismatch"
            )
        return raw
    required = {
        "schema_version",
        "analysis_status",
        "confirmed_candidate_sha256",
        "confirmed_authorization_candidate_sha256",
        "implementation_commit",
        "jet_route_commit",
        "incremental_bundle_sha256",
        "authorization",
    }
    authorization = record.get("authorization") if isinstance(record, dict) else None
    if (
        not isinstance(record, dict)
        or set(record) != required
        or record.get("schema_version") != 1
        or record.get("analysis_status")
        != "confirmed_scheduler_pressure_response_validation_and_materialization"
        or record.get("confirmed_candidate_sha256") != PROTOCOL_SHA256
        or not isinstance(record.get("confirmed_authorization_candidate_sha256"), str)
        or not re.fullmatch(
            r"[0-9a-f]{64}", record["confirmed_authorization_candidate_sha256"]
        )
        or not isinstance(record.get("implementation_commit"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", record["implementation_commit"])
        or not isinstance(record.get("jet_route_commit"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", record["jet_route_commit"])
        or not isinstance(record.get("incremental_bundle_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", record["incremental_bundle_sha256"])
        or authorization
        != {
            "exact_image_validation": True,
            "materialize_three_fresh_pools": True,
            "scheduler_arm_rollout": False,
            "counterfactual_replay": False,
            "learner_training": False,
        }
    ):
        raise SchedulerPressureResponseMaterializationError(
            "materialization authorization contract mismatch"
        )
    return raw


def _load_confirmed_candidate(path: Path) -> bytes:
    raw = path.read_bytes()
    if base._sha_bytes(raw) != PROTOCOL_SHA256:
        raise SchedulerPressureResponseMaterializationError(
            "confirmed protocol byte hash mismatch"
        )
    try:
        protocol = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SchedulerPressureResponseMaterializationError(
            "confirmed protocol is invalid JSON"
        ) from error
    if not isinstance(protocol, dict):
        raise SchedulerPressureResponseMaterializationError(
            "confirmed protocol must be an object"
        )
    expected = {
        "schema_version": 1,
        "analysis_status": "candidate_zero_update_scheduler_pressure_response_surface",
        "calibration_only": True,
        "confirmatory_eligible": False,
        "implementation_authorized": False,
        "materialization_authorized": False,
        "eos_launch_authorized": False,
        "counterfactual_replay_authorized": False,
        "training_authorized": False,
        "requires_separate_final_plan_confirmation_before_any_eos_launch": True,
    }
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise SchedulerPressureResponseMaterializationError(
            "confirmed protocol labels mismatch"
        )
    design = protocol.get("design")
    if not isinstance(design, dict) or (
        design.get("fresh_pool_order_seeds"),
        design.get("selection_seeds"),
        design.get("generation_study_seeds"),
        design.get("prompt_groups_per_pool"),
        design.get("dispatch_cohorts"),
    ) != (
        list(ORIGINAL_REPLICATION_SEEDS),
        [value[0] for value in ORIGINAL_REPLICATION_SEEDS.values()],
        [value[1] for value in ORIGINAL_REPLICATION_SEEDS.values()],
        32,
        8,
    ):
        raise SchedulerPressureResponseMaterializationError(
            "confirmed replication or pool geometry mismatch"
        )
    return raw


def _load_amendment(path: Path) -> bytes:
    raw = path.read_bytes()
    try:
        record = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SchedulerPressureResponseMaterializationError(
            "concurrency amendment is invalid JSON"
        ) from error
    new_pool = (
        record.get("prospective_replications", {}).get("new_pool_required", {})
        if isinstance(record, dict)
        else {}
    )
    if (
        base._sha_bytes(raw) != AMENDMENT_SHA256
        or not isinstance(record, dict)
        or record.get("candidate_status") != "awaiting_exact_user_confirmation"
        or record.get("requires_exact_hash_confirmation") is not True
        or record.get("non_retroactivity", {}).get("reclassify_replication_49001")
        is not False
        or record.get("prospective_replications", {}).get("replication_order")
        != [49002, 49003, 49004]
        or (
            new_pool.get("order_seed"),
            new_pool.get("selection_seed"),
            new_pool.get("generation_study_seed"),
            new_pool.get("status"),
        )
        != (49004, 2026091004, 69004, "not_materialized")
    ):
        raise SchedulerPressureResponseMaterializationError(
            "concurrency amendment contract mismatch"
        )
    return raw


def _load_replacement_amendment(path: Path) -> bytes:
    raw = path.read_bytes()
    try:
        record = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SchedulerPressureResponseMaterializationError(
            "replacement amendment is invalid JSON"
        ) from error
    replacement = (
        record.get("prospective_replacement", {}) if isinstance(record, dict) else {}
    )
    authorization = (
        record.get("authorization_if_exactly_confirmed", {})
        if isinstance(record, dict)
        else {}
    )
    if (
        base._sha_bytes(raw) != REPLACEMENT_AMENDMENT_SHA256
        or not isinstance(record, dict)
        or record.get("status") != "candidate_awaiting_exact_confirmation"
        or record.get("analysis_status")
        != "prospective_runtime_and_replacement_amendment"
        or record.get("integrity_decisions", {}).get("replication_49004_consumed")
        is not True
        or record.get("integrity_decisions", {}).get("same_seed_retry_allowed")
        is not False
        or (
            replacement.get("replacement_order_seed"),
            replacement.get("selection_seed"),
            replacement.get("generation_study_seed"),
            replacement.get("fresh_pool_required"),
            replacement.get("replacement_replication_order"),
        )
        != (49005, 2026091005, 69005, True, [49002, 49003, 49005])
        or authorization.get("fresh_pool_49005_materialization_and_validation")
        is not True
        or any(
            authorization.get(field) is not False
            for field in (
                "scheduler_arm_49005",
                "counterfactual_replay",
                "learner_training",
                "population_claim",
            )
        )
    ):
        raise SchedulerPressureResponseMaterializationError(
            "replacement amendment contract mismatch"
        )
    return raw


def materialize(
    *,
    output_dir: Path,
    protocol_path: Path,
    authorization_path: Path,
    amendment_path: Path | None,
    order_seed: int,
    key: bytes,
) -> None:
    """Materialize the named replication with its locked fresh-selection seed."""
    try:
        selection_seed, generation_seed = REPLICATION_SEEDS[order_seed]
    except KeyError as error:
        raise SchedulerPressureResponseMaterializationError(
            f"unsupported pressure-response order seed {order_seed}"
        ) from error
    protocol_raw = _load_confirmed_candidate(protocol_path)
    authorization_raw = _load_materialization_authorization(
        authorization_path, order_seed=order_seed
    )
    amendment_raw = None
    if order_seed in {49004, 49005}:
        if amendment_path is None:
            raise SchedulerPressureResponseMaterializationError(
                f"{order_seed} requires its confirmed amendment candidate"
            )
        amendment_raw = (
            _load_amendment(amendment_path)
            if order_seed == 49004
            else _load_replacement_amendment(amendment_path)
        )
    elif amendment_path is not None:
        raise SchedulerPressureResponseMaterializationError(
            "legacy replications do not accept a concurrency amendment"
        )
    spec = base.StructuredGenerationMaterializationSpec(
        design_id=DESIGN_ID,
        analysis_status="controlled_zero_update_scheduler_pressure_response_pool",
        protocol_id=PROTOCOL_SHA256,
        selection_seed=selection_seed,
        order_seed=order_seed,
        generation_seed=generation_seed,
        source_split="scheduler_pressure_response",
        protocol_filename="scheduler_pressure_response_protocol.candidate.v1.json",
        identity_field="protocol_id",
        pair_count=16,
    )
    base._materialize_from_protocol(
        output_dir=output_dir,
        protocol_raw=protocol_raw,
        key=key,
        spec=spec,
        additional_files=(
            {
                "scheduler_pressure_response_concurrency_amendment.candidate.v1.json": amendment_raw,
                "scheduler_pressure_response_concurrency_amendment_confirmation.v1.json": authorization_raw,
            }
            if order_seed == 49004
            else {
                "scheduler_pressure_response_49004_replacement_runtime_amendment.candidate.v1.json": amendment_raw,
                "scheduler_pressure_response_49004_replacement_runtime_amendment_confirmation.v1.json": authorization_raw,
            }
            if order_seed == 49005
            else {
                "scheduler_pressure_response_materialization_authorization.v1.json": authorization_raw
            }
        ),
        additional_report_hashes=(
            {
                "concurrency_amendment_sha256": base._sha_bytes(amendment_raw),
                "concurrency_amendment_confirmation_sha256": base._sha_bytes(
                    authorization_raw
                ),
            }
            if order_seed == 49004
            else {
                "replacement_runtime_amendment_sha256": base._sha_bytes(amendment_raw),
                "replacement_runtime_amendment_confirmation_sha256": base._sha_bytes(
                    authorization_raw
                ),
            }
            if order_seed == 49005
            else {
                "materialization_authorization_sha256": base._sha_bytes(
                    authorization_raw
                )
            }
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--amendment", type=Path)
    parser.add_argument("--order-seed", type=int, required=True)
    parser.add_argument(
        "--hmac-key-env", default="STRUCTURED_GENERATION_PROMPT_HMAC_KEY"
    )
    args = parser.parse_args()
    key = os.environ.get(args.hmac_key_env)
    if key is None:
        raise SchedulerPressureResponseMaterializationError("HMAC key is missing")
    materialize(
        output_dir=args.output_dir,
        protocol_path=args.protocol,
        authorization_path=args.authorization,
        amendment_path=args.amendment,
        order_seed=args.order_seed,
        key=key.encode(),
    )


if __name__ == "__main__":
    main()
