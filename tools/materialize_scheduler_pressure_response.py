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
REPLICATION_SEEDS: Final[dict[int, tuple[int, int]]] = {
    49001: (2026091001, 69001),
    49002: (2026091002, 69002),
    49003: (2026091003, 69003),
}


class SchedulerPressureResponseMaterializationError(ValueError):
    """The confirmed protocol or requested replication is inconsistent."""


def _load_materialization_authorization(path: Path) -> bytes:
    raw = path.read_bytes()
    try:
        record = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SchedulerPressureResponseMaterializationError(
            "materialization authorization is invalid JSON"
        ) from error
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
        list(REPLICATION_SEEDS),
        [value[0] for value in REPLICATION_SEEDS.values()],
        [value[1] for value in REPLICATION_SEEDS.values()],
        32,
        8,
    ):
        raise SchedulerPressureResponseMaterializationError(
            "confirmed replication or pool geometry mismatch"
        )
    return raw


def materialize(
    *,
    output_dir: Path,
    protocol_path: Path,
    authorization_path: Path,
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
    authorization_raw = _load_materialization_authorization(authorization_path)
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
        additional_files={
            "scheduler_pressure_response_materialization_authorization.v1.json": authorization_raw
        },
        additional_report_hashes={
            "materialization_authorization_sha256": base._sha_bytes(authorization_raw)
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
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
        order_seed=args.order_seed,
        key=key.encode(),
    )


if __name__ == "__main__":
    main()
