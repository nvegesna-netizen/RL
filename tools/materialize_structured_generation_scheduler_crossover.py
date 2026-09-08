# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Materialize one confirmed fresh structured scheduler-crossover pool."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Final

from tools import materialize_structured_generation_latency_pool as base


DESIGN_ID: Final[str] = "structured_generation_scheduler_crossover_v1"
PROTOCOL_SHA256: Final[str] = (
    "ddd7536aa3d67351974629c2c56fbe41456b9e3b48b0d26e47ef7f7e2728f7ef"
)
REPLICATION_SEEDS: Final[dict[int, tuple[int, int]]] = {
    46001: (2026090801, 65001),
    46002: (2026090802, 65002),
    46003: (2026090803, 65003),
    46004: (2026090804, 65004),
}


class StructuredSchedulerCrossoverMaterializationError(ValueError):
    """The confirmed protocol or requested replication is inconsistent."""


def _load_confirmed_candidate(path: Path) -> bytes:
    raw = path.read_bytes()
    if base._sha_bytes(raw) != PROTOCOL_SHA256:
        raise StructuredSchedulerCrossoverMaterializationError(
            "confirmed protocol byte hash mismatch"
        )
    try:
        protocol = json.loads(raw)
    except json.JSONDecodeError as error:
        raise StructuredSchedulerCrossoverMaterializationError(
            "confirmed protocol is invalid JSON"
        ) from error
    if not isinstance(protocol, dict):
        raise StructuredSchedulerCrossoverMaterializationError(
            "confirmed protocol must be an object"
        )
    expected_labels = {
        "analysis_status": "candidate_controlled_natural_latency_scheduler_crossover",
        "calibration_only": True,
        "candidate_status": "awaiting_explicit_user_confirmation",
        "confirmatory_eligible": False,
        "schema_version": 1,
    }
    if any(protocol.get(key) != value for key, value in expected_labels.items()):
        raise StructuredSchedulerCrossoverMaterializationError(
            "confirmed protocol labels mismatch"
        )
    authorization = protocol.get("authorization")
    if not isinstance(authorization, dict) or any(
        value is not False for value in authorization.values()
    ):
        raise StructuredSchedulerCrossoverMaterializationError(
            "candidate authorization fields were mutated"
        )
    observed = {}
    for replication in protocol.get("replications", []):
        if not isinstance(replication, dict):
            raise StructuredSchedulerCrossoverMaterializationError(
                "invalid replication entry"
            )
        order_seed = replication.get("order_seed")
        selection_seed = replication.get("selection_seed")
        generation_seed = replication.get("generation_study_seed")
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (order_seed, selection_seed, generation_seed)
        ):
            raise StructuredSchedulerCrossoverMaterializationError(
                "replication seeds must be integers"
            )
        observed[order_seed] = (selection_seed, generation_seed)
    if observed != REPLICATION_SEEDS:
        raise StructuredSchedulerCrossoverMaterializationError(
            "confirmed replication seeds mismatch"
        )
    return raw


def materialize(
    *, output_dir: Path, protocol_path: Path, order_seed: int, key: bytes
) -> None:
    """Materialize the named replication with its locked fresh-selection seed."""
    try:
        selection_seed, generation_seed = REPLICATION_SEEDS[order_seed]
    except KeyError as error:
        raise StructuredSchedulerCrossoverMaterializationError(
            f"unsupported crossover order seed {order_seed}"
        ) from error
    protocol_raw = _load_confirmed_candidate(protocol_path)
    spec = base.StructuredGenerationMaterializationSpec(
        design_id=DESIGN_ID,
        analysis_status="controlled_natural_latency_scheduler_crossover_pool",
        protocol_id=PROTOCOL_SHA256,
        selection_seed=selection_seed,
        order_seed=order_seed,
        generation_seed=generation_seed,
        source_split="scheduler_crossover",
        protocol_filename="crossover_protocol.candidate.v1.json",
        identity_field="protocol_id",
    )
    base._materialize_from_protocol(
        output_dir=output_dir,
        protocol_raw=protocol_raw,
        key=key,
        spec=spec,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--order-seed", type=int, required=True)
    parser.add_argument(
        "--hmac-key-env", default="STRUCTURED_GENERATION_PROMPT_HMAC_KEY"
    )
    args = parser.parse_args()
    key = os.environ.get(args.hmac_key_env)
    if key is None:
        raise StructuredSchedulerCrossoverMaterializationError("HMAC key is missing")
    materialize(
        output_dir=args.output_dir,
        protocol_path=args.protocol,
        order_seed=args.order_seed,
        key=key.encode(),
    )


if __name__ == "__main__":
    main()
