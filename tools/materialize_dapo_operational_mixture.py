# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Materialize the three confirmed fresh DAPO operational-mixture pools."""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from tools import materialize_dapo_operational_latency_discovery as base
from tools import materialize_dapo_scheduler_crossover as crossover


DESIGN_ID: Final[str] = "dapo_math_operational_mixture_v1"
PROTOCOL_SHA256: Final[str] = (
    "e9942c4afbec0b3b622079d1208eaba6a7c83a7759fc307cfd47293e797a9f63"
)
CONFIRMATION_SHA256: Final[str] = (
    "f6a55f89f641d46b3942fbc05eac46f071665eb21435211f035dc3ad99faf969"
)
POOL_SPECS: Final[tuple[tuple[int, int, tuple[int, int], int], ...]] = (
    (50001, 2026091301, (70001, 70011), 71001),
    (50002, 2026091302, (70002, 70012), 71002),
    (50003, 2026091303, (70003, 70013), 71003),
)
ARM_EXECUTION_ORDERS: Final[dict[int, tuple[str, ...]]] = {
    50001: (
        "l0_in_order",
        "l0_ready_first",
        "l1_ready_first",
        "l1_in_order",
        "l3_in_order",
        "l3_ready_first",
    ),
    50002: (
        "l1_in_order",
        "l1_ready_first",
        "l3_ready_first",
        "l3_in_order",
        "l0_ready_first",
        "l0_in_order",
    ),
    50003: (
        "l3_in_order",
        "l3_ready_first",
        "l0_in_order",
        "l0_ready_first",
        "l1_ready_first",
        "l1_in_order",
    ),
}
PROMPTS_PER_POOL: Final[int] = 32
PRIOR_DISCOVERY_PROMPTS: Final[int] = 48
PRIOR_CROSSOVER_PROMPTS: Final[int] = 64


class DapoOperationalMixtureMaterializationError(ValueError):
    """The confirmed protocol or fresh-pool materialization is invalid."""


def _load_bound_json(path: Path, expected_sha256: str, label: str) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise DapoOperationalMixtureMaterializationError(
            f"cannot load {label}"
        ) from error
    if base._sha_bytes(raw) != expected_sha256 or not isinstance(value, dict):
        raise DapoOperationalMixtureMaterializationError(f"{label} hash mismatch")
    return value


def _load_protocol(path: Path) -> bytes:
    raw = path.read_bytes()
    value = _load_bound_json(path, PROTOCOL_SHA256, "protocol")
    expected = {
        "schema_version": 1,
        "analysis_status": "candidate_dapo_operational_mixture_zero_update_shadow_study",
        "candidate_status": "awaiting_exact_user_confirmation",
        "implementation_authorized": False,
        "materialization_authorized": False,
        "reference_collection_authorized": False,
        "scheduler_arm_authorized": False,
    }
    if any(
        value.get(key) != expected_value for key, expected_value in expected.items()
    ):
        raise DapoOperationalMixtureMaterializationError("protocol labels mismatch")
    return raw


def _load_confirmation(path: Path) -> bytes:
    raw = path.read_bytes()
    value = _load_bound_json(path, CONFIRMATION_SHA256, "confirmation")
    expected = {
        "schema_version": 1,
        "candidate_sha256": PROTOCOL_SHA256,
        "confirmed": True,
        "implementation_authorized": True,
        "local_validation_authorized": True,
        "materialization_authorized": False,
        "reference_collection_authorized": False,
        "scheduler_arm_authorized": False,
    }
    if any(
        value.get(key) != expected_value for key, expected_value in expected.items()
    ):
        raise DapoOperationalMixtureMaterializationError(
            "confirmation authorization boundary mismatch"
        )
    return raw


def _load_intervening_ids(path: Path) -> tuple[set[str], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise DapoOperationalMixtureMaterializationError(
            "intervening-study ledger is invalid JSON"
        ) from error
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("complete_through_materialization_start") is not True
        or not isinstance(value.get("canonical_prompt_sha256"), list)
    ):
        raise DapoOperationalMixtureMaterializationError(
            "intervening-study ledger contract mismatch"
        )
    identities = value["canonical_prompt_sha256"]
    if any(
        not isinstance(item, str)
        or len(item) != 64
        or any(character not in "0123456789abcdef" for character in item)
        for item in identities
    ) or len(identities) != len(set(identities)):
        raise DapoOperationalMixtureMaterializationError(
            "intervening-study identities are invalid or duplicated"
        )
    return set(identities), raw


def _prior_ids(
    prompts: Sequence[base.UniquePrompt],
) -> tuple[set[str], set[str]]:
    discovery = base._select_disjoint_pools(prompts)
    discovery_ids = {
        item.canonical_sha256 for pool in discovery.values() for item in pool
    }
    crossover_pools, reported_discovery_ids = crossover._select_holdout_pools(prompts)
    crossover_ids = {
        item.canonical_sha256 for pool in crossover_pools.values() for item in pool
    }
    if (
        discovery_ids != reported_discovery_ids
        or len(discovery_ids) != PRIOR_DISCOVERY_PROMPTS
        or len(crossover_ids) != PRIOR_CROSSOVER_PROMPTS
        or discovery_ids.intersection(crossover_ids)
    ):
        raise DapoOperationalMixtureMaterializationError(
            "prior-study exclusion reconstruction mismatch"
        )
    return discovery_ids, crossover_ids


def _select_fresh_pools(
    prompts: Sequence[base.UniquePrompt], intervening_ids: set[str]
) -> tuple[dict[int, tuple[base.UniquePrompt, ...]], set[str], set[str]]:
    discovery_ids, crossover_ids = _prior_ids(prompts)
    excluded = discovery_ids | crossover_ids | intervening_ids
    remaining = [item for item in prompts if item.canonical_sha256 not in excluded]
    output: dict[int, tuple[base.UniquePrompt, ...]] = {}
    for selection_seed, _, _, _ in POOL_SPECS:
        random.Random(selection_seed).shuffle(remaining)
        output[selection_seed] = tuple(remaining[:PROMPTS_PER_POOL])
        remaining = remaining[PROMPTS_PER_POOL:]
    selected_ids = [item.canonical_sha256 for pool in output.values() for item in pool]
    if (
        len(selected_ids) != len(POOL_SPECS) * PROMPTS_PER_POOL
        or len(selected_ids) != len(set(selected_ids))
        or excluded.intersection(selected_ids)
    ):
        raise DapoOperationalMixtureMaterializationError(
            "fresh operational-mixture pools overlap or violate exclusions"
        )
    return output, discovery_ids, crossover_ids


def materialize(
    *,
    output_dir: Path,
    protocol_path: Path,
    confirmation_path: Path,
    intervening_ledger_path: Path,
    key: bytes,
) -> None:
    """Create three immutable pools after exact source and freshness validation."""
    if len(key) < 32:
        raise DapoOperationalMixtureMaterializationError("HMAC key is too short")
    protocol_raw = _load_protocol(protocol_path)
    confirmation_raw = _load_confirmation(confirmation_path)
    intervening_ids, intervening_raw = _load_intervening_ids(intervening_ledger_path)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.", dir=output_dir.parent
    ) as temporary_name:
        root = Path(temporary_name)
        root.chmod(0o700)
        tokenizer, snapshot_raw, model_weights_sha256 = base._snapshot_model(root)
        parquet_path = Path(
            hf_hub_download(
                repo_id=base.DATASET_REPO,
                repo_type="dataset",
                filename=base.DATASET_FILE,
                revision=base.DATASET_REVISION,
            )
        )
        if base._sha_path(parquet_path) != base.DATASET_FILE_SHA256:
            raise DapoOperationalMixtureMaterializationError(
                "dataset file hash mismatch"
            )
        dataset = load_dataset(
            "parquet", data_files={"train": str(parquet_path)}, split="train"
        )
        prompts, conflicts, audit = base._deduplicate_dataset(dataset)
        base._validate_source_audit(conflicts, audit)
        selected, discovery_ids, crossover_ids = _select_fresh_pools(
            prompts, intervening_ids
        )
        prompt_template = (
            Path(__file__).parents[1] / "examples/prompts/cot.txt"
        ).read_text()
        reports = []
        for (
            selection_seed,
            dispatch_seed,
            reference_seeds,
            scheduler_seed,
        ) in POOL_SPECS:
            report = base._make_pool(
                root=root,
                tokenizer=tokenizer,
                prompt_template=prompt_template,
                key=key,
                protocol_raw=protocol_raw,
                snapshot_raw=snapshot_raw,
                model_weights_sha256=model_weights_sha256,
                selection_seed=selection_seed,
                generation_seed=dispatch_seed,
                prompts=selected[selection_seed],
                design_id=DESIGN_ID,
                analysis_status="controlled_dapo_operational_mixture_pool",
                protocol_sha256=PROTOCOL_SHA256,
            )
            report.update(
                reference_generation_seeds=list(reference_seeds),
                scheduler_generation_seed=scheduler_seed,
                scheduler_selection_seed=dispatch_seed,
                arm_execution_order=list(ARM_EXECUTION_ORDERS[selection_seed]),
            )
            reports.append(report)
        base._write(
            root / "operational_mixture_protocol.candidate.v1.json", protocol_raw
        )
        base._write(root / "operational_mixture_confirmation.v1.json", confirmation_raw)
        base._write(root / "intervening_study_ledger.v1.json", intervening_raw)
        exclusion_ledger = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "conflicted_identity_count": len(conflicts),
            "prior_discovery_identity_count": len(discovery_ids),
            "prior_crossover_identity_count": len(crossover_ids),
            "intervening_identity_count": len(intervening_ids),
            "excluded_prompt_ids": sorted(
                base._opaque(key, "operational-mixture-exclusion", identity)
                for identity in discovery_ids | crossover_ids | intervening_ids
            ),
        }
        base._write(
            root / "private_exclusion_ledger.v1.json",
            json.dumps(exclusion_ledger, indent=2, sort_keys=True).encode() + b"\n",
        )
        audit_record = crossover._audit_record(audit)
        base._write(
            root / "dataset_audit.v1.json",
            json.dumps(audit_record, indent=2, sort_keys=True).encode() + b"\n",
        )
        report = {
            "schema_version": 1,
            "status": "passed",
            "design_id": DESIGN_ID,
            "protocol_sha256": PROTOCOL_SHA256,
            "confirmation_sha256": CONFIRMATION_SHA256,
            "intervening_ledger_sha256": base._sha_bytes(intervening_raw),
            "model_weights_sha256": model_weights_sha256,
            "dataset_audit": audit_record,
            "pools": reports,
        }
        base._write(
            root / "materialization_report.v1.json",
            json.dumps(report, indent=2, sort_keys=True).encode() + b"\n",
        )
        hash_lines = [
            f"{base._sha_path(path)}  {path.name}\n"
            for path in sorted(root.iterdir())
            if path.is_file() and path.name != "SHA256SUMS"
        ]
        base._write(root / "SHA256SUMS", "".join(hash_lines).encode())
        for directory in [root, *(path for path in root.rglob("*") if path.is_dir())]:
            directory.chmod(0o700)
        for path in root.rglob("*"):
            if path.is_file():
                path.chmod(0o600)
        root.rename(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--intervening-ledger", type=Path, required=True)
    parser.add_argument("--hmac-key-env", default="DAPO_OPERATIONAL_MIXTURE_HMAC_KEY")
    args = parser.parse_args()
    key = os.environ.get(args.hmac_key_env)
    if key is None:
        raise DapoOperationalMixtureMaterializationError("HMAC key is missing")
    materialize(
        output_dir=args.output_dir,
        protocol_path=args.protocol,
        confirmation_path=args.confirmation,
        intervening_ledger_path=args.intervening_ledger,
        key=key.encode(),
    )


if __name__ == "__main__":
    main()
