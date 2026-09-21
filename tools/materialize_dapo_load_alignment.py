# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Materialize fresh DAPO pools for the confirmed load-alignment study."""

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
from tools import materialize_dapo_operational_mixture as prior_mixture
from tools import materialize_dapo_scheduler_crossover as crossover


DESIGN_ID: Final[str] = "dapo_math_load_alignment_v1"
PROTOCOL_SHA256: Final[str] = (
    "00e9c7d481a101cd79d3b155ada9b395346e7de0d6ca8e6e9f827846d2dc4147"
)
CONFIRMATION_SHA256: Final[str] = (
    "da1d796992a0a0d1c285dd8c834322131f30709d3f75c5cf7f07ea5faa5a844c"
)
POOL_SPECS: Final[tuple[tuple[int, int, tuple[int, int], int], ...]] = (
    (51001, 2026092001, (72001, 72011), 73001),
    (51002, 2026092002, (72002, 72012), 73002),
    (51003, 2026092003, (72003, 72013), 73003),
)
ARM_EXECUTION_ORDERS: Final[dict[int, tuple[str, ...]]] = {
    51001: (
        "l0_natural_in_order",
        "l0_natural_ready_first",
        "l3_natural_in_order",
        "l3_natural_ready_first",
        "l3_high_load_delayed_in_order",
        "l3_high_load_delayed_ready_first",
        "l3_low_load_delayed_ready_first",
        "l3_low_load_delayed_in_order",
    ),
    51002: (
        "l3_high_load_delayed_ready_first",
        "l3_high_load_delayed_in_order",
        "l3_low_load_delayed_in_order",
        "l3_low_load_delayed_ready_first",
        "l0_natural_ready_first",
        "l0_natural_in_order",
        "l3_natural_ready_first",
        "l3_natural_in_order",
    ),
    51003: (
        "l3_low_load_delayed_in_order",
        "l3_low_load_delayed_ready_first",
        "l3_natural_in_order",
        "l3_natural_ready_first",
        "l3_high_load_delayed_ready_first",
        "l3_high_load_delayed_in_order",
        "l0_natural_in_order",
        "l0_natural_ready_first",
    ),
}
PROMPTS_PER_POOL: Final[int] = 32


class DapoLoadAlignmentMaterializationError(ValueError):
    """The load-alignment authorization or materialization is invalid."""


def _load_bound_json(path: Path, expected_sha256: str, label: str) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise DapoLoadAlignmentMaterializationError(f"cannot load {label}") from error
    if base._sha_bytes(raw) != expected_sha256 or not isinstance(value, dict):
        raise DapoLoadAlignmentMaterializationError(f"{label} hash mismatch")
    return value


def _load_protocol(path: Path) -> bytes:
    raw = path.read_bytes()
    value = _load_bound_json(path, PROTOCOL_SHA256, "protocol")
    expected = {
        "schema_version": 1,
        "analysis_status": "candidate_dapo_load_alignment_signed_control_zero_update_study",
        "candidate_status": "awaiting_exact_user_confirmation",
        "implementation_authorized": False,
        "materialization_authorized": False,
        "reference_collection_authorized": False,
        "scheduler_arm_authorized": False,
        "eos_launch_authorized": False,
    }
    if any(
        value.get(key) != expected_value for key, expected_value in expected.items()
    ):
        raise DapoLoadAlignmentMaterializationError("protocol labels mismatch")
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
        "eos_launch_authorized": False,
        "materialization_authorized": False,
        "reference_collection_authorized": False,
        "scheduler_arm_authorized": False,
    }
    if any(
        value.get(key) != expected_value for key, expected_value in expected.items()
    ):
        raise DapoLoadAlignmentMaterializationError(
            "confirmation authorization boundary mismatch"
        )
    return raw


def _load_materialization_authority(path: Path) -> tuple[bytes, str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise DapoLoadAlignmentMaterializationError(
            "cannot load materialization authority"
        ) from error
    expected = {
        "schema_version": 1,
        "analysis_status": "dapo_load_alignment_materialization_confirmation",
        "candidate_sha256": PROTOCOL_SHA256,
        "confirmed": True,
        "eos_launch_authorized": True,
        "materialization_authorized": True,
        "reference_collection_authorized": True,
        "scheduler_arm_authorized": False,
        "counterfactual_replay_authorized": False,
        "learner_training_authorized": False,
        "population_claim_authorized": False,
    }
    if not isinstance(value, dict) or any(
        value.get(key) != expected_value for key, expected_value in expected.items()
    ):
        raise DapoLoadAlignmentMaterializationError(
            "materialization authority boundary mismatch"
        )
    return raw, base._sha_bytes(raw)


def _load_intervening_ids(path: Path) -> tuple[set[str], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise DapoLoadAlignmentMaterializationError(
            "invalid prior-study ledger"
        ) from error
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("complete_through_materialization_start") is not True
        or not isinstance(value.get("canonical_prompt_sha256"), list)
    ):
        raise DapoLoadAlignmentMaterializationError(
            "prior-study ledger contract mismatch"
        )
    identities = value["canonical_prompt_sha256"]
    if any(
        not isinstance(item, str)
        or len(item) != 64
        or any(character not in "0123456789abcdef" for character in item)
        for item in identities
    ) or len(identities) != len(set(identities)):
        raise DapoLoadAlignmentMaterializationError("prior-study identities invalid")
    return set(identities), raw


def _reconstructed_prior_ids(
    prompts: Sequence[base.UniquePrompt],
) -> tuple[set[str], set[str], set[str]]:
    discovery = base._select_disjoint_pools(prompts)
    discovery_ids = {
        item.canonical_sha256 for pool in discovery.values() for item in pool
    }
    crossover_pools, reported = crossover._select_holdout_pools(prompts)
    crossover_ids = {
        item.canonical_sha256 for pool in crossover_pools.values() for item in pool
    }
    mixture_pools, mixture_discovery, mixture_crossover = (
        prior_mixture._select_fresh_pools(prompts, set())
    )
    mixture_ids = {
        item.canonical_sha256 for pool in mixture_pools.values() for item in pool
    }
    if (
        discovery_ids != reported
        or discovery_ids != mixture_discovery
        or crossover_ids != mixture_crossover
        or len(discovery_ids) != 48
        or len(crossover_ids) != 64
        or len(mixture_ids) != 96
        or discovery_ids & crossover_ids
        or (discovery_ids | crossover_ids) & mixture_ids
    ):
        raise DapoLoadAlignmentMaterializationError(
            "prior-study exclusion reconstruction mismatch"
        )
    return discovery_ids, crossover_ids, mixture_ids


def _select_fresh_pools(
    prompts: Sequence[base.UniquePrompt], ledger_ids: set[str]
) -> tuple[dict[int, tuple[base.UniquePrompt, ...]], dict[str, set[str]]]:
    discovery_ids, crossover_ids, mixture_ids = _reconstructed_prior_ids(prompts)
    excluded = discovery_ids | crossover_ids | mixture_ids | ledger_ids
    remaining = [item for item in prompts if item.canonical_sha256 not in excluded]
    output: dict[int, tuple[base.UniquePrompt, ...]] = {}
    for selection_seed, _, _, _ in POOL_SPECS:
        random.Random(selection_seed).shuffle(remaining)
        output[selection_seed] = tuple(remaining[:PROMPTS_PER_POOL])
        remaining = remaining[PROMPTS_PER_POOL:]
    selected = [item.canonical_sha256 for pool in output.values() for item in pool]
    if len(selected) != 96 or len(set(selected)) != 96 or excluded & set(selected):
        raise DapoLoadAlignmentMaterializationError("fresh pools violate exclusions")
    return output, {
        "discovery": discovery_ids,
        "crossover": crossover_ids,
        "operational_mixture": mixture_ids,
        "ledger": ledger_ids,
    }


def materialize(
    *,
    output_dir: Path,
    protocol_path: Path,
    confirmation_path: Path,
    materialization_authority_path: Path,
    prior_study_ledger_path: Path,
    key: bytes,
) -> None:
    """Create candidate-bound pools after exact source and freshness validation."""
    if len(key) < 32:
        raise DapoLoadAlignmentMaterializationError("HMAC key is too short")
    protocol_raw = _load_protocol(protocol_path)
    confirmation_raw = _load_confirmation(confirmation_path)
    materialization_authority_raw, materialization_authority_sha256 = (
        _load_materialization_authority(materialization_authority_path)
    )
    ledger_ids, ledger_raw = _load_intervening_ids(prior_study_ledger_path)
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
            raise DapoLoadAlignmentMaterializationError("dataset file hash mismatch")
        dataset = load_dataset(
            "parquet", data_files={"train": str(parquet_path)}, split="train"
        )
        prompts, conflicts, audit = base._deduplicate_dataset(dataset)
        base._validate_source_audit(conflicts, audit)
        selected, exclusions = _select_fresh_pools(prompts, ledger_ids)
        prompt_template = (
            Path(__file__).parents[1] / "examples/prompts/cot.txt"
        ).read_text()
        reports = []
        for pool_seed, dispatch_seed, reference_seeds, scheduler_seed in POOL_SPECS:
            report = base._make_pool(
                root=root,
                tokenizer=tokenizer,
                prompt_template=prompt_template,
                key=key,
                protocol_raw=protocol_raw,
                snapshot_raw=snapshot_raw,
                model_weights_sha256=model_weights_sha256,
                selection_seed=pool_seed,
                generation_seed=dispatch_seed,
                prompts=selected[pool_seed],
                design_id=DESIGN_ID,
                analysis_status="controlled_dapo_load_alignment_pool",
                protocol_sha256=PROTOCOL_SHA256,
            )
            report.update(
                reference_generation_seeds=list(reference_seeds),
                scheduler_generation_seed=scheduler_seed,
                scheduler_selection_seed=dispatch_seed,
                arm_execution_order=list(ARM_EXECUTION_ORDERS[pool_seed]),
            )
            reports.append(report)
        base._write(root / "load_alignment_protocol.candidate.v1.json", protocol_raw)
        base._write(root / "load_alignment_confirmation.v1.json", confirmation_raw)
        base._write(
            root / "load_alignment_materialization_authority.v1.json",
            materialization_authority_raw,
        )
        base._write(root / "prior_study_ledger.v1.json", ledger_raw)
        exclusion_union = set().union(*exclusions.values())
        exclusion_record = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "counts": {name: len(values) for name, values in exclusions.items()},
            "excluded_prompt_ids": sorted(
                base._opaque(key, "load-alignment-exclusion", identity)
                for identity in exclusion_union
            ),
        }
        base._write(
            root / "private_exclusion_ledger.v1.json",
            json.dumps(exclusion_record, indent=2, sort_keys=True).encode() + b"\n",
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
            "materialization_authority_sha256": materialization_authority_sha256,
            "prior_study_ledger_sha256": base._sha_bytes(ledger_raw),
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
    parser.add_argument("--materialization-authority", type=Path, required=True)
    parser.add_argument("--prior-study-ledger", type=Path, required=True)
    parser.add_argument("--hmac-key-env", default="DAPO_LOAD_ALIGNMENT_HMAC_KEY")
    args = parser.parse_args()
    key = os.environ.get(args.hmac_key_env)
    if key is None:
        raise DapoLoadAlignmentMaterializationError("HMAC key is missing")
    materialize(
        output_dir=args.output_dir,
        protocol_path=args.protocol,
        confirmation_path=args.confirmation,
        materialization_authority_path=args.materialization_authority,
        prior_study_ledger_path=args.prior_study_ledger,
        key=key.encode(),
    )


if __name__ == "__main__":
    main()
