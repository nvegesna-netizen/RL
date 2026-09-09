# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Materialize four fresh, disjoint DAPO scheduler-crossover holdout pools."""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from tools import materialize_dapo_operational_latency_discovery as base


DESIGN_ID: Final[str] = "dapo_math_scheduler_crossover_v1"
PROTOCOL_SHA256: Final[str] = (
    "c49f9604225db847eded1d298c592938299fb3c3d508cf60d2b598f1e8b604c7"
)
CONFIRMATION_SHA256: Final[str] = (
    "a73a4b56744cc31fe5a50f4cccb2b1dd985357ca1879b8377df19c6b1622e91e"
)
POOL_SPECS: Final[tuple[tuple[int, int, int], ...]] = (
    (48001, 58001, 68001),
    (48002, 58002, 68002),
    (48003, 58003, 68003),
    (48004, 58004, 68004),
)
PROMPTS_PER_POOL: Final[int] = 16
DISCOVERY_PROMPTS: Final[int] = 48


class DapoCrossoverMaterializationError(ValueError):
    """The protocol, source, or generated holdout materialization is invalid."""


def _load_protocol(path: Path) -> bytes:
    raw = path.read_bytes()
    if base._sha_bytes(raw) != PROTOCOL_SHA256:
        raise DapoCrossoverMaterializationError("protocol byte hash mismatch")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise DapoCrossoverMaterializationError("protocol is invalid JSON") from error
    expected_labels = {
        "analysis_status": "candidate_controlled_dapo_natural_latency_scheduler_crossover",
        "calibration_only": True,
        "candidate_status": "awaiting_explicit_user_confirmation",
        "confirmatory_eligible": False,
        "schema_version": 1,
    }
    if not isinstance(value, dict) or any(
        value.get(key) != expected for key, expected in expected_labels.items()
    ):
        raise DapoCrossoverMaterializationError("protocol labels mismatch")
    authorization = value.get("authorization")
    if not isinstance(authorization, dict) or any(
        item is not False for item in authorization.values()
    ):
        raise DapoCrossoverMaterializationError(
            "candidate authorization fields were mutated"
        )
    expected_replications = [
        {
            "arm_execution_order": list(order),
            "dispatch_order_seed": dispatch_seed,
            "generation_study_seed": generation_seed,
            "pool_selection_seed": selection_seed,
            "replication_id": f"replication_{selection_seed}",
        }
        for (selection_seed, dispatch_seed, generation_seed), order in zip(
            POOL_SPECS,
            (
                ("ready_first", "in_order"),
                ("in_order", "ready_first"),
                ("in_order", "ready_first"),
                ("ready_first", "in_order"),
            ),
            strict=True,
        )
    ]
    if value.get("replications") != expected_replications:
        raise DapoCrossoverMaterializationError("replication contract mismatch")
    return raw


def _load_confirmation(path: Path) -> bytes:
    raw = path.read_bytes()
    if base._sha_bytes(raw) != CONFIRMATION_SHA256:
        raise DapoCrossoverMaterializationError("confirmation byte hash mismatch")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise DapoCrossoverMaterializationError(
            "confirmation is invalid JSON"
        ) from error
    expected_authorization = {
        "counterfactual_replay": False,
        "exact_image_validation": True,
        "implementation": True,
        "learner_training": False,
        "materialization_of_four_fresh_pools": True,
        "read_only_pool_validation": True,
        "scheduler_arm_launch": False,
    }
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("confirmed_candidate_sha256") != PROTOCOL_SHA256
        or value.get("authorization") != expected_authorization
    ):
        raise DapoCrossoverMaterializationError("confirmation contract mismatch")
    return raw


def _select_holdout_pools(
    prompts: Sequence[base.UniquePrompt],
) -> tuple[dict[int, tuple[base.UniquePrompt, ...]], set[str]]:
    discovery = base._select_disjoint_pools(prompts)
    discovery_ids = {
        item.canonical_sha256 for pool in discovery.values() for item in pool
    }
    if len(discovery_ids) != DISCOVERY_PROMPTS:
        raise DapoCrossoverMaterializationError("discovery exclusion is not exact")
    remaining = [
        item for item in prompts if item.canonical_sha256 not in discovery_ids
    ]
    output: dict[int, tuple[base.UniquePrompt, ...]] = {}
    for selection_seed, _, _ in POOL_SPECS:
        random.Random(selection_seed).shuffle(remaining)
        output[selection_seed] = tuple(remaining[:PROMPTS_PER_POOL])
        remaining = remaining[PROMPTS_PER_POOL:]
    holdout_ids = [
        item.canonical_sha256 for pool in output.values() for item in pool
    ]
    if (
        len(holdout_ids) != 4 * PROMPTS_PER_POOL
        or len(set(holdout_ids)) != len(holdout_ids)
        or discovery_ids.intersection(holdout_ids)
    ):
        raise DapoCrossoverMaterializationError(
            "holdout pools are overlapping or not fresh"
        )
    return output, discovery_ids


def _audit_record(audit: Mapping[str, int]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_repo": base.DATASET_REPO,
        "dataset_revision": base.DATASET_REVISION,
        "dataset_file": base.DATASET_FILE,
        "dataset_file_sha256": base.DATASET_FILE_SHA256,
        **audit,
    }


def materialize(
    *, output_dir: Path, protocol_path: Path, confirmation_path: Path, key: bytes
) -> None:
    """Create all four frozen holdout pools and their shared model snapshot."""
    if len(key) < 32:
        raise DapoCrossoverMaterializationError("HMAC key is too short")
    protocol_raw = _load_protocol(protocol_path)
    confirmation_raw = _load_confirmation(confirmation_path)
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
            raise DapoCrossoverMaterializationError("dataset file hash mismatch")
        dataset = load_dataset(
            "parquet", data_files={"train": str(parquet_path)}, split="train"
        )
        prompts, conflicts, audit = base._deduplicate_dataset(dataset)
        base._validate_source_audit(conflicts, audit)
        selected, discovery_ids = _select_holdout_pools(prompts)
        prompt_template = (
            Path(__file__).parents[1] / "examples/prompts/cot.txt"
        ).read_text()
        reports = [
            base._make_pool(
                root=root,
                tokenizer=tokenizer,
                prompt_template=prompt_template,
                key=key,
                protocol_raw=protocol_raw,
                snapshot_raw=snapshot_raw,
                model_weights_sha256=model_weights_sha256,
                selection_seed=selection_seed,
                generation_seed=generation_seed,
                prompts=selected[selection_seed],
                design_id=DESIGN_ID,
                analysis_status="controlled_dapo_scheduler_crossover_pool",
                protocol_sha256=PROTOCOL_SHA256,
            )
            for selection_seed, _, generation_seed in POOL_SPECS
        ]
        base._write(root / "scheduler_crossover_protocol.candidate.v1.json", protocol_raw)
        base._write(root / "scheduler_crossover_confirmation.v1.json", confirmation_raw)
        exclusion_ledger = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "conflicted_identity_count": len(conflicts),
            "discovery_identity_count": len(discovery_ids),
            "conflicted_prompt_ids": sorted(
                base._opaque(key, "excluded-conflict", item.canonical_sha256)
                for item in conflicts
            ),
            "discovery_prompt_ids": sorted(
                base._opaque(key, "excluded-discovery", item)
                for item in discovery_ids
            ),
        }
        base._write(
            root / "private_holdout_exclusion_ledger.v1.json",
            json.dumps(exclusion_ledger, indent=2, sort_keys=True).encode() + b"\n",
        )
        audit_record = _audit_record(audit)
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
            "discovery_protocol_sha256": base.PROTOCOL_SHA256,
            "discovery_prompt_identities_excluded": len(discovery_ids),
            "conflict_prompt_identities_excluded": len(conflicts),
            "model_repo": base.MODEL_REPO,
            "model_revision": base.MODEL_REVISION,
            "model_weights_sha256": model_weights_sha256,
            "dataset_audit": audit_record,
            "holdout_exclusion_ledger_sha256": base._sha_path(
                root / "private_holdout_exclusion_ledger.v1.json"
            ),
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
    parser.add_argument("--hmac-key-env", default="DAPO_CROSSOVER_PROMPT_HMAC_KEY")
    args = parser.parse_args()
    key = os.environ.get(args.hmac_key_env)
    if key is None:
        raise DapoCrossoverMaterializationError("HMAC key is missing")
    materialize(
        output_dir=args.output_dir,
        protocol_path=args.protocol,
        confirmation_path=args.confirmation,
        key=key.encode(),
    )


if __name__ == "__main__":
    main()
