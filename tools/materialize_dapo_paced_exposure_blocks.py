# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Atomically materialize six disjoint private paced-exposure pools."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import random
import tempfile

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from nemo_rl.algorithms.async_utils.fixed_pool import (
    load_fixed_pool_manifest,
    validate_dapo_paced_exposure_manifest_design,
    validate_fixed_pool_materialization,
)
from nemo_rl.algorithms.async_utils.paced_exposure import PacedExposureExecutionSpec
from tools import materialize_dapo_operational_latency_discovery as base
from tools import materialize_dapo_paced_exposure as single


BLOCKS = 6
PROMPTS_PER_BLOCK = 64


def select_disjoint_pools(
    prompts: Sequence[base.UniquePrompt],
    *,
    ledger_ids: set[str],
    selection_seeds: Sequence[int],
) -> tuple[tuple[base.UniquePrompt, ...], ...]:
    """Apply frozen seeds sequentially without replacement."""
    if len(selection_seeds) != BLOCKS or len(set(selection_seeds)) != BLOCKS:
        raise ValueError("expected six unique selection seeds")
    reconstructed = set().union(*single.prior._reconstructed_prior_ids(prompts))
    if not reconstructed <= ledger_ids:
        raise ValueError("prior ledger omitted reconstructed study identities")
    available = [
        prompt for prompt in prompts if prompt.canonical_sha256 not in ledger_ids
    ]
    if len(available) < BLOCKS * PROMPTS_PER_BLOCK:
        raise ValueError("insufficient fresh unique prompts for six blocks")
    pools = []
    for seed in selection_seeds:
        random.Random(seed).shuffle(available)
        pools.append(tuple(available[:PROMPTS_PER_BLOCK]))
        available = available[PROMPTS_PER_BLOCK:]
    identities = [prompt.canonical_sha256 for pool in pools for prompt in pool]
    if len(identities) != BLOCKS * PROMPTS_PER_BLOCK or len(set(identities)) != len(
        identities
    ):
        raise ValueError("selected P2b pools are not disjoint")
    return tuple(pools)


def hardlink_snapshot(source: Path, destination: Path, snapshot_raw: bytes) -> None:
    """Give each manifest a local validated snapshot without duplicating bytes."""
    source_model = source / "model_snapshot"
    destination_model = destination / "model_snapshot"
    destination_model.mkdir(parents=True)
    single.base._write(destination / "model_snapshot_manifest.v1.json", snapshot_raw)
    for item in sorted(source_model.rglob("*")):
        relative = item.relative_to(source_model)
        target = destination_model / relative
        if item.is_symlink():
            raise ValueError("model snapshot contains a symbolic link")
        if item.is_dir():
            target.mkdir()
        elif item.is_file():
            os.link(item, target)


def proposed_seeds(spec: PacedExposureExecutionSpec) -> set[int]:
    return {
        spec.selection_seed,
        spec.source_shuffle_seed,
        *spec.calibration_generation_seeds,
        *spec.cadence_pair_generation_seeds,
    }


def materialize(
    *,
    output_root: Path,
    spec_paths: Sequence[Path],
    spec_sha256: Sequence[str],
    identity_ledger_path: Path,
    seed_ledger_path: Path,
    key: bytes,
) -> None:
    if len(key) < 32:
        raise ValueError("HMAC key must contain at least 32 bytes")
    if len(spec_paths) != BLOCKS or len(spec_sha256) != BLOCKS:
        raise ValueError("expected six execution specs and hashes")
    spec_raw = [
        single.read_bound(path, digest) for path, digest in zip(spec_paths, spec_sha256)
    ]
    specs = [PacedExposureExecutionSpec.model_validate_json(raw) for raw in spec_raw]
    identity_hashes = {spec.prior_identity_ledger_sha256 for spec in specs}
    seed_hashes = {spec.prior_seed_ledger_sha256 for spec in specs}
    if len(identity_hashes) != 1 or len(seed_hashes) != 1:
        raise ValueError("P2b execution specs do not share prior-ledger bindings")
    identity_raw = single.read_bound(identity_ledger_path, identity_hashes.pop())
    seed_raw = single.read_bound(seed_ledger_path, seed_hashes.pop())
    identities = single.IdentityLedger.model_validate_json(identity_raw)
    seeds = single.SeedLedger.model_validate_json(seed_raw)
    proposed = [value for spec in specs for value in proposed_seeds(spec)]
    if len(proposed) != BLOCKS * 7 or len(set(proposed)) != len(proposed):
        raise ValueError("P2b execution specs do not contain 42 unique seeds")
    if set(proposed).intersection(seeds.used_or_reserved_seeds):
        raise ValueError("P2b execution reuses a previously used or reserved seed")
    if output_root.exists():
        raise FileExistsError("refusing to overwrite existing P2b pool root")
    output_root.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".paced-p2b-", dir=output_root.parent
    ) as temporary:
        root = Path(temporary)
        root.chmod(0o700)
        first_root = root / "block-0" / "pool"
        first_root.mkdir(parents=True)
        tokenizer, snapshot_raw, weights_sha = base._snapshot_model(first_root)
        parquet = Path(
            hf_hub_download(
                repo_id=base.DATASET_REPO,
                repo_type="dataset",
                filename=base.DATASET_FILE,
                revision=base.DATASET_REVISION,
            )
        )
        if base._sha_path(parquet) != base.DATASET_FILE_SHA256:
            raise ValueError("pinned DAPO dataset checksum mismatch")
        dataset = load_dataset(
            "parquet", data_files={"train": str(parquet)}, split="train"
        )
        prompts, conflicts, audit = base._deduplicate_dataset(dataset)
        base._validate_source_audit(conflicts, audit)
        ledger_ids = set(identities.canonical_prompt_sha256)
        pools = select_disjoint_pools(
            prompts,
            ledger_ids=ledger_ids,
            selection_seeds=[spec.selection_seed for spec in specs],
        )
        template = (Path(__file__).parents[1] / "examples/prompts/cot.txt").read_text()
        block_reports = []
        for index, (spec, raw, digest, selected) in enumerate(
            zip(specs, spec_raw, spec_sha256, pools)
        ):
            block_root = root / f"block-{index}" / "pool"
            if index:
                block_root.mkdir(parents=True)
                hardlink_snapshot(first_root, block_root, snapshot_raw)
            report = base._make_pool(
                root=block_root,
                tokenizer=tokenizer,
                prompt_template=template,
                key=key,
                protocol_raw=raw,
                snapshot_raw=snapshot_raw,
                model_weights_sha256=weights_sha,
                selection_seed=spec.selection_seed,
                generation_seed=spec.source_shuffle_seed,
                prompts=selected,
                design_id=spec.source_design_id,
                analysis_status="prospective_p2b_geometry_transfer_pool",
                protocol_sha256=digest,
            )
            base._write(block_root / "execution_spec.v1.json", raw)
            base._write(block_root / "prior_identity_ledger.v1.json", identity_raw)
            base._write(block_root / "prior_seed_ledger.v1.json", seed_raw)
            manifest_path = (
                block_root / f"fixed_pool_manifest.v1.{spec.selection_seed}.json"
            )
            manifest = load_fixed_pool_manifest(manifest_path)
            validate_fixed_pool_materialization(manifest)
            validate_dapo_paced_exposure_manifest_design(manifest)
            report.update(
                block=index,
                execution_spec_sha256=digest,
                prior_identity_ledger_sha256=spec.prior_identity_ledger_sha256,
                prior_seed_ledger_sha256=spec.prior_seed_ledger_sha256,
                excluded_prior_identities=len(ledger_ids),
                dataset_audit=audit,
                sequential_without_replacement=True,
                status="materialized_not_generated",
            )
            base._write(
                block_root / "materialization_report.v1.json",
                json.dumps(report, indent=2, sort_keys=True).encode() + b"\n",
            )
            hash_lines = [
                f"{base._sha_path(item)}  {item.name}\n"
                for item in sorted(block_root.iterdir())
                if item.is_file()
            ]
            base._write(block_root / "SHA256SUMS", "".join(hash_lines).encode())
            block_reports.append(
                {
                    "block": index,
                    "manifest_sha256": base._sha_path(manifest_path),
                    "execution_spec_sha256": digest,
                }
            )
        base._write(
            root / "materialization_report.v1.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "analysis_status": "prospective_p2b_geometry_transfer_materialization",
                    "blocks": block_reports,
                    "independent_blocks": BLOCKS,
                    "prompt_groups_per_block": PROMPTS_PER_BLOCK,
                    "total_unique_prompt_groups": BLOCKS * PROMPTS_PER_BLOCK,
                    "excluded_prior_identities": len(ledger_ids),
                    "dataset_audit": audit,
                    "status": "materialized_not_generated",
                },
                indent=2,
                sort_keys=True,
            ).encode()
            + b"\n",
        )
        for item in root.rglob("*"):
            item.chmod(0o700 if item.is_dir() else 0o600)
        root.rename(output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execution-spec", type=Path, action="append", required=True)
    parser.add_argument("--execution-spec-sha256", action="append", required=True)
    parser.add_argument("--identity-ledger", type=Path, required=True)
    parser.add_argument("--seed-ledger", type=Path, required=True)
    args = parser.parse_args()
    key = os.environ.get("DAPO_PACED_EXPOSURE_HMAC_KEY")
    if key is None:
        raise ValueError("private paced-pool HMAC key is missing")
    materialize(
        output_root=args.output_root,
        spec_paths=args.execution_spec,
        spec_sha256=args.execution_spec_sha256,
        identity_ledger_path=args.identity_ledger,
        seed_ledger_path=args.seed_ledger,
        key=key.encode(),
    )


if __name__ == "__main__":
    main()
