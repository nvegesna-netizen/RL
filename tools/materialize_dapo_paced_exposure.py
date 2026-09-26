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

"""Materialize one private paced-pilot pool from launcher-bound prior ledgers."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, model_validator

from nemo_rl.algorithms.async_utils.paced_exposure import (
    PacedExposureExecutionSpec,
    Seed,
    Sha256Hex,
)
from nemo_rl.algorithms.async_utils.fixed_pool import (
    load_fixed_pool_manifest,
    validate_dapo_paced_exposure_manifest_design,
    validate_fixed_pool_materialization,
)
from tools import materialize_dapo_operational_latency_discovery as base
from tools import materialize_dapo_load_alignment as prior


class IdentityLedger(BaseModel, extra="forbid", frozen=True):
    """Private canonical identities; provenance must be checked by the launcher."""

    canonical_prompt_sha256: tuple[Sha256Hex, ...]
    source_artifact_sha256: tuple[Sha256Hex, ...]

    @model_validator(mode="after")
    def _unique(self) -> IdentityLedger:
        if len(self.canonical_prompt_sha256) != len(set(self.canonical_prompt_sha256)):
            raise ValueError("prior identity ledger contains duplicates")
        if not self.source_artifact_sha256 or not self.canonical_prompt_sha256:
            raise ValueError("prior identity evidence is missing")
        return self


class SeedLedger(BaseModel, extra="forbid", frozen=True):
    used_or_reserved_seeds: tuple[Seed, ...]
    source_artifact_sha256: tuple[Sha256Hex, ...]

    @model_validator(mode="after")
    def _nonempty(self) -> SeedLedger:
        if not self.used_or_reserved_seeds or not self.source_artifact_sha256:
            raise ValueError("prior seed evidence is missing")
        return self


def read_bound(path: Path, expected_sha256: str) -> bytes:
    """Read only the exact file approved in the execution spec or launch route."""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("paced materialization input hash mismatch")
    return raw


def select_fresh_pool(
    prompts: Sequence[base.UniquePrompt], *, ledger_ids: set[str], selection_seed: int
) -> tuple[base.UniquePrompt, ...]:
    """Uniformly sample 64 eligible identities without replacing rejected rows."""
    reconstructed = set().union(*prior._reconstructed_prior_ids(prompts))
    if not reconstructed <= ledger_ids:
        raise ValueError("prior ledger omitted reconstructed study identities")
    available = [
        prompt for prompt in prompts if prompt.canonical_sha256 not in ledger_ids
    ]
    if len(available) < 64:
        raise ValueError("insufficient fresh unique prompts")
    random.Random(selection_seed).shuffle(available)
    selected = tuple(available[:64])
    if len({prompt.canonical_sha256 for prompt in selected}) != 64:
        raise ValueError("deduplicated input contains repeated selected identities")
    return selected


def materialize(
    *,
    output_dir: Path,
    spec_path: Path,
    spec_sha256: str,
    identity_ledger_path: Path,
    seed_ledger_path: Path,
    key: bytes,
) -> None:
    """Create a private pool once; never substitute prompts or overwrite output."""
    if len(key) < 32:
        raise ValueError("HMAC key must contain at least 32 bytes")
    spec_raw = read_bound(spec_path, spec_sha256)
    spec = PacedExposureExecutionSpec.model_validate_json(spec_raw)
    identity_raw = read_bound(identity_ledger_path, spec.prior_identity_ledger_sha256)
    seed_raw = read_bound(seed_ledger_path, spec.prior_seed_ledger_sha256)
    identities = IdentityLedger.model_validate_json(identity_raw)
    seeds = SeedLedger.model_validate_json(seed_raw)
    proposed = {
        spec.selection_seed,
        spec.source_shuffle_seed,
        *spec.calibration_generation_seeds,
        *spec.cadence_pair_generation_seeds,
    }
    if proposed.intersection(seeds.used_or_reserved_seeds):
        raise ValueError("paced execution reuses a previously used or reserved seed")
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite existing paced pool")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".paced-pool-", dir=output_dir.parent
    ) as temporary:
        root = Path(temporary)
        root.chmod(0o700)
        tokenizer, snapshot_raw, weights_sha = base._snapshot_model(root)
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
        selected = select_fresh_pool(
            prompts, ledger_ids=ledger_ids, selection_seed=spec.selection_seed
        )
        template = (Path(__file__).parents[1] / "examples/prompts/cot.txt").read_text()
        report = base._make_pool(
            root=root,
            tokenizer=tokenizer,
            prompt_template=template,
            key=key,
            protocol_raw=spec_raw,
            snapshot_raw=snapshot_raw,
            model_weights_sha256=weights_sha,
            selection_seed=spec.selection_seed,
            generation_seed=spec.source_shuffle_seed,
            prompts=selected,
            design_id=spec.source_design_id,
            analysis_status="exploratory_paced_exposure_pool",
            protocol_sha256=spec_sha256,
        )
        base._write(root / "execution_spec.v1.json", spec_raw)
        base._write(root / "prior_identity_ledger.v1.json", identity_raw)
        base._write(root / "prior_seed_ledger.v1.json", seed_raw)
        manifest = load_fixed_pool_manifest(
            root / f"fixed_pool_manifest.v1.{spec.selection_seed}.json"
        )
        validate_fixed_pool_materialization(manifest)
        validate_dapo_paced_exposure_manifest_design(manifest)
        report.update(
            execution_spec_sha256=spec_sha256,
            prior_identity_ledger_sha256=spec.prior_identity_ledger_sha256,
            prior_seed_ledger_sha256=spec.prior_seed_ledger_sha256,
            excluded_prior_identities=len(ledger_ids),
            dataset_audit=audit,
            status="materialized_not_generated",
        )
        base._write(
            root / "materialization_report.v1.json",
            json.dumps(report, indent=2, sort_keys=True).encode() + b"\n",
        )
        hash_lines = [
            f"{base._sha_path(item)}  {item.name}\n"
            for item in sorted(root.iterdir())
            if item.is_file()
        ]
        base._write(root / "SHA256SUMS", "".join(hash_lines).encode())
        for item in root.rglob("*"):
            item.chmod(0o700 if item.is_dir() else 0o600)
        root.rename(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-spec", type=Path, required=True)
    parser.add_argument("--execution-spec-sha256", required=True)
    parser.add_argument("--identity-ledger", type=Path, required=True)
    parser.add_argument("--seed-ledger", type=Path, required=True)
    args = parser.parse_args()
    key = os.environ.get("DAPO_PACED_EXPOSURE_HMAC_KEY")
    if key is None:
        raise ValueError("private paced-pool HMAC key is missing")
    materialize(
        output_dir=args.output_dir,
        spec_path=args.execution_spec,
        spec_sha256=args.execution_spec_sha256,
        identity_ledger_path=args.identity_ledger,
        seed_ledger_path=args.seed_ledger,
        key=key.encode(),
    )


if __name__ == "__main__":
    main()
