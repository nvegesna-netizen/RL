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

"""Materialize fresh DAPO pools for common-input mechanism validation."""

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

from tools import materialize_dapo_load_alignment as prior_load_alignment
from tools import materialize_dapo_operational_latency_discovery as base
from tools import materialize_dapo_scheduler_crossover as crossover


DESIGN_ID: Final[str] = "dapo_math_load_alignment_common_input_validation_v1"
PROTOCOL_SHA256: Final[str] = (
    "9891ce5626413ecad8df286bb89077bb68fdc43b2772379b64bf49af5e80376e"
)
CONFIRMATION_SHA256: Final[str] = (
    "e0ffb535548b2b89b222cc247b3afd6a63cb58120f61ab49c6c4810374c79266"
)
POOL_SEEDS: Final[tuple[int, ...]] = (55001, 55002, 55003)
CALIBRATION_SEEDS: Final[dict[int, tuple[int, int]]] = {
    55001: (77001, 77101),
    55002: (77002, 77102),
    55003: (77003, 77103),
}
VALIDATION_SEEDS: Final[dict[int, tuple[int, int]]] = {
    55001: (77201, 77301),
    55002: (77202, 77302),
    55003: (77203, 77303),
}
PROMPTS_PER_POOL: Final[int] = 32


class DapoFreshCommonInputMaterializationError(ValueError):
    """The fresh common-input authority or materialization is invalid."""


def _load_json(path: Path, label: str) -> tuple[bytes, dict[str, object]]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise DapoFreshCommonInputMaterializationError(
            f"cannot load {label}"
        ) from error
    if not isinstance(value, dict):
        raise DapoFreshCommonInputMaterializationError(f"{label} must be an object")
    return raw, value


def _load_protocol(path: Path) -> bytes:
    raw, value = _load_json(path, "protocol")
    if base._sha_bytes(raw) != PROTOCOL_SHA256 or (
        value.get("schema_version"),
        value.get("analysis_status"),
        value.get("candidate_status"),
        value.get("requires_exact_hash_confirmation"),
    ) != (
        1,
        "candidate_dapo_load_alignment_fresh_common_input_validation_protocol",
        "awaiting_exact_user_confirmation",
        True,
    ):
        raise DapoFreshCommonInputMaterializationError("protocol binding mismatch")
    return raw


def _load_protocol_confirmation(path: Path) -> bytes:
    raw, value = _load_json(path, "protocol confirmation")
    expected_authorized = {
        "implement_followup_code_and_local_tests": True,
        "prepare_no_rollout_validation_candidate": True,
        "eos_launch": False,
        "fresh_pool_materialization": False,
        "calibration_generation": False,
        "validation_generation": False,
        "live_scheduler_arm": False,
        "learner_replay": False,
        "learner_training": False,
        "population_claim": False,
    }
    if base._sha_bytes(raw) != CONFIRMATION_SHA256 or (
        value.get("schema_version"),
        value.get("analysis_status"),
        value.get("candidate_sha256"),
        value.get("authorized"),
    ) != (
        1,
        "dapo_load_alignment_fresh_common_input_validation_protocol_confirmation",
        PROTOCOL_SHA256,
        expected_authorized,
    ):
        raise DapoFreshCommonInputMaterializationError(
            "protocol confirmation binding mismatch"
        )
    return raw


def _load_execution_authority(path: Path) -> tuple[bytes, str]:
    raw, value = _load_json(path, "materialization/calibration authority")
    expected_authorized = {
        "eos_launch": True,
        "fresh_pool_materialization": True,
        "calibration_generation": True,
        "validation_generation": False,
        "live_scheduler_arm": False,
        "learner_replay": False,
        "learner_training": False,
        "population_claim": False,
    }
    if (
        value.get("schema_version") != 1
        or value.get("analysis_status")
        != "dapo_load_alignment_fresh_common_input_materialization_calibration_confirmation"
        or value.get("confirmed") is not True
        or value.get("protocol_sha256") != PROTOCOL_SHA256
        or value.get("protocol_confirmation_sha256") != CONFIRMATION_SHA256
        or value.get("authorization") != expected_authorized
    ):
        raise DapoFreshCommonInputMaterializationError(
            "materialization/calibration authority mismatch"
        )
    return raw, base._sha_bytes(raw)


def _load_prior_study_ids(path: Path) -> tuple[set[str], bytes]:
    raw, value = _load_json(path, "prior-study ledger")
    identities = value.get("canonical_prompt_sha256")
    if (
        value.get("schema_version") != 1
        or value.get("complete_through_materialization_start") is not True
        or value.get("includes_closed_load_alignment_pool_identities") is not True
        or not isinstance(identities, list)
        or len(identities) < 96
        or len(identities) != len(set(identities))
        or any(
            not isinstance(item, str)
            or len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
            for item in identities
        )
    ):
        raise DapoFreshCommonInputMaterializationError(
            "prior-study ledger contract mismatch"
        )
    return set(identities), raw


def _select_fresh_pools(
    prompts: Sequence[base.UniquePrompt], ledger_ids: set[str]
) -> tuple[dict[int, tuple[base.UniquePrompt, ...]], dict[str, set[str]]]:
    discovery_ids, crossover_ids, mixture_ids = (
        prior_load_alignment._reconstructed_prior_ids(prompts)
    )
    excluded = discovery_ids | crossover_ids | mixture_ids | ledger_ids
    remaining = [item for item in prompts if item.canonical_sha256 not in excluded]
    pools: dict[int, tuple[base.UniquePrompt, ...]] = {}
    for seed in POOL_SEEDS:
        random.Random(seed).shuffle(remaining)
        pools[seed] = tuple(remaining[:PROMPTS_PER_POOL])
        remaining = remaining[PROMPTS_PER_POOL:]
    selected = {item.canonical_sha256 for pool in pools.values() for item in pool}
    if len(selected) != len(POOL_SEEDS) * PROMPTS_PER_POOL or selected & excluded:
        raise DapoFreshCommonInputMaterializationError(
            "fresh pools violate exclusion or disjointness contract"
        )
    return pools, {
        "discovery": discovery_ids,
        "crossover": crossover_ids,
        "operational_mixture": mixture_ids,
        "complete_prior_study_ledger": ledger_ids,
    }


def materialize(
    *,
    output_dir: Path,
    protocol_path: Path,
    protocol_confirmation_path: Path,
    execution_authority_path: Path,
    prior_study_ledger_path: Path,
    key: bytes,
) -> None:
    """Create three candidate-bound pools after exact freshness validation."""
    if len(key) < 32:
        raise DapoFreshCommonInputMaterializationError("HMAC key is too short")
    protocol_raw = _load_protocol(protocol_path)
    confirmation_raw = _load_protocol_confirmation(protocol_confirmation_path)
    authority_raw, authority_sha256 = _load_execution_authority(
        execution_authority_path
    )
    ledger_ids, ledger_raw = _load_prior_study_ids(prior_study_ledger_path)
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
            raise DapoFreshCommonInputMaterializationError("dataset file hash mismatch")
        dataset = load_dataset(
            "parquet", data_files={"train": str(parquet_path)}, split="train"
        )
        prompts, conflicts, audit = base._deduplicate_dataset(dataset)
        base._validate_source_audit(conflicts, audit)
        selected, exclusions = _select_fresh_pools(prompts, ledger_ids)
        prompt_template = (
            Path(__file__).parents[1] / "examples/prompts/cot.txt"
        ).read_text()
        pool_reports: list[dict[str, object]] = []
        for pool_seed in POOL_SEEDS:
            report = base._make_pool(
                root=root,
                tokenizer=tokenizer,
                prompt_template=prompt_template,
                key=key,
                protocol_raw=protocol_raw,
                snapshot_raw=snapshot_raw,
                model_weights_sha256=model_weights_sha256,
                selection_seed=pool_seed,
                generation_seed=pool_seed,
                prompts=selected[pool_seed],
                design_id=DESIGN_ID,
                analysis_status="controlled_dapo_common_input_validation_pool",
                protocol_sha256=PROTOCOL_SHA256,
            )
            report.update(
                calibration_generation_seeds=list(CALIBRATION_SEEDS[pool_seed]),
                validation_generation_seeds=list(VALIDATION_SEEDS[pool_seed]),
            )
            pool_reports.append(report)
        base._write(root / "protocol.candidate.v1.json", protocol_raw)
        base._write(root / "protocol_confirmation.v1.json", confirmation_raw)
        base._write(root / "execution_authority.v1.json", authority_raw)
        base._write(root / "prior_study_ledger.v1.json", ledger_raw)
        exclusion_union = set().union(*exclusions.values())
        exclusion_record = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "counts": {name: len(values) for name, values in exclusions.items()},
            "excluded_prompt_ids": sorted(
                base._opaque(key, "fresh-common-input-exclusion", identity)
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
            "protocol_confirmation_sha256": CONFIRMATION_SHA256,
            "execution_authority_sha256": authority_sha256,
            "prior_study_ledger_sha256": base._sha_bytes(ledger_raw),
            "model_weights_sha256": model_weights_sha256,
            "dataset_audit": audit_record,
            "pools": pool_reports,
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
    parser.add_argument("--protocol-confirmation", type=Path, required=True)
    parser.add_argument("--execution-authority", type=Path, required=True)
    parser.add_argument("--prior-study-ledger", type=Path, required=True)
    parser.add_argument("--hmac-key-env", default="DAPO_FRESH_COMMON_INPUT_HMAC_KEY")
    args = parser.parse_args()
    key = os.environ.get(args.hmac_key_env)
    if key is None:
        raise DapoFreshCommonInputMaterializationError("HMAC key is missing")
    materialize(
        output_dir=args.output_dir,
        protocol_path=args.protocol,
        protocol_confirmation_path=args.protocol_confirmation,
        execution_authority_path=args.execution_authority,
        prior_study_ledger_path=args.prior_study_ledger,
        key=key.encode(),
    )


if __name__ == "__main__":
    main()
