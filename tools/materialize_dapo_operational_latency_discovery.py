# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Materialize three disjoint DAPO operational-latency discovery pools."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer

from nemo_rl.algorithms.async_utils.fixed_pool import compute_fixed_pool_id


DESIGN_ID: Final[str] = "dapo_math_operational_latency_discovery_v1"
PROTOCOL_SHA256: Final[str] = (
    "24cb689d6f600b64fe9a986c69482d44547ca3fd7e9858730e32d8e12afb77c9"
)
DATASET_REPO: Final[str] = "BytedTsinghua-SIA/DAPO-Math-17k"
DATASET_REVISION: Final[str] = "65877096c24ffa7abc4e4fa5edb95cf3413a5674"
DATASET_FILE: Final[str] = "data/dapo-math-17k.parquet"
DATASET_FILE_SHA256: Final[str] = (
    "534375d6bb8630d22ab46a56e11f2ffec1d288d8f7d04099bc82d68948705941"
)
MODEL_REPO: Final[str] = "Qwen/Qwen2.5-Math-7B"
MODEL_REVISION: Final[str] = "b101308fe89651ea5ce025f25317fea6fc07e96e"
SOURCE_IDS: Final[tuple[str, str]] = ("dapo_math_a", "dapo_math_b")
POOL_SPECS: Final[tuple[tuple[int, int], ...]] = (
    (47001, 67001),
    (47002, 67002),
    (47003, 67003),
)
PROMPTS_PER_POOL: Final[int] = 16
COHORT_SIZE: Final[int] = 4


class DapoOperationalMaterializationError(ValueError):
    """The protocol, source data, or generated materialization is invalid."""


@dataclass(frozen=True, slots=True)
class UniquePrompt:
    """One unique canonical DAPO prompt and its source lineage."""

    canonical_sha256: str
    prompt: str
    ground_truth: str
    first_source_index: int
    source_extra_index: str
    duplicate_count: int


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _opaque(key: bytes, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _write(path: Path, value: bytes) -> None:
    path.write_bytes(value)
    path.chmod(0o600)


def _load_protocol(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    if _sha_bytes(raw) != PROTOCOL_SHA256:
        raise DapoOperationalMaterializationError("protocol byte hash mismatch")
    try:
        protocol = json.loads(raw)
    except json.JSONDecodeError as error:
        raise DapoOperationalMaterializationError("protocol is invalid JSON") from error
    if not isinstance(protocol, dict):
        raise DapoOperationalMaterializationError("protocol must be an object")
    expected = {
        "analysis_status": "candidate_operational_workload_latency_discovery",
        "calibration_only": True,
        "candidate_status": "awaiting_explicit_user_confirmation",
        "confirmatory_eligible": False,
        "schema_version": 1,
    }
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise DapoOperationalMaterializationError("protocol labels mismatch")
    authorization = protocol.get("authorization")
    if not isinstance(authorization, dict) or any(authorization.values()):
        raise DapoOperationalMaterializationError(
            "candidate protocol must not contain an authorization"
        )
    dataset = protocol.get("dataset")
    if not isinstance(dataset, dict) or (
        dataset.get("repo_id"),
        dataset.get("revision"),
        dataset.get("file_sha256"),
        dataset.get("split"),
    ) != (DATASET_REPO, DATASET_REVISION, DATASET_FILE_SHA256, "train"):
        raise DapoOperationalMaterializationError("dataset pin mismatch")
    runtime = protocol.get("runtime")
    if not isinstance(runtime, dict) or (
        runtime.get("prompt_groups_per_pool"),
        runtime.get("num_generations_per_prompt"),
        runtime.get("train_steps"),
        runtime.get("physical_weight_version"),
    ) != (PROMPTS_PER_POOL, 16, 0, 0):
        raise DapoOperationalMaterializationError("runtime contract mismatch")
    pools = protocol.get("pools")
    observed_specs = (
        tuple(
            (pool.get("prompt_selection_seed"), pool.get("generation_seed"))
            for pool in pools
        )
        if isinstance(pools, list) and all(isinstance(pool, dict) for pool in pools)
        else ()
    )
    if observed_specs != POOL_SPECS:
        raise DapoOperationalMaterializationError("pool seed schedule mismatch")
    return protocol, raw


def _snapshot_model(root: Path) -> tuple[Any, bytes, str]:
    model_root = root / "model_snapshot"
    snapshot_download(repo_id=MODEL_REPO, revision=MODEL_REVISION, local_dir=model_root)
    if any(path.is_symlink() for path in model_root.rglob("*")):
        raise DapoOperationalMaterializationError(
            "model snapshot contains a symbolic link"
        )
    records = [
        {
            "path": str(path.relative_to(model_root)),
            "bytes": path.stat().st_size,
            "sha256": _sha_path(path),
        }
        for path in sorted(model_root.rglob("*"))
        if path.is_file() and ".cache" not in path.parts
    ]
    weight_records = [
        record for record in records if str(record["path"]).endswith(".safetensors")
    ]
    if not weight_records:
        raise DapoOperationalMaterializationError(
            "model snapshot contains no safetensor weights"
        )
    weights_sha256 = (
        str(weight_records[0]["sha256"])
        if len(weight_records) == 1 and weight_records[0]["path"] == "model.safetensors"
        else _sha_bytes(_canonical(weight_records))
    )
    snapshot_raw = json.dumps(records, indent=2, sort_keys=True).encode() + b"\n"
    _write(root / "model_snapshot_manifest.v1.json", snapshot_raw)
    return AutoTokenizer.from_pretrained(model_root), snapshot_raw, weights_sha256


def _parse_source_row(
    row: Mapping[str, object], source_index: int
) -> tuple[str, str, str]:
    prompt = row.get("prompt")
    reward_model = row.get("reward_model")
    extra_info = row.get("extra_info")
    if (
        not isinstance(prompt, list)
        or len(prompt) != 1
        or not isinstance(prompt[0], dict)
        or prompt[0].get("role") != "user"
        or not isinstance(prompt[0].get("content"), str)
        or not prompt[0]["content"]
        or not isinstance(reward_model, dict)
        or not isinstance(reward_model.get("ground_truth"), str)
    ):
        raise DapoOperationalMaterializationError(
            f"invalid DAPO row schema at source index {source_index}"
        )
    source_extra_index = ""
    if isinstance(extra_info, dict) and isinstance(extra_info.get("index"), str):
        source_extra_index = extra_info["index"]
    return prompt[0]["content"], reward_model["ground_truth"], source_extra_index


def _deduplicate_dataset(
    dataset: Sequence[Mapping[str, object]],
) -> tuple[list[UniquePrompt], dict[str, int]]:
    unique: dict[str, tuple[str, str, int, str, int]] = {}
    for source_index, row in enumerate(dataset):
        prompt, ground_truth, extra_index = _parse_source_row(row, source_index)
        canonical_sha = _sha_bytes(_canonical([{"role": "user", "content": prompt}]))
        prior = unique.get(canonical_sha)
        if prior is None:
            unique[canonical_sha] = (prompt, ground_truth, source_index, extra_index, 1)
        else:
            prior_prompt, prior_truth, first_index, prior_extra, count = prior
            if prior_prompt != prompt or prior_truth != ground_truth:
                raise DapoOperationalMaterializationError(
                    "one canonical prompt identity has conflicting ground truths"
                )
            unique[canonical_sha] = (
                prior_prompt,
                prior_truth,
                first_index,
                prior_extra,
                count + 1,
            )
    prompts = [
        UniquePrompt(
            canonical_sha256=canonical_sha,
            prompt=value[0],
            ground_truth=value[1],
            first_source_index=value[2],
            source_extra_index=value[3],
            duplicate_count=value[4],
        )
        for canonical_sha, value in sorted(unique.items())
    ]
    counts = [item.duplicate_count for item in prompts]
    if not counts:
        raise DapoOperationalMaterializationError("DAPO dataset contains no prompts")
    return prompts, {
        "published_rows": len(dataset),
        "unique_canonical_prompts": len(prompts),
        "duplicate_rows": len(dataset) - len(prompts),
        "minimum_multiplicity": min(counts),
        "maximum_multiplicity": max(counts),
        "conflicting_ground_truth_count": 0,
    }


def _select_disjoint_pools(
    prompts: Sequence[UniquePrompt],
) -> dict[int, tuple[UniquePrompt, ...]]:
    if len(prompts) < len(POOL_SPECS) * PROMPTS_PER_POOL:
        raise DapoOperationalMaterializationError("too few unique DAPO prompts")
    remaining = list(prompts)
    output: dict[int, tuple[UniquePrompt, ...]] = {}
    for selection_seed, _ in POOL_SPECS:
        rng = random.Random(selection_seed)
        rng.shuffle(remaining)
        chosen = tuple(remaining[:PROMPTS_PER_POOL])
        remaining = remaining[PROMPTS_PER_POOL:]
        output[selection_seed] = chosen
    identities = [item.canonical_sha256 for pool in output.values() for item in pool]
    if len(identities) != len(set(identities)):
        raise DapoOperationalMaterializationError("selected pools are not disjoint")
    return output


def _rendered_token_ids(tokenizer: Any, prompt_template: str, prompt: str) -> list[int]:
    formatted = prompt_template.format(prompt)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": formatted}],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    return [
        int(value)
        for value in tokenizer(rendered, add_special_tokens=False)["input_ids"]
    ]


def _make_pool(
    *,
    root: Path,
    tokenizer: Any,
    prompt_template: str,
    key: bytes,
    protocol_raw: bytes,
    snapshot_raw: bytes,
    model_weights_sha256: str,
    selection_seed: int,
    generation_seed: int,
    prompts: Sequence[UniquePrompt],
) -> dict[str, object]:
    source_records: dict[str, list[dict[str, object]]] = {
        source_id: [] for source_id in SOURCE_IDS
    }
    record_by_identity: dict[str, tuple[str, int, dict[str, object]]] = {}
    for selection_index, prompt in enumerate(prompts):
        source_id = SOURCE_IDS[selection_index % len(SOURCE_IDS)]
        token_ids = _rendered_token_ids(tokenizer, prompt_template, prompt.prompt)
        record: dict[str, object] = {
            "input": prompt.prompt,
            "output": prompt.ground_truth,
            "source_id": source_id,
            "source_dataset_id": DATASET_REPO,
            "source_revision": DATASET_REVISION,
            "source_split": "train",
            "source_dataset_index": prompt.first_source_index,
            "source_prompt_id": _opaque(key, DESIGN_ID, prompt.canonical_sha256),
            "repeated_prompt_cluster_id": _opaque(
                key, DESIGN_ID, "cluster", prompt.canonical_sha256
            ),
            "canonical_prompt_sha256": prompt.canonical_sha256,
            "source_extra_index": prompt.source_extra_index,
            "source_duplicate_count": prompt.duplicate_count,
            "input_token_count": len(token_ids),
            "input_token_ids_sha256": _sha_bytes(_canonical(token_ids)),
            "selection_seed": selection_seed,
            "model_revision": MODEL_REVISION,
        }
        row = len(source_records[source_id])
        source_records[source_id].append(record)
        record_by_identity[prompt.canonical_sha256] = (source_id, row, record)

    filenames = {
        source_id: f"{source_id}_{selection_seed}.jsonl" for source_id in SOURCE_IDS
    }
    raw_sources = {
        source_id: b"".join(_canonical(row) + b"\n" for row in records)
        for source_id, records in source_records.items()
    }
    for source_id, raw in raw_sources.items():
        _write(root / filenames[source_id], raw)

    dispatch_order = list(prompts)
    random.Random(generation_seed).shuffle(dispatch_order)
    offsets = {SOURCE_IDS[0]: 0, SOURCE_IDS[1]: len(source_records[SOURCE_IDS[0]])}
    items = []
    design_items = []
    for ordinal, prompt in enumerate(dispatch_order):
        source_id, row, record = record_by_identity[prompt.canonical_sha256]
        source_prompt_id = str(record["source_prompt_id"])
        items.append(
            {
                "ordinal": ordinal,
                "pool_item_id": _opaque(
                    key, DESIGN_ID, selection_seed, generation_seed, ordinal
                ),
                "source_prompt_id": source_prompt_id,
                "source_id": source_id,
                "source_dataset_index": prompt.first_source_index,
                "dataset_index": offsets[source_id] + row,
                "task_name": source_id,
                "repeated_prompt_cluster_id": record["repeated_prompt_cluster_id"],
                "dispatch_cohort": ordinal // COHORT_SIZE,
                "decorrelation_block": f"cohort-{ordinal // COHORT_SIZE}",
                "matching_pair_id": f"unique-prompt-{prompt.canonical_sha256}",
                "materialized_source_row": row,
                "materialized_record_sha256": _sha_bytes(_canonical(record)),
                "input_token_count": record["input_token_count"],
                "input_token_ids_sha256": record["input_token_ids_sha256"],
            }
        )
        design_items.append(
            {
                "source_pool_ordinal": ordinal,
                "source_prompt_id": source_prompt_id,
                "task_name": source_id,
                "canonical_prompt_sha256": prompt.canonical_sha256,
                "source_dataset_index": prompt.first_source_index,
                "source_duplicate_count": prompt.duplicate_count,
                "rendered_prompt_tokens": record["input_token_count"],
            }
        )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "id_namespace_fingerprint": _sha_bytes(key),
        "design_protocol_sha256": _sha_bytes(protocol_raw),
        "order_seed": selection_seed,
        "model_revision": MODEL_REVISION,
        "model_weights_sha256": model_weights_sha256,
        "model_snapshot_manifest_sha256": _sha_bytes(snapshot_raw),
        "model_snapshot_path": "model_snapshot",
        "sources": [
            {
                "source_id": source_id,
                "dataset_id": DATASET_REPO,
                "revision": DATASET_REVISION,
                "split": "train",
                "materialized_file": filenames[source_id],
                "content_sha256": _sha_bytes(raw_sources[source_id]),
            }
            for source_id in SOURCE_IDS
        ],
        "items": items,
    }
    manifest["pool_id"] = compute_fixed_pool_id(manifest)
    manifest_raw = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    manifest_name = f"fixed_pool_manifest.v1.{selection_seed}.json"
    _write(root / manifest_name, manifest_raw)
    design = {
        "schema_version": 1,
        "analysis_status": "candidate_operational_workload_latency_discovery",
        "calibration_only": True,
        "confirmatory_eligible": False,
        "protocol_sha256": PROTOCOL_SHA256,
        "fixed_pool_id": manifest["pool_id"],
        "selection_seed": selection_seed,
        "generation_seed": generation_seed,
        "items": design_items,
    }
    design_raw = json.dumps(design, indent=2, sort_keys=True).encode() + b"\n"
    design_name = f"selection_design.v1.{selection_seed}.json"
    _write(root / design_name, design_raw)
    return {
        "selection_seed": selection_seed,
        "generation_seed": generation_seed,
        "manifest": manifest_name,
        "manifest_sha256": _sha_bytes(manifest_raw),
        "selection_design": design_name,
        "selection_design_sha256": _sha_bytes(design_raw),
        "pool_id": manifest["pool_id"],
        "prompt_groups": len(items),
    }


def materialize(*, output_dir: Path, protocol_path: Path, key: bytes) -> None:
    """Create all three immutable discovery pools and one shared model snapshot."""
    if len(key) < 32:
        raise DapoOperationalMaterializationError("HMAC key is too short")
    _, protocol_raw = _load_protocol(protocol_path)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.", dir=output_dir.parent
    ) as temporary_name:
        root = Path(temporary_name)
        root.chmod(0o700)
        tokenizer, snapshot_raw, model_weights_sha256 = _snapshot_model(root)
        parquet_path = Path(
            hf_hub_download(
                repo_id=DATASET_REPO,
                repo_type="dataset",
                filename=DATASET_FILE,
                revision=DATASET_REVISION,
            )
        )
        if _sha_path(parquet_path) != DATASET_FILE_SHA256:
            raise DapoOperationalMaterializationError("dataset file hash mismatch")
        dataset = load_dataset(
            "parquet", data_files={"train": str(parquet_path)}, split="train"
        )
        prompts, audit = _deduplicate_dataset(dataset)
        selected = _select_disjoint_pools(prompts)
        prompt_template = (
            Path(__file__).parents[1] / "examples/prompts/cot.txt"
        ).read_text()
        reports = [
            _make_pool(
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
            )
            for selection_seed, generation_seed in POOL_SPECS
        ]
        _write(root / "operational_discovery_protocol.candidate.v1.json", protocol_raw)
        audit_record = {
            "schema_version": 1,
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "dataset_file": DATASET_FILE,
            "dataset_file_sha256": DATASET_FILE_SHA256,
            **audit,
        }
        _write(
            root / "dataset_audit.v1.json",
            json.dumps(audit_record, indent=2, sort_keys=True).encode() + b"\n",
        )
        report = {
            "schema_version": 1,
            "status": "passed",
            "design_id": DESIGN_ID,
            "protocol_sha256": PROTOCOL_SHA256,
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "model_weights_sha256": model_weights_sha256,
            "dataset_audit": audit_record,
            "pools": reports,
        }
        _write(
            root / "materialization_report.v1.json",
            json.dumps(report, indent=2, sort_keys=True).encode() + b"\n",
        )
        hash_lines = [
            f"{_sha_path(path)}  {path.name}\n"
            for path in sorted(root.iterdir())
            if path.is_file() and path.name != "SHA256SUMS"
        ]
        _write(root / "SHA256SUMS", "".join(hash_lines).encode())
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
    parser.add_argument("--hmac-key-env", default="DAPO_OPERATIONAL_PROMPT_HMAC_KEY")
    args = parser.parse_args()
    key = os.environ.get(args.hmac_key_env)
    if key is None:
        raise DapoOperationalMaterializationError("HMAC key is missing")
    materialize(
        output_dir=args.output_dir, protocol_path=args.protocol, key=key.encode()
    )


if __name__ == "__main__":
    main()
