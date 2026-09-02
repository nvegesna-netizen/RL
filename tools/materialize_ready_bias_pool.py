# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Materialize the preregistered GSM8K/AIME2024 fixed-pool calibration.

Raw prompts and answers are written only to the private output directory. The
fixed-pool manifests contain source coordinates, token/content digests, and
opaque keyed identities, but never prompt or completion text.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from nemo_rl.algorithms.async_utils.fixed_pool import compute_fixed_pool_id


GSM8K_REPO = "openai/gsm8k"
GSM8K_REVISION = "e53f048856ff4f594e959d75785d2c2d37b678ee"
AIME_REPO = "HuggingFaceH4/aime_2024"
AIME_REVISION = "2fe88a2f1091d5048c0f36abc874fb997b3dd99a"
MODEL_REPO = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
MODEL_WEIGHTS_SHA256 = (
    "a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796"
)
PROMPT_FILE_SHA256 = "a3575cc34f8bbd8ed5107a0d58003acf4277baf18d29409c7e61e0946e25b031"
SELECTION_SEED = 20260901
ORDER_SEEDS = (42001, 42002, 42003)
PROMPTS_PER_TASK = 24
PAIR_CALIPER_TOKENS = 16
COHORT_SIZE = 4
MAX_INPUT_TOKENS = 256


def _canonical(record: object) -> bytes:
    return json.dumps(
        record, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _opaque(key: bytes, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.chmod(0o600)
    temporary.replace(path)


def _render_tokens(tokenizer: Any, prompt_template: str, problem: str) -> list[int]:
    formatted = prompt_template.format(problem)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": formatted}],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    values = tokenizer(
        rendered,
        return_tensors=None,
        add_special_tokens=False,
    )["input_ids"]
    return [int(value) for value in values]


def _maximum_caliper_matching(
    left_lengths: Mapping[int, int],
    right_lengths: Mapping[int, int],
    *,
    seed: int,
    caliper: int,
) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    left_order = list(left_lengths)
    rng.shuffle(left_order)
    tie_rank = {index: rng.random() for index in right_lengths}
    candidates = {
        left: sorted(
            (
                right
                for right, length in right_lengths.items()
                if abs(left_lengths[left] - length) <= caliper
            ),
            key=lambda right: (
                abs(left_lengths[left] - right_lengths[right]),
                tie_rank[right],
                right,
            ),
        )
        for left in left_order
    }
    right_to_left: dict[int, int] = {}

    def augment(left: int, seen: set[int]) -> bool:
        for right in candidates[left]:
            if right in seen:
                continue
            seen.add(right)
            prior = right_to_left.get(right)
            if prior is None or augment(prior, seen):
                right_to_left[right] = left
                return True
        return False

    for left in left_order:
        augment(left, set())
    pairs = [(left, right) for right, left in right_to_left.items()]
    rng.shuffle(pairs)
    return pairs


def _source_record(
    *,
    key: bytes,
    source_id: str,
    dataset_id: str,
    revision: str,
    source_index: int,
    problem: str,
    answer: str,
) -> dict[str, object]:
    problem_sha = _sha(problem.encode("utf-8"))
    source_prompt_id = _opaque(
        key,
        "source-prompt-v1",
        dataset_id,
        revision,
        "train",
        source_index,
        problem_sha,
    )
    return {
        "input": problem,
        "output": answer,
        "source_id": source_id,
        "source_dataset_index": source_index,
        "source_prompt_id": source_prompt_id,
        "repeated_prompt_cluster_id": source_prompt_id,
    }


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> bytes:
    raw = b"".join(_canonical(record) + b"\n" for record in records)
    _atomic_write(path, raw)
    return raw


def _build_manifest(
    *,
    output_dir: Path,
    key: bytes,
    order_seed: int,
    design_sha: str,
    source_records: Mapping[str, Sequence[Mapping[str, object]]],
    source_raw: Mapping[str, bytes],
    source_specs: Mapping[str, tuple[str, str]],
    token_ids: Mapping[tuple[str, int], Sequence[int]],
    matching_pairs: Sequence[tuple[int, int]],
    model_snapshot_manifest_sha256: str,
) -> dict[str, object]:
    rng = random.Random(order_seed)
    pair_order = list(range(len(matching_pairs)))
    rng.shuffle(pair_order)
    source_rows = {
        source_id: {
            int(record["source_dataset_index"]): row
            for row, record in enumerate(records)
        }
        for source_id, records in source_records.items()
    }
    source_offsets: dict[str, int] = {}
    offset = 0
    for source_id in ("gsm8k", "AIME2024"):
        source_offsets[source_id] = offset
        offset += len(source_records[source_id])

    ordered: list[tuple[str, int, int]] = []
    for cohort, pair_start in enumerate(range(0, len(pair_order), 2)):
        pair_ids = pair_order[pair_start : pair_start + 2]
        cohort_items: list[tuple[str, int, int]] = []
        for pair_id in pair_ids:
            aime_index, gsm_index = matching_pairs[pair_id]
            cohort_items.extend(
                (("gsm8k", gsm_index, pair_id), ("AIME2024", aime_index, pair_id))
            )
        rng.shuffle(cohort_items)
        ordered.extend(cohort_items)
        assert len(cohort_items) == COHORT_SIZE and cohort == pair_start // 2

    items: list[dict[str, object]] = []
    for ordinal, (source_id, source_index, pair_id) in enumerate(ordered):
        row = source_rows[source_id][source_index]
        record = source_records[source_id][row]
        tokens = list(token_ids[(source_id, source_index)])
        cohort = ordinal // COHORT_SIZE
        items.append(
            {
                "ordinal": ordinal,
                "pool_item_id": _opaque(
                    key, "pool-item-v1", order_seed, ordinal, record["source_prompt_id"]
                ),
                "source_prompt_id": record["source_prompt_id"],
                "source_id": source_id,
                "source_dataset_index": source_index,
                "dataset_index": source_offsets[source_id] + row,
                "task_name": source_id,
                "repeated_prompt_cluster_id": record["repeated_prompt_cluster_id"],
                "dispatch_cohort": cohort,
                "decorrelation_block": f"cohort-{cohort}",
                "matching_pair_id": f"pair-{pair_id}",
                "materialized_source_row": row,
                "materialized_record_sha256": _sha(_canonical(record)),
                "input_token_count": len(tokens),
                "input_token_ids_sha256": _sha(_canonical(tokens)),
            }
        )

    sources = []
    for source_id in ("gsm8k", "AIME2024"):
        dataset_id, revision = source_specs[source_id]
        sources.append(
            {
                "source_id": source_id,
                "dataset_id": dataset_id,
                "revision": revision,
                "split": "train",
                "materialized_file": f"{source_id}.jsonl",
                "content_sha256": _sha(source_raw[source_id]),
            }
        )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "id_namespace_fingerprint": _sha(key),
        "design_protocol_sha256": design_sha,
        "order_seed": order_seed,
        "model_revision": MODEL_REVISION,
        "model_weights_sha256": MODEL_WEIGHTS_SHA256,
        "model_snapshot_manifest_sha256": model_snapshot_manifest_sha256,
        "model_snapshot_path": "model_snapshot",
        "sources": sources,
        "items": items,
    }
    manifest["pool_id"] = compute_fixed_pool_id(manifest)
    return manifest


def materialize(output_dir: Path, prompt_file: Path, key: bytes) -> None:
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    output_dir.chmod(0o700)
    prompt_template = prompt_file.read_text(encoding="utf-8")
    if _sha(prompt_file.read_bytes()) != PROMPT_FILE_SHA256:
        raise RuntimeError("study CoT prompt SHA-256 mismatch")
    design_sha = _sha(Path(__file__).read_bytes())

    model_dir = output_dir / "model_snapshot"
    snapshot_download(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        local_dir=model_dir,
    )
    for directory in [
        model_dir,
        *(path for path in model_dir.rglob("*") if path.is_dir()),
    ]:
        directory.chmod(0o700)
    for path in model_dir.rglob("*"):
        if path.is_file():
            path.chmod(0o600)
    weights_path = model_dir / "model.safetensors"
    if _sha_file(weights_path) != MODEL_WEIGHTS_SHA256:
        raise RuntimeError("pinned Qwen model.safetensors SHA-256 mismatch")
    snapshot_records = [
        {
            "path": str(path.relative_to(model_dir)),
            "bytes": path.stat().st_size,
            "sha256": _sha_file(path),
        }
        for path in sorted(model_dir.rglob("*"))
        if path.is_file() and ".cache" not in path.parts
    ]
    snapshot_manifest_bytes = (
        json.dumps(snapshot_records, indent=2, sort_keys=True).encode() + b"\n"
    )
    _atomic_write(
        output_dir / "model_snapshot_manifest.v1.json",
        snapshot_manifest_bytes,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    gsm = load_dataset(
        GSM8K_REPO,
        "main",
        split="train",
        revision=GSM8K_REVISION,
    )
    aime = load_dataset(AIME_REPO, split="train", revision=AIME_REVISION)
    if len(gsm) != 7473 or set(gsm.column_names) != {"question", "answer"}:
        raise RuntimeError("pinned GSM8K split cardinality/schema mismatch")
    if len(aime) != 30 or set(aime.column_names) != {
        "id",
        "problem",
        "solution",
        "answer",
        "url",
        "year",
    }:
        raise RuntimeError("pinned AIME2024 split cardinality/schema mismatch")
    gsm_tokens = {
        index: _render_tokens(tokenizer, prompt_template, str(row["question"]))
        for index, row in enumerate(gsm)
    }
    aime_tokens = {
        index: _render_tokens(tokenizer, prompt_template, str(row["problem"]))
        for index, row in enumerate(aime)
    }
    eligible_aime = {
        index: len(tokens)
        for index, tokens in aime_tokens.items()
        if len(tokens) <= MAX_INPUT_TOKENS
    }
    eligible_gsm = {
        index: len(tokens)
        for index, tokens in gsm_tokens.items()
        if len(tokens) <= MAX_INPUT_TOKENS
    }
    pairs = _maximum_caliper_matching(
        eligible_aime,
        eligible_gsm,
        seed=SELECTION_SEED,
        caliper=PAIR_CALIPER_TOKENS,
    )
    if len(pairs) < PROMPTS_PER_TASK:
        raise RuntimeError(
            f"only {len(pairs)} cross-task pairs satisfy the token caliper"
        )
    pairs = pairs[:PROMPTS_PER_TASK]
    selected_aime = sorted(left for left, _ in pairs)
    selected_gsm = sorted(right for _, right in pairs)

    gsm_records = [
        _source_record(
            key=key,
            source_id="gsm8k",
            dataset_id=GSM8K_REPO,
            revision=GSM8K_REVISION,
            source_index=index,
            problem=str(gsm[index]["question"]),
            answer=str(gsm[index]["answer"]).split("####")[-1].strip(),
        )
        for index in selected_gsm
    ]
    aime_records = [
        _source_record(
            key=key,
            source_id="AIME2024",
            dataset_id=AIME_REPO,
            revision=AIME_REVISION,
            source_index=index,
            problem=str(aime[index]["problem"]),
            answer=str(aime[index]["answer"]),
        )
        for index in selected_aime
    ]
    source_records = {"gsm8k": gsm_records, "AIME2024": aime_records}
    source_raw = {
        source_id: _write_jsonl(output_dir / f"{source_id}.jsonl", records)
        for source_id, records in source_records.items()
    }
    token_ids = {
        **{("gsm8k", index): gsm_tokens[index] for index in selected_gsm},
        **{("AIME2024", index): aime_tokens[index] for index in selected_aime},
    }
    source_specs = {
        "gsm8k": (GSM8K_REPO, GSM8K_REVISION),
        "AIME2024": (AIME_REPO, AIME_REVISION),
    }
    manifest_paths = []
    for order_seed in ORDER_SEEDS:
        manifest = _build_manifest(
            output_dir=output_dir,
            key=key,
            order_seed=order_seed,
            design_sha=design_sha,
            source_records=source_records,
            source_raw=source_raw,
            source_specs=source_specs,
            token_ids=token_ids,
            matching_pairs=pairs,
            model_snapshot_manifest_sha256=_sha(snapshot_manifest_bytes),
        )
        path = output_dir / f"fixed_pool_manifest.v1.{order_seed}.json"
        _atomic_write(
            path, json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
        )
        manifest_paths.append(path.name)

    report = {
        "schema_version": 1,
        "design_protocol_sha256": design_sha,
        "selection_seed": SELECTION_SEED,
        "order_seeds": list(ORDER_SEEDS),
        "prompts_per_task": PROMPTS_PER_TASK,
        "cohort_size": COHORT_SIZE,
        "task_count_per_cohort": {"gsm8k": 2, "AIME2024": 2},
        "pair_caliper_tokens": PAIR_CALIPER_TOKENS,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "observed_max_pair_delta_tokens": max(
            abs(len(aime_tokens[a]) - len(gsm_tokens[g])) for a, g in pairs
        ),
        "dataset_fingerprints": {
            "gsm8k": getattr(gsm, "_fingerprint", None),
            "AIME2024": getattr(aime, "_fingerprint", None),
        },
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_weights_sha256": MODEL_WEIGHTS_SHA256,
        "prompt_file_sha256": _sha(prompt_file.read_bytes()),
        "manifests": manifest_paths,
    }
    _atomic_write(
        output_dir / "materialization_report.v1.json",
        json.dumps(report, indent=2, sort_keys=True).encode() + b"\n",
    )
    hash_lines = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            hash_lines.append(f"{_sha(path.read_bytes())}  {path.name}\n")
    _atomic_write(output_dir / "SHA256SUMS", "".join(hash_lines).encode())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument(
        "--hmac-key-env",
        default="READY_BIAS_PROMPT_HMAC_KEY",
        help="Environment variable containing the non-persisted ID key",
    )
    args = parser.parse_args()
    key_value = os.environ.get(args.hmac_key_env)
    if key_value is None or len(key_value.encode()) < 32:
        raise RuntimeError("prompt-ID HMAC key is missing or shorter than 32 bytes")
    materialize(args.output_dir, args.prompt_file, key_value.encode())


if __name__ == "__main__":
    main()
