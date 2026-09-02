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

"""Materialize a fixed OpenMathInstruct-2 solution-length feasibility pool.

Raw prompts, answers, source metadata, and the exclusion ledger are written
only to the mode-0700 output directory. The fixed-pool manifest contains only
source coordinates, content digests, and opaque keyed identities.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import random
import stat
import unicodedata
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from nemo_rl.algorithms.async_utils.fixed_pool import compute_fixed_pool_id


DATASET_REPO = "nvidia/OpenMathInstruct-2"
DATASET_REVISION = "469216e3f46f4dacf476b382e192485ea51a143e"
DATASET_SPLIT = "train_1M"
DATASET_CARDINALITY = 1_000_000
DATASET_SCHEMA = {
    "problem",
    "generated_solution",
    "expected_answer",
    "problem_source",
}
MODEL_REPO = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
MODEL_WEIGHTS_SHA256 = (
    "a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796"
)
PROMPT_FILE_SHA256 = "a3575cc34f8bbd8ed5107a0d58003acf4277baf18d29409c7e61e0946e25b031"
SELECTION_SEED = 20260902
ORDER_SEED = 43001
PAIRS = 8
COHORT_SIZE = 4
MAX_INPUT_TOKENS = 256
INPUT_CALIPER_TOKENS = 8
ANSWER_CALIPER_TOKENS = 4
SHORT_REFERENCE_RANGE = (32, 96)
LONG_REFERENCE_RANGE = (256, 384)
MIN_REFERENCE_MEDIAN_RATIO = 3.0
MAX_RETAINED_CANDIDATES_PER_STRATUM = 4096
MATCHING_ALGORITHM_VERSION = "bounded_reservoir_lazy_augmenting_v1"
BOOTSTRAP_SEED = 20260904
BOOTSTRAP_REPS = 10_000
SOURCE_IDS = ("openmath_short", "openmath_long")
STRATUM_LABELS = {SOURCE_IDS[0]: "short", SOURCE_IDS[1]: "long"}


@dataclass(frozen=True, slots=True)
class Candidate:
    """One eligible, normalized OpenMath source row."""

    source_index: int
    problem: str
    answer: str
    problem_source: str
    normalized_problem_sha256: str
    input_token_ids: tuple[int, ...]
    reference_solution_token_count: int
    reference_answer_token_count: int


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


def _normalize_text(value: str) -> str:
    return unicodedata.normalize(
        "NFC", value.replace("\r\n", "\n").replace("\r", "\n")
    ).strip()


def _token_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    values = tokenizer(
        text,
        return_tensors=None,
        add_special_tokens=False,
    )["input_ids"]
    return tuple(int(value) for value in values)


def _render_tokens(
    tokenizer: Any, prompt_template: str, problem: str
) -> tuple[int, ...]:
    formatted = prompt_template.format(problem)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": formatted}],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    return _token_ids(tokenizer, rendered)


def _validate_dataset_contract(dataset: Any) -> None:
    if len(dataset) != DATASET_CARDINALITY:
        raise RuntimeError(
            "pinned OpenMathInstruct-2 split cardinality mismatch: "
            f"expected {DATASET_CARDINALITY}, got {len(dataset)}"
        )
    if set(dataset.column_names) != DATASET_SCHEMA:
        raise RuntimeError("pinned OpenMathInstruct-2 split column mismatch")
    features = getattr(dataset, "features", None)
    if not isinstance(features, Mapping) or set(features) != DATASET_SCHEMA:
        raise RuntimeError("pinned OpenMathInstruct-2 feature schema mismatch")
    non_string = sorted(
        name
        for name, feature in features.items()
        if getattr(feature, "dtype", None) != "string"
    )
    if non_string:
        raise RuntimeError(
            f"pinned OpenMathInstruct-2 features are not strings: {non_string}"
        )


def _build_candidates(
    dataset: Any,
    tokenizer: Any,
    prompt_template: str,
) -> tuple[list[Candidate], list[Candidate], dict[str, int]]:
    multiplicity: Counter[str] = Counter()
    rows_scanned_first_pass = 0
    for index, row in enumerate(dataset):
        rows_scanned_first_pass += 1
        for field in DATASET_SCHEMA:
            if not isinstance(row[field], str):
                raise RuntimeError(
                    f"OpenMath row {index} field {field!r} is not a string"
                )
        problem = _normalize_text(row["problem"])
        multiplicity[_sha(problem.encode("utf-8"))] += 1

    short: list[Candidate] = []
    long: list[Candidate] = []
    counts = Counter(
        {
            "rows_scanned_problem_multiplicity": rows_scanned_first_pass,
            "rows_scanned_candidate_selection": 0,
            "rendered_prompts_tokenized": 0,
            "reference_solutions_tokenized": 0,
            "reference_answers_tokenized": 0,
            "eligible_openmath_short": 0,
            "eligible_openmath_long": 0,
            "rejected_empty_field": 0,
            "rejected_repeated_problem": 0,
            "rejected_input_too_long": 0,
            "rejected_outside_reference_bands": 0,
        }
    )
    short_rng = random.Random(SELECTION_SEED)
    long_rng = random.Random(SELECTION_SEED ^ 0x5F3759DF)

    def retain(
        reservoir: list[Candidate],
        candidate: Candidate,
        *,
        eligible_count: int,
        rng: random.Random,
    ) -> None:
        if len(reservoir) < MAX_RETAINED_CANDIDATES_PER_STRATUM:
            reservoir.append(candidate)
            return
        replacement = rng.randrange(eligible_count)
        if replacement < MAX_RETAINED_CANDIDATES_PER_STRATUM:
            reservoir[replacement] = candidate

    for index, row in enumerate(dataset):
        counts["rows_scanned_candidate_selection"] += 1
        problem = _normalize_text(row["problem"])
        problem_sha = _sha(problem.encode("utf-8"))
        generated_solution = _normalize_text(row["generated_solution"])
        answer = _normalize_text(row["expected_answer"])
        problem_source = _normalize_text(row["problem_source"])
        if not problem or not generated_solution or not answer or not problem_source:
            counts["rejected_empty_field"] += 1
            continue
        if multiplicity[problem_sha] != 1:
            counts["rejected_repeated_problem"] += 1
            continue
        counts["rendered_prompts_tokenized"] += 1
        input_token_ids = _render_tokens(tokenizer, prompt_template, problem)
        if len(input_token_ids) > MAX_INPUT_TOKENS:
            counts["rejected_input_too_long"] += 1
            continue
        counts["reference_solutions_tokenized"] += 1
        reference_length = len(_token_ids(tokenizer, generated_solution))
        counts["reference_answers_tokenized"] += 1
        answer_length = len(_token_ids(tokenizer, answer))
        candidate = Candidate(
            source_index=index,
            problem=problem,
            answer=answer,
            problem_source=problem_source,
            normalized_problem_sha256=problem_sha,
            input_token_ids=input_token_ids,
            reference_solution_token_count=reference_length,
            reference_answer_token_count=answer_length,
        )
        if SHORT_REFERENCE_RANGE[0] <= reference_length <= SHORT_REFERENCE_RANGE[1]:
            counts["eligible_openmath_short"] += 1
            retain(
                short,
                candidate,
                eligible_count=counts["eligible_openmath_short"],
                rng=short_rng,
            )
        elif LONG_REFERENCE_RANGE[0] <= reference_length <= LONG_REFERENCE_RANGE[1]:
            counts["eligible_openmath_long"] += 1
            retain(
                long,
                candidate,
                eligible_count=counts["eligible_openmath_long"],
                rng=long_rng,
            )
        else:
            counts["rejected_outside_reference_bands"] += 1
    counts["retained_openmath_short"] = len(short)
    counts["retained_openmath_long"] = len(long)
    return short, long, dict(sorted(counts.items()))


def _index_long_candidates(
    long: Sequence[Candidate],
) -> dict[tuple[str, int, int], list[int]]:
    buckets: dict[tuple[str, int, int], list[int]] = {}
    for index, candidate in enumerate(long):
        key = (
            candidate.problem_source,
            len(candidate.input_token_ids),
            candidate.reference_answer_token_count,
        )
        buckets.setdefault(key, []).append(index)
    for indices in buckets.values():
        indices.sort(key=lambda index: long[index].source_index)
    return buckets


def _compatible_long_indices(
    candidate: Candidate,
    buckets: Mapping[tuple[str, int, int], Sequence[int]],
) -> Iterator[int]:
    bucket_keys = []
    input_length = len(candidate.input_token_ids)
    answer_length = candidate.reference_answer_token_count
    for input_delta in range(-INPUT_CALIPER_TOKENS, INPUT_CALIPER_TOKENS + 1):
        candidate_input_length = input_length + input_delta
        if candidate_input_length < 0:
            continue
        for answer_delta in range(-ANSWER_CALIPER_TOKENS, ANSWER_CALIPER_TOKENS + 1):
            candidate_answer_length = answer_length + answer_delta
            if candidate_answer_length < 0:
                continue
            key = (
                candidate.problem_source,
                candidate_input_length,
                candidate_answer_length,
            )
            if key in buckets:
                bucket_keys.append((abs(input_delta), abs(answer_delta), key))
    for _, _, key in sorted(bucket_keys):
        yield from buckets[key]


def _find_bounded_matching(
    short: Sequence[Candidate],
    long: Sequence[Candidate],
    *,
    seed: int,
) -> tuple[list[tuple[Candidate, Candidate]], dict[str, int]]:
    """Find eight pairs within bounded reservoirs, without a maximality claim."""
    rng = random.Random(seed)
    short_order = list(range(len(short)))
    rng.shuffle(short_order)
    buckets = _index_long_candidates(long)
    long_to_short: dict[int, int] = {}
    stats = Counter(
        {
            "short_reservoir_size": len(short),
            "long_reservoir_size": len(long),
            "long_index_buckets": len(buckets),
            "short_candidates_examined": 0,
            "augment_calls": 0,
            "compatibility_edges_examined": 0,
        }
    )

    def augment(short_index: int, seen: set[int]) -> bool:
        stats["augment_calls"] += 1
        for long_index in _compatible_long_indices(short[short_index], buckets):
            stats["compatibility_edges_examined"] += 1
            if long_index in seen:
                continue
            seen.add(long_index)
            prior = long_to_short.get(long_index)
            if prior is None or augment(prior, seen):
                long_to_short[long_index] = short_index
                return True
        return False

    for short_index in short_order:
        stats["short_candidates_examined"] += 1
        augment(short_index, set())
        if len(long_to_short) == PAIRS:
            break
    pairs = [
        (short[short_index], long[long_index])
        for long_index, short_index in long_to_short.items()
    ]
    rng.shuffle(pairs)
    stats["matches_found"] = len(pairs)
    return pairs, dict(sorted(stats.items()))


def _median(values: Sequence[int]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _select_pairs(
    short: Sequence[Candidate],
    long: Sequence[Candidate],
) -> tuple[list[tuple[Candidate, Candidate]], dict[str, int]]:
    matches, matching_counts = _find_bounded_matching(short, long, seed=SELECTION_SEED)
    if len(matches) < PAIRS:
        raise RuntimeError(
            f"only {len(matches)} OpenMath pairs satisfy all locked calipers"
        )
    selected = matches[:PAIRS]
    short_hashes = {candidate.normalized_problem_sha256 for candidate, _ in selected}
    long_hashes = {candidate.normalized_problem_sha256 for _, candidate in selected}
    if (
        short_hashes & long_hashes
        or len(short_hashes) != PAIRS
        or len(long_hashes) != PAIRS
    ):
        raise RuntimeError("selected OpenMath problems are not disjoint singletons")
    short_median = _median(
        [candidate.reference_solution_token_count for candidate, _ in selected]
    )
    long_median = _median(
        [candidate.reference_solution_token_count for _, candidate in selected]
    )
    if long_median / short_median < MIN_REFERENCE_MEDIAN_RATIO:
        raise RuntimeError("selected OpenMath reference median ratio is below 3.0")
    return selected, matching_counts


def _source_record(
    *,
    key: bytes,
    source_id: str,
    candidate: Candidate,
    pair_id: int,
) -> dict[str, object]:
    source_prompt_id = _opaque(
        key,
        "source-prompt-v1",
        DATASET_REPO,
        DATASET_REVISION,
        DATASET_SPLIT,
        candidate.source_index,
        candidate.normalized_problem_sha256,
    )
    cluster_id = _opaque(
        key,
        "normalized-problem-cluster-v1",
        DATASET_REPO,
        DATASET_REVISION,
        DATASET_SPLIT,
        candidate.normalized_problem_sha256,
    )
    return {
        "input": candidate.problem,
        "output": candidate.answer,
        "source_id": source_id,
        "source_dataset_id": DATASET_REPO,
        "source_revision": DATASET_REVISION,
        "source_split": DATASET_SPLIT,
        "source_dataset_index": candidate.source_index,
        "source_prompt_id": source_prompt_id,
        "repeated_prompt_cluster_id": cluster_id,
        "selection_stratum": source_id,
        "problem_source": candidate.problem_source,
        "matching_pair_id": f"pair-{pair_id}",
        "normalized_problem_sha256": candidate.normalized_problem_sha256,
        "input_token_count": len(candidate.input_token_ids),
        "input_token_ids_sha256": _sha(_canonical(candidate.input_token_ids)),
        "reference_solution_token_count": candidate.reference_solution_token_count,
        "reference_answer_token_count": candidate.reference_answer_token_count,
        "selection_seed": SELECTION_SEED,
        "model_revision": MODEL_REVISION,
        "prompt_file_sha256": PROMPT_FILE_SHA256,
    }


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> bytes:
    raw = b"".join(_canonical(record) + b"\n" for record in records)
    _atomic_write(path, raw)
    return raw


def _build_manifest(
    *,
    key: bytes,
    design_sha: str,
    source_records: Mapping[str, Sequence[Mapping[str, object]]],
    source_raw: Mapping[str, bytes],
    selected_pairs: Sequence[tuple[Candidate, Candidate]],
    model_snapshot_manifest_sha256: str,
) -> dict[str, object]:
    rng = random.Random(ORDER_SEED)
    pair_order = list(range(PAIRS))
    rng.shuffle(pair_order)
    source_rows = {
        source_id: {
            int(record["source_dataset_index"]): row
            for row, record in enumerate(records)
        }
        for source_id, records in source_records.items()
    }
    offsets = {
        SOURCE_IDS[0]: 0,
        SOURCE_IDS[1]: len(source_records[SOURCE_IDS[0]]),
    }
    records_by_coordinate = {
        (source_id, int(record["source_dataset_index"])): record
        for source_id, records in source_records.items()
        for record in records
    }

    ordered: list[tuple[str, Candidate, int]] = []
    for pair_start in range(0, PAIRS, 2):
        cohort_items: list[tuple[str, Candidate, int]] = []
        for pair_id in pair_order[pair_start : pair_start + 2]:
            short, long = selected_pairs[pair_id]
            cohort_items.extend(
                ((SOURCE_IDS[0], short, pair_id), (SOURCE_IDS[1], long, pair_id))
            )
        rng.shuffle(cohort_items)
        ordered.extend(cohort_items)

    items: list[dict[str, object]] = []
    for ordinal, (source_id, candidate, pair_id) in enumerate(ordered):
        record = records_by_coordinate[(source_id, candidate.source_index)]
        row = source_rows[source_id][candidate.source_index]
        cohort = ordinal // COHORT_SIZE
        items.append(
            {
                "ordinal": ordinal,
                "pool_item_id": _opaque(
                    key,
                    "pool-item-v1",
                    ORDER_SEED,
                    ordinal,
                    record["source_prompt_id"],
                ),
                "source_prompt_id": record["source_prompt_id"],
                "source_id": source_id,
                "source_dataset_index": candidate.source_index,
                "dataset_index": offsets[source_id] + row,
                "task_name": source_id,
                "repeated_prompt_cluster_id": record["repeated_prompt_cluster_id"],
                "dispatch_cohort": cohort,
                "decorrelation_block": f"cohort-{cohort}",
                "matching_pair_id": f"pair-{pair_id}",
                "materialized_source_row": row,
                "materialized_record_sha256": _sha(_canonical(record)),
                "input_token_count": len(candidate.input_token_ids),
                "input_token_ids_sha256": _sha(_canonical(candidate.input_token_ids)),
            }
        )

    sources = [
        {
            "source_id": source_id,
            "dataset_id": DATASET_REPO,
            "revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
            "materialized_file": f"{source_id}.jsonl",
            "content_sha256": _sha(source_raw[source_id]),
        }
        for source_id in SOURCE_IDS
    ]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "id_namespace_fingerprint": _sha(key),
        "design_protocol_sha256": design_sha,
        "order_seed": ORDER_SEED,
        "model_revision": MODEL_REVISION,
        "model_weights_sha256": MODEL_WEIGHTS_SHA256,
        "model_snapshot_manifest_sha256": model_snapshot_manifest_sha256,
        "model_snapshot_path": "model_snapshot",
        "sources": sources,
        "items": items,
    }
    manifest["pool_id"] = compute_fixed_pool_id(manifest)
    return manifest


def _snapshot_model(output_dir: Path) -> tuple[Any, bytes]:
    model_dir = output_dir / "model_snapshot"
    snapshot_download(repo_id=MODEL_REPO, revision=MODEL_REVISION, local_dir=model_dir)
    for path in model_dir.rglob("*"):
        if path.is_symlink():
            raise RuntimeError("pinned model snapshot contains a symbolic link")
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
    snapshot_manifest = (
        json.dumps(snapshot_records, indent=2, sort_keys=True).encode() + b"\n"
    )
    _atomic_write(output_dir / "model_snapshot_manifest.v1.json", snapshot_manifest)
    return AutoTokenizer.from_pretrained(model_dir), snapshot_manifest


def _validate_private_modes(output_dir: Path) -> None:
    if stat.S_IMODE(output_dir.stat().st_mode) != 0o700:
        raise RuntimeError("materialization directory mode is not 0700")
    for path in output_dir.rglob("*"):
        expected = 0o700 if path.is_dir() else 0o600
        if stat.S_IMODE(path.stat().st_mode) != expected:
            raise RuntimeError(
                f"private artifact mode mismatch for {path.relative_to(output_dir)}"
            )


def materialize(output_dir: Path, prompt_file: Path, key: bytes) -> None:
    if len(key) < 32:
        raise RuntimeError("prompt-ID HMAC key is shorter than 32 bytes")
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    output_dir.chmod(0o700)
    prompt_raw = prompt_file.read_bytes()
    if _sha(prompt_raw) != PROMPT_FILE_SHA256:
        raise RuntimeError("study CoT prompt SHA-256 mismatch")
    prompt_template = prompt_raw.decode("utf-8")
    design_sha = _sha(Path(__file__).read_bytes())

    tokenizer, snapshot_manifest = _snapshot_model(output_dir)
    dataset = load_dataset(
        DATASET_REPO,
        split=DATASET_SPLIT,
        revision=DATASET_REVISION,
    )
    _validate_dataset_contract(dataset)
    short_candidates, long_candidates, selection_counts = _build_candidates(
        dataset, tokenizer, prompt_template
    )
    selected_pairs, matching_counts = _select_pairs(short_candidates, long_candidates)

    records: dict[str, list[dict[str, object]]] = {
        SOURCE_IDS[0]: [],
        SOURCE_IDS[1]: [],
    }
    for pair_id, (short, long) in enumerate(selected_pairs):
        records[SOURCE_IDS[0]].append(
            _source_record(
                key=key,
                source_id=SOURCE_IDS[0],
                candidate=short,
                pair_id=pair_id,
            )
        )
        records[SOURCE_IDS[1]].append(
            _source_record(
                key=key,
                source_id=SOURCE_IDS[1],
                candidate=long,
                pair_id=pair_id,
            )
        )
    source_raw = {
        source_id: _write_jsonl(output_dir / f"{source_id}.jsonl", source_records)
        for source_id, source_records in records.items()
    }
    exclusion_rows = sorted(
        (
            {
                "dataset_id": DATASET_REPO,
                "revision": DATASET_REVISION,
                "split": DATASET_SPLIT,
                "source_dataset_index": candidate.source_index,
                "normalized_problem_sha256": candidate.normalized_problem_sha256,
            }
            for pair in selected_pairs
            for candidate in pair
        ),
        key=lambda row: int(row["source_dataset_index"]),
    )
    exclusion_raw = (
        json.dumps(
            {"schema_version": 1, "excluded_problem_clusters": exclusion_rows},
            indent=2,
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    _atomic_write(output_dir / "private_exclusion_ledger.v1.json", exclusion_raw)

    manifest = _build_manifest(
        key=key,
        design_sha=design_sha,
        source_records=records,
        source_raw=source_raw,
        selected_pairs=selected_pairs,
        model_snapshot_manifest_sha256=_sha(snapshot_manifest),
    )
    manifest_name = f"fixed_pool_manifest.v1.{ORDER_SEED}.json"
    manifest_raw = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    _atomic_write(output_dir / manifest_name, manifest_raw)

    records_by_prompt_id = {
        str(record["source_prompt_id"]): record
        for source_records in records.values()
        for record in source_records
    }
    design_items = []
    manifest_items = manifest["items"]
    if not isinstance(manifest_items, list):
        raise RuntimeError("constructed fixed-pool items are not a list")
    for item in manifest_items:
        if not isinstance(item, Mapping):
            raise RuntimeError("constructed fixed-pool item is not an object")
        record = records_by_prompt_id[str(item["source_prompt_id"])]
        design_items.append(
            {
                "source_pool_ordinal": item["ordinal"],
                "source_prompt_id": item["source_prompt_id"],
                "task_name": item["task_name"],
                "pair_id": item["matching_pair_id"],
                "stratum": STRATUM_LABELS[str(record["selection_stratum"])],
                "problem_source": record["problem_source"],
                "rendered_prompt_tokens": record["input_token_count"],
                "reference_solution_tokens": record["reference_solution_token_count"],
                "reference_answer_token_count": record["reference_answer_token_count"],
            }
        )
    selection_design = {
        "schema_version": 1,
        "analysis_status": "preregistered_pre_generation",
        "calibration_only": True,
        "confirmatory_eligible": False,
        "fixed_pool_id": manifest["pool_id"],
        "fixed_pool_manifest_sha256": _sha(manifest_raw),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "items": design_items,
    }
    selection_design_raw = (
        json.dumps(selection_design, indent=2, sort_keys=True).encode() + b"\n"
    )
    selection_design_name = "selection_design.v1.json"
    _atomic_write(output_dir / selection_design_name, selection_design_raw)

    short_lengths = [pair[0].reference_solution_token_count for pair in selected_pairs]
    long_lengths = [pair[1].reference_solution_token_count for pair in selected_pairs]
    report = {
        "schema_version": 1,
        "design_protocol_sha256": design_sha,
        "dataset_id": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "dataset_split": DATASET_SPLIT,
        "dataset_cardinality": len(dataset),
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "selection_seed": SELECTION_SEED,
        "order_seed": ORDER_SEED,
        "pairs": PAIRS,
        "cohort_size": COHORT_SIZE,
        "task_count_per_cohort": {SOURCE_IDS[0]: 2, SOURCE_IDS[1]: 2},
        "max_input_tokens": MAX_INPUT_TOKENS,
        "input_caliper_tokens": INPUT_CALIPER_TOKENS,
        "answer_caliper_tokens": ANSWER_CALIPER_TOKENS,
        "matching_algorithm_version": MATCHING_ALGORITHM_VERSION,
        "max_retained_candidates_per_stratum": (MAX_RETAINED_CANDIDATES_PER_STRATUM),
        "reference_token_ranges": {
            SOURCE_IDS[0]: list(SHORT_REFERENCE_RANGE),
            SOURCE_IDS[1]: list(LONG_REFERENCE_RANGE),
        },
        "selection_counts": selection_counts,
        "matching_counts": matching_counts,
        "selected_reference_token_medians": {
            SOURCE_IDS[0]: _median(short_lengths),
            SOURCE_IDS[1]: _median(long_lengths),
        },
        "selected_reference_median_ratio": _median(long_lengths)
        / _median(short_lengths),
        "observed_max_input_pair_delta_tokens": max(
            abs(len(short.input_token_ids) - len(long.input_token_ids))
            for short, long in selected_pairs
        ),
        "observed_max_answer_pair_delta_tokens": max(
            abs(short.reference_answer_token_count - long.reference_answer_token_count)
            for short, long in selected_pairs
        ),
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_weights_sha256": MODEL_WEIGHTS_SHA256,
        "prompt_file_sha256": _sha(prompt_raw),
        "private_exclusion_ledger_sha256": _sha(exclusion_raw),
        "selection_design_sha256": _sha(selection_design_raw),
        "selection_design": selection_design_name,
        "manifest": manifest_name,
    }
    _atomic_write(
        output_dir / "materialization_report.v1.json",
        json.dumps(report, indent=2, sort_keys=True).encode() + b"\n",
    )
    hash_lines = [
        f"{_sha_file(path)}  {path.name}\n"
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    _atomic_write(output_dir / "SHA256SUMS", "".join(hash_lines).encode())
    _validate_private_modes(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument(
        "--hmac-key-env",
        default="OPENMATH_LATENCY_PROMPT_HMAC_KEY",
        help="Environment variable containing the non-persisted ID key",
    )
    args = parser.parse_args()
    key_value = os.environ.get(args.hmac_key_env)
    if key_value is None:
        raise RuntimeError("prompt-ID HMAC key is missing")
    materialize(args.output_dir, args.prompt_file, key_value.encode())


if __name__ == "__main__":
    main()
