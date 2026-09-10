"""Tests for strict fixed-pool manifests and collection configuration."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from nemo_rl.algorithms.async_utils.fixed_pool import (
    FixedPoolDataset,
    FixedPoolManifestError,
    compute_fixed_pool_id,
    fixed_pool_collate_fn,
    load_fixed_pool_manifest,
    validate_fixed_pool_manifest_design,
    validate_fixed_pool_materialization,
    validate_fixed_pool_trace,
    validate_openmath_latency_feasibility_manifest_design,
)
from nemo_rl.algorithms.async_utils.scheduler_trace import (
    JsonlSchedulerTraceSink,
    SchedulerEventType,
    SchedulerTraceValidationError,
)
from nemo_rl.algorithms.grpo import GRPOConfig
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.algorithms.single_controller_utils.config import (
    AsyncRLConfig,
    FixedPoolCollectionConfig,
    MasterConfig,
    SchedulerTraceConfig,
    validate_single_controller_config,
)
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _manifest_record() -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": 1,
        "id_namespace_fingerprint": _digest("private-id-namespace"),
        "design_protocol_sha256": _digest("design-protocol-v1"),
        "order_seed": 20260901,
        "model_revision": _digest("model-revision"),
        "model_weights_sha256": _digest("model-weights"),
        "model_snapshot_manifest_sha256": _digest("model-snapshot-manifest"),
        "model_snapshot_path": "model_snapshot",
        "sources": [
            {
                "source_id": "gsm8k",
                "dataset_id": "openai/gsm8k",
                "revision": "0123456789abcdef",
                "split": "train",
                "materialized_file": "gsm8k.jsonl",
                "content_sha256": _digest("gsm8k-source-content"),
            },
            {
                "source_id": "aime2024",
                "dataset_id": "HuggingFaceH4/aime_2024",
                "revision": "fedcba9876543210",
                "split": "train",
                "materialized_file": "aime2024.jsonl",
                "content_sha256": _digest("aime-source-content"),
            },
        ],
        "items": [
            {
                "ordinal": 0,
                "pool_item_id": _digest("item-0"),
                "source_prompt_id": _digest("prompt-2"),
                "source_id": "gsm8k",
                "source_dataset_index": 20,
                "dataset_index": 2,
                "task_name": "fast-math",
                "repeated_prompt_cluster_id": _digest("cluster-2"),
                "dispatch_cohort": 0,
                "decorrelation_block": "input-length-bin-0",
                "matching_pair_id": "pair-0",
                "materialized_source_row": 0,
                "materialized_record_sha256": _digest("record-0"),
                "input_token_count": 1,
                "input_token_ids_sha256": hashlib.sha256(b"[3]").hexdigest(),
            },
            {
                "ordinal": 1,
                "pool_item_id": _digest("item-1"),
                "source_prompt_id": _digest("prompt-0"),
                "source_id": "aime2024",
                "source_dataset_index": 7,
                "dataset_index": 0,
                "task_name": "slow-math",
                "repeated_prompt_cluster_id": _digest("cluster-0"),
                "dispatch_cohort": 1,
                "decorrelation_block": "input-length-bin-0",
                "matching_pair_id": "pair-0",
                "materialized_source_row": 0,
                "materialized_record_sha256": _digest("record-1"),
                "input_token_count": 1,
                "input_token_ids_sha256": hashlib.sha256(b"[1]").hexdigest(),
            },
        ],
    }
    record["pool_id"] = compute_fixed_pool_id(record)
    return record


def _write_manifest(path: Path, record: dict[str, object]) -> bytes:
    raw = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()
    path.write_bytes(raw)
    return raw


def _write_openmath_manifest(
    tmp_path: Path,
    *,
    record_mutation: Callable[[str, int, dict[str, object]], None] | None = None,
) -> Path:
    sources = []
    items = []
    rows_by_source: dict[str, list[dict[str, object]]] = {}
    source_ids = ("openmath_short", "openmath_long")
    for source_offset, source_id in enumerate(source_ids):
        rows = []
        for row_index in range(8):
            record = {
                "input": f"private problem {source_id} {row_index}",
                "output": str(row_index),
                "source_id": source_id,
                "source_dataset_id": "nvidia/OpenMathInstruct-2",
                "source_revision": "469216e3f46f4dacf476b382e192485ea51a143e",
                "source_split": "train_1M",
                "source_dataset_index": source_offset * 100 + row_index,
                "source_prompt_id": _digest(f"{source_id}-prompt-{row_index}"),
                "repeated_prompt_cluster_id": _digest(
                    f"{source_id}-cluster-{row_index}"
                ),
                "reference_solution_token_count": (
                    64 + row_index if source_id == "openmath_short" else 300 + row_index
                ),
                "reference_answer_token_count": (
                    3 + row_index if source_id == "openmath_short" else 6 + row_index
                ),
                "normalized_problem_sha256": _digest(
                    f"{source_id}-problem-{row_index}"
                ),
                "problem_source": "synthetic_math",
                "selection_stratum": source_id,
                "matching_pair_id": f"pair-{row_index}",
                "input_token_count": 100 + row_index + 5 * source_offset,
                "input_token_ids_sha256": _digest(
                    f"openmath-input-tokens-{source_offset}-{row_index}"
                ),
                "selection_seed": 20260902,
                "model_revision": "8faed761d45a263340a0528343f099c05c9a4323",
                "prompt_file_sha256": (
                    "a3575cc34f8bbd8ed5107a0d58003acf4277baf18d29409c7e61e0946e25b031"
                ),
            }
            if record_mutation is not None:
                record_mutation(source_id, row_index, record)
            rows.append(record)
        rows_by_source[source_id] = rows
        raw = b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            for row in rows
        )
        materialized_file = f"{source_id}.jsonl"
        (tmp_path / materialized_file).write_bytes(raw)
        sources.append(
            {
                "source_id": source_id,
                "dataset_id": "nvidia/OpenMathInstruct-2",
                "revision": "469216e3f46f4dacf476b382e192485ea51a143e",
                "split": "train_1M",
                "materialized_file": materialized_file,
                "content_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )

    ordinal = 0
    for cohort in range(4):
        for row_index in (cohort * 2, cohort * 2 + 1):
            for source_offset, source_id in enumerate(source_ids):
                record = rows_by_source[source_id][row_index]
                items.append(
                    {
                        "ordinal": ordinal,
                        "pool_item_id": _digest(f"openmath-item-{ordinal}"),
                        "source_prompt_id": record["source_prompt_id"],
                        "source_id": source_id,
                        "source_dataset_index": record["source_dataset_index"],
                        "dataset_index": source_offset * 8 + row_index,
                        "task_name": source_id,
                        "repeated_prompt_cluster_id": record[
                            "repeated_prompt_cluster_id"
                        ],
                        "dispatch_cohort": cohort,
                        "decorrelation_block": f"cohort-{cohort}",
                        "matching_pair_id": f"pair-{row_index}",
                        "materialized_source_row": row_index,
                        "materialized_record_sha256": hashlib.sha256(
                            json.dumps(
                                record, sort_keys=True, separators=(",", ":")
                            ).encode()
                        ).hexdigest(),
                        "input_token_count": record["input_token_count"],
                        "input_token_ids_sha256": record["input_token_ids_sha256"],
                    }
                )
                ordinal += 1

    manifest_record: dict[str, object] = {
        "schema_version": 1,
        "id_namespace_fingerprint": _digest("openmath-id-namespace"),
        "design_protocol_sha256": _digest("openmath-design-protocol"),
        "order_seed": 43001,
        "model_revision": "8faed761d45a263340a0528343f099c05c9a4323",
        "model_weights_sha256": (
            "a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796"
        ),
        "model_snapshot_manifest_sha256": _digest("model-snapshot-manifest"),
        "model_snapshot_path": "model_snapshot",
        "sources": sources,
        "items": items,
    }
    manifest_record["pool_id"] = compute_fixed_pool_id(manifest_record)
    path = tmp_path / "openmath_pool.json"
    _write_manifest(path, manifest_record)
    return path


def test_manifest_round_trip_and_exact_file_digest(tmp_path: Path) -> None:
    path = tmp_path / "pool.json"
    record = _manifest_record()
    raw = _write_manifest(path, record)

    manifest = load_fixed_pool_manifest(path)

    assert manifest.pool_id == record["pool_id"]
    assert manifest.manifest_sha256 == hashlib.sha256(raw).hexdigest()
    assert [item.dataset_index for item in manifest.items] == [2, 0]
    assert manifest.num_dispatch_cohorts == 2
    assert manifest.cohort_sizes == (1, 1)
    assert [source.source_id for source in manifest.sources] == ["gsm8k", "aime2024"]


def test_openmath_latency_feasibility_design_accepts_bound_pool(
    tmp_path: Path,
) -> None:
    manifest = load_fixed_pool_manifest(_write_openmath_manifest(tmp_path))

    validate_openmath_latency_feasibility_manifest_design(manifest)
    validate_fixed_pool_manifest_design(manifest, "openmath_latency_feasibility_v1")


def test_openmath_latency_feasibility_design_rejects_source_revision(
    tmp_path: Path,
) -> None:
    manifest = load_fixed_pool_manifest(_write_openmath_manifest(tmp_path))
    sources = (
        manifest.sources[0].model_copy(update={"revision": "wrong-revision"}),
        manifest.sources[1],
    )

    with pytest.raises(FixedPoolManifestError, match="pinned dataset revision"):
        validate_openmath_latency_feasibility_manifest_design(
            replace(manifest, sources=sources)
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("order_seed", 43002, "order_seed must be 43001"),
        ("model_revision", _digest("wrong-model-revision"), "pinned model"),
        ("model_weights_sha256", _digest("wrong-model-weights"), "pinned model"),
    ],
)
def test_openmath_latency_feasibility_design_rejects_manifest_provenance(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    manifest = load_fixed_pool_manifest(_write_openmath_manifest(tmp_path))

    with pytest.raises(FixedPoolManifestError, match=match):
        validate_openmath_latency_feasibility_manifest_design(
            replace(manifest, **{field: value})
        )


def test_openmath_latency_feasibility_design_rejects_unbalanced_cohort(
    tmp_path: Path,
) -> None:
    manifest = load_fixed_pool_manifest(_write_openmath_manifest(tmp_path))
    items = list(manifest.items)
    short_index = next(
        index
        for index, item in enumerate(items)
        if item.dispatch_cohort == 0 and item.task_name == "openmath_short"
    )
    long_index = next(
        index
        for index, item in enumerate(items)
        if item.dispatch_cohort == 1 and item.task_name == "openmath_long"
    )
    items[short_index] = items[short_index].model_copy(
        update={"dispatch_cohort": 1, "decorrelation_block": "cohort-1"}
    )
    items[long_index] = items[long_index].model_copy(
        update={"dispatch_cohort": 0, "decorrelation_block": "cohort-0"}
    )

    with pytest.raises(FixedPoolManifestError, match=r"balanced 2\+2 block"):
        validate_openmath_latency_feasibility_manifest_design(
            replace(manifest, items=tuple(items))
        )


def test_openmath_latency_feasibility_design_rejects_input_caliper(
    tmp_path: Path,
) -> None:
    def mutate(source_id: str, row_index: int, record: dict[str, object]) -> None:
        if source_id == "openmath_long" and row_index == 0:
            record["input_token_count"] = 109

    manifest = load_fixed_pool_manifest(
        _write_openmath_manifest(tmp_path, record_mutation=mutate)
    )

    with pytest.raises(FixedPoolManifestError, match="input-token caliper"):
        validate_openmath_latency_feasibility_manifest_design(manifest)


def test_openmath_latency_feasibility_design_rejects_pair_across_cohorts(
    tmp_path: Path,
) -> None:
    manifest = load_fixed_pool_manifest(_write_openmath_manifest(tmp_path))
    items = list(manifest.items)
    first_long = next(
        index
        for index, item in enumerate(items)
        if item.matching_pair_id == "pair-0" and item.task_name == "openmath_long"
    )
    second_long = next(
        index
        for index, item in enumerate(items)
        if item.matching_pair_id == "pair-2" and item.task_name == "openmath_long"
    )
    first_cohort = items[first_long].dispatch_cohort
    second_cohort = items[second_long].dispatch_cohort
    items[first_long] = items[first_long].model_copy(
        update={
            "dispatch_cohort": second_cohort,
            "decorrelation_block": f"cohort-{second_cohort}",
        }
    )
    items[second_long] = items[second_long].model_copy(
        update={
            "dispatch_cohort": first_cohort,
            "decorrelation_block": f"cohort-{first_cohort}",
        }
    )

    with pytest.raises(FixedPoolManifestError, match="crosses dispatch cohorts"):
        validate_openmath_latency_feasibility_manifest_design(
            replace(manifest, items=tuple(items))
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("reference_solution_token_count", 97, "outside the openmath_short band"),
        ("reference_answer_token_count", 11, "answer-token caliper"),
        ("normalized_problem_sha256", "not-a-hash", "lowercase SHA-256"),
        ("problem_source", "", "requires problem_source"),
        ("source_revision", "wrong-revision", "bound source metadata mismatch"),
        ("selection_seed", 20260903, "bound source metadata mismatch"),
        ("model_revision", _digest("wrong-model-revision"), "bound source metadata"),
        ("prompt_file_sha256", _digest("wrong-prompt"), "bound source metadata"),
    ],
)
def test_openmath_latency_feasibility_design_rejects_bound_metadata(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    def mutate(source_id: str, row_index: int, record: dict[str, object]) -> None:
        if source_id == "openmath_short" and row_index == 0:
            record[field] = value

    manifest = load_fixed_pool_manifest(
        _write_openmath_manifest(tmp_path, record_mutation=mutate)
    )

    with pytest.raises(FixedPoolManifestError, match=match):
        validate_openmath_latency_feasibility_manifest_design(manifest)


@pytest.mark.parametrize("schema_change", ["extra", "missing"])
def test_openmath_latency_feasibility_design_rejects_record_schema_drift(
    tmp_path: Path, schema_change: str
) -> None:
    def mutate(source_id: str, row_index: int, record: dict[str, object]) -> None:
        if source_id == "openmath_short" and row_index == 0:
            if schema_change == "extra":
                record["unregistered_metadata"] = "unexpected"
            else:
                del record["output"]

    manifest = load_fixed_pool_manifest(
        _write_openmath_manifest(tmp_path, record_mutation=mutate)
    )

    with pytest.raises(FixedPoolManifestError, match="record schema mismatch"):
        validate_openmath_latency_feasibility_manifest_design(manifest)


def test_openmath_latency_feasibility_design_rejects_cross_source_pair(
    tmp_path: Path,
) -> None:
    def mutate(source_id: str, row_index: int, record: dict[str, object]) -> None:
        if source_id == "openmath_long" and row_index == 0:
            record["problem_source"] = "different_source"

    manifest = load_fixed_pool_manifest(
        _write_openmath_manifest(tmp_path, record_mutation=mutate)
    )

    with pytest.raises(FixedPoolManifestError, match="crosses problem sources"):
        validate_openmath_latency_feasibility_manifest_design(manifest)


def test_openmath_latency_feasibility_design_rejects_duplicate_problem(
    tmp_path: Path,
) -> None:
    def mutate(source_id: str, row_index: int, record: dict[str, object]) -> None:
        if source_id == "openmath_long" and row_index == 0:
            record["normalized_problem_sha256"] = _digest("openmath_short-problem-0")

    manifest = load_fixed_pool_manifest(
        _write_openmath_manifest(tmp_path, record_mutation=mutate)
    )

    with pytest.raises(FixedPoolManifestError, match="16 distinct normalized"):
        validate_openmath_latency_feasibility_manifest_design(manifest)


def test_openmath_latency_feasibility_design_rejects_long_input(
    tmp_path: Path,
) -> None:
    def mutate(source_id: str, row_index: int, record: dict[str, object]) -> None:
        if source_id == "openmath_short" and row_index == 0:
            record["input_token_count"] = 257

    manifest = load_fixed_pool_manifest(
        _write_openmath_manifest(tmp_path, record_mutation=mutate)
    )

    with pytest.raises(FixedPoolManifestError, match="256-token bound"):
        validate_openmath_latency_feasibility_manifest_design(manifest)


def test_openmath_latency_feasibility_design_rejects_median_ratio(
    tmp_path: Path,
) -> None:
    def mutate(source_id: str, _row_index: int, record: dict[str, object]) -> None:
        record["reference_solution_token_count"] = (
            96 if source_id == "openmath_short" else 256
        )

    manifest = load_fixed_pool_manifest(
        _write_openmath_manifest(tmp_path, record_mutation=mutate)
    )

    with pytest.raises(FixedPoolManifestError, match="median ratio is below 3.0"):
        validate_openmath_latency_feasibility_manifest_design(manifest)


def test_fixed_pool_design_dispatch_rejects_unknown_design(tmp_path: Path) -> None:
    manifest = load_fixed_pool_manifest(_write_openmath_manifest(tmp_path))

    with pytest.raises(FixedPoolManifestError, match="unsupported fixed-pool"):
        validate_fixed_pool_manifest_design(manifest, "unknown")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda record: record.update(unexpected=True), "extra_forbidden"),
        (
            lambda record: record["items"][1].update(ordinal=2),  # type: ignore[index,union-attr]
            "ordinals must be contiguous",
        ),
        (
            lambda record: record["items"][1].update(  # type: ignore[index,union-attr]
                pool_item_id=record["items"][0]["pool_item_id"]  # type: ignore[index]
            ),
            "pool_item_id values must be unique",
        ),
        (
            lambda record: record["items"][1].update(dispatch_cohort=2),  # type: ignore[index,union-attr]
            "cohort IDs must be contiguous",
        ),
        (
            lambda record: record["items"][0].update(source_prompt_id="not-a-hash"),  # type: ignore[index,union-attr]
            "string_pattern_mismatch",
        ),
        (
            lambda record: record["items"][0].update(source_id="unknown"),  # type: ignore[index,union-attr]
            "reference unknown sources",
        ),
        (
            lambda record: record["items"][1].update(  # type: ignore[index,union-attr]
                source_id="gsm8k",
                source_dataset_index=20,
            ),
            "source coordinates must preserve",
        ),
    ],
)
def test_manifest_rejects_structural_mutations(
    tmp_path: Path, mutation, match: str
) -> None:
    record = _manifest_record()
    mutation(record)
    # Recompute when possible so the structural invariant, rather than a stale
    # pool digest, is what rejects the mutation.
    try:
        record["pool_id"] = compute_fixed_pool_id(record)
    except FixedPoolManifestError:
        pass
    path = tmp_path / "bad.json"
    _write_manifest(path, record)

    with pytest.raises(FixedPoolManifestError, match=match):
        load_fixed_pool_manifest(path)


def test_manifest_rejects_stale_pool_id(tmp_path: Path) -> None:
    record = _manifest_record()
    record["sources"][0]["revision"] = "different-revision"  # type: ignore[index]
    path = tmp_path / "stale.json"
    _write_manifest(path, record)

    with pytest.raises(FixedPoolManifestError, match="pool_id does not match"):
        load_fixed_pool_manifest(path)


def test_manifest_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":1,"schema_version":1}\n')

    with pytest.raises(FixedPoolManifestError, match="duplicate JSON key"):
        load_fixed_pool_manifest(path)


def test_materialized_sources_are_bound_to_manifest_items(tmp_path: Path) -> None:
    record = copy.deepcopy(_manifest_record())
    model_dir = tmp_path / "model_snapshot"
    model_dir.mkdir()
    weights = b"small-test-weights"
    (model_dir / "model.safetensors").write_bytes(weights)
    snapshot_manifest = (
        json.dumps(
            [
                {
                    "path": "model.safetensors",
                    "bytes": len(weights),
                    "sha256": hashlib.sha256(weights).hexdigest(),
                }
            ],
            indent=2,
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    (tmp_path / "model_snapshot_manifest.v1.json").write_bytes(snapshot_manifest)
    record["model_weights_sha256"] = hashlib.sha256(weights).hexdigest()
    record["model_snapshot_manifest_sha256"] = hashlib.sha256(
        snapshot_manifest
    ).hexdigest()
    record["sources"] = [record["sources"][0]]  # type: ignore[index]
    item = record["items"][0]  # type: ignore[index]
    item["dataset_index"] = 0
    item["materialized_source_row"] = 0
    materialized = {
        "input": "private prompt",
        "output": "private answer",
        "source_id": item["source_id"],
        "source_dataset_index": item["source_dataset_index"],
        "source_prompt_id": item["source_prompt_id"],
        "repeated_prompt_cluster_id": item["repeated_prompt_cluster_id"],
    }
    canonical = json.dumps(materialized, sort_keys=True, separators=(",", ":")).encode()
    raw = canonical + b"\n"
    (tmp_path / "gsm8k.jsonl").write_bytes(raw)
    record["sources"][0]["content_sha256"] = hashlib.sha256(raw).hexdigest()  # type: ignore[index]
    item["materialized_record_sha256"] = hashlib.sha256(canonical).hexdigest()
    record["items"] = [item]
    record["pool_id"] = compute_fixed_pool_id(record)
    manifest_path = tmp_path / "pool.json"
    _write_manifest(manifest_path, record)
    manifest = load_fixed_pool_manifest(manifest_path)

    validate_fixed_pool_materialization(manifest)

    (tmp_path / "gsm8k.jsonl").write_bytes(raw.replace(b"private prompt", b"changed"))
    with pytest.raises(FixedPoolManifestError, match="source SHA mismatch"):
        validate_fixed_pool_materialization(manifest)


def _datum(idx: int, task_name: str) -> DatumSpec:
    return {
        "message_log": [
            {
                "role": "user",
                "content": f"private prompt {idx}",
                "token_ids": torch.tensor([idx + 1]),
            }
        ],
        "length": 1,
        "extra_env_info": None,
        "loss_multiplier": 1.0,
        "idx": idx,
        "task_name": task_name,
    }


def test_dataset_and_collate_preserve_only_opaque_manifest_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pool.json"
    _write_manifest(path, _manifest_record())
    manifest = load_fixed_pool_manifest(path)
    dataset = FixedPoolDataset(
        [
            _datum(0, "slow-math"),
            _datum(1, "unused"),
            _datum(2, "fast-math"),
        ],
        manifest,
    )

    first, second = dataset[0], dataset[1]
    assert [first["idx"], second["idx"]] == [2, 0]
    first_batch = fixed_pool_collate_fn([first])
    second_batch = fixed_pool_collate_fn([second])
    assert first_batch["source_pool_ordinal"] == [0]
    assert second_batch["source_pool_ordinal"] == [1]
    assert first_batch["source_prompt_id"] == [manifest.items[0].source_prompt_id]
    assert second_batch["source_prompt_id"] == [manifest.items[1].source_prompt_id]
    assert all(
        "private prompt" not in value
        for value in first_batch["source_prompt_id"] + second_batch["source_prompt_id"]
    )
    with pytest.raises(FixedPoolManifestError, match="must not cross"):
        fixed_pool_collate_fn([first, second])


@pytest.mark.parametrize(
    ("dataset", "match"),
    [
        (
            [_datum(99, "slow-math"), _datum(1, "unused"), _datum(2, "fast-math")],
            "expected dataset idx 0",
        ),
        (
            [_datum(0, "wrong-task"), _datum(1, "unused"), _datum(2, "fast-math")],
            "expected task 'slow-math'",
        ),
    ],
)
def test_dataset_fails_closed_on_source_mismatch(
    tmp_path: Path, dataset: list[DatumSpec], match: str
) -> None:
    path = tmp_path / "pool.json"
    _write_manifest(path, _manifest_record())
    wrapped = FixedPoolDataset(dataset, load_fixed_pool_manifest(path))

    with pytest.raises(FixedPoolManifestError, match=match):
        wrapped[1]


def _fixed_pool_master_config(**overrides) -> MasterConfig:
    async_rl = AsyncRLConfig(
        max_inflight_prompts=2,
        max_buffered_rollouts=2,
        scheduler_trace=SchedulerTraceConfig(enabled=True, path="trace.jsonl"),
        fixed_pool=FixedPoolCollectionConfig(
            enabled=True,
            manifest_path="pool.json",
        ),
    )
    values = {
        "data": {"shuffle": False, "use_multiple_dataloader": False},
        "grpo": GRPOConfig.model_construct(
            use_dynamic_sampling=False,
            val_period=0,
            val_at_start=False,
            val_at_end=False,
            stop_at_validation_metric=None,
        ),
        "checkpointing": {"enabled": False},
        "async_rl": async_rl,
    }
    values.update(overrides)
    return MasterConfig.model_construct(**values)


def test_fixed_pool_config_skips_training_only_sampler_constraints() -> None:
    config = _fixed_pool_master_config(policy={"train_global_batch_size": -1})
    validate_single_controller_config(config)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda config: setattr(
                config.async_rl,
                "scheduler_trace",
                SchedulerTraceConfig(enabled=False),
            ),
            "scheduler_trace.enabled must be true",
        ),
        (
            lambda config: config.data.update(shuffle=True),
            "data.shuffle must be false",
        ),
        (
            lambda config: config.data.update(use_multiple_dataloader=True),
            "use_multiple_dataloader is unsupported",
        ),
        (
            lambda config: setattr(config.grpo, "use_dynamic_sampling", True),
            "use_dynamic_sampling must be false",
        ),
        (
            lambda config: config.checkpointing.update(enabled=True),
            "checkpointing must be disabled",
        ),
    ],
)
def test_fixed_pool_config_rejects_incompatible_modes(mutation, match: str) -> None:
    config = _fixed_pool_master_config()
    mutation(config)

    with pytest.raises(ValueError, match=match):
        validate_single_controller_config(config)


def test_fixed_pool_config_requires_manifest_path() -> None:
    with pytest.raises(ValidationError, match="manifest_path is required"):
        FixedPoolCollectionConfig(enabled=True)


def test_fixed_pool_config_preserves_ready_bias_default() -> None:
    config = FixedPoolCollectionConfig()

    assert config.design_id == "ready_bias_v1"


def test_fixed_pool_config_accepts_openmath_design() -> None:
    config = FixedPoolCollectionConfig(
        enabled=True,
        manifest_path="pool.json",
        design_id="openmath_latency_feasibility_v1",
    )

    assert config.design_id == "openmath_latency_feasibility_v1"


def test_fixed_pool_config_accepts_structured_generation_design() -> None:
    config = FixedPoolCollectionConfig(
        enabled=True,
        manifest_path="pool.json",
        design_id="structured_generation_latency_v1",
    )

    assert config.design_id == "structured_generation_latency_v1"


def test_fixed_pool_config_accepts_structured_scheduler_crossover_design() -> None:
    config = FixedPoolCollectionConfig(
        enabled=True,
        manifest_path="pool.json",
        design_id="structured_generation_scheduler_crossover_v1",
    )

    assert config.design_id == "structured_generation_scheduler_crossover_v1"


def test_fixed_pool_config_accepts_dapo_scheduler_crossover_design() -> None:
    config = FixedPoolCollectionConfig(
        enabled=True,
        manifest_path="pool.json",
        design_id="dapo_math_scheduler_crossover_v1",
    )

    assert config.design_id == "dapo_math_scheduler_crossover_v1"


def test_fixed_pool_config_rejects_unknown_design() -> None:
    with pytest.raises(ValidationError, match="literal_error"):
        FixedPoolCollectionConfig(design_id="unknown")  # type: ignore[arg-type]


def _write_complete_fixed_pool_trace(
    trace_path: Path,
    manifest_path: Path,
    *,
    include_selection: bool = False,
) -> None:
    manifest = load_fixed_pool_manifest(manifest_path)

    async def write() -> None:
        sink = JsonlSchedulerTraceSink(
            trace_path,
            trace_run_id="fixed-pool-test",
            process_epoch="process-0",
        )
        await sink.start()
        boundary = {
            "run_mode": "fixed_pool",
            "pool_id": manifest.pool_id,
            "pool_manifest_sha256": manifest.manifest_sha256,
            "model_revision": manifest.model_revision,
            "model_weights_sha256": manifest.model_weights_sha256,
            "trainer_version": 0,
        }
        sink.emit(
            SchedulerEventType.RUN_STARTED,
            **boundary,
            scalar_summaries={"planned_prompt_groups": len(manifest.items)},
        )
        for item in manifest.items:
            admission_id = f"cohort-{item.dispatch_cohort}"
            sink.emit(
                SchedulerEventType.ADMISSION_GRANTED,
                admission_id=admission_id,
                sampler_dispatch_index=item.dispatch_cohort,
                trainer_version=0,
                scalar_summaries={"expected_prompt_groups": 1},
            )
            identity = {
                "logical_group_id": f"group-{item.ordinal}",
                "attempt_id": f"attempt-{item.ordinal}",
                "admission_id": admission_id,
                "prompt_idx": item.dataset_index,
                "task_name": item.task_name,
                "source_prompt_id": item.source_prompt_id,
                "repeated_prompt_cluster_id": item.repeated_prompt_cluster_id,
                "source_pool_ordinal": item.ordinal,
                "dispatch_cohort": item.dispatch_cohort,
                "trainer_version": 0,
            }
            sink.emit(
                SchedulerEventType.ATTEMPT_DISPATCHED,
                **identity,
                start_weight_version=0,
            )
            sink.emit(
                SchedulerEventType.ROLLOUT_COMPLETED,
                **identity,
                scalar_summaries={"completion_count": 2},
            )
            sink.emit(
                SchedulerEventType.GROUP_READY,
                **identity,
                start_weight_version=0,
                end_weight_version=0,
            )
            sink.emit(
                SchedulerEventType.GROUP_ARCHIVED,
                **identity,
                terminal_reason="fixed_pool_archive",
            )
        if include_selection:
            sink.emit(
                SchedulerEventType.SELECT_DECISION,
                trainer_version=0,
                min_prompt_groups=1,
                max_prompt_groups=1,
                eligible_prompt_groups=0,
                ready_prompt_groups=0,
                scalar_summaries={"selected_prompt_groups": 0},
            )
        sink.emit(
            SchedulerEventType.RUN_ENDED,
            **boundary,
            terminal_reason="fixed_pool_complete",
            live_logical_group_ids=(),
            scalar_summaries={
                "completed_train_steps": 0,
                "collected_prompt_groups": len(manifest.items),
            },
        )
        await sink.close()

    asyncio.run(write())


def test_fixed_pool_trace_requires_exact_complete_archived_lifecycle(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "pool.json"
    _write_manifest(manifest_path, _manifest_record())
    trace_path = tmp_path / "trace.jsonl"
    _write_complete_fixed_pool_trace(trace_path, manifest_path)

    report = validate_fixed_pool_trace(
        trace_path,
        load_fixed_pool_manifest(manifest_path),
        expected_completions_per_group=2,
    )

    assert report.planned_prompt_groups == 2
    assert report.dispatched_prompt_groups == 2
    assert report.completed_prompt_groups == 2
    assert report.ready_prompt_groups == 2
    assert report.archived_prompt_groups == 2
    assert report.physical_weight_version == 0


def test_fixed_pool_trace_rejects_any_scheduler_selection(tmp_path: Path) -> None:
    manifest_path = tmp_path / "pool.json"
    _write_manifest(manifest_path, _manifest_record())
    trace_path = tmp_path / "trace.jsonl"
    _write_complete_fixed_pool_trace(
        trace_path,
        manifest_path,
        include_selection=True,
    )

    with pytest.raises(SchedulerTraceValidationError, match="forbidden lifecycle"):
        validate_fixed_pool_trace(
            trace_path,
            load_fixed_pool_manifest(manifest_path),
            expected_completions_per_group=2,
        )


def _collector_batch(cohort: int, ordinals: tuple[int, ...]) -> BatchedDataDict:
    return BatchedDataDict(
        {
            "idx": list(ordinals),
            "task_name": ["math"] * len(ordinals),
            "source_pool_ordinal": list(ordinals),
            "source_prompt_id": [_digest(f"prompt-{i}") for i in ordinals],
            "repeated_prompt_cluster_id": [_digest(f"cluster-{i}") for i in ordinals],
            "dispatch_cohort": [cohort] * len(ordinals),
        }
    )


def _collector_controller(dataloader, rollout_manager, trace):
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    controller = object.__new__(controller_cls)
    controller._fixed_pool_manifest = SimpleNamespace(
        items=tuple(range(4)),
        pool_id=_digest("pool"),
        manifest_sha256=_digest("manifest"),
    )
    controller._async_cfg = SimpleNamespace(max_inflight_prompts=2)
    controller._buffer_capacity = asyncio.Semaphore(2)
    controller._dataloader = dataloader
    controller._rollout_manager = rollout_manager
    controller._scheduler_trace = trace
    controller._sampler_fingerprint = _digest("fixed-pool-sampler")
    controller._trainer_version = 0
    controller._collected_groups = 0
    return controller


def test_collector_dispatches_in_manifest_order_and_archives_out_of_order() -> None:
    class Trace:
        def __init__(self) -> None:
            self.admissions = []

        def emit(self, event_type, **fields) -> None:
            assert event_type is SchedulerEventType.ADMISSION_GRANTED
            self.admissions.append(fields)

    class Manager:
        def __init__(self) -> None:
            self.dispatched = []
            self.archived = []

        async def generate_and_push(
            self, prompt, *, target_step, admission_id, dispatch_started_event
        ):
            ordinal = prompt["source_pool_ordinal"]
            self.dispatched.append(ordinal)
            dispatch_started_event.set()
            await asyncio.sleep((3 - ordinal) * 0.001)
            return SimpleNamespace(ordinal=ordinal)

        async def archive_fixed_pool_group(self, handle) -> None:
            self.archived.append(handle.ordinal)

    trace = Trace()
    manager = Manager()
    controller = _collector_controller(
        [_collector_batch(0, (0, 1)), _collector_batch(1, (2, 3))],
        manager,
        trace,
    )

    asyncio.run(controller._collect_fixed_pool())

    assert manager.dispatched == [0, 1, 2, 3]
    assert manager.archived != manager.dispatched
    assert sorted(manager.archived) == manager.dispatched
    assert controller._collected_groups == 4
    assert [item["sampler_dispatch_index"] for item in trace.admissions] == [0, 1]


def test_collector_cancels_started_tasks_when_later_batch_iteration_fails() -> None:
    class Trace:
        def emit(self, event_type, **fields) -> None:
            del event_type, fields

    class Manager:
        def __init__(self) -> None:
            self.cancelled = 0

        async def generate_and_push(
            self, prompt, *, target_step, admission_id, dispatch_started_event
        ):
            del prompt, target_step, admission_id
            dispatch_started_event.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise

        async def archive_fixed_pool_group(self, handle) -> None:
            raise AssertionError(f"unexpected archive: {handle}")

    class FailingLoader:
        def __iter__(self):
            yield _collector_batch(0, (0, 1))
            raise RuntimeError("later batch failed")

    manager = Manager()
    controller = _collector_controller(FailingLoader(), manager, Trace())

    with pytest.raises(RuntimeError, match="later batch failed"):
        asyncio.run(controller._collect_fixed_pool())

    assert manager.cancelled == 2
    assert controller._buffer_capacity._value == 2


def test_collector_does_not_hang_if_manager_fails_before_dispatch_signal() -> None:
    class Trace:
        def emit(self, event_type, **fields) -> None:
            del event_type, fields

    class Manager:
        async def generate_and_push(self, prompt, **kwargs):
            del prompt, kwargs
            raise RuntimeError("failed before dispatch signal")

        async def archive_fixed_pool_group(self, handle) -> None:
            raise AssertionError(f"unexpected archive: {handle}")

    controller = _collector_controller(
        [_collector_batch(0, (0, 1))], Manager(), Trace()
    )
    controller._fixed_pool_manifest.items = tuple(range(2))

    async def run() -> None:
        with pytest.raises(RuntimeError, match="failed before dispatch signal"):
            await asyncio.wait_for(controller._collect_fixed_pool(), timeout=0.5)

    asyncio.run(run())
    assert controller._buffer_capacity._value == 2
