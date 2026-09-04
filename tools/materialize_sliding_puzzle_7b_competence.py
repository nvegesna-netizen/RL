# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Rematerialize the exposed puzzle pool against the pinned 7B model."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from nemo_rl.algorithms.async_utils.fixed_pool import (
    compute_fixed_pool_id,
    load_fixed_pool_manifest,
    validate_fixed_pool_manifest_design,
    validate_fixed_pool_materialization,
)


MODEL_REPO: Final[str] = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION: Final[str] = "a09a35458c702b33eeacc393d103063234e8bc28"
MODEL_WEIGHTS_SHA256: Final[str] = (
    "456f5eff514d78f7b0ef52a057118046acf15d86606655767db068cabf5f49f7"
)
GENERATION_CONFIG_SHA256: Final[str] = (
    "3a8f9087e486054c8a4a08dae2e5a3ba62e23da212b5b8c08bc42cb983c3459f"
)
PLAN_ID: Final[str] = "3b2b0dbbbc0e799c9668d3649155eca0f97b8b4fc4436d0c2e430d3c700208eb"
PLAN_SHA256: Final[str] = (
    "81912121165377eb2de725f71e61a2fef96af74ea9a1d92085695f338e029494"
)
SOURCE_MANIFEST_SHA256: Final[str] = (
    "6a16ad62e35aaf9f07610ae96c43b7988b6a25887a279d00e5547b79e411dd64"
)
SOURCE_POOL_ID: Final[str] = (
    "8ac5a1a26685947c375836769b45d458ac7e5ea296d66a62c4312e7d8fc0c6c6"
)
SOURCE_SELECTION_DESIGN_SHA256: Final[str] = (
    "4c3598bb501c8d34c38266fff2d0cdaa5691eb37337c07be91449b1d5491b9e9"
)
EXPECTED_WEIGHT_FILES: Final[tuple[tuple[str, int, str], ...]] = (
    (
        "model-00001-of-00004.safetensors",
        3_945_441_440,
        "a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7",
    ),
    (
        "model-00002-of-00004.safetensors",
        3_864_726_352,
        "f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185",
    ),
    (
        "model-00003-of-00004.safetensors",
        3_864_726_424,
        "8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5",
    ),
    (
        "model-00004-of-00004.safetensors",
        3_556_377_672,
        "1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd",
    ),
)


class CompetenceMaterializationError(ValueError):
    """The frozen competence plan or its source pool is inconsistent."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: bytes) -> None:
    path.write_bytes(value)
    path.chmod(0o600)


def _load_plan(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    if _sha256_bytes(raw) != PLAN_SHA256:
        raise CompetenceMaterializationError("competence plan byte hash mismatch")
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as error:
        raise CompetenceMaterializationError(
            "competence plan is invalid JSON"
        ) from error
    if not isinstance(plan, dict):
        raise CompetenceMaterializationError("competence plan must be an object")
    if set(plan) != {
        "analysis_status",
        "attempt_limit",
        "calibration_only",
        "confirmatory_eligible",
        "expected_boards",
        "hard_board_metrics_gate",
        "latency_or_readiness_inference_allowed",
        "model",
        "plan_id",
        "replay_authorized",
        "runtime",
        "schema_version",
        "source_pool",
        "thresholds",
        "training_authorized",
    }:
        raise CompetenceMaterializationError("competence plan keys do not match v1")
    without_id = {key: value for key, value in plan.items() if key != "plan_id"}
    if plan.get("plan_id") != _sha256_bytes(_canonical(without_id)):
        raise CompetenceMaterializationError("competence plan ID mismatch")
    expected = {
        "analysis_status": "calibration_only_exposed_pool_model_competence",
        "attempt_limit": 1,
        "calibration_only": True,
        "confirmatory_eligible": False,
        "expected_boards": 16,
        "hard_board_metrics_gate": False,
        "latency_or_readiness_inference_allowed": False,
        "plan_id": PLAN_ID,
        "replay_authorized": False,
        "schema_version": 1,
        "training_authorized": False,
    }
    if any(plan.get(key) != value for key, value in expected.items()):
        raise CompetenceMaterializationError("competence plan violates frozen labels")
    model = plan.get("model")
    expected_weights = [
        {"path": path, "bytes": size, "sha256": sha256}
        for path, size, sha256 in EXPECTED_WEIGHT_FILES
    ]
    if not isinstance(model, dict) or model != {
        "generation_config_sha256": GENERATION_CONFIG_SHA256,
        "sampling_source": "pinned_generation_config_supported_fields",
        "repo_id": MODEL_REPO,
        "revision": MODEL_REVISION,
        "weight_files": expected_weights,
        "weight_manifest_sha256": MODEL_WEIGHTS_SHA256,
    }:
        raise CompetenceMaterializationError("competence model pin mismatch")
    source = plan.get("source_pool")
    if not isinstance(source, dict) or source != {
        "board_exposure_status": "previously_exposed_calibration_pool",
        "fixed_pool_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "pool_id": SOURCE_POOL_ID,
        "selection_design_sha256": SOURCE_SELECTION_DESIGN_SHA256,
    }:
        raise CompetenceMaterializationError("competence source-pool pin mismatch")
    if plan.get("runtime") != {
        "generation_study_seed": 63001,
        "max_inflight_prompts": 4,
        "max_moves": 12,
        "max_new_tokens_per_turn": 128,
        "max_rollout_turns": 12,
        "max_total_sequence_length": 2048,
        "num_generations_per_prompt": 2,
        "repetition_penalty": 1.0,
        "temperature": 0.7,
        "top_k": 20,
        "top_p": 0.8,
    } or plan.get("thresholds") != {
        "action_format_valid_rate_min": 0.9,
        "all_expected_completions_required": True,
        "easy_max_turn_rate_max": 0.25,
        "easy_solve_rate_min": 0.5,
        "easy_truncation_rate_max": 0.25,
        "movement_legal_rate_min": 0.75,
    }:
        raise CompetenceMaterializationError(
            "competence runtime or thresholds mismatch"
        )
    return plan, raw


def _snapshot_model(root: Path) -> tuple[Any, bytes]:
    model_root = root / "model_snapshot"
    snapshot_download(repo_id=MODEL_REPO, revision=MODEL_REVISION, local_dir=model_root)
    if any(path.is_symlink() for path in model_root.rglob("*")):
        raise CompetenceMaterializationError("model snapshot contains a symbolic link")
    for relative, expected_size, expected_sha in EXPECTED_WEIGHT_FILES:
        path = model_root / relative
        if path.stat().st_size != expected_size or _sha256_path(path) != expected_sha:
            raise CompetenceMaterializationError(f"model weight mismatch: {relative}")
    if _sha256_path(model_root / "generation_config.json") != GENERATION_CONFIG_SHA256:
        raise CompetenceMaterializationError("generation config hash mismatch")
    records = [
        {
            "path": str(path.relative_to(model_root)),
            "bytes": path.stat().st_size,
            "sha256": _sha256_path(path),
        }
        for path in sorted(model_root.rglob("*"))
        if path.is_file() and ".cache" not in path.parts
    ]
    weight_records = [
        record for record in records if str(record["path"]).endswith(".safetensors")
    ]
    if _sha256_bytes(_canonical(weight_records)) != MODEL_WEIGHTS_SHA256:
        raise CompetenceMaterializationError("model weight-manifest hash mismatch")
    snapshot_manifest = json.dumps(records, indent=2, sort_keys=True).encode() + b"\n"
    _write(root / "model_snapshot_manifest.v1.json", snapshot_manifest)
    return AutoTokenizer.from_pretrained(model_root), snapshot_manifest


def _load_source_rows(manifest: Any) -> dict[str, list[dict[str, object]]]:
    rows_by_source: dict[str, list[dict[str, object]]] = {}
    for source in manifest.sources:
        path = manifest.manifest_path.parent / source.materialized_file
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if any(not isinstance(row, dict) for row in rows):
            raise CompetenceMaterializationError(
                "source pool contains a non-object row"
            )
        rows_by_source[source.source_id] = rows
    return rows_by_source


def _retokenize(tokenizer: Any, record: Mapping[str, object]) -> dict[str, object]:
    messages = record.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 1
        or not isinstance(messages[0], dict)
        or messages[0].get("role") != "user"
        or not isinstance(messages[0].get("content"), str)
    ):
        raise CompetenceMaterializationError(
            "source record requires one rendered user message"
        )
    rendered_content = messages[0]["content"]
    input_ids = tokenizer(
        rendered_content, return_tensors=None, add_special_tokens=False
    )["input_ids"]
    updated = dict(record)
    updated["model_revision"] = MODEL_REVISION
    updated["input_token_count"] = len(input_ids)
    updated["input_token_ids_sha256"] = _sha256_bytes(
        _canonical([int(value) for value in input_ids])
    )
    return updated


def _build_manifest(
    *,
    source_manifest: Any,
    plan_sha256: str,
    source_rows: Mapping[str, Sequence[Mapping[str, object]]],
    source_raw: Mapping[str, bytes],
    snapshot_manifest_sha256: str,
) -> dict[str, object]:
    item_rows = {
        (source_id, row): record
        for source_id, rows in source_rows.items()
        for row, record in enumerate(rows)
    }
    items = []
    for source_item in source_manifest.items:
        record = item_rows[(source_item.source_id, source_item.materialized_source_row)]
        items.append(
            {
                "ordinal": source_item.ordinal,
                "pool_item_id": _sha256_bytes(
                    _canonical(
                        [
                            "sliding-puzzle-7b-competence-item-v1",
                            SOURCE_POOL_ID,
                            source_item.ordinal,
                            MODEL_REVISION,
                        ]
                    )
                ),
                "source_prompt_id": source_item.source_prompt_id,
                "source_id": source_item.source_id,
                "source_dataset_index": source_item.source_dataset_index,
                "dataset_index": source_item.dataset_index,
                "task_name": source_item.task_name,
                "repeated_prompt_cluster_id": source_item.repeated_prompt_cluster_id,
                "dispatch_cohort": source_item.dispatch_cohort,
                "decorrelation_block": source_item.decorrelation_block,
                "matching_pair_id": source_item.matching_pair_id,
                "materialized_source_row": source_item.materialized_source_row,
                "materialized_record_sha256": _sha256_bytes(_canonical(record)),
                "input_token_count": record["input_token_count"],
                "input_token_ids_sha256": record["input_token_ids_sha256"],
            }
        )
    sources = [
        {
            "source_id": source.source_id,
            "dataset_id": source.dataset_id,
            "revision": source.revision,
            "split": source.split,
            "materialized_file": source.materialized_file,
            "content_sha256": _sha256_bytes(source_raw[source.source_id]),
        }
        for source in source_manifest.sources
    ]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "id_namespace_fingerprint": _sha256_bytes(
            _canonical([source_manifest.id_namespace_fingerprint, PLAN_ID])
        ),
        "design_protocol_sha256": plan_sha256,
        "order_seed": source_manifest.order_seed,
        "model_revision": MODEL_REVISION,
        "model_weights_sha256": MODEL_WEIGHTS_SHA256,
        "model_snapshot_manifest_sha256": snapshot_manifest_sha256,
        "model_snapshot_path": "model_snapshot",
        "sources": sources,
        "items": items,
    }
    manifest["pool_id"] = compute_fixed_pool_id(manifest)
    return manifest


def _set_private_modes(root: Path) -> None:
    for directory in [root, *(path for path in root.rglob("*") if path.is_dir())]:
        directory.chmod(0o700)
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(0o600)


def materialize(*, source_path: Path, plan_path: Path, output_dir: Path) -> None:
    """Create a 7B-bound copy of the exact exposed 16-board pool."""
    plan, plan_raw = _load_plan(plan_path)
    if _sha256_path(source_path) != SOURCE_MANIFEST_SHA256:
        raise CompetenceMaterializationError("source manifest hash mismatch")
    selection_design = source_path.parent / "selection_design.v1.json"
    if _sha256_path(selection_design) != SOURCE_SELECTION_DESIGN_SHA256:
        raise CompetenceMaterializationError("source selection-design hash mismatch")
    source_manifest = load_fixed_pool_manifest(source_path)
    validate_fixed_pool_materialization(source_manifest)
    validate_fixed_pool_manifest_design(
        source_manifest, "sliding_puzzle_latency_feasibility_v1"
    )
    if source_manifest.pool_id != SOURCE_POOL_ID:
        raise CompetenceMaterializationError("source pool ID mismatch")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.", dir=output_dir.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        temporary.chmod(0o700)
        tokenizer, snapshot_manifest = _snapshot_model(temporary)
        original_rows = _load_source_rows(source_manifest)
        rows = {
            source_id: [_retokenize(tokenizer, record) for record in source_rows]
            for source_id, source_rows in original_rows.items()
        }
        source_raw = {
            source_id: b"".join(_canonical(record) + b"\n" for record in source_rows)
            for source_id, source_rows in rows.items()
        }
        for source in source_manifest.sources:
            _write(temporary / source.materialized_file, source_raw[source.source_id])
        _write(temporary / "competence_plan.v1.json", plan_raw)
        plan_sha256 = _sha256_bytes(plan_raw)
        manifest = _build_manifest(
            source_manifest=source_manifest,
            plan_sha256=plan_sha256,
            source_rows=rows,
            source_raw=source_raw,
            snapshot_manifest_sha256=_sha256_bytes(snapshot_manifest),
        )
        manifest_raw = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
        manifest_path = temporary / "fixed_pool_manifest.v1.7b_competence.json"
        _write(manifest_path, manifest_raw)
        board_hashes = sorted(
            str(record["board_state_sha256"])
            for source_rows in rows.values()
            for record in source_rows
        )
        source_board_hashes = sorted(
            str(record["board_state_sha256"])
            for source_rows in original_rows.values()
            for record in source_rows
        )
        if board_hashes != source_board_hashes or len(set(board_hashes)) != 16:
            raise CompetenceMaterializationError("source board identity changed")
        lineage = {
            "schema_version": 1,
            "analysis_status": plan["analysis_status"],
            "plan_id": PLAN_ID,
            "plan_sha256": plan_sha256,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_pool_id": SOURCE_POOL_ID,
            "source_selection_design_sha256": SOURCE_SELECTION_DESIGN_SHA256,
            "board_state_sha256": board_hashes,
            "materialized_manifest_sha256": _sha256_bytes(manifest_raw),
            "materialized_pool_id": manifest["pool_id"],
        }
        _write(
            temporary / "source_lineage.v1.json",
            json.dumps(lineage, indent=2, sort_keys=True).encode() + b"\n",
        )
        report = {
            "schema_version": 1,
            "status": "passed",
            "design_id": "sliding_puzzle_7b_competence_v1",
            "analysis_status": plan["analysis_status"],
            "plan_id": PLAN_ID,
            "plan_sha256": plan_sha256,
            "source_lineage_sha256": _sha256_path(temporary / "source_lineage.v1.json"),
            "manifest_sha256": _sha256_bytes(manifest_raw),
            "pool_id": manifest["pool_id"],
        }
        _write(
            temporary / "materialization_report.v1.json",
            json.dumps(report, indent=2, sort_keys=True).encode() + b"\n",
        )
        loaded = load_fixed_pool_manifest(manifest_path)
        validate_fixed_pool_materialization(loaded)
        validate_fixed_pool_manifest_design(loaded, "sliding_puzzle_7b_competence_v1")
        hash_paths = [
            temporary / source.materialized_file for source in source_manifest.sources
        ] + [
            temporary / "competence_plan.v1.json",
            manifest_path,
            temporary / "model_snapshot_manifest.v1.json",
            temporary / "source_lineage.v1.json",
            temporary / "materialization_report.v1.json",
        ]
        checksum = b"".join(
            f"{_sha256_path(path)}  {path.name}\n".encode() for path in hash_paths
        )
        _write(temporary / "SHA256SUMS", checksum)
        _set_private_modes(temporary)
        temporary.rename(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    materialize(
        source_path=args.source_manifest,
        plan_path=args.plan,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
