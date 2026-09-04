# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Materialize the exposed puzzle pool with a compact action-only prompt."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from nemo_rl.algorithms.async_utils.fixed_pool import (
    compute_fixed_pool_id,
    load_fixed_pool_manifest,
    validate_fixed_pool_manifest_design,
    validate_fixed_pool_materialization,
)
from nemo_rl.environments.games.sliding_puzzle import SlidingPuzzleGameLogic
from tools import materialize_sliding_puzzle_7b_competence as base


PLAN_ID: Final[str] = "9d1d45624504ce4370c8e51a888e1160ee3d916869c1bcb5b4e1a363f0c068b1"
PLAN_SHA256: Final[str] = (
    "8fdb6d007560e1d260dcd2cbb2a0ddbae863a26e7f5cb15c67c81eea9c1de454"
)
DESIGN_ID: Final[str] = "sliding_puzzle_7b_compact_prompt_v2"
PROMPT_BUILDER_VERSION: Final[str] = "compact_action_only_v2"
OUTPUT_MANIFEST: Final[str] = "fixed_pool_manifest.v2.7b_compact_action.json"


class CompactActionMaterializationError(ValueError):
    """The compact-action plan or its source pool is inconsistent."""


def _load_plan(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    if base._sha256_bytes(raw) != PLAN_SHA256:
        raise CompactActionMaterializationError(
            "compact-action plan byte hash mismatch"
        )
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as error:
        raise CompactActionMaterializationError(
            "compact-action plan is invalid JSON"
        ) from error
    if not isinstance(plan, dict):
        raise CompactActionMaterializationError("compact-action plan must be an object")
    without_id = {key: value for key, value in plan.items() if key != "plan_id"}
    if (
        plan.get("plan_id") != PLAN_ID
        or base._sha256_bytes(base._canonical(without_id)) != PLAN_ID
    ):
        raise CompactActionMaterializationError("compact-action plan ID mismatch")
    expected_labels = {
        "analysis_status": "calibration_only_exposed_pool_prompt_competence",
        "attempt_limit": 1,
        "calibration_only": True,
        "confirmatory_eligible": False,
        "expected_boards": 16,
        "hard_board_metrics_gate": False,
        "latency_or_readiness_inference_allowed": False,
        "replay_authorized": False,
        "schema_version": 2,
        "training_authorized": False,
    }
    if any(plan.get(key) != value for key, value in expected_labels.items()):
        raise CompactActionMaterializationError(
            "compact-action plan violates frozen labels"
        )
    expected_weights = [
        {"path": path, "bytes": size, "sha256": sha256}
        for path, size, sha256 in base.EXPECTED_WEIGHT_FILES
    ]
    if plan.get("model") != {
        "generation_config_sha256": base.GENERATION_CONFIG_SHA256,
        "repo_id": base.MODEL_REPO,
        "revision": base.MODEL_REVISION,
        "sampling_source": "pinned_generation_config_supported_fields",
        "weight_files": expected_weights,
        "weight_manifest_sha256": base.MODEL_WEIGHTS_SHA256,
    }:
        raise CompactActionMaterializationError("compact-action model pin mismatch")
    if plan.get("source_pool") != {
        "board_exposure_status": "previously_exposed_calibration_pool",
        "fixed_pool_manifest_sha256": base.SOURCE_MANIFEST_SHA256,
        "pool_id": base.SOURCE_POOL_ID,
        "selection_design_sha256": base.SOURCE_SELECTION_DESIGN_SHA256,
    }:
        raise CompactActionMaterializationError("compact-action source pin mismatch")
    if plan.get("prompt_intervention") != {
        "builder_version": PROMPT_BUILDER_VERSION,
        "changes_only_prompt": True,
        "legacy_builder_version": "legacy_sliding_puzzle_prompt_v1",
        "paired_prior_pipeline": 66188117,
        "response_contract": "exactly_one_action_tag_and_no_other_text",
    }:
        raise CompactActionMaterializationError(
            "compact-action prompt intervention mismatch"
        )
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
        raise CompactActionMaterializationError(
            "compact-action runtime or thresholds mismatch"
        )
    return plan, raw


def _prompt_for_game_state(game_state: Mapping[str, object]) -> str:
    rendered_board = SlidingPuzzleGameLogic.render(dict(game_state))
    return (
        "Solve this 3x3 sliding puzzle by moving one tile into the empty cell.\n"
        "The goal is:\n1 2 3\n4 5 6\n7 8 0\n\n"
        f"Current board:\n{rendered_board}\n\n"
        "Directions name the tile's motion into the empty cell: up moves the tile "
        "below the empty cell; down moves the tile above it; left moves the tile "
        "to its right; right moves the tile to its left.\n"
        "Reply with exactly one legal move and no other text. Use exactly one of:\n"
        "<action>up</action>\n<action>down</action>\n"
        "<action>left</action>\n<action>right</action>"
    )


def _rewrite_prompt(tokenizer: Any, record: Mapping[str, object]) -> dict[str, object]:
    extra = record.get("extra_env_info")
    game_state = extra.get("game_state") if isinstance(extra, dict) else None
    if not isinstance(game_state, dict):
        raise CompactActionMaterializationError("source record has no game state")
    prompt = _prompt_for_game_state(game_state)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    ).strip()
    input_ids = tokenizer(rendered, return_tensors=None, add_special_tokens=False)[
        "input_ids"
    ]
    updated = dict(record)
    updated["messages"] = [{"role": "user", "content": rendered}]
    updated["model_revision"] = base.MODEL_REVISION
    updated["prompt_builder_version"] = PROMPT_BUILDER_VERSION
    updated["input_token_count"] = len(input_ids)
    updated["input_token_ids_sha256"] = base._sha256_bytes(
        base._canonical([int(value) for value in input_ids])
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
                "pool_item_id": base._sha256_bytes(
                    base._canonical(
                        [
                            "sliding-puzzle-7b-compact-action-item-v2",
                            base.SOURCE_POOL_ID,
                            source_item.ordinal,
                            base.MODEL_REVISION,
                            PLAN_ID,
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
                "materialized_record_sha256": base._sha256_bytes(
                    base._canonical(record)
                ),
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
            "content_sha256": base._sha256_bytes(source_raw[source.source_id]),
        }
        for source in source_manifest.sources
    ]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "id_namespace_fingerprint": base._sha256_bytes(
            base._canonical([source_manifest.id_namespace_fingerprint, PLAN_ID])
        ),
        "design_protocol_sha256": plan_sha256,
        "order_seed": source_manifest.order_seed,
        "model_revision": base.MODEL_REVISION,
        "model_weights_sha256": base.MODEL_WEIGHTS_SHA256,
        "model_snapshot_manifest_sha256": snapshot_manifest_sha256,
        "model_snapshot_path": "model_snapshot",
        "sources": sources,
        "items": items,
    }
    manifest["pool_id"] = compute_fixed_pool_id(manifest)
    return manifest


def materialize(*, source_path: Path, plan_path: Path, output_dir: Path) -> None:
    """Create the compact-action copy of the exact exposed 16-board pool."""
    plan, plan_raw = _load_plan(plan_path)
    if base._sha256_path(source_path) != base.SOURCE_MANIFEST_SHA256:
        raise CompactActionMaterializationError("source manifest hash mismatch")
    selection_design = source_path.parent / "selection_design.v1.json"
    if base._sha256_path(selection_design) != base.SOURCE_SELECTION_DESIGN_SHA256:
        raise CompactActionMaterializationError("source selection-design hash mismatch")
    source_manifest = load_fixed_pool_manifest(source_path)
    validate_fixed_pool_materialization(source_manifest)
    validate_fixed_pool_manifest_design(
        source_manifest, "sliding_puzzle_latency_feasibility_v1"
    )
    if source_manifest.pool_id != base.SOURCE_POOL_ID:
        raise CompactActionMaterializationError("source pool ID mismatch")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.", dir=output_dir.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        temporary.chmod(0o700)
        tokenizer, snapshot_manifest = base._snapshot_model(temporary)
        original_rows = base._load_source_rows(source_manifest)
        rows = {
            source_id: [_rewrite_prompt(tokenizer, record) for record in source_rows]
            for source_id, source_rows in original_rows.items()
        }
        source_raw = {
            source_id: b"".join(
                base._canonical(record) + b"\n" for record in source_rows
            )
            for source_id, source_rows in rows.items()
        }
        for source in source_manifest.sources:
            base._write(
                temporary / source.materialized_file, source_raw[source.source_id]
            )
        base._write(temporary / "competence_plan.v2.json", plan_raw)
        plan_sha256 = base._sha256_bytes(plan_raw)
        manifest = _build_manifest(
            source_manifest=source_manifest,
            plan_sha256=plan_sha256,
            source_rows=rows,
            source_raw=source_raw,
            snapshot_manifest_sha256=base._sha256_bytes(snapshot_manifest),
        )
        manifest_raw = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
        manifest_path = temporary / OUTPUT_MANIFEST
        base._write(manifest_path, manifest_raw)
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
            raise CompactActionMaterializationError("source board identity changed")
        lineage = {
            "schema_version": 2,
            "analysis_status": plan["analysis_status"],
            "plan_id": PLAN_ID,
            "plan_sha256": plan_sha256,
            "source_manifest_sha256": base.SOURCE_MANIFEST_SHA256,
            "source_pool_id": base.SOURCE_POOL_ID,
            "source_selection_design_sha256": base.SOURCE_SELECTION_DESIGN_SHA256,
            "board_state_sha256": board_hashes,
            "materialized_manifest_sha256": base._sha256_bytes(manifest_raw),
            "materialized_pool_id": manifest["pool_id"],
        }
        base._write(
            temporary / "source_lineage.v2.json",
            json.dumps(lineage, indent=2, sort_keys=True).encode() + b"\n",
        )
        report = {
            "schema_version": 2,
            "status": "passed",
            "design_id": DESIGN_ID,
            "analysis_status": plan["analysis_status"],
            "plan_id": PLAN_ID,
            "plan_sha256": plan_sha256,
            "source_lineage_sha256": base._sha256_path(
                temporary / "source_lineage.v2.json"
            ),
            "manifest_sha256": base._sha256_bytes(manifest_raw),
            "pool_id": manifest["pool_id"],
        }
        base._write(
            temporary / "materialization_report.v2.json",
            json.dumps(report, indent=2, sort_keys=True).encode() + b"\n",
        )
        loaded = load_fixed_pool_manifest(manifest_path)
        validate_fixed_pool_materialization(loaded)
        validate_fixed_pool_manifest_design(loaded, DESIGN_ID)
        hash_paths = [
            temporary / source.materialized_file for source in source_manifest.sources
        ] + [
            temporary / "competence_plan.v2.json",
            manifest_path,
            temporary / "model_snapshot_manifest.v1.json",
            temporary / "source_lineage.v2.json",
            temporary / "materialization_report.v2.json",
        ]
        checksum = b"".join(
            f"{base._sha256_path(path)}  {path.name}\n".encode() for path in hash_paths
        )
        base._write(temporary / "SHA256SUMS", checksum)
        base._set_private_modes(temporary)
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
