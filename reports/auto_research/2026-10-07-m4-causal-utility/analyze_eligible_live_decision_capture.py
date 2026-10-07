#!/usr/bin/env python3
"""Independently authenticate an outcome-excluded eligible live capsule."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from nemo_rl.algorithms.async_utils.conditional_m4_capsule import (
    tensor_content_sha256,
)
from nemo_rl.data_plane.schema import DP_TRAIN_FIELDS


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path, *, chunk_bytes: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _recursive_inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"checkpoint contains symlink: {path}")
        if path.is_file():
            rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
    return rows


def analyze(capsule_dir: Path, protocol_path: Path) -> dict[str, Any]:
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    manifest_path = capsule_dir / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    eligibility = protocol["eligibility"]
    capsule_contract = protocol["capsule"]
    rows = manifest.get("groups", [])

    checks: dict[str, bool] = {
        "protocol_schema": protocol.get("schema")
        == "m4-shield-eligible-live-causal-v1",
        "manifest_schema": manifest.get("schema")
        == "m4-shield-eligible-live-capsule-v1",
        "status": manifest.get("status")
        == "QUALIFIED_OUTCOME_EXCLUDED_ELIGIBLE_CAPTURE",
        "learner_version_matches_updates": manifest.get("learner_version")
        == manifest.get("train_steps_before_capture"),
        "parameter_hash_unchanged": manifest.get("parameter_hash_unchanged") is True
        and manifest.get("parameter_sha256_before")
        == manifest.get("parameter_sha256_after"),
        "captured_update_not_executed": manifest.get("captured_update_executed")
        is False,
        "outcomes_unopened": manifest.get("post_update_outcomes_opened") is False,
        "captured_optimizer_steps_zero": manifest.get(
            "optimizer_steps_at_captured_decision"
        )
        == 0,
        "captured_scheduler_steps_zero": manifest.get(
            "scheduler_steps_at_captured_decision"
        )
        == 0,
        "row_count": len(rows)
        == capsule_contract["frontier_tensor_groups"]
        + capsule_contract["heldout_tensor_groups"],
        "group_authentication": manifest.get("group_authentication_sha256")
        == _sha256(_canonical_json(rows)),
    }

    expected_fields = tuple(DP_TRAIN_FIELDS)
    group_ids: list[str] = []
    sample_ids: list[str] = []
    paths_valid = True
    raw_hashes_valid = True
    tensor_hashes_valid = True
    fields_valid = True
    shapes_valid = True
    root = capsule_dir.resolve()
    for expected_index, row in enumerate(rows):
        group_ids.append(row["group_id"])
        sample_ids.extend(row["sample_ids"])
        raw_path = (capsule_dir / row["raw_file"]).resolve()
        if (
            root not in raw_path.parents
            or raw_path.name != f"group-{expected_index:02d}.pt"
        ):
            paths_valid = False
            continue
        if not raw_path.is_file() or _file_sha256(raw_path) != row["raw_file_sha256"]:
            raw_hashes_valid = False
            continue
        fields = torch.load(raw_path, map_location="cpu", weights_only=True)
        if (
            tuple(fields) != expected_fields
            or tuple(row["field_names"]) != expected_fields
        ):
            fields_valid = False
        if not all(
            isinstance(fields.get(name), torch.Tensor) for name in expected_fields
        ):
            fields_valid = False
            continue
        if tensor_content_sha256(fields) != row["tensor_content_sha256"]:
            tensor_hashes_valid = False
        if {name: list(fields[name].shape) for name in sorted(expected_fields)} != row[
            "tensor_shapes"
        ]:
            shapes_valid = False
        if not all(
            fields[name].shape[0] == len(row["sample_ids"]) for name in expected_fields
        ):
            shapes_valid = False

    frontier_count = capsule_contract["frontier_tensor_groups"]
    frontier_ids = set(group_ids[:frontier_count])
    heldout_ids = set(group_ids[frontier_count:])
    base_ids = set(manifest.get("base_group_ids", []))
    shield_ids = set(manifest.get("shield_group_ids", []))
    role_sets = {
        role: {row["group_id"] for row in rows if role in row["roles"]}
        for role in ("base", "shield", "heldout")
    }
    decision = manifest.get("decision", {})
    proposals = decision.get("proposals", {})
    base = proposals.get("reward_variance_risk", {})
    shield = proposals.get("m4_shield", {})
    exact_variance = all(
        math.isclose(
            float(base.get(field, math.nan)),
            float(shield.get(field, math.nan)),
            rel_tol=0.0,
            abs_tol=eligibility["numeric_tolerance"],
        )
        for field in (
            "proposed_reward_variance_sum",
            "proposed_imminent_reward_variance_sum",
        )
    )
    fifo_tokens = int(decision.get("baseline_valid_actor_tokens", 0))
    shield_tokens = int(shield.get("proposed_valid_actor_tokens", 0))
    checks.update(
        {
            "paths": paths_valid,
            "raw_file_hashes": raw_hashes_valid,
            "tensor_content_hashes": tensor_hashes_valid,
            "required_fields": fields_valid,
            "tensor_shapes": shapes_valid,
            "unique_group_ids": len(group_ids) == len(set(group_ids)),
            "unique_sample_ids": len(sample_ids) == len(set(sample_ids)),
            "heldout_disjoint": frontier_ids.isdisjoint(heldout_ids),
            "base_from_frontier": len(base_ids) == eligibility["selected_groups"]
            and base_ids.issubset(frontier_ids),
            "shield_from_frontier": len(shield_ids) == eligibility["selected_groups"]
            and shield_ids.issubset(frontier_ids),
            "actions_differ": base_ids != shield_ids,
            "minimum_overlap": len(base_ids & shield_ids)
            >= eligibility["minimum_base_overlap_groups"],
            "roles_match": role_sets["base"] == base_ids
            and role_sets["shield"] == shield_ids
            and role_sets["heldout"] == heldout_ids,
            "exact_search": base.get("status") == "proposed"
            and shield.get("status") == "proposed"
            and base.get("search_strategy") == "exact"
            and shield.get("search_strategy") == "exact",
            "actual_is_shield": decision.get("actuation_scorer") == "m4_shield"
            and decision.get("proposal_matches_actual") is True,
            "actual_ids_are_shield": set(decision.get("actual_selected_group_ids", []))
            == shield_ids,
            "token_band": fifo_tokens > 0
            and eligibility["minimum_fifo_token_multiplier"] * fifo_tokens
            <= shield_tokens
            <= eligibility["maximum_fifo_token_multiplier"] * fifo_tokens,
            "reward_variance_preserved": exact_variance,
            "strict_imminent_m4_gain": float(
                shield.get("proposed_imminent_l1", math.nan)
            )
            > float(base.get("proposed_imminent_l1", math.nan))
            + eligibility["numeric_tolerance"],
        }
    )

    checkpoint_root = (capsule_dir / manifest.get("checkpoint_root", "")).resolve()
    checkpoint_within_capsule = root in checkpoint_root.parents
    checkpoint_rows = (
        _recursive_inventory(checkpoint_root) if checkpoint_within_capsule else []
    )
    checks.update(
        {
            "checkpoint_within_capsule": checkpoint_within_capsule,
            "checkpoint_nonempty": bool(checkpoint_rows),
            "checkpoint_inventory": checkpoint_rows == manifest.get("checkpoint_files"),
            "checkpoint_authentication": manifest.get(
                "checkpoint_authentication_sha256"
            )
            == _sha256(_canonical_json(checkpoint_rows)),
        }
    )

    passed = all(checks.values())
    return {
        "schema": "m4-shield-eligible-live-capture-qualification-result-v1",
        "status": (
            "PASS_OUTCOME_EXCLUDED_ELIGIBLE_CAPTURE"
            if passed
            else "FAIL_ELIGIBLE_CAPTURE_GATE"
        ),
        "protocol_sha256": _sha256(protocol_bytes),
        "manifest_sha256": _sha256(manifest_bytes),
        "learner_version": manifest.get("learner_version"),
        "checks": checks,
        "base_group_ids": sorted(base_ids),
        "shield_group_ids": sorted(shield_ids),
        "heldout_group_ids": sorted(heldout_ids),
        "post_update_outcomes_opened": False,
        "claim_boundary": protocol["boundaries"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.capsule_dir, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_json(result))
    print(json.dumps({"output": str(args.output), "status": result["status"]}))
    if result["status"] != "PASS_OUTCOME_EXCLUDED_ELIGIBLE_CAPTURE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
