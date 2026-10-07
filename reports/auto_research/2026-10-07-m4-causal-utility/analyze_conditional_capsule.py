#!/usr/bin/env python3
"""Authenticate a conditional-M4 no-update shared capsule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from nemo_rl.algorithms.async_utils.conditional_m4_capsule import (
    tensor_content_sha256,
)


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


def analyze(capsule_dir: Path, protocol_path: Path) -> dict[str, Any]:
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    manifest_path = capsule_dir / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    design = protocol["frozen_design"]
    contract = protocol["capsule_contract"]
    rows = manifest["groups"]
    expected_inventory = {"manifest.json"} | {
        f"groups/group-{index:02d}.pt"
        for index in range(design["candidate_groups"] + design["heldout_groups"])
    }
    actual_inventory = {
        str(path.relative_to(capsule_dir))
        for path in capsule_dir.rglob("*")
        if path.is_file()
    }

    checks: dict[str, bool] = {
        "schema": manifest.get("schema") == "conditional-m4-shared-capsule-v1",
        "status": manifest.get("status") == "QUALIFIED_NO_UPDATE_CAPSULE",
        "frontier_count": manifest.get("shared_frontier_groups")
        == design["candidate_groups"],
        "heldout_count": manifest.get("disjoint_heldout_groups")
        == design["heldout_groups"],
        "frontier_budget": 1
        <= manifest.get("frontiers_examined", 0)
        <= design["maximum_frontiers"],
        "parameter_hash_unchanged": manifest.get("parameter_hash_unchanged") is True
        and manifest.get("parameter_sha256_before")
        == manifest.get("parameter_sha256_after"),
        "learner_version_zero": manifest.get("learner_version")
        == design["learner_version"],
        "finish_train_step_calls_zero": manifest.get("finish_train_step_calls")
        == contract["finish_train_step_calls"],
        "optimizer_steps_zero": manifest.get("optimizer_steps")
        == contract["optimizer_steps"],
        "scheduler_steps_zero": manifest.get("scheduler_steps")
        == contract["scheduler_steps"],
        "learning_outcomes_unopened": manifest.get("learning_outcomes_opened") is False,
        "row_count": len(rows) == design["candidate_groups"] + design["heldout_groups"],
        "group_authentication": manifest.get("group_authentication_sha256")
        == _sha256(_canonical_json(rows)),
        "exact_inventory": actual_inventory == expected_inventory,
    }

    expected_fields = tuple(contract["required_fields"])
    group_ids: list[str] = []
    sample_ids: list[str] = []
    raw_hashes_valid = True
    tensor_hashes_valid = True
    fields_valid = True
    shapes_valid = True
    paths_valid = True
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
        expected_shapes = {
            name: list(fields[name].shape) for name in sorted(expected_fields)
        }
        if expected_shapes != row["tensor_shapes"]:
            shapes_valid = False
        if not all(
            fields[name].shape[0] == len(row["sample_ids"]) for name in expected_fields
        ):
            shapes_valid = False

    selection = manifest["selection"]
    frontier_ids = set(group_ids[: design["candidate_groups"]])
    heldout_ids = set(group_ids[design["candidate_groups"] :])
    comparator_ids = set(selection["comparator_group_ids"])
    treatment_ids = set(selection["treatment_group_ids"])
    role_sets = {
        role: {row["group_id"] for row in rows if role in row["roles"]}
        for role in ("comparator", "treatment", "heldout")
    }
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
            "comparator_from_frontier": len(comparator_ids) == design["batch_groups"]
            and comparator_ids.issubset(frontier_ids),
            "treatment_from_frontier": len(treatment_ids) == design["batch_groups"]
            and treatment_ids.issubset(frontier_ids),
            "arms_differ": comparator_ids != treatment_ids,
            "roles_match_selection": role_sets["comparator"] == comparator_ids
            and role_sets["treatment"] == treatment_ids
            and role_sets["heldout"] == heldout_ids,
            "predicted_gain": selection["relative_predicted_norm_gain"]
            >= design["minimum_relative_predicted_norm_gain"],
            "reward_variance_retained": selection["reward_variance_retention"]
            >= design["minimum_reward_variance_retention"],
            "token_band": design["minimum_token_ratio"]
            <= selection["token_ratio"]
            <= design["maximum_token_ratio"],
        }
    )
    passed = all(checks.values())
    return {
        "schema": "conditional-m4-capsule-qualification-result-v1",
        "status": "PASS_NO_UPDATE_CAPSULE" if passed else "FAIL_CAPSULE_GATE",
        "protocol_sha256": _sha256(protocol_bytes),
        "manifest_sha256": _sha256(manifest_bytes),
        "group_authentication_sha256": manifest.get("group_authentication_sha256"),
        "checks": checks,
        "frontiers_examined": manifest.get("frontiers_examined"),
        "relative_predicted_norm_gain": selection.get("relative_predicted_norm_gain"),
        "reward_variance_retention": selection.get("reward_variance_retention"),
        "token_ratio": selection.get("token_ratio"),
        "shared_arm_groups": selection.get("shared_group_count"),
        "claim_boundary": protocol["claim_boundary"],
        "decision": protocol["decision_ladder"]["pass" if passed else "fail_integrity"],
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
    if result["status"] != "PASS_NO_UPDATE_CAPSULE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
