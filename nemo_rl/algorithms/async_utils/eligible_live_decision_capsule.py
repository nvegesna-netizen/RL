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

"""Outcome-excluded capsules for eligible live M4-Shield decisions."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from nemo_rl.algorithms.async_utils.conditional_m4_capsule import (
    CapsuleGroup,
    tensor_content_sha256,
)
from nemo_rl.algorithms.async_utils.opportunity_at_risk import (
    OPPORTUNITY_GROUP_ID_KEY,
)
from nemo_rl.data_plane.schema import DP_TRAIN_FIELDS

_NUMERIC_TOLERANCE = 1e-12


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


def _file_sha256(path: Path, *, chunk_bytes: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def recursive_file_inventory(root: Path) -> list[dict[str, Any]]:
    """Hash every regular file under ``root`` and reject symlinks."""
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {root}")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"checkpoint inventory forbids symlink: {path}")
        if not path.is_file():
            continue
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    if not rows:
        raise ValueError("checkpoint inventory is empty")
    return rows


def _proposal(decision: Mapping[str, Any], scorer: str) -> Mapping[str, Any]:
    proposals = decision.get("proposals")
    if not isinstance(proposals, Mapping):
        raise ValueError("eligible decision lacks proposals")
    proposal = proposals.get(scorer)
    if not isinstance(proposal, Mapping) or proposal.get("status") != "proposed":
        raise ValueError(f"eligible decision lacks exact {scorer} proposal")
    if proposal.get("search_strategy") != "exact":
        raise ValueError(f"eligible decision {scorer} search was not exact")
    return proposal


def validate_eligible_decision(
    decision: Mapping[str, Any],
    *,
    base_group_ids: Sequence[str],
    shield_group_ids: Sequence[str],
) -> None:
    """Fail closed unless the retained event satisfies the frozen eligibility rule."""
    if decision.get("skip_reason") is not None:
        raise ValueError("eligible decision was skipped")
    if decision.get("candidate_group_count") != 8:
        raise ValueError("eligible decision must contain exactly eight candidates")
    if decision.get("actuation_scorer") != "m4_shield":
        raise ValueError("eligible decision did not actuate M4-Shield")
    if decision.get("proposal_matches_actual") is not True:
        raise ValueError("eligible Shield proposal did not match the live action")
    base = _proposal(decision, "reward_variance_risk")
    shield = _proposal(decision, "m4_shield")
    if tuple(base.get("proposed_group_ids", ())) != tuple(base_group_ids):
        raise ValueError("retained base IDs disagree with the decision")
    if tuple(shield.get("proposed_group_ids", ())) != tuple(shield_group_ids):
        raise ValueError("retained Shield IDs disagree with the decision")
    if tuple(base_group_ids) == tuple(shield_group_ids):
        raise ValueError("eligible decision requires distinct actions")
    if len(base_group_ids) != 4 or len(shield_group_ids) != 4:
        raise ValueError("eligible decision requires two four-group actions")
    if len(set(base_group_ids) & set(shield_group_ids)) < 2:
        raise ValueError("eligible Shield action retained fewer than two base groups")
    for field in (
        "proposed_reward_variance_sum",
        "proposed_imminent_reward_variance_sum",
    ):
        base_value = float(base[field])
        shield_value = float(shield[field])
        if not math.isclose(
            base_value,
            shield_value,
            rel_tol=0.0,
            abs_tol=_NUMERIC_TOLERANCE,
        ):
            raise ValueError(f"eligible Shield action changed {field}")
    if not (
        float(shield["proposed_imminent_l1"])
        > float(base["proposed_imminent_l1"]) + _NUMERIC_TOLERANCE
    ):
        raise ValueError("eligible Shield action lacks strict imminent M4 gain")
    fifo_tokens = int(decision["baseline_valid_actor_tokens"])
    shield_tokens = int(shield["proposed_valid_actor_tokens"])
    if not (0.98 * fifo_tokens <= shield_tokens <= 1.02 * fifo_tokens):
        raise ValueError("eligible Shield action violates the frozen token band")
    if tuple(decision.get("actual_selected_group_ids", ())) != tuple(shield_group_ids):
        raise ValueError("eligible live action IDs disagree with the Shield action")


def write_eligible_live_decision_capsule(
    output_dir: Path,
    *,
    frontier: Sequence[CapsuleGroup],
    heldout: Sequence[CapsuleGroup],
    decision: Mapping[str, Any],
    base_group_ids: Sequence[str],
    shield_group_ids: Sequence[str],
    learner_version: int,
    train_steps_before_capture: int,
    parameter_sha256_before: str,
    parameter_sha256_after: str,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    """Persist exact tensors and an authenticated pre-update checkpoint manifest."""
    if (output_dir / "manifest.json").exists() or (output_dir / "groups").exists():
        raise FileExistsError(f"eligible live capsule already exists: {output_dir}")
    if learner_version != train_steps_before_capture or learner_version < 0:
        raise ValueError("learner version and completed update count disagree")
    if parameter_sha256_before != parameter_sha256_after:
        raise RuntimeError("model parameters changed during eligible capture")
    if len(frontier) != 8 or len(heldout) != 4:
        raise ValueError(
            "eligible live capsule requires eight frontier and four heldout groups"
        )
    validate_eligible_decision(
        decision,
        base_group_ids=base_group_ids,
        shield_group_ids=shield_group_ids,
    )

    groups = tuple(frontier) + tuple(heldout)
    group_ids = tuple(
        group.meta.extra_info.get(OPPORTUNITY_GROUP_ID_KEY) for group in groups
    )
    if not all(isinstance(group_id, str) and group_id for group_id in group_ids):
        raise ValueError("eligible live capsule groups require nonempty IDs")
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("frontier and heldout groups must be disjoint")
    frontier_ids = set(group_ids[:8])
    if set(base_group_ids) - frontier_ids or set(shield_group_ids) - frontier_ids:
        raise ValueError("base or Shield action is outside the retained frontier")

    output_dir.mkdir(parents=True, exist_ok=True)
    groups_dir = output_dir / "groups"
    groups_dir.mkdir()
    group_rows: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        fields: dict[str, torch.Tensor] = {}
        for field in DP_TRAIN_FIELDS:
            value = group.fields.get(field)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"eligible capsule field {field!r} is not a tensor")
            fields[field] = value.detach().cpu().contiguous().clone()
        sample_count = len(group.meta.sample_ids)
        if not all(value.shape[0] == sample_count for value in fields.values()):
            raise ValueError("eligible capsule tensor batch dimensions disagree")
        raw_path = groups_dir / f"group-{index:02d}.pt"
        torch.save(fields, raw_path, pickle_protocol=2)
        group_id = str(group_ids[index])
        roles: list[str] = []
        if group_id in base_group_ids:
            roles.append("base")
        if group_id in shield_group_ids:
            roles.append("shield")
        if index >= 8:
            roles.append("heldout")
        group_rows.append(
            {
                "field_names": list(DP_TRAIN_FIELDS),
                "group_id": group_id,
                "index": index,
                "raw_file": str(raw_path.relative_to(output_dir)),
                "raw_file_sha256": _file_sha256(raw_path),
                "roles": roles,
                "sample_ids": list(group.meta.sample_ids),
                "tensor_content_sha256": tensor_content_sha256(fields),
                "tensor_shapes": {
                    name: list(value.shape) for name, value in sorted(fields.items())
                },
            }
        )

    checkpoint_rows = recursive_file_inventory(checkpoint_dir)
    manifest: dict[str, Any] = {
        "schema": "m4-shield-eligible-live-capsule-v1",
        "status": "QUALIFIED_OUTCOME_EXCLUDED_ELIGIBLE_CAPTURE",
        "learner_version": learner_version,
        "train_steps_before_capture": train_steps_before_capture,
        "frontier_groups": 8,
        "heldout_groups": 4,
        "base_group_ids": list(base_group_ids),
        "shield_group_ids": list(shield_group_ids),
        "decision": dict(decision),
        "parameter_sha256_before": parameter_sha256_before,
        "parameter_sha256_after": parameter_sha256_after,
        "parameter_hash_unchanged": True,
        "captured_update_executed": False,
        "post_update_outcomes_opened": False,
        "optimizer_steps_at_captured_decision": 0,
        "scheduler_steps_at_captured_decision": 0,
        "groups": group_rows,
        "checkpoint_root": str(checkpoint_dir.relative_to(output_dir)),
        "checkpoint_files": checkpoint_rows,
        "group_authentication_sha256": hashlib.sha256(
            _canonical_json(group_rows)
        ).hexdigest(),
        "checkpoint_authentication_sha256": hashlib.sha256(
            _canonical_json(checkpoint_rows)
        ).hexdigest(),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_bytes(_canonical_json(manifest))
    return manifest | {"manifest_sha256": _file_sha256(manifest_path)}
