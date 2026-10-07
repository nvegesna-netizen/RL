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

"""Shared-frontier, no-update capsule support for the conditional M4 bridge."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from nemo_rl.algorithms.async_utils.opportunity_at_risk import (
    OPPORTUNITY_GROUP_ID_KEY,
    OPPORTUNITY_L1_KEY,
    OPPORTUNITY_VALID_TOKENS_KEY,
)
from nemo_rl.algorithms.async_utils.opportunity_at_risk_v2 import (
    OPPORTUNITY_REWARD_MEAN_KEY,
    OPPORTUNITY_REWARD_VARIANCE_KEY,
    OARSV2Candidate,
    select_two_sided_oars_v2,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import DP_TRAIN_FIELDS
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

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


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path, *, chunk_bytes: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ConditionalM4Selection:
    """One comparator/treatment contrast chosen from a shared frontier."""

    comparator_group_ids: tuple[str, ...]
    treatment_group_ids: tuple[str, ...]
    comparator_predicted_norm_sum: float
    treatment_predicted_norm_sum: float
    relative_predicted_norm_gain: float
    comparator_reward_variance: float
    treatment_reward_variance: float
    comparator_tokens: int
    treatment_tokens: int
    minimum_tokens: int
    maximum_tokens: int
    feasible_combination_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparator_group_ids": list(self.comparator_group_ids),
            "treatment_group_ids": list(self.treatment_group_ids),
            "comparator_predicted_norm_sum": self.comparator_predicted_norm_sum,
            "treatment_predicted_norm_sum": self.treatment_predicted_norm_sum,
            "relative_predicted_norm_gain": self.relative_predicted_norm_gain,
            "comparator_reward_variance": self.comparator_reward_variance,
            "treatment_reward_variance": self.treatment_reward_variance,
            "reward_variance_retention": (
                self.treatment_reward_variance / self.comparator_reward_variance
                if self.comparator_reward_variance > 0
                else 1.0
            ),
            "comparator_tokens": self.comparator_tokens,
            "treatment_tokens": self.treatment_tokens,
            "token_ratio": self.treatment_tokens / self.comparator_tokens,
            "minimum_tokens": self.minimum_tokens,
            "maximum_tokens": self.maximum_tokens,
            "feasible_combination_count": self.feasible_combination_count,
            "shared_group_count": len(
                set(self.comparator_group_ids) & set(self.treatment_group_ids)
            ),
        }


@dataclass(frozen=True)
class CapsuleGroup:
    """Prepared version-zero tensors and provenance for one prompt group."""

    meta: KVBatchMeta
    fields: BatchedDataDict[Any]


def candidate_from_meta(
    meta: KVBatchMeta, *, ready_timestamp_ns: int
) -> OARSV2Candidate:
    """Build the registered scheduler candidate from prepared group metadata."""
    info = meta.extra_info
    group_id = info.get(OPPORTUNITY_GROUP_ID_KEY)
    if not isinstance(group_id, str) or not group_id:
        raise ValueError("capsule candidate lacks a nonempty opportunity group ID")
    values = {
        "l1": info.get(OPPORTUNITY_L1_KEY),
        "valid_actor_tokens": info.get(OPPORTUNITY_VALID_TOKENS_KEY),
        "reward_mean": info.get(OPPORTUNITY_REWARD_MEAN_KEY),
        "reward_variance": info.get(OPPORTUNITY_REWARD_VARIANCE_KEY),
    }
    if not isinstance(values["valid_actor_tokens"], int) or isinstance(
        values["valid_actor_tokens"], bool
    ):
        raise ValueError("valid actor tokens must be an integer")
    numeric = ("l1", "reward_mean", "reward_variance")
    if not all(
        isinstance(values[name], (int, float))
        and not isinstance(values[name], bool)
        and math.isfinite(float(values[name]))
        for name in numeric
    ):
        raise ValueError("capsule candidate has nonfinite opportunity metadata")
    if (
        float(values["l1"]) < 0
        or int(values["valid_actor_tokens"]) < 1
        or float(values["reward_variance"]) < 0
    ):
        raise ValueError("capsule candidate has invalid nonnegative metadata")
    return OARSV2Candidate(
        group_id=group_id,
        start_weight_version=0,
        ready_timestamp_ns=ready_timestamp_ns,
        l1=float(values["l1"]),
        l2=0.0,
        valid_actor_tokens=int(values["valid_actor_tokens"]),
        reward_mean=float(values["reward_mean"]),
        reward_variance=float(values["reward_variance"]),
    )


def predict_gradient_norm(
    candidate: OARSV2Candidate,
    *,
    coefficients: Sequence[float],
    intercept: float,
) -> float:
    """Apply the frozen positive-M4 deployment model."""
    if candidate.l1 <= 0:
        return 0.0
    if len(coefficients) != 4:
        raise ValueError("conditional M4 deployment model requires four coefficients")
    features = (
        math.log1p(candidate.valid_actor_tokens),
        candidate.reward_mean,
        candidate.reward_variance,
        math.log1p(candidate.l1),
    )
    prediction = math.exp(
        intercept
        + math.fsum(
            value * weight for value, weight in zip(features, coefficients, strict=True)
        )
    )
    if not math.isfinite(prediction) or prediction < 0:
        raise ValueError("conditional M4 prediction is invalid")
    return prediction


def select_conditional_m4_contrast(
    candidates: Sequence[OARSV2Candidate],
    *,
    coefficients: Sequence[float],
    intercept: float,
    minimum_service_multiplier: float,
    maximum_service_multiplier: float,
    batch_groups: int,
) -> ConditionalM4Selection:
    """Choose a reward-variance comparator and conditional-M4 treatment."""
    if len(candidates) < batch_groups or batch_groups < 1:
        raise ValueError("candidate frontier is smaller than the batch cardinality")
    ordered = tuple(sorted(candidates, key=lambda row: row.group_id))
    if len({row.group_id for row in ordered}) != len(ordered):
        raise ValueError("candidate group IDs must be unique")
    fifo_ids = tuple(row.group_id for row in candidates[:batch_groups])
    comparator = select_two_sided_oars_v2(
        ordered,
        baseline_group_ids=fifo_ids,
        current_train_weight=0,
        decision_timestamp_ns=max(row.ready_timestamp_ns for row in ordered),
        max_staleness_versions=0,
        scorer="reward_variance_risk",
        minimum_service_multiplier=minimum_service_multiplier,
        maximum_service_multiplier=maximum_service_multiplier,
        exact_search_max_candidates=len(ordered),
        deadline_ns=None,
    )
    by_id = {row.group_id: row for row in ordered}
    comparator_rows = tuple(
        by_id[group_id] for group_id in comparator.proposed_group_ids
    )
    comparator_variance = math.fsum(row.reward_variance for row in comparator_rows)
    comparator_tokens = sum(row.valid_actor_tokens for row in comparator_rows)
    comparator_score = math.fsum(
        predict_gradient_norm(row, coefficients=coefficients, intercept=intercept)
        for row in comparator_rows
    )

    best_rows = comparator_rows
    best_ids = tuple(sorted(comparator.proposed_group_ids))
    best_key = (
        comparator_score,
        comparator_variance,
        -comparator_tokens,
    )
    feasible = 0
    for rows in itertools.combinations(ordered, batch_groups):
        tokens = sum(row.valid_actor_tokens for row in rows)
        if tokens < comparator.minimum_tokens or tokens > comparator.maximum_tokens:
            continue
        variance = math.fsum(row.reward_variance for row in rows)
        if variance + _NUMERIC_TOLERANCE < comparator_variance:
            continue
        feasible += 1
        score = math.fsum(
            predict_gradient_norm(row, coefficients=coefficients, intercept=intercept)
            for row in rows
        )
        ids = tuple(row.group_id for row in rows)
        key = (score, variance, -tokens)
        if key > best_key or (key == best_key and ids < best_ids):
            best_rows = rows
            best_ids = ids
            best_key = key
    if feasible < 1:
        raise RuntimeError("registered reward-variance comparator is not feasible")
    treatment_score = best_key[0]
    relative_gain = (
        treatment_score / comparator_score - 1.0 if comparator_score > 0 else 0.0
    )
    return ConditionalM4Selection(
        comparator_group_ids=tuple(comparator.proposed_group_ids),
        treatment_group_ids=tuple(row.group_id for row in best_rows),
        comparator_predicted_norm_sum=comparator_score,
        treatment_predicted_norm_sum=treatment_score,
        relative_predicted_norm_gain=relative_gain,
        comparator_reward_variance=comparator_variance,
        treatment_reward_variance=math.fsum(row.reward_variance for row in best_rows),
        comparator_tokens=comparator_tokens,
        treatment_tokens=sum(row.valid_actor_tokens for row in best_rows),
        minimum_tokens=comparator.minimum_tokens,
        maximum_tokens=comparator.maximum_tokens,
        feasible_combination_count=feasible,
    )


def tensor_content_sha256(fields: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(fields):
        tensor = fields[name].detach().cpu().contiguous()
        header = _canonical_json(
            {"dtype": str(tensor.dtype), "name": name, "shape": list(tensor.shape)}
        )
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(tensor.view(torch.uint8).reshape(-1).numpy().tobytes())
    return digest.hexdigest()


def write_capsule(
    output_dir: Path,
    *,
    frontier: Sequence[CapsuleGroup],
    heldout: Sequence[CapsuleGroup],
    selection: ConditionalM4Selection,
    parameter_sha256_before: str,
    parameter_sha256_after: str,
    frontiers_examined: int,
) -> dict[str, Any]:
    """Write raw tensor files plus a compact, canonical authentication manifest."""
    if output_dir.exists():
        raise FileExistsError(f"capsule output already exists: {output_dir}")
    if parameter_sha256_before != parameter_sha256_after:
        raise RuntimeError(
            "model parameters changed during no-update capsule collection"
        )
    all_groups = tuple(frontier) + tuple(heldout)
    group_ids = [
        group.meta.extra_info.get(OPPORTUNITY_GROUP_ID_KEY) for group in all_groups
    ]
    if not all(isinstance(group_id, str) and group_id for group_id in group_ids):
        raise ValueError("capsule groups require nonempty IDs")
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("capsule frontier and heldout groups must be disjoint")
    if not set(selection.comparator_group_ids).issubset(
        set(group_ids[: len(frontier)])
    ):
        raise ValueError("comparator is not contained in the shared frontier")
    if not set(selection.treatment_group_ids).issubset(set(group_ids[: len(frontier)])):
        raise ValueError("treatment is not contained in the shared frontier")

    output_dir.mkdir(parents=True)
    groups_dir = output_dir / "groups"
    groups_dir.mkdir()
    rows: list[dict[str, Any]] = []
    for index, group in enumerate(all_groups):
        fields: dict[str, torch.Tensor] = {}
        for field in DP_TRAIN_FIELDS:
            value = group.fields.get(field)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"capsule field {field!r} is not a tensor")
            fields[field] = value.detach().cpu().contiguous().clone()
        sample_count = len(group.meta.sample_ids)
        if not all(value.shape[0] == sample_count for value in fields.values()):
            raise ValueError("capsule tensor batch dimensions disagree")
        raw_path = groups_dir / f"group-{index:02d}.pt"
        # PyTorch's restricted ``weights_only`` loader intentionally supports
        # the conservative protocol-2 instruction set used by torch.save's
        # default.  Keep capsule files readable without relaxing safe loading.
        torch.save(fields, raw_path, pickle_protocol=2)
        group_id = str(group_ids[index])
        roles: list[str] = []
        if group_id in selection.comparator_group_ids:
            roles.append("comparator")
        if group_id in selection.treatment_group_ids:
            roles.append("treatment")
        if index >= len(frontier):
            roles.append("heldout")
        rows.append(
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
    manifest = {
        "schema": "conditional-m4-shared-capsule-v1",
        "status": "QUALIFIED_NO_UPDATE_CAPSULE",
        "frontiers_examined": frontiers_examined,
        "shared_frontier_groups": len(frontier),
        "disjoint_heldout_groups": len(heldout),
        "selection": selection.to_dict(),
        "parameter_sha256_before": parameter_sha256_before,
        "parameter_sha256_after": parameter_sha256_after,
        "parameter_hash_unchanged": True,
        "learner_version": 0,
        "finish_train_step_calls": 0,
        "optimizer_steps": 0,
        "scheduler_steps": 0,
        "learning_outcomes_opened": False,
        "groups": rows,
    }
    group_authentication = _sha256(_canonical_json(rows))
    manifest["group_authentication_sha256"] = group_authentication
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_bytes(_canonical_json(manifest))
    return manifest | {"manifest_sha256": _file_sha256(manifest_path)}
