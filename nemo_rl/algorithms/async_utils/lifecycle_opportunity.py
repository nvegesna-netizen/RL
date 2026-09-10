# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Arm-blind post-run GRPO opportunity reconstruction from lifecycle facts."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from nemo_rl.algorithms.async_utils.gradient_opportunity import (
    GRPOAdvantageEstimatorProtocol,
    GRPOOpportunityInputs,
    GroupOpportunitySummary,
    compute_grpo_gradient_opportunity,
)

SCHEMA_VERSION = 2
DERIVATION_METHOD = "lifecycle_derived_grpo_opportunity_v1"
ESTIMATOR_SETTINGS: dict[str, object] = {
    "baseline_algorithm": "nemo_rl_grpo_v1",
    "normalize_rewards": True,
    "normalization_epsilon": 1e-6,
    "reduction_dtype": "float32",
    "reward_dtype": "float32",
    "use_leave_one_out_baseline": True,
}


class LifecycleOpportunityError(ValueError):
    """Raised when lifecycle evidence cannot support exact reconstruction."""


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LifecycleOpportunityError(f"{name} must be an integer")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LifecycleOpportunityError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise LifecycleOpportunityError(f"{name} must be finite")
    return result


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise LifecycleOpportunityError(f"{name} must be a nonempty string")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise LifecycleOpportunityError(f"{name} must be boolean")
    return value


def _canonical_line(row: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            row,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _project_sibling_sources(
    lifecycle_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[dict[str, object]]], str, str]:
    """Project only Q-permitted sibling fields from the full lifecycle ledger."""
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    run_ids: set[str] = set()
    clock_ids: set[str] = set()
    sequences: set[int] = set()
    for expected_sequence, row in enumerate(lifecycle_rows):
        if _integer(row.get("schema_version"), name="schema_version") != 4:
            raise LifecycleOpportunityError(
                "lifecycle reconstruction requires schema v4"
            )
        sequence = _integer(row.get("sequence"), name="sequence")
        if sequence != expected_sequence:
            raise LifecycleOpportunityError("lifecycle sequence must be contiguous")
        controller_sequence = _integer(
            row.get("controller_sequence"), name="controller_sequence"
        )
        if controller_sequence in sequences:
            raise LifecycleOpportunityError("duplicate controller_sequence")
        sequences.add(controller_sequence)
        run_ids.add(_string(row.get("run_id"), name="run_id"))
        clock_ids.add(_string(row.get("clock_domain_id"), name="clock_domain_id"))
        if row.get("stage") != "sibling_done":
            continue
        group_id = _string(row.get("group_id"), name="group_id")
        # This projection is the enforced arm-blind boundary. No release,
        # readiness, removal, learner, or terminal-disposition field survives it.
        groups[group_id].append(
            {
                "group_id": group_id,
                "controller_sequence": controller_sequence,
                "start_weight_version": _integer(
                    row.get("start_weight_version"), name="start_weight_version"
                ),
                "trajectory_id": _string(
                    row.get("trajectory_id"), name="trajectory_id"
                ),
                "sibling_idx": _integer(row.get("sibling_idx"), name="sibling_idx"),
                "reward": _number(row.get("reward"), name="reward"),
                "assistant_tokens": _integer(
                    row.get("assistant_tokens"), name="assistant_tokens"
                ),
                "truncated": _boolean(row.get("truncated"), name="truncated"),
            }
        )
    if len(run_ids) != 1 or len(clock_ids) != 1:
        raise LifecycleOpportunityError(
            "lifecycle ledger must contain one run and clock domain"
        )
    if not groups:
        raise LifecycleOpportunityError("lifecycle ledger has no sibling_done events")
    return groups, next(iter(run_ids)), next(iter(clock_ids))


def _derive_group(
    group_id: str,
    sources: Sequence[Mapping[str, object]],
    *,
    estimator: GRPOAdvantageEstimatorProtocol,
    siblings_per_group: int,
) -> tuple[GroupOpportunitySummary, int]:
    if len(sources) != siblings_per_group:
        raise LifecycleOpportunityError(
            f"{group_id}: expected {siblings_per_group} siblings, got {len(sources)}"
        )
    by_index: dict[int, Mapping[str, object]] = {}
    for source in sources:
        index = _integer(source.get("sibling_idx"), name="sibling_idx")
        if index in by_index or not 0 <= index < siblings_per_group:
            raise LifecycleOpportunityError(f"{group_id}: invalid sibling permutation")
        by_index[index] = source
    if set(by_index) != set(range(siblings_per_group)):
        raise LifecycleOpportunityError(f"{group_id}: incomplete sibling permutation")
    ordered = tuple(by_index[index] for index in range(siblings_per_group))
    start_versions = {
        _integer(source.get("start_weight_version"), name="start_weight_version")
        for source in ordered
    }
    if len(start_versions) != 1:
        raise LifecycleOpportunityError(f"{group_id}: mixed sibling start versions")
    start_weight_version = next(iter(start_versions))

    sample_ids = tuple(
        _string(source.get("trajectory_id"), name="trajectory_id") for source in ordered
    )
    expected_ids = tuple(f"{group_id}_g{index}" for index in range(siblings_per_group))
    if sample_ids != expected_ids or len(set(sample_ids)) != siblings_per_group:
        raise LifecycleOpportunityError(f"{group_id}: sibling sample identity mismatch")
    rewards = torch.tensor(
        [_number(source.get("reward"), name="reward") for source in ordered],
        dtype=torch.float32,
    )
    if not bool(((rewards == 0) | (rewards == 1)).all().item()):
        raise LifecycleOpportunityError(f"{group_id}: reward outside binary support")
    token_counts = tuple(
        _integer(source.get("assistant_tokens"), name="assistant_tokens")
        for source in ordered
    )
    if any(count < 0 for count in token_counts):
        raise LifecycleOpportunityError(f"{group_id}: negative assistant token count")
    actor_mask = torch.zeros(
        (siblings_per_group, max(token_counts, default=0) + 1), dtype=torch.float32
    )
    for sibling_index, count in enumerate(token_counts):
        actor_mask[sibling_index, 1 : count + 1] = 1
    inputs = GRPOOpportunityInputs(
        prompt_ids=torch.zeros(siblings_per_group, dtype=torch.long),
        rewards=rewards,
        actor_mask=actor_mask,
        repeated_batch={"total_reward": rewards},
        estimator_kwargs={},
    )
    summary = compute_grpo_gradient_opportunity(
        {},
        group_id=group_id,
        sample_ids=sample_ids,
        start_weight_version=start_weight_version,
        truncation=tuple(
            _boolean(source.get("truncated"), name="truncated") for source in ordered
        ),
        estimator=estimator,
        prepare_inputs=lambda _: inputs,
    )
    source_max = max(
        _integer(source.get("controller_sequence"), name="controller_sequence")
        for source in ordered
    )
    return summary, source_max


def derive_lifecycle_opportunity_rows(
    lifecycle_rows: Sequence[Mapping[str, Any]],
    *,
    estimator: GRPOAdvantageEstimatorProtocol,
    loss_settings: Mapping[str, object],
    siblings_per_group: int,
    train_batch_size: int,
) -> tuple[dict[str, Any], ...]:
    """Derive a schema-v2 opportunity ledger after the active run window."""
    if siblings_per_group < 2 or train_batch_size <= 0:
        raise LifecycleOpportunityError("invalid group or train-batch size")
    if (
        type(estimator).__name__ != "GRPOAdvantageEstimator"
        or getattr(estimator, "normalize_rewards", None) is not True
        or getattr(estimator, "use_leave_one_out_baseline", None) is not True
    ):
        raise LifecycleOpportunityError(
            "estimator does not match normalized leave-one-out GRPO"
        )
    required_loss_keys = {
        "disable_ppo_ratio",
        "positive_example_nll_weight",
        "sequence_level_importance_ratios",
        "token_level_loss",
        "use_cispo",
    }
    if set(loss_settings) != required_loss_keys:
        raise LifecycleOpportunityError(
            "loss settings do not match the frozen contract"
        )

    sibling_groups, run_id, clock_domain_id = _project_sibling_sources(lifecycle_rows)
    base = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "clock_domain_id": clock_domain_id,
        "derivation_method": DERIVATION_METHOD,
    }
    rows: list[dict[str, Any]] = [
        {
            **base,
            "event_type": "header",
            "estimator_name": type(estimator).__name__,
            "estimator_settings": dict(ESTIMATOR_SETTINGS),
            "loss_settings": dict(loss_settings),
            "source_lifecycle_schema_version": 4,
        }
    ]
    derived_groups: list[tuple[int, GroupOpportunitySummary]] = []
    opportunity_sample_ids: set[str] = set()
    for group_id, sources in sibling_groups.items():
        if len(sources) < siblings_per_group:
            # Bounded shutdown may cancel an out-of-window rollout mid-generation.
            # Such a group has no computable Q. The strict join rejects missing Q
            # in every registered opportunity-required window.
            continue
        summary, source_max = _derive_group(
            group_id,
            sources,
            estimator=estimator,
            siblings_per_group=siblings_per_group,
        )
        duplicate = opportunity_sample_ids.intersection(summary.sample_ids)
        if duplicate:
            raise LifecycleOpportunityError(
                "sample identity appears in multiple groups"
            )
        opportunity_sample_ids.update(summary.sample_ids)
        derived_groups.append((source_max, summary))
    for source_max, summary in sorted(derived_groups, key=lambda item: item[0]):
        rows.append(
            {
                **base,
                "event_type": "group",
                "source_controller_sequence_max": source_max,
                **summary.to_dict(),
            }
        )

    learner_events: dict[int, Mapping[str, Any]] = {}
    selected_by_previous: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in lifecycle_rows:
        if row.get("stage") == "learner_version_advanced":
            learner_version = _integer(
                row.get("learner_weight_version"), name="learner_weight_version"
            )
            if learner_version in learner_events:
                raise LifecycleOpportunityError("duplicate learner version transition")
            learner_events[learner_version] = row
        elif row.get("stage") == "removed" and row.get("removal_reason") == "selected":
            previous = _integer(
                row.get("learner_weight_version"), name="learner_weight_version"
            )
            selected_by_previous[previous].append(row)

    dangling = set(selected_by_previous).difference(
        learner_version - 1 for learner_version in learner_events
    )
    if dangling:
        raise LifecycleOpportunityError("selected groups lack a learner transition")
    completed_samples: set[str] = set()
    for learner_version, learner_event in sorted(learner_events.items()):
        previous = learner_version - 1
        selected = selected_by_previous.get(previous, [])
        sample_ids: list[str] = []
        source_sequences: list[int] = []
        for removal in sorted(
            selected,
            key=lambda row: _integer(
                row.get("controller_sequence"), name="controller_sequence"
            ),
        ):
            raw_ids = removal.get("sample_ids")
            if not isinstance(raw_ids, list):
                raise LifecycleOpportunityError("selected removal lacks sample IDs")
            sample_ids.extend(_string(value, name="sample_id") for value in raw_ids)
            source_sequences.append(
                _integer(removal.get("controller_sequence"), name="controller_sequence")
            )
        learner_sequence = _integer(
            learner_event.get("controller_sequence"), name="controller_sequence"
        )
        if (
            len(sample_ids) != train_batch_size
            or len(set(sample_ids)) != train_batch_size
            or not source_sequences
            or max(source_sequences) >= learner_sequence
        ):
            raise LifecycleOpportunityError(
                f"learner version {learner_version}: invalid selected-step identity"
            )
        if not set(sample_ids).issubset(opportunity_sample_ids):
            raise LifecycleOpportunityError("selected sample lacks derived opportunity")
        if completed_samples.intersection(sample_ids):
            raise LifecycleOpportunityError("sample appears in multiple learner steps")
        completed_samples.update(sample_ids)
        rows.append(
            {
                **base,
                "event_type": "train_step_completed",
                "source_controller_sequence_max": learner_sequence,
                "previous_learner_version": previous,
                "learner_version": learner_version,
                "sample_ids": sample_ids,
            }
        )
    for row in rows:
        _canonical_line(row)
    return tuple(rows)


def flush_lifecycle_opportunity_rows(
    rows: Sequence[Mapping[str, Any]], output_path: str | Path
) -> None:
    """Atomically write a canonical lifecycle-derived opportunity ledger."""
    if not rows or rows[0].get("event_type") != "header":
        raise LifecycleOpportunityError("derived ledger must begin with a header")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            for row in rows:
                output.write(_canonical_line(row))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def flush_lifecycle_derivation_summary(
    *,
    output_path: str | Path,
    elapsed_ns: int,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Atomically report post-window reconstruction cost separately from duty."""
    if elapsed_ns < 0 or not rows:
        raise LifecycleOpportunityError("invalid derivation summary")
    summary = {
        "schema_version": 1,
        "measurement_scope": "post_active_window_lifecycle_opportunity_derivation",
        "derivation_method": DERIVATION_METHOD,
        "elapsed_ns": elapsed_ns,
        "group_count": sum(row.get("event_type") == "group" for row in rows),
        "completed_step_count": sum(
            row.get("event_type") == "train_step_completed" for row in rows
        ),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(_canonical_line(summary))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
