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

"""Exact pre-release GRPO coefficient-opportunity summaries and ledger."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import torch


class GRPOAdvantageEstimatorProtocol(Protocol):
    """Structural type for the configured production GRPO estimator."""

    def compute_advantage(
        self,
        prompt_ids: torch.Tensor,
        rewards: torch.Tensor,
        mask: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return loss-shaped GRPO advantages."""
        ...


class ControllerSequencer(Protocol):
    """Shared controller sequencer used by lifecycle and opportunity ledgers."""

    def __call__(self) -> tuple[int, int]:
        """Return ``(controller_sequence, monotonic_timestamp_ns)``."""
        ...


@dataclass(frozen=True)
class GRPOOpportunityInputs:
    """Inputs produced by the shared production advantage-preparation helper.

    All tensors must already be detached CPU tensors with the exact production
    casts applied. ``actor_mask`` is the production
    ``token_mask * sample_mask[:, None]`` tensor.
    """

    prompt_ids: torch.Tensor
    rewards: torch.Tensor
    actor_mask: torch.Tensor
    repeated_batch: Mapping[str, torch.Tensor]
    estimator_kwargs: Mapping[str, torch.Tensor]


OpportunityInputPreparation = Callable[[Mapping[str, Any]], GRPOOpportunityInputs]


@dataclass(frozen=True)
class SiblingOpportunitySummary:
    """Scalar-only opportunity summary for one sibling rollout."""

    sample_id: str
    sibling_index: int
    reward: float
    scalar_advantage: float
    valid_actor_tokens: int
    opportunity: float
    truncated: bool


@dataclass(frozen=True)
class GroupOpportunitySummary:
    """Scalar-only exact GRPO opportunity summary for one prompt group."""

    group_id: str
    sample_ids: tuple[str, ...]
    start_weight_version: int
    opportunity: float
    l2_coefficient_mass: float
    signed_coefficient_mass: float
    positive_coefficient_mass: float
    negative_coefficient_mass: float
    valid_actor_tokens: int
    nonzero_advantage_siblings: int
    nonzero_advantage_tokens: int
    siblings: tuple[SiblingOpportunitySummary, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary without raw training data."""
        result = asdict(self)
        result["sample_ids"] = list(self.sample_ids)
        result["siblings"] = [asdict(sibling) for sibling in self.siblings]
        return result


def _iter_named_tensors(
    inputs: GRPOOpportunityInputs,
) -> Sequence[tuple[str, torch.Tensor]]:
    tensors: list[tuple[str, torch.Tensor]] = [
        ("prompt_ids", inputs.prompt_ids),
        ("rewards", inputs.rewards),
        ("actor_mask", inputs.actor_mask),
    ]
    tensors.extend(
        (f"repeated_batch.{name}", tensor)
        for name, tensor in inputs.repeated_batch.items()
    )
    tensors.extend(
        (f"estimator_kwargs.{name}", tensor)
        for name, tensor in inputs.estimator_kwargs.items()
    )
    return tensors


def _validate_prepared_inputs(inputs: GRPOOpportunityInputs) -> None:
    """Validate the shared helper's detached-CPU and finite-value contract."""
    if inputs.rewards.ndim != 1:
        raise ValueError(f"rewards must have shape [batch], got {inputs.rewards.shape}")
    if inputs.actor_mask.ndim != 2:
        raise ValueError(
            f"actor_mask must have shape [batch, sequence], got {inputs.actor_mask.shape}"
        )
    batch_size = inputs.rewards.shape[0]
    if inputs.actor_mask.shape[0] != batch_size:
        raise ValueError("actor_mask and rewards batch dimensions must match")
    if inputs.prompt_ids.ndim < 1 or inputs.prompt_ids.shape[0] != batch_size:
        raise ValueError("prompt_ids and rewards batch dimensions must match")
    if inputs.actor_mask.shape[1] < 1:
        raise ValueError("actor_mask must retain the leading loss-alignment position")

    reserved_kwargs = {"prompt_ids", "rewards", "mask", "repeated_batch"}
    overlap = reserved_kwargs.intersection(inputs.estimator_kwargs)
    if overlap:
        raise ValueError(
            "estimator_kwargs contains reserved argument(s): "
            + ", ".join(sorted(overlap))
        )

    for name, tensor in _iter_named_tensors(inputs):
        if tensor.device.type != "cpu":
            raise ValueError(f"{name} must be prepared on CPU")
        if tensor.requires_grad:
            raise ValueError(f"{name} must be detached")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{name} contains a nonfinite value")

    mask_is_binary = (inputs.actor_mask == 0) | (inputs.actor_mask == 1)
    if not bool(mask_is_binary.all().item()):
        raise ValueError("actor_mask must contain only zero or one")


def compute_grpo_gradient_opportunity(
    train_batch: Mapping[str, Any],
    *,
    group_id: str,
    sample_ids: Sequence[str],
    start_weight_version: int,
    truncation: Sequence[bool],
    estimator: GRPOAdvantageEstimatorProtocol,
    prepare_inputs: OpportunityInputPreparation,
) -> GroupOpportunitySummary:
    """Compute exact scalar GRPO opportunity from the definitive train batch.

    The injected ``prepare_inputs`` callable is the shared production helper;
    this function deliberately does not duplicate production casting or mask
    construction.

    Args:
        train_batch: Definitive, already packed-source training batch.
        group_id: Authoritative prompt-group identifier.
        sample_ids: Authoritative sibling sample identifiers in batch order.
        start_weight_version: Learner version captured at reservation.
        truncation: Per-sibling truncation flags in batch order.
        estimator: Configured production GRPO advantage estimator object.
        prepare_inputs: Shared production advantage-input preparation callable.

    Returns:
        Immutable scalar-only group opportunity summary.

    Raises:
        ValueError: If identity, tensor, shape, finiteness, or GRPO scalar
            invariants are violated.
        TypeError: If the configured estimator returns a non-tensor value.
    """
    if not group_id:
        raise ValueError("group_id must be nonempty")
    if start_weight_version < 0:
        raise ValueError("start_weight_version must be nonnegative")
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_ids must be nonempty and unique")

    inputs = prepare_inputs(train_batch)
    _validate_prepared_inputs(inputs)
    batch_size = inputs.rewards.shape[0]
    if len(sample_ids) != batch_size or len(truncation) != batch_size:
        raise ValueError(
            "sample_ids and truncation must match the prepared input batch size"
        )

    advantages = estimator.compute_advantage(
        prompt_ids=inputs.prompt_ids,
        rewards=inputs.rewards,
        mask=inputs.actor_mask,
        repeated_batch=dict(inputs.repeated_batch),
        **dict(inputs.estimator_kwargs),
    )
    if not isinstance(advantages, torch.Tensor):
        raise TypeError("the production estimator must return a torch.Tensor")
    if advantages.device.type != "cpu":
        raise ValueError("production advantages must be computed on CPU")
    if advantages.requires_grad:
        raise ValueError("production advantages must be detached")
    if advantages.shape != inputs.actor_mask.shape:
        raise ValueError(
            "advantages and actor_mask must have identical shapes, got "
            f"{advantages.shape} and {inputs.actor_mask.shape}"
        )
    if not bool(torch.isfinite(advantages).all().item()):
        raise ValueError("production advantages contain a nonfinite value")

    scalar_advantages = advantages[:, 0]
    if not bool((advantages == scalar_advantages.unsqueeze(-1)).all().item()):
        raise ValueError("GRPO advantages must be scalar-constant within each sibling")

    aligned_advantages = advantages[:, 1:].to(dtype=torch.float64)
    aligned_mask = inputs.actor_mask[:, 1:].to(dtype=torch.float64)
    coefficients = aligned_advantages * aligned_mask
    absolute_coefficients = coefficients.abs()

    siblings: list[SiblingOpportunitySummary] = []
    for sibling_index, sample_id in enumerate(sample_ids):
        sibling_mask = aligned_mask[sibling_index]
        valid_actor_tokens_float = float(sibling_mask.sum(dtype=torch.float64).item())
        valid_actor_tokens = int(valid_actor_tokens_float)
        if valid_actor_tokens_float != valid_actor_tokens:
            raise ValueError("actor_mask valid-token count must be integral")
        siblings.append(
            SiblingOpportunitySummary(
                sample_id=sample_id,
                sibling_index=sibling_index,
                reward=float(inputs.rewards[sibling_index].item()),
                scalar_advantage=float(scalar_advantages[sibling_index].item()),
                valid_actor_tokens=valid_actor_tokens,
                opportunity=float(
                    absolute_coefficients[sibling_index].sum(dtype=torch.float64).item()
                ),
                truncated=bool(truncation[sibling_index]),
            )
        )

    positive_mass = coefficients.clamp_min(0).sum(dtype=torch.float64)
    negative_mass = (-coefficients.clamp_max(0)).sum(dtype=torch.float64)
    opportunity = absolute_coefficients.sum(dtype=torch.float64)
    squared_mass = coefficients.square().sum(dtype=torch.float64)
    nonzero_advantage = scalar_advantages != 0
    nonzero_token_mask = aligned_mask.bool() & nonzero_advantage.unsqueeze(-1)

    summary = GroupOpportunitySummary(
        group_id=group_id,
        sample_ids=tuple(sample_ids),
        start_weight_version=start_weight_version,
        opportunity=float(opportunity.item()),
        l2_coefficient_mass=math.sqrt(float(squared_mass.item())),
        signed_coefficient_mass=float(coefficients.sum(dtype=torch.float64).item()),
        positive_coefficient_mass=float(positive_mass.item()),
        negative_coefficient_mass=float(negative_mass.item()),
        valid_actor_tokens=int(aligned_mask.sum(dtype=torch.float64).item()),
        nonzero_advantage_siblings=int(nonzero_advantage.sum().item()),
        nonzero_advantage_tokens=int(nonzero_token_mask.sum().item()),
        siblings=tuple(siblings),
    )
    if not math.isclose(
        summary.opportunity,
        math.fsum(sibling.opportunity for sibling in summary.siblings),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("group and sibling opportunity sums disagree")
    return summary


def _canonical_json_line(event: Mapping[str, Any]) -> bytes:
    """Encode one event canonically and reject nonfinite JSON numbers."""
    return (
        json.dumps(
            event,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


class GradientOpportunityRecorder:
    """In-memory scalar opportunity and completed-train-step JSONL ledger."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        run_id: str,
        clock_domain_id: str,
        sequencer: ControllerSequencer,
    ) -> None:
        if not run_id or not clock_domain_id:
            raise ValueError("run_id and clock_domain_id must be nonempty")
        self.run_id = run_id
        self.clock_domain_id = clock_domain_id
        self._sequencer = sequencer
        self._events: list[dict[str, Any]] = []
        self._last_controller_sequence: int | None = None
        self._last_timestamp_ns: int | None = None
        self._header_written = False
        self._group_ids: set[str] = set()
        self._opportunity_sample_ids: set[str] = set()
        self._completed_sample_ids: set[str] = set()

    def _append(self, event_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        controller_sequence, timestamp_ns = self._sequencer()
        if (
            isinstance(controller_sequence, bool)
            or not isinstance(controller_sequence, int)
            or controller_sequence < 0
        ):
            raise ValueError("controller_sequence must be a nonnegative integer")
        if (
            isinstance(timestamp_ns, bool)
            or not isinstance(timestamp_ns, int)
            or timestamp_ns < 0
        ):
            raise ValueError("timestamp_ns must be a nonnegative integer")
        if (
            self._last_controller_sequence is not None
            and controller_sequence <= self._last_controller_sequence
        ):
            raise ValueError("controller_sequence must increase strictly")
        if (
            self._last_timestamp_ns is not None
            and timestamp_ns < self._last_timestamp_ns
        ):
            raise ValueError("timestamp_ns must be monotonic")

        event = {
            "schema_version": self.SCHEMA_VERSION,
            "event_type": event_type,
            "run_id": self.run_id,
            "clock_domain_id": self.clock_domain_id,
            "controller_sequence": controller_sequence,
            "timestamp_ns": timestamp_ns,
            **dict(payload),
        }
        _canonical_json_line(event)
        self._events.append(event)
        self._last_controller_sequence = controller_sequence
        self._last_timestamp_ns = timestamp_ns
        return event

    def append_header(
        self,
        *,
        estimator_name: str,
        estimator_settings: Mapping[str, Any],
        loss_settings: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Append the unique ledger header."""
        if self._events or self._header_written:
            raise ValueError("the ledger header must be the first and only header")
        if not estimator_name:
            raise ValueError("estimator_name must be nonempty")
        event = self._append(
            "header",
            {
                "estimator_name": estimator_name,
                "estimator_settings": dict(estimator_settings),
                "loss_settings": dict(loss_settings),
            },
        )
        self._header_written = True
        return event

    def append_group(self, summary: GroupOpportunitySummary) -> dict[str, Any]:
        """Append one unique scalar-only prompt-group opportunity event."""
        if not self._header_written:
            raise ValueError("append_header must precede group events")
        if summary.group_id in self._group_ids:
            raise ValueError(f"duplicate group_id={summary.group_id!r}")
        duplicate_sample_ids = self._opportunity_sample_ids.intersection(
            summary.sample_ids
        )
        if duplicate_sample_ids:
            raise ValueError(
                "duplicate opportunity sample ID(s): "
                + ", ".join(sorted(duplicate_sample_ids))
            )
        event = self._append("group", summary.to_dict())
        self._group_ids.add(summary.group_id)
        self._opportunity_sample_ids.update(summary.sample_ids)
        return event

    def append_train_step_completed(
        self,
        *,
        previous_learner_version: int,
        learner_version: int,
        sample_ids: Sequence[str],
    ) -> dict[str, Any]:
        """Append IDs belonging to one completed controller learner step."""
        if not self._header_written:
            raise ValueError("append_header must precede train-step events")
        if learner_version != previous_learner_version + 1:
            raise ValueError("learner version must advance by exactly one")
        ordered_sample_ids = tuple(sample_ids)
        if not ordered_sample_ids or len(ordered_sample_ids) != len(
            set(ordered_sample_ids)
        ):
            raise ValueError("completed-step sample_ids must be nonempty and unique")
        unknown = set(ordered_sample_ids).difference(self._opportunity_sample_ids)
        if unknown:
            raise ValueError(
                "completed-step sample ID(s) lack opportunity records: "
                + ", ".join(sorted(unknown))
            )
        duplicate = set(ordered_sample_ids).intersection(self._completed_sample_ids)
        if duplicate:
            raise ValueError(
                "sample ID(s) already belong to a completed step: "
                + ", ".join(sorted(duplicate))
            )
        event = self._append(
            "train_step_completed",
            {
                "previous_learner_version": previous_learner_version,
                "learner_version": learner_version,
                "sample_ids": list(ordered_sample_ids),
            },
        )
        self._completed_sample_ids.update(ordered_sample_ids)
        return event

    def snapshot(self) -> tuple[dict[str, Any], ...]:
        """Return a detached JSON round-trip copy of all events."""
        return tuple(json.loads(_canonical_json_line(event)) for event in self._events)

    def flush_jsonl(self, output_path: str | Path) -> None:
        """Atomically publish the current canonical ledger snapshot."""
        if not self._header_written:
            raise ValueError("cannot flush an opportunity ledger without a header")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "wb") as output:
                for event in self._events:
                    output.write(_canonical_json_line(event))
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
