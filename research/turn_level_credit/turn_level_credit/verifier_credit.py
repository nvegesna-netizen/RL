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

"""Pure verifier score-to-credit transforms for multi-turn trajectories."""

import math
from dataclasses import dataclass
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, model_validator


class VerifierCreditTransformConfig(BaseModel):
    """Configuration for converting bounded verifier scores into turn credit.

    The retrospective and hindsight definitions follow the low-cost components
    of TCPO. The unit weights are neutral research defaults, not claimed paper
    hyperparameters; experiments must record any overrides explicitly.

    Attributes:
        mode: Score-to-credit transform to apply.
        success_threshold: Score at which a previous state counts as successful.
        progress_weight: Multiplier for improvement over the best prior score.
        preservation_weight: Credit for preserving a prior successful best score.
        regression_weight: Multiplier for score loss after prior success.
        hindsight_weight: Multiplier for eligible leave-one-out hindsight credit.
        normalization: Optional post-transform normalization within prompt-turn groups.
        normalization_epsilon: Variance stabilizer for group z-score normalization.
        early_turn_discount: Multiplicative early-turn prior, raised to turn index.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    mode: Literal[
        "raw",
        "adjacent_delta",
        "retrospective",
        "retrospective_hindsight",
    ] = "raw"
    success_threshold: float = 1.0
    progress_weight: float = 1.0
    preservation_weight: float = 1.0
    regression_weight: float = 1.0
    hindsight_weight: float = 1.0
    normalization: Literal["none", "group_zscore"] = "none"
    normalization_epsilon: float = 1.0e-6
    early_turn_discount: float = 1.0

    @model_validator(mode="after")
    def _validate_ranges(self) -> "VerifierCreditTransformConfig":
        if not 0.0 < self.success_threshold <= 1.0:
            raise ValueError("success_threshold must be in (0, 1]")
        for field_name in (
            "progress_weight",
            "preservation_weight",
            "regression_weight",
            "hindsight_weight",
        ):
            if getattr(self, field_name) < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.normalization_epsilon <= 0.0:
            raise ValueError("normalization_epsilon must be positive")
        if not 0.0 < self.early_turn_discount <= 1.0:
            raise ValueError("early_turn_discount must be in (0, 1]")
        return self


@dataclass(frozen=True)
class VerifierScoreBatch:
    """Padded verifier scores and prompt groups for completed trajectories."""

    scores: torch.Tensor
    mask: torch.Tensor
    prompt_group_ids: torch.Tensor


@dataclass(frozen=True)
class RetrospectiveCredit:
    """Retrospective credit and its mutually exclusive components."""

    credit: torch.Tensor
    best_previous_score: torch.Tensor
    improvement: torch.Tensor
    preservation_mask: torch.Tensor
    regression: torch.Tensor


@dataclass(frozen=True)
class HindsightCredit:
    """Leave-one-out hindsight credit and its reference diagnostics."""

    credit: torch.Tensor
    future_best_score: torch.Tensor
    eligibility_mask: torch.Tensor
    reference_count: torch.Tensor
    leave_one_out_baseline: torch.Tensor


def _validate_score_batch(batch: VerifierScoreBatch) -> None:
    """Validate score, mask, and group invariants before a transform runs."""
    scores = batch.scores
    mask = batch.mask
    prompt_group_ids = batch.prompt_group_ids
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
        raise ValueError("verifier scores must have non-empty shape [batch, turns]")
    if not scores.is_floating_point():
        raise TypeError("verifier scores must use a floating-point dtype")
    if mask.shape != scores.shape or mask.dtype != torch.bool:
        raise ValueError("verifier mask must be boolean with the score tensor shape")
    if prompt_group_ids.shape != (scores.shape[0],):
        raise ValueError("prompt_group_ids must have shape [batch]")
    if prompt_group_ids.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("prompt_group_ids must use an integer dtype")
    if scores.device != mask.device or scores.device != prompt_group_ids.device:
        raise ValueError("scores, mask, and prompt_group_ids must share one device")
    if not bool(mask.any(dim=1).all().item()):
        raise ValueError("every verifier trajectory must contain an observed turn")
    if scores.shape[1] > 1 and bool(((~mask[:, :-1]) & mask[:, 1:]).any().item()):
        raise ValueError("verifier masks must be prefix-contiguous")
    observed_scores = scores[mask]
    if not bool(torch.isfinite(observed_scores).all().item()):
        raise ValueError("observed verifier scores must be finite")
    if bool(((observed_scores < 0.0) | (observed_scores > 1.0)).any().item()):
        raise ValueError("observed verifier scores must be in [0, 1]")


def _validate_credit_tensor(
    credit: torch.Tensor,
    batch: VerifierScoreBatch,
) -> None:
    """Validate a padded credit tensor against an already described batch."""
    _validate_score_batch(batch)
    if credit.shape != batch.scores.shape:
        raise ValueError("credit must have the verifier score tensor shape")
    if not credit.is_floating_point():
        raise TypeError("credit must use a floating-point dtype")
    if credit.device != batch.scores.device:
        raise ValueError("credit and verifier scores must share one device")
    if not bool(torch.isfinite(credit[batch.mask]).all().item()):
        raise ValueError("observed credit must be finite")


def _best_previous_scores(batch: VerifierScoreBatch) -> torch.Tensor:
    """Return the best observed score strictly before each turn."""
    masked_scores = batch.scores.masked_fill(~batch.mask, float("-inf"))
    prefix_best = masked_scores.cummax(dim=1).values
    best_previous = torch.zeros_like(batch.scores)
    best_previous[:, 1:] = prefix_best[:, :-1]
    return best_previous.masked_fill(~batch.mask, 0.0)


def raw_verifier_credit(batch: VerifierScoreBatch) -> torch.Tensor:
    """Use the current verifier score as the current turn's credit."""
    _validate_score_batch(batch)
    return batch.scores.masked_fill(~batch.mask, 0.0)


def adjacent_score_delta_credit(batch: VerifierScoreBatch) -> torch.Tensor:
    """Use current minus previous verifier score, with a zero initial score."""
    _validate_score_batch(batch)
    previous_scores = torch.zeros_like(batch.scores)
    previous_scores[:, 1:] = batch.scores[:, :-1]
    return (batch.scores - previous_scores).masked_fill(~batch.mask, 0.0)


def retrospective_verifier_credit(
    batch: VerifierScoreBatch,
    *,
    config: VerifierCreditTransformConfig,
) -> RetrospectiveCredit:
    """Compute best-prior progress, success preservation, and regression credit."""
    _validate_score_batch(batch)
    best_previous = _best_previous_scores(batch)
    improvement = (batch.scores - best_previous).clamp_min(0.0)
    improvement = improvement.masked_fill(~batch.mask, 0.0)
    prior_success = (best_previous >= config.success_threshold) & batch.mask
    preservation_mask = (
        prior_success & (improvement == 0.0) & (batch.scores >= best_previous)
    )
    regression = (best_previous - batch.scores).clamp_min(0.0)
    regression = (regression * prior_success.to(regression.dtype)).masked_fill(
        ~batch.mask, 0.0
    )
    credit = (
        config.progress_weight * improvement
        + config.preservation_weight * preservation_mask.to(batch.scores.dtype)
        - config.regression_weight * regression
    ).masked_fill(~batch.mask, 0.0)
    return RetrospectiveCredit(
        credit=credit,
        best_previous_score=best_previous,
        improvement=improvement,
        preservation_mask=preservation_mask,
        regression=regression,
    )


def _future_best_scores(batch: VerifierScoreBatch) -> torch.Tensor:
    """Return the best observed score at or after each turn."""
    masked_scores = batch.scores.masked_fill(~batch.mask, float("-inf"))
    future_best = torch.flip(
        torch.flip(masked_scores, dims=(1,)).cummax(dim=1).values,
        dims=(1,),
    )
    return future_best.masked_fill(~batch.mask, 0.0)


def hindsight_leave_one_out_credit(
    batch: VerifierScoreBatch,
    *,
    success_threshold: float,
) -> HindsightCredit:
    """Compare eligible future-best scores within each prompt and turn.

    A turn is eligible only when it is pre-success and does not improve over
    the best prior score. The reference excludes the current trajectory and
    includes only eligible trajectories from the same prompt and turn. Credit
    falls back to zero when no such peer exists.
    """
    _validate_score_batch(batch)
    if not 0.0 < success_threshold <= 1.0:
        raise ValueError("success_threshold must be in (0, 1]")
    best_previous = _best_previous_scores(batch)
    improvement = (batch.scores - best_previous).clamp_min(0.0)
    eligibility = (
        batch.mask & (best_previous < success_threshold) & (improvement == 0.0)
    )
    future_best = _future_best_scores(batch)

    _, inverse_group_ids = torch.unique(
        batch.prompt_group_ids,
        sorted=True,
        return_inverse=True,
    )
    group_count = int(inverse_group_ids.max().item()) + 1
    credit = torch.zeros_like(batch.scores)
    baseline = torch.zeros_like(batch.scores)
    reference_count = torch.zeros_like(batch.mask, dtype=torch.int64)
    for turn_index in range(batch.scores.shape[1]):
        eligible = eligibility[:, turn_index]
        eligible_values = future_best[:, turn_index] * eligible.to(batch.scores.dtype)

        group_sums = torch.zeros(
            group_count,
            dtype=batch.scores.dtype,
            device=batch.scores.device,
        )
        group_sums.scatter_add_(0, inverse_group_ids, eligible_values)
        group_counts = torch.zeros(
            group_count,
            dtype=torch.int64,
            device=batch.scores.device,
        )
        group_counts.scatter_add_(
            0,
            inverse_group_ids,
            eligible.to(torch.int64),
        )

        peer_counts = group_counts[inverse_group_ids] - eligible.to(torch.int64)
        peer_sums = group_sums[inverse_group_ids] - eligible_values
        has_reference = eligible & (peer_counts > 0)
        turn_baseline = torch.where(
            has_reference,
            peer_sums / peer_counts.clamp_min(1).to(batch.scores.dtype),
            torch.zeros_like(peer_sums),
        )
        baseline[:, turn_index] = turn_baseline
        reference_count[:, turn_index] = torch.where(
            eligible,
            peer_counts,
            torch.zeros_like(peer_counts),
        )
        credit[:, turn_index] = torch.where(
            has_reference,
            future_best[:, turn_index] - turn_baseline,
            torch.zeros_like(turn_baseline),
        )

    return HindsightCredit(
        credit=credit.masked_fill(~batch.mask, 0.0),
        future_best_score=future_best,
        eligibility_mask=eligibility,
        reference_count=reference_count,
        leave_one_out_baseline=baseline,
    )


def normalize_credit_within_prompt_turn(
    credit: torch.Tensor,
    batch: VerifierScoreBatch,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Z-score observed credit within each prompt and turn index.

    Groups with fewer than two observed trajectories retain their raw credit.
    A constant group with at least two observations maps to zero. Padded values
    are ignored and replaced with zero.
    """
    _validate_credit_tensor(credit, batch)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")

    _, inverse_group_ids = torch.unique(
        batch.prompt_group_ids,
        sorted=True,
        return_inverse=True,
    )
    group_count = int(inverse_group_ids.max().item()) + 1
    working_dtype = torch.float64 if credit.dtype == torch.float64 else torch.float32
    observed_credit = credit.masked_fill(~batch.mask, 0.0).to(working_dtype)
    normalized = torch.zeros_like(observed_credit)

    for turn_index in range(batch.scores.shape[1]):
        observed = batch.mask[:, turn_index]
        values = observed_credit[:, turn_index]
        observed_values = values * observed.to(working_dtype)

        group_sums = torch.zeros(
            group_count,
            dtype=working_dtype,
            device=credit.device,
        )
        group_sums.scatter_add_(0, inverse_group_ids, observed_values)
        group_counts = torch.zeros(
            group_count,
            dtype=torch.int64,
            device=credit.device,
        )
        group_counts.scatter_add_(0, inverse_group_ids, observed.to(torch.int64))

        counts = group_counts[inverse_group_ids]
        safe_counts = counts.clamp_min(1).to(working_dtype)
        means = group_sums[inverse_group_ids] / safe_counts
        centered = (values - means) * observed.to(working_dtype)
        group_centered_square_sums = torch.zeros_like(group_sums)
        group_centered_square_sums.scatter_add_(
            0,
            inverse_group_ids,
            centered.square(),
        )
        variances = group_centered_square_sums[inverse_group_ids] / safe_counts
        zscores = centered / torch.sqrt(variances + epsilon)
        normalizable = observed & (counts >= 2)
        normalized[:, turn_index] = torch.where(
            normalizable,
            zscores,
            torch.where(observed, values, torch.zeros_like(values)),
        )

    return normalized.to(credit.dtype)


def postprocess_verifier_credit(
    credit: torch.Tensor,
    batch: VerifierScoreBatch,
    *,
    config: VerifierCreditTransformConfig,
) -> torch.Tensor:
    """Apply configured prompt-turn normalization and early-turn weighting."""
    if config.normalization == "group_zscore":
        credit = normalize_credit_within_prompt_turn(
            credit,
            batch,
            epsilon=config.normalization_epsilon,
        )
    else:
        _validate_credit_tensor(credit, batch)
        credit = credit.masked_fill(~batch.mask, 0.0)

    turn_indices = torch.arange(
        batch.scores.shape[1],
        dtype=credit.dtype,
        device=credit.device,
    )
    early_turn_weights = config.early_turn_discount**turn_indices
    return (credit * early_turn_weights.unsqueeze(0)).masked_fill(~batch.mask, 0.0)


def compute_verifier_credit(
    batch: VerifierScoreBatch,
    *,
    config: VerifierCreditTransformConfig,
) -> torch.Tensor:
    """Dispatch the configured pure score-to-credit transform."""
    if config.mode == "raw":
        credit = raw_verifier_credit(batch)
    elif config.mode == "adjacent_delta":
        credit = adjacent_score_delta_credit(batch)
    elif config.mode == "retrospective":
        credit = retrospective_verifier_credit(batch, config=config).credit
    elif config.mode == "retrospective_hindsight":
        retrospective = retrospective_verifier_credit(batch, config=config)
        hindsight = hindsight_leave_one_out_credit(
            batch,
            success_threshold=config.success_threshold,
        )
        credit = retrospective.credit + config.hindsight_weight * hindsight.credit
    else:
        raise AssertionError(f"Unhandled verifier credit mode: {config.mode}")
    return postprocess_verifier_credit(credit, batch, config=config)
