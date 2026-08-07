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

"""Mechanical signal gates for frozen fixed-horizon math-repair rollouts."""

from dataclasses import dataclass
from typing import Any

import torch

from turn_level_credit.verifier_credit import (
    VerifierCreditTransformConfig,
    VerifierScoreBatch,
    compute_verifier_credit,
    hindsight_leave_one_out_credit,
)


@dataclass(frozen=True)
class MathRepairCalibrationThresholds:
    """Predeclared minimum evidence required before an optimizer experiment."""

    expected_turns: int = 3
    minimum_prompts: int = 32
    minimum_samples: int = 256
    minimum_generations_per_prompt: int = 4
    minimum_first_turn_success: float = 0.02
    maximum_first_turn_success: float = 0.80
    minimum_any_turn_success: float = 0.05
    maximum_any_turn_success: float = 0.95
    minimum_repair_fraction: float = 0.01
    minimum_nonzero_credit_fraction: float = 0.005
    minimum_credit_std: float = 1.0e-3


def _score_from_feedback(content: object, *, prompt: int, generation: int) -> float:
    """Read the deliberately explicit verifier feedback protocol."""
    if not isinstance(content, str):
        raise TypeError(
            f"prompt {prompt} generation {generation} has non-text feedback"
        )
    if content.startswith("Verifier: correct."):
        return 1.0
    if content.startswith("Verifier: incorrect."):
        return 0.0
    raise ValueError(
        f"prompt {prompt} generation {generation} has unknown verifier feedback"
    )


def _initial_prompt(conversation: list[object], *, record_number: int) -> str:
    """Return the logged user prompt used to recover rollout groups."""
    if not conversation or not isinstance(conversation[0], dict):
        raise ValueError(
            f"validation record {record_number} has no initial prompt message"
        )
    first_message = conversation[0]
    content = first_message.get("content")
    if first_message.get("role") != "user" or not isinstance(content, str):
        raise ValueError(
            f"validation record {record_number} must start with a text user prompt"
        )
    return content


def parse_math_repair_validation_records(
    records: list[dict[str, Any]],
    *,
    generations_per_prompt: int,
) -> tuple[VerifierScoreBatch, torch.Tensor, dict[int, int]]:
    """Convert NeMo-RL validation records into score traces and prompt groups."""
    if not records:
        raise ValueError("math-repair calibration contains no validation records")
    if generations_per_prompt < 1:
        raise ValueError("generations_per_prompt must be positive")
    if len(records) % generations_per_prompt != 0:
        raise ValueError(
            "validation record count is not divisible by generations_per_prompt"
        )
    score_rows: list[list[float]] = []
    terminal_success: list[float] = []
    group_ids: list[int] = []
    group_sizes: dict[int, int] = {}
    prompt_to_group: dict[str, int] = {}

    for record_number, record in enumerate(records):
        logged_index = record.get("idx")
        if logged_index != record_number:
            raise ValueError("validation record indices must be contiguous and ordered")
        contents = record.get("content")
        rewards = record.get("rewards")
        if not isinstance(contents, list) or not isinstance(rewards, list):
            raise TypeError(
                f"validation record {record_number} requires content and reward lists"
            )
        if len(contents) != 1 or len(rewards) != 1:
            raise ValueError(
                f"validation record {record_number} must contain one logged sample"
            )
        for conversation, raw_reward in zip(contents, rewards, strict=True):
            if not isinstance(conversation, list):
                raise TypeError(
                    f"validation record {record_number} is not a conversation"
                )
            prompt_text = _initial_prompt(
                conversation,
                record_number=record_number,
            )
            group_id = prompt_to_group.setdefault(prompt_text, len(prompt_to_group))
            generation_index = group_sizes.get(group_id, 0)
            feedback_scores = [
                _score_from_feedback(
                    message.get("content"),
                    prompt=group_id,
                    generation=generation_index,
                )
                for message in conversation
                if isinstance(message, dict) and message.get("role") == "environment"
            ]
            if not feedback_scores:
                raise ValueError(
                    f"prompt {group_id} generation {generation_index} has no verifier turns"
                )
            if isinstance(raw_reward, bool) or not isinstance(raw_reward, (int, float)):
                raise TypeError(
                    f"prompt {group_id} generation {generation_index} has invalid reward"
                )
            outcome = float(raw_reward)
            if outcome not in (0.0, 1.0):
                raise ValueError("math-repair terminal rewards must be binary")
            expected_outcome = float(any(score == 1.0 for score in feedback_scores))
            if outcome != expected_outcome:
                raise ValueError(
                    f"prompt {group_id} generation {generation_index} terminal reward "
                    "does not match its verifier trace"
                )
            score_rows.append(feedback_scores)
            terminal_success.append(outcome)
            group_ids.append(group_id)
            group_sizes[group_id] = group_sizes.get(group_id, 0) + 1

    if any(size != generations_per_prompt for size in group_sizes.values()):
        raise ValueError("logged prompt groups do not match generations_per_prompt")
    if group_ids != sorted(group_ids):
        raise ValueError("validation rollouts for each prompt must be contiguous")

    turn_counts = {len(scores) for scores in score_rows}
    if len(turn_counts) != 1:
        raise ValueError("math-repair validation traces have inconsistent horizons")
    scores = torch.tensor(score_rows, dtype=torch.float32)
    return (
        VerifierScoreBatch(
            scores=scores,
            mask=torch.ones_like(scores, dtype=torch.bool),
            prompt_group_ids=torch.tensor(group_ids, dtype=torch.int64),
        ),
        torch.tensor(terminal_success, dtype=torch.float32),
        group_sizes,
    )


def evaluate_math_repair_calibration(
    records: list[dict[str, Any]],
    *,
    generations_per_prompt: int = 8,
    thresholds: MathRepairCalibrationThresholds = MathRepairCalibrationThresholds(),
) -> dict[str, Any]:
    """Evaluate frozen outcome, repair, grouping, and estimator signal gates."""
    batch, terminal_success, group_sizes = parse_math_repair_validation_records(
        records,
        generations_per_prompt=generations_per_prompt,
    )
    if thresholds.expected_turns < 2:
        raise ValueError("expected_turns must be at least two")
    first_success = batch.scores[:, 0] == 1.0
    later_success = (batch.scores[:, 1:] == 1.0).any(dim=1)
    repaired = (~first_success) & later_success
    adjacent_pairs = batch.scores[:, :-1], batch.scores[:, 1:]
    preserved_count = int(
        ((adjacent_pairs[0] == 1.0) & (adjacent_pairs[1] == 1.0)).sum().item()
    )
    regression_count = int(
        ((adjacent_pairs[0] == 1.0) & (adjacent_pairs[1] == 0.0)).sum().item()
    )

    metrics: dict[str, float | int] = {
        "prompt_count": len(group_sizes),
        "sample_count": batch.scores.shape[0],
        "observed_turns": batch.scores.shape[1],
        "minimum_generations_per_prompt": min(group_sizes.values()),
        "maximum_generations_per_prompt": max(group_sizes.values()),
        "first_turn_success_fraction": float(first_success.float().mean().item()),
        "any_turn_success_fraction": float(terminal_success.mean().item()),
        "repair_after_initial_failure_fraction": float(repaired.float().mean().item()),
        "preserved_success_transition_count": preserved_count,
        "regression_after_success_transition_count": regression_count,
    }
    credit_modes = (
        "raw",
        "adjacent_delta",
        "retrospective",
        "retrospective_hindsight",
    )
    for mode in credit_modes:
        credit = compute_verifier_credit(
            batch,
            config=VerifierCreditTransformConfig(mode=mode),
        )
        observed = credit[batch.mask]
        metrics[f"{mode}_credit_std"] = float(observed.std(unbiased=False).item())
        metrics[f"{mode}_nonzero_credit_fraction"] = float(
            (observed != 0.0).float().mean().item()
        )

    hindsight = hindsight_leave_one_out_credit(batch, success_threshold=1.0)
    observed_hindsight = hindsight.credit[batch.mask]
    metrics["hindsight_reference_fraction"] = float(
        (hindsight.reference_count[batch.mask] > 0).float().mean().item()
    )
    metrics["hindsight_nonzero_credit_fraction"] = float(
        (observed_hindsight != 0.0).float().mean().item()
    )

    checks = {
        "expected_fixed_horizon": batch.scores.shape[1] == thresholds.expected_turns,
        "minimum_prompt_count": len(group_sizes) >= thresholds.minimum_prompts,
        "minimum_sample_count": batch.scores.shape[0] >= thresholds.minimum_samples,
        "minimum_group_size": min(group_sizes.values())
        >= thresholds.minimum_generations_per_prompt,
        "uniform_group_size": len(set(group_sizes.values())) == 1,
        "first_turn_success_off_floor": metrics["first_turn_success_fraction"]
        >= thresholds.minimum_first_turn_success,
        "first_turn_success_off_ceiling": metrics["first_turn_success_fraction"]
        <= thresholds.maximum_first_turn_success,
        "any_turn_success_off_floor": metrics["any_turn_success_fraction"]
        >= thresholds.minimum_any_turn_success,
        "any_turn_success_off_ceiling": metrics["any_turn_success_fraction"]
        <= thresholds.maximum_any_turn_success,
        "repair_is_observed": metrics["repair_after_initial_failure_fraction"]
        >= thresholds.minimum_repair_fraction,
        "preservation_is_observed": preserved_count > 0,
        "regression_is_observed": regression_count > 0,
        "hindsight_has_peer_references": metrics["hindsight_reference_fraction"] > 0.0,
        "hindsight_is_nonzero": metrics["hindsight_nonzero_credit_fraction"]
        >= thresholds.minimum_nonzero_credit_fraction,
    }
    for mode in credit_modes:
        checks[f"{mode}_credit_is_non_degenerate"] = (
            metrics[f"{mode}_credit_std"] >= thresholds.minimum_credit_std
            and metrics[f"{mode}_nonzero_credit_fraction"]
            >= thresholds.minimum_nonzero_credit_fraction
        )
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    return {
        "decision": "pass" if not failed_checks else "fail",
        "metrics": metrics,
        "checks": checks,
        "failed_checks": failed_checks,
    }
