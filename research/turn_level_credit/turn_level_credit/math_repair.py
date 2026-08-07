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

"""Fixed-horizon verifier feedback for research-only math repair rollouts."""

import math
from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict

import torch
from pydantic import BaseModel, ConfigDict, model_validator

from nemo_rl.data.interfaces import LLMMessageLogType

VERIFIER_SCORE_KEY = "reward/verifier_score"
TERMINAL_SUCCESS_KEY = "reward/terminal_success"


class MathRepairConfig(BaseModel):
    """Configuration for a fixed-budget sequence of verifier-scored answers."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    num_workers: int = 1
    max_turns: int = 3
    success_threshold: float = 1.0
    math_verify_impl: Literal["hf_math_verify", "dapo_math_verify"] = "hf_math_verify"

    @model_validator(mode="after")
    def _validate_ranges(self) -> "MathRepairConfig":
        if self.num_workers < 1:
            raise ValueError("math repair num_workers must be positive")
        if self.max_turns < 2:
            raise ValueError("math repair max_turns must be at least two")
        if not 0.0 < self.success_threshold <= 1.0:
            raise ValueError("math repair success_threshold must be in (0, 1]")
        return self


class MathRepairMetadata(TypedDict):
    """Per-trajectory verifier state carried between repair turns."""

    ground_truth: str
    turn_index: NotRequired[int]
    first_success_turn: NotRequired[int | None]
    best_score: NotRequired[float]


@dataclass(frozen=True)
class MathRepairTurnBatch:
    """Validated state transition for one batched verifier call."""

    observations: list[dict[str, str]]
    metadata: list[MathRepairMetadata]
    verifier_scores: torch.Tensor
    terminal_success: torch.Tensor
    terminateds: torch.Tensor


def _validate_metadata(
    metadata: MathRepairMetadata,
    *,
    config: MathRepairConfig,
    row: int,
) -> tuple[int, int | None, float]:
    """Validate and unpack one trajectory's pre-turn repair state."""
    ground_truth = metadata.get("ground_truth")
    if not isinstance(ground_truth, str) or not ground_truth.strip():
        raise ValueError(f"math repair row {row} requires a non-empty ground_truth")

    previous_turn = metadata.get("turn_index", 0)
    if isinstance(previous_turn, bool) or not isinstance(previous_turn, int):
        raise TypeError(f"math repair row {row} turn_index must be an integer")
    if not 0 <= previous_turn < config.max_turns:
        raise ValueError(
            f"math repair row {row} turn_index must be in [0, {config.max_turns})"
        )

    first_success_turn = metadata.get("first_success_turn")
    if first_success_turn is not None:
        if isinstance(first_success_turn, bool) or not isinstance(
            first_success_turn, int
        ):
            raise TypeError(
                f"math repair row {row} first_success_turn must be an integer or None"
            )
        if not 1 <= first_success_turn <= previous_turn:
            raise ValueError(
                f"math repair row {row} first_success_turn must describe a prior turn"
            )

    best_score = metadata.get("best_score", 0.0)
    if isinstance(best_score, bool) or not isinstance(best_score, (int, float)):
        raise TypeError(f"math repair row {row} best_score must be numeric")
    best_score = float(best_score)
    if not math.isfinite(best_score) or not 0.0 <= best_score <= 1.0:
        raise ValueError(f"math repair row {row} best_score must be in [0, 1]")
    if first_success_turn is not None and best_score < config.success_threshold:
        raise ValueError(
            f"math repair row {row} records success without a successful best_score"
        )
    if first_success_turn is None and best_score >= config.success_threshold:
        raise ValueError(
            f"math repair row {row} has a successful best_score without a "
            "first_success_turn"
        )
    return previous_turn, first_success_turn, best_score


def advance_math_repair_turns(
    scores: list[float],
    metadata: list[MathRepairMetadata],
    *,
    config: MathRepairConfig,
) -> MathRepairTurnBatch:
    """Advance fixed-horizon metadata from one verifier score per trajectory.

    Success never terminates a training trajectory early. The final turn emits
    the any-turn success outcome exactly once, while every turn emits its raw
    verifier score as a separate named component.
    """
    if len(scores) != len(metadata):
        raise ValueError("math repair scores and metadata batches must align")
    if not scores:
        raise ValueError("math repair cannot advance an empty batch")

    observations: list[dict[str, str]] = []
    updated_metadata: list[MathRepairMetadata] = []
    verifier_scores: list[float] = []
    terminal_success: list[float] = []
    terminateds: list[bool] = []

    for row, (raw_score, row_metadata) in enumerate(zip(scores, metadata, strict=True)):
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise TypeError(f"math repair row {row} verifier score must be numeric")
        score = float(raw_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(
                f"math repair row {row} verifier score must be finite and in [0, 1]"
            )
        previous_turn, first_success_turn, best_score = _validate_metadata(
            row_metadata,
            config=config,
            row=row,
        )
        current_turn = previous_turn + 1
        successful = score >= config.success_threshold
        if successful and first_success_turn is None:
            first_success_turn = current_turn
        best_score = max(best_score, score)
        terminated = current_turn == config.max_turns
        outcome = float(terminated and first_success_turn is not None)

        feedback = (
            "Verifier: correct. Re-check the solution and return the final answer "
            "again."
            if successful
            else "Verifier: incorrect. Recompute carefully and return a corrected "
            "final answer."
        )
        observations.append({"role": "environment", "content": feedback})
        updated_metadata.append(
            MathRepairMetadata(
                ground_truth=row_metadata["ground_truth"],
                turn_index=current_turn,
                first_success_turn=first_success_turn,
                best_score=best_score,
            )
        )
        verifier_scores.append(score)
        terminal_success.append(outcome)
        terminateds.append(terminated)

    return MathRepairTurnBatch(
        observations=observations,
        metadata=updated_metadata,
        verifier_scores=torch.tensor(verifier_scores, dtype=torch.float32),
        terminal_success=torch.tensor(terminal_success, dtype=torch.float32),
        terminateds=torch.tensor(terminateds, dtype=torch.bool),
    )


def _latest_assistant_responses(
    message_log_batch: list[LLMMessageLogType],
) -> list[str]:
    """Extract exactly the current candidate answer from each message log."""
    responses = []
    for row, conversation in enumerate(message_log_batch):
        assistant_messages = [
            message for message in conversation if message.get("role") == "assistant"
        ]
        if not assistant_messages:
            raise ValueError(f"math repair row {row} has no assistant response")
        content = assistant_messages[-1].get("content")
        if not isinstance(content, str):
            raise TypeError(
                f"math repair row {row} assistant response content must be a string"
            )
        responses.append(content)
    return responses
