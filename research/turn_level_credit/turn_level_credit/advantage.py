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

"""GRPO advantage composition with token-aligned native turn credit."""

from dataclasses import dataclass
from typing import Any

import torch

from turn_level_credit.config import TurnCreditConfig
from turn_level_credit.trace import (
    compute_environment_credit,
    scatter_turn_credit,
    turn_batch_from_mapping,
)
from turn_level_credit.verifier_credit import (
    VerifierScoreBatch,
    compute_verifier_credit,
)


@dataclass(frozen=True)
class TurnCreditAdvantageDiagnostics:
    """Finite first-update scale and support diagnostics on trainable tokens."""

    valid_token_count: int
    macro_std: float
    auxiliary_std: float
    weighted_auxiliary_std: float
    auxiliary_nonzero_fraction: float
    composed_std: float


def _advantage_diagnostics(
    *,
    macro_advantage: torch.Tensor,
    auxiliary_advantage: torch.Tensor,
    advantage_mask: torch.Tensor,
    macro_weight: float,
    turn_weight: float,
) -> TurnCreditAdvantageDiagnostics:
    """Summarize exactly the trainable token values entering composition."""
    if (
        macro_advantage.shape != auxiliary_advantage.shape
        or macro_advantage.shape != advantage_mask.shape
    ):
        raise ValueError("Advantage diagnostics require equal tensor shapes")
    valid_token_count = int(advantage_mask.sum().item())
    if valid_token_count == 0:
        raise ValueError("Advantage diagnostics require at least one trainable token")
    macro_values = macro_advantage[advantage_mask]
    auxiliary_values = auxiliary_advantage[advantage_mask]
    composed_values = macro_weight * macro_values + turn_weight * auxiliary_values
    if not bool(
        torch.isfinite(macro_values).all().item()
        and torch.isfinite(auxiliary_values).all().item()
        and torch.isfinite(composed_values).all().item()
    ):
        raise ValueError("Turn-credit advantage components must be finite")

    def population_std(values: torch.Tensor) -> float:
        return float(values.float().std(unbiased=False).item())

    return TurnCreditAdvantageDiagnostics(
        valid_token_count=valid_token_count,
        macro_std=population_std(macro_values),
        auxiliary_std=population_std(auxiliary_values),
        weighted_auxiliary_std=population_std(turn_weight * auxiliary_values),
        auxiliary_nonzero_fraction=float((auxiliary_values != 0).float().mean().item()),
        composed_std=population_std(composed_values),
    )


def _log_advantage_diagnostics(
    diagnostics: TurnCreditAdvantageDiagnostics,
) -> None:
    """Emit one parseable research-local diagnostic line per update."""
    print(
        "TURN_CREDIT_ADVANTAGE_METRICS "
        f"valid_token_count={diagnostics.valid_token_count} "
        f"macro_std={diagnostics.macro_std:.9g} "
        f"auxiliary_std={diagnostics.auxiliary_std:.9g} "
        f"weighted_auxiliary_std={diagnostics.weighted_auxiliary_std:.9g} "
        "auxiliary_nonzero_fraction="
        f"{diagnostics.auxiliary_nonzero_fraction:.9g} "
        f"composed_std={diagnostics.composed_std:.9g}",
        flush=True,
    )


def _prompt_group_ids(
    prompt_ids: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Match core GRPO's exact padded-prompt row equivalence relation."""
    if (
        prompt_ids.ndim != 2
        or prompt_ids.shape[0] != batch_size
        or prompt_ids.shape[1] == 0
    ):
        raise ValueError(
            "Verifier prompt IDs must have shape [turn-credit batch, prompt tokens]"
        )
    if prompt_ids.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("Verifier prompt IDs must use an integer token dtype")
    _, inverse_group_ids = torch.unique(
        prompt_ids,
        dim=0,
        sorted=True,
        return_inverse=True,
    )
    return inverse_group_ids.to(device=device, dtype=torch.int64)


class TurnLevelGRPOAdvantageEstimator:
    """Wrap the core GRPO estimator with auxiliary turn credit."""

    def __init__(
        self,
        *,
        base_estimator: Any,
        config: TurnCreditConfig,
    ) -> None:
        self.base_estimator = base_estimator
        self.config = config

    def compute_advantage(
        self,
        prompt_ids: torch.Tensor,
        rewards: torch.Tensor,
        mask: torch.Tensor,
        *,
        repeated_batch: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute macro GRPO advantage plus token-aligned turn credit."""
        macro_advantage = self.base_estimator.compute_advantage(
            prompt_ids=prompt_ids,
            rewards=rewards,
            mask=mask,
            repeated_batch=repeated_batch,
            **kwargs,
        )

        turn_batch = turn_batch_from_mapping(repeated_batch)
        if turn_batch.batch_size != mask.shape[0]:
            raise ValueError(
                "Turn-credit batch size does not match token advantage batch size"
            )
        advantage_mask = mask.bool()

        if self.config.turn_weight == 0.0 and self.config.macro_weight == 1.0:
            scatter_turn_credit(
                torch.zeros_like(turn_batch.credit_rewards),
                turn_batch,
                advantage_mask,
            )
            return macro_advantage

        if self.config.verifier_transform is None:
            credit = compute_environment_credit(
                turn_batch,
                mode=self.config.environment_mode,
                discount=self.config.discount,
            )
        else:
            prompt_group_ids = _prompt_group_ids(
                prompt_ids,
                batch_size=turn_batch.batch_size,
                device=turn_batch.credit_rewards.device,
            )
            credit = compute_verifier_credit(
                VerifierScoreBatch(
                    scores=turn_batch.credit_rewards,
                    mask=turn_batch.mask,
                    prompt_group_ids=prompt_group_ids,
                ),
                config=self.config.verifier_transform,
            )
        auxiliary_advantage = scatter_turn_credit(
            credit,
            turn_batch,
            advantage_mask,
        )
        _log_advantage_diagnostics(
            _advantage_diagnostics(
                macro_advantage=macro_advantage,
                auxiliary_advantage=auxiliary_advantage,
                advantage_mask=advantage_mask,
                macro_weight=self.config.macro_weight,
                turn_weight=self.config.turn_weight,
            )
        )
        advantages = (
            self.config.macro_weight * macro_advantage
            + self.config.turn_weight * auxiliary_advantage
        )
        return advantages * advantage_mask.to(dtype=advantages.dtype)
