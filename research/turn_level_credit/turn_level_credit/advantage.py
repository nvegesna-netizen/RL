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
        advantages = (
            self.config.macro_weight * macro_advantage
            + self.config.turn_weight * auxiliary_advantage
        )
        return advantages * advantage_mask.to(dtype=advantages.dtype)
