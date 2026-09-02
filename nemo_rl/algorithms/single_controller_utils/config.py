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

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from nemo_rl.algorithms.async_utils.controlled_release import (
    ControlledReleaseDelayConfig,
)
from nemo_rl.algorithms.async_utils.staleness_sampler import (
    InOrderSamplerConfig,
    SamplerConfig,
    required_buffer_capacity_for_config,
)
from nemo_rl.algorithms.grpo import GRPOConfig, GRPOLoggerConfig
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.data import DataConfig
from nemo_rl.data_plane.interfaces import DataPlaneConfig
from nemo_rl.distributed.virtual_cluster import ClusterConfig
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.utils.checkpoint import CheckpointingConfig

# ── User-facing SingleController configs ────────────────────────────────────


class GradientOpportunityAuditConfig(BaseModel, frozen=True):
    """Default-off pre-release GRPO coefficient-opportunity audit."""

    enabled: bool = False
    output_path: Optional[str] = None
    observer_duty_path: Optional[str] = None


class AsyncRLConfig(BaseModel, extra="allow"):
    # Staleness policy shared by the rollout and train pumps.
    sampler: SamplerConfig = Field(
        default_factory=InOrderSamplerConfig,
    )
    # Recompute generation KV caches after each weight update.
    recompute_kv_cache_after_weight_updates: bool = False
    # Min ready groups the streaming trainer waits for before dispatching a batch.
    min_groups_for_streaming_train: int = 32
    # Cap on in-flight generate_and_push calls in the rollout pump.
    max_inflight_prompts: int = 32
    # Cap on unconsumed rollout groups buffered in the DataPlane (backpressure).
    max_buffered_rollouts: int = 64
    # Enable per-rollout diagnostic prints (prompt content / completion previews).
    diagnostics: bool = False
    # Optional controller-local JSONL path for prompt-group lifecycle events.
    lifecycle_audit_path: Optional[str] = None
    # Default-off research intervention applied after sibling generation.
    controlled_release_delay: ControlledReleaseDelayConfig = Field(
        default_factory=ControlledReleaseDelayConfig
    )
    # Default-off all-group GRPO coefficient-opportunity ledger.
    gradient_opportunity_audit: GradientOpportunityAuditConfig = Field(
        default_factory=GradientOpportunityAuditConfig
    )


class MasterConfig(BaseModel, extra="allow"):
    policy: PolicyConfig
    loss_fn: ClippedPGLossConfig
    env: dict[str, Any]
    data: DataConfig
    grpo: GRPOConfig
    logger: GRPOLoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig
    data_plane: DataPlaneConfig
    async_rl: AsyncRLConfig


def validate_sampler_buffer_capacity(
    async_config: AsyncRLConfig,
    *,
    required_capacity: Optional[int],
    sampler_name: str,
) -> None:
    """Validate that backpressure cannot deadlock the selected sampler."""
    if (
        required_capacity is not None
        and async_config.max_buffered_rollouts < required_capacity
    ):
        raise ValueError(
            f"max_buffered_rollouts ({async_config.max_buffered_rollouts}) is below "
            f"the {sampler_name} sampler's required capacity "
            f"({required_capacity}); the rollout pump would deadlock waiting for "
            f"buffer slots."
        )


def validate_single_controller_config(master_config: MasterConfig) -> None:
    """Validate cross-section SingleController constraints before setup."""
    async_config = master_config.async_rl
    release_config = async_config.controlled_release_delay
    opportunity_config = async_config.gradient_opportunity_audit
    if release_config.enabled:
        if not async_config.lifecycle_audit_path:
            raise ValueError(
                "async_rl.controlled_release_delay.enabled=true requires "
                "async_rl.lifecycle_audit_path"
            )
        if bool(master_config.env.get("should_use_nemo_gym")):
            raise ValueError(
                "controlled release delay is supported only by native async rollouts"
            )
    if opportunity_config.enabled:
        if not release_config.enabled:
            raise ValueError(
                "async_rl.gradient_opportunity_audit.enabled=true requires "
                "async_rl.controlled_release_delay.enabled=true"
            )
        if not opportunity_config.output_path:
            raise ValueError(
                "async_rl.gradient_opportunity_audit.enabled=true requires "
                "async_rl.gradient_opportunity_audit.output_path"
            )
        if not opportunity_config.observer_duty_path:
            raise ValueError(
                "async_rl.gradient_opportunity_audit.enabled=true requires "
                "async_rl.gradient_opportunity_audit.observer_duty_path"
            )
        assert async_config.lifecycle_audit_path is not None
        audit_paths = (
            Path(async_config.lifecycle_audit_path).resolve(),
            Path(opportunity_config.output_path).resolve(),
            Path(opportunity_config.observer_duty_path).resolve(),
        )
        if len(set(audit_paths)) != len(audit_paths):
            raise ValueError(
                "lifecycle, opportunity, and observer-duty audits require "
                "distinct paths"
            )
        if master_config.grpo.adv_estimator.name != "grpo":
            raise ValueError(
                "gradient opportunity audit currently requires the native GRPO "
                "advantage estimator"
            )
        loss_config = master_config.loss_fn
        unsupported_loss = (
            loss_config.disable_ppo_ratio
            or not loss_config.token_level_loss
            or loss_config.sequence_level_importance_ratios
            or loss_config.use_cispo
            or loss_config.positive_example_nll_weight != 0
        )
        if unsupported_loss:
            raise ValueError(
                "gradient opportunity audit requires ordinary token-level clipped "
                "PG (PPO ratio enabled, no sequence-level ratio, CISPO, or "
                "positive-example NLL)"
            )
    num_prompts_per_step = master_config.grpo.num_prompts_per_step
    if num_prompts_per_step < async_config.min_groups_for_streaming_train:
        raise ValueError(
            f"grpo.num_prompts_per_step ({num_prompts_per_step}) "
            f"must be >= async_rl.min_groups_for_streaming_train "
            f"({async_config.min_groups_for_streaming_train})"
        )

    rl_step_samples = (
        num_prompts_per_step * master_config.grpo.num_generations_per_prompt
    )
    train_global_batch_size = master_config.policy["train_global_batch_size"]
    if rl_step_samples != train_global_batch_size:
        raise ValueError(
            "num_prompts_per_step * num_generations_per_prompt "
            f"({rl_step_samples}) must equal policy.train_global_batch_size "
            f"({train_global_batch_size}) so that one RL step maps to exactly one "
            "optimizer.step. Multi-mini-step inside a single RL step is not "
            "supported on the SC split path."
        )

    required_capacity = required_buffer_capacity_for_config(
        async_config.sampler,
        num_prompts_per_step,
    )
    validate_sampler_buffer_capacity(
        async_config,
        required_capacity=required_capacity,
        sampler_name=async_config.sampler.name,
    )

    # A non-zero reference-policy KL penalty makes the loss read
    # ``reference_policy_logprobs``, but the SC train pump only computes them
    # when ``skip_reference_policy_logprobs_calculation`` is false (see
    # SingleControllerActor._reference_logprobs_required). Catch the
    # inconsistent pair at setup instead of a mid-training KeyError.
    reference_policy_kl_penalty = getattr(
        master_config.loss_fn, "reference_policy_kl_penalty", 0
    )
    if (
        reference_policy_kl_penalty
        and master_config.grpo.skip_reference_policy_logprobs_calculation
    ):
        raise ValueError(
            "loss_fn.reference_policy_kl_penalty="
            f"{reference_policy_kl_penalty} requires reference_policy_logprobs, "
            "but grpo.skip_reference_policy_logprobs_calculation=true skips "
            "computing them on the SingleController path. Set "
            "grpo.skip_reference_policy_logprobs_calculation=false, or set "
            "loss_fn.reference_policy_kl_penalty=0."
        )


# ── Internal SingleController configs ────────────────────────────────────


@dataclass
class AdvantageConfig:
    """Internal DataPlane field mapping for advantage calculation."""

    output_field: str = "advantages"
    prompt_ids_field: str = "prompt_ids_for_adv"
    reward_field: str = "total_reward"
    token_mask_field: str = "token_mask"
    sample_mask_field: str = "sample_mask"
    repeated_batch_fields: list[str] = field(default_factory=list)
    policy_logprobs_field: str = "prev_logprobs"
    reference_logprobs_field: str = "reference_policy_logprobs"
