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
from typing import Any, Literal, Optional, TypeAlias

from pydantic import BaseModel, Field, PositiveInt, model_validator

from nemo_rl.algorithms.async_utils.staleness_sampler import (
    CustomSamplerConfig,
    InOrderSamplerConfig,
    ReadyFirstSamplerConfig,
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

FixedPoolDesignId: TypeAlias = Literal[
    "ready_bias_v1",
    "openmath_latency_feasibility_v1",
    "openmath_termination_headroom_v1",
    "sliding_puzzle_latency_feasibility_v1",
    "sliding_puzzle_7b_competence_v1",
    "sliding_puzzle_7b_compact_prompt_v2",
]


class SchedulerTraceConfig(BaseModel, extra="forbid"):
    """Controller-local, metadata-only scheduler lifecycle trace."""

    enabled: bool = False
    path: Optional[str] = None
    max_queue_events: PositiveInt = 4096
    flush_every: PositiveInt = 64

    @model_validator(mode="after")
    def _require_path_when_enabled(self) -> "SchedulerTraceConfig":
        if self.enabled and not self.path:
            raise ValueError("async_rl.scheduler_trace.path is required when enabled")
        return self


class FixedPoolCollectionConfig(BaseModel, extra="forbid"):
    """Scheduler-neutral, fixed-policy source-pool collection."""

    enabled: bool = False
    manifest_path: Optional[str] = None
    # Strict manifest design contract; ready_bias_v1 preserves the original default.
    design_id: FixedPoolDesignId = "ready_bias_v1"

    @model_validator(mode="after")
    def _require_manifest_when_enabled(self) -> "FixedPoolCollectionConfig":
        if self.enabled and not self.manifest_path:
            raise ValueError(
                "async_rl.fixed_pool.manifest_path is required when enabled"
            )
        return self


class SchedulerAssayConfig(BaseModel, extra="forbid"):
    """Zero-update production-sampler assay over a fixed source pool."""

    enabled: bool = False
    plan_path: Optional[str] = None
    arm_id: Optional[str] = None
    order_seed: Optional[int] = None

    @model_validator(mode="after")
    def _require_identity_when_enabled(self) -> "SchedulerAssayConfig":
        if self.enabled and (
            not self.plan_path or not self.arm_id or self.order_seed is None
        ):
            raise ValueError(
                "async_rl.scheduler_assay plan_path, arm_id, and order_seed are "
                "required when enabled"
            )
        return self


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
    # Versioned group lifecycle trace. Disabled is a task/file-free no-op.
    scheduler_trace: SchedulerTraceConfig = Field(
        default_factory=SchedulerTraceConfig,
    )
    # Finite, no-training collection over a predeclared prompt-group manifest.
    fixed_pool: FixedPoolCollectionConfig = Field(
        default_factory=FixedPoolCollectionConfig,
    )
    # Controlled production-sampler selection with no learner updates.
    scheduler_assay: SchedulerAssayConfig = Field(
        default_factory=SchedulerAssayConfig,
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
    if async_config.fixed_pool.enabled:
        assay_config = async_config.scheduler_assay
        if not async_config.scheduler_trace.enabled:
            raise ValueError(
                "async_rl.scheduler_trace.enabled must be true for fixed-pool "
                "collection"
            )
        if master_config.data["shuffle"]:
            raise ValueError("data.shuffle must be false for fixed-pool collection")
        if master_config.data.get("use_multiple_dataloader"):
            raise ValueError(
                "data.use_multiple_dataloader is unsupported for fixed-pool collection"
            )
        if master_config.grpo.use_dynamic_sampling:
            raise ValueError(
                "grpo.use_dynamic_sampling must be false for fixed-pool collection"
            )
        if (
            master_config.grpo.val_period > 0
            or master_config.grpo.val_at_start
            or master_config.grpo.val_at_end
            or master_config.grpo.stop_at_validation_metric is not None
        ):
            raise ValueError(
                "validation and validation-based early stopping must be disabled "
                "for fixed-pool collection"
            )
        if master_config.checkpointing["enabled"]:
            raise ValueError("checkpointing must be disabled for fixed-pool collection")
        if async_config.max_inflight_prompts < 1:
            raise ValueError(
                "async_rl.max_inflight_prompts must be positive for fixed-pool "
                "collection"
            )
        if async_config.max_buffered_rollouts < async_config.max_inflight_prompts:
            raise ValueError(
                "async_rl.max_buffered_rollouts must be at least "
                "async_rl.max_inflight_prompts for fixed-pool collection"
            )
        if assay_config.enabled:
            if async_config.fixed_pool.design_id != "ready_bias_v1":
                raise ValueError(
                    "scheduler assay schema v1 requires fixed_pool.design_id="
                    "ready_bias_v1"
                )
            if not isinstance(
                async_config.sampler,
                (ReadyFirstSamplerConfig, InOrderSamplerConfig),
            ):
                raise ValueError(
                    "scheduler assay requires ready_first or in_order sampler"
                )
            if master_config.grpo.num_prompts_per_step != 4:
                raise ValueError("scheduler assay requires grpo.num_prompts_per_step=4")
            if async_config.min_groups_for_streaming_train != 4:
                raise ValueError(
                    "scheduler assay requires min_groups_for_streaming_train=4"
                )
            if async_config.max_inflight_prompts != 4:
                raise ValueError("scheduler assay requires max_inflight_prompts=4")
            if async_config.max_buffered_rollouts != 16:
                raise ValueError("scheduler assay requires max_buffered_rollouts=16")
            if master_config.grpo.max_num_epochs != 1:
                raise ValueError("scheduler assay requires grpo.max_num_epochs=1")
            lookahead = (
                async_config.sampler.max_staleness_versions
                if isinstance(async_config.sampler, ReadyFirstSamplerConfig)
                else async_config.sampler.max_lookahead_versions
            )
            if lookahead != 3:
                raise ValueError("scheduler assay requires sampler lookahead=3")
        # Collection performs one initial weight sync and no learner or sampler
        # operations. Assay mode performs sampler operations but no learner or
        # loss operations, so training-only batch/loss constraints do not apply.
        return
    if async_config.scheduler_assay.enabled:
        raise ValueError(
            "async_rl.scheduler_assay.enabled=true requires fixed_pool.enabled=true"
        )
    if async_config.scheduler_trace.enabled and isinstance(
        async_config.sampler, CustomSamplerConfig
    ):
        raise ValueError(
            "scheduler tracing currently requires a built-in sampler so exact "
            "eligible/selected/evicted group identities can be recorded"
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

    if isinstance(async_config.sampler, ReadyFirstSamplerConfig):
        if not master_config.loss_fn.use_importance_sampling_correction:
            raise ValueError(
                "the ready_first sampler requires "
                "loss_fn.use_importance_sampling_correction=true"
            )
        if master_config.loss_fn.force_on_policy_ratio:
            raise ValueError(
                "the ready_first sampler requires "
                "loss_fn.force_on_policy_ratio=false so prev_logprobs are used"
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
