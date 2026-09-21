# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Frozen plan types for the DAPO load-alignment signed-control study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, StringConstraints, ValidationError, model_validator


Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
GitCommitHex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
SamplerName: TypeAlias = Literal["ready_first", "in_order"]
PressureLevel: TypeAlias = Literal["l0", "l3"]
DelayTarget: TypeAlias = Literal[
    "none", "fixed_higher_load_half", "fixed_lower_load_half"
]
CONFIRMED_PROTOCOL_SHA256 = (
    "00e9c7d481a101cd79d3b155ada9b395346e7de0d6ca8e6e9f827846d2dc4147"
)
IMPLEMENTATION_CONFIRMATION_SHA256 = (
    "da1d796992a0a0d1c285dd8c834322131f30709d3f75c5cf7f07ea5faa5a844c"
)


class DapoLoadAlignmentPlanError(ValueError):
    """The load-alignment plan is malformed or inconsistent."""


class DapoLoadAlignmentArm(BaseModel, extra="forbid", frozen=True):
    arm_id: str
    sampler: SamplerName
    pressure_level: PressureLevel
    sampler_lookahead_versions: Literal[0, 3]
    max_buffered_rollouts: Literal[4, 16]
    delay_target: DelayTarget
    release_delay_seconds: float

    @model_validator(mode="after")
    def _validate_arm(self) -> "DapoLoadAlignmentArm":
        if (self.sampler_lookahead_versions, self.max_buffered_rollouts) != {
            "l0": (0, 4),
            "l3": (3, 16),
        }[self.pressure_level]:
            raise ValueError("pressure level does not match lookahead and buffer")
        if self.pressure_level == "l0" and self.delay_target != "none":
            raise ValueError("L0 is the natural negative control")
        expected_delay = 0.0 if self.delay_target == "none" else 16.0
        if self.release_delay_seconds != expected_delay:
            raise ValueError("release delay does not match delay target")
        condition = {
            "none": "natural",
            "fixed_higher_load_half": "high_load_delayed",
            "fixed_lower_load_half": "low_load_delayed",
        }[self.delay_target]
        expected_id = f"{self.pressure_level}_{condition}_{self.sampler}"
        if self.arm_id != expected_id:
            raise ValueError(f"load-alignment arm ID must be {expected_id!r}")
        return self


class DapoLoadAlignmentPool(BaseModel, extra="forbid", frozen=True):
    replication_id: str
    order_seed: Literal[51001, 51002, 51003]
    scheduler_selection_seed: Literal[2026092001, 2026092002, 2026092003]
    reference_generation_seeds: tuple[int, int]
    scheduler_generation_seed: Literal[73001, 73002, 73003]
    arm_execution_order: tuple[str, ...]
    pool_id: Sha256Hex
    manifest_sha256: Sha256Hex
    private_reference_manifest_sha256: Sha256Hex
    fixed_lower_load_prompt_ids: tuple[Sha256Hex, ...]
    fixed_higher_load_prompt_ids: tuple[Sha256Hex, ...]


class DapoLoadAlignmentThresholds(BaseModel, extra="forbid", frozen=True):
    backend_length_termination_rate_max_each_arm: float
    maximum_active_generation_groups_required_each_arm: int
    signed_pressure_median_min: float
    signed_pressure_replications_exceeding_l0_min: int
    high_load_delayed_promotion_median_min: float
    high_load_delayed_replications_at_or_above_min: int
    low_load_delayed_promotion_median_max: float
    low_load_delayed_replications_at_or_below_max: int
    signed_separation_median_min: float
    natural_lower_load_promotion_median_min: float
    natural_lower_load_replications_at_or_above_min: int
    natural_lower_load_negative_replications_max: int
    natural_easier_promotion_median_min: float
    natural_easier_replications_at_or_above_min: int
    natural_easier_negative_replications_max: int


class DapoLoadAlignmentPlan(BaseModel, extra="forbid", frozen=True):
    schema_version: Literal[1]
    analysis_status: Literal["controlled_dapo_load_alignment_signed_control"]
    calibration_only: Literal[True]
    confirmatory_eligible: Literal[False]
    population_claim_authorized: Literal[False]
    counterfactual_replay_authorized: Literal[False]
    training_authorized: Literal[False]
    confirmed_protocol_sha256: Sha256Hex
    implementation_confirmation_sha256: Sha256Hex
    materialization_confirmation_sha256: Sha256Hex
    reference_result_sha256: Sha256Hex
    final_plan_confirmation_sha256: Sha256Hex
    analysis_code_commit: GitCommitHex
    expected_base_commit: GitCommitHex
    expected_image_sha256: Sha256Hex
    source_design_id: Literal["dapo_math_load_alignment_v1"]
    model_revision: Literal["b101308fe89651ea5ce025f25317fea6fc07e96e"]
    model_weights_sha256: Literal[
        "70a914a3466bf064ca88ae875cb33c366856147e475a2b8da54e7a3d94d8074a"
    ]
    pools: tuple[DapoLoadAlignmentPool, ...]
    arms: tuple[DapoLoadAlignmentArm, ...]
    prompt_groups: Literal[32]
    completions_per_group: Literal[16]
    selection_groups_per_step: Literal[4]
    selection_steps: Literal[8]
    max_inflight_prompts: Literal[4]
    max_total_sequence_length: Literal[6144]
    data_max_input_seq_length: Literal[2048]
    hf_config_override_max_position_embeddings: Literal[6144]
    max_new_tokens: Literal[4096]
    temperature: float
    top_p: float
    replication_order: tuple[Literal[51001, 51002, 51003], ...]
    thresholds: DapoLoadAlignmentThresholds
    plan_id: Sha256Hex

    @model_validator(mode="after")
    def _validate_protocol(self) -> "DapoLoadAlignmentPlan":
        if (
            self.confirmed_protocol_sha256 != CONFIRMED_PROTOCOL_SHA256
            or self.implementation_confirmation_sha256
            != IMPLEMENTATION_CONFIRMATION_SHA256
        ):
            raise ValueError("plan is not bound to the confirmed protocol")
        expected_bindings = {
            51001: (2026092001, (72001, 72011), 73001),
            51002: (2026092002, (72002, 72012), 73002),
            51003: (2026092003, (72003, 72013), 73003),
        }
        observed = {
            pool.order_seed: (
                pool.scheduler_selection_seed,
                pool.reference_generation_seeds,
                pool.scheduler_generation_seed,
            )
            for pool in self.pools
        }
        if observed != expected_bindings or self.replication_order != (
            51001,
            51002,
            51003,
        ):
            raise ValueError("load-alignment replication bindings mismatch")
        if tuple(pool.order_seed for pool in self.pools) != self.replication_order:
            raise ValueError("load-alignment pool order mismatch")
        if (self.temperature, self.top_p) != (1.0, 0.7):
            raise ValueError("load-alignment sampling parameters mismatch")
        expected_arms = {
            f"l0_natural_{sampler}" for sampler in ("in_order", "ready_first")
        } | {
            f"l3_{condition}_{sampler}"
            for condition in ("natural", "high_load_delayed", "low_load_delayed")
            for sampler in ("in_order", "ready_first")
        }
        if len(self.arms) != 8 or {arm.arm_id for arm in self.arms} != expected_arms:
            raise ValueError("load-alignment plan requires the locked eight arms")
        for pool in self.pools:
            if (
                len(pool.arm_execution_order) != 8
                or set(pool.arm_execution_order) != expected_arms
            ):
                raise ValueError("each pool must bind every arm exactly once")
            lower = set(pool.fixed_lower_load_prompt_ids)
            higher = set(pool.fixed_higher_load_prompt_ids)
            if len(lower) != 16 or len(higher) != 16 or lower & higher:
                raise ValueError("fixed load halves must be disjoint 16-prompt sets")
        expected_thresholds = DapoLoadAlignmentThresholds(
            backend_length_termination_rate_max_each_arm=0.2,
            maximum_active_generation_groups_required_each_arm=4,
            signed_pressure_median_min=0.5,
            signed_pressure_replications_exceeding_l0_min=2,
            high_load_delayed_promotion_median_min=1 / 28,
            high_load_delayed_replications_at_or_above_min=2,
            low_load_delayed_promotion_median_max=-1 / 28,
            low_load_delayed_replications_at_or_below_max=2,
            signed_separation_median_min=1 / 14,
            natural_lower_load_promotion_median_min=1 / 28,
            natural_lower_load_replications_at_or_above_min=2,
            natural_lower_load_negative_replications_max=1,
            natural_easier_promotion_median_min=1 / 28,
            natural_easier_replications_at_or_above_min=2,
            natural_easier_negative_replications_max=1,
        )
        if self.thresholds != expected_thresholds:
            raise ValueError("load-alignment thresholds mismatch")
        return self

    def arm(self, arm_id: str) -> DapoLoadAlignmentArm:
        matches = [arm for arm in self.arms if arm.arm_id == arm_id]
        if len(matches) != 1:
            raise DapoLoadAlignmentPlanError(f"unknown arm {arm_id!r}")
        return matches[0]

    def pool(self, order_seed: int) -> DapoLoadAlignmentPool:
        matches = [pool for pool in self.pools if pool.order_seed == order_seed]
        if len(matches) != 1:
            raise DapoLoadAlignmentPlanError(f"unknown pool {order_seed}")
        return matches[0]

    def delayed_prompt_ids(self, arm_id: str, order_seed: int) -> frozenset[str]:
        arm = self.arm(arm_id)
        pool = self.pool(order_seed)
        if arm.delay_target == "fixed_lower_load_half":
            return frozenset(pool.fixed_lower_load_prompt_ids)
        if arm.delay_target == "fixed_higher_load_half":
            return frozenset(pool.fixed_higher_load_prompt_ids)
        return frozenset()


def canonical_dapo_load_alignment_plan(plan: DapoLoadAlignmentPlan) -> bytes:
    return json.dumps(
        plan.model_dump(mode="json", exclude={"plan_id"}),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def compute_dapo_load_alignment_plan_id(plan: DapoLoadAlignmentPlan) -> str:
    return hashlib.sha256(canonical_dapo_load_alignment_plan(plan)).hexdigest()


def load_dapo_load_alignment_plan(path: str | Path) -> DapoLoadAlignmentPlan:
    plan_path = Path(path)
    try:
        plan = DapoLoadAlignmentPlan.model_validate_json(plan_path.read_bytes())
    except (OSError, ValidationError) as error:
        raise DapoLoadAlignmentPlanError(
            f"invalid plan {plan_path}: {error}"
        ) from error
    expected = compute_dapo_load_alignment_plan_id(plan)
    if plan.plan_id != expected:
        raise DapoLoadAlignmentPlanError(
            f"plan ID mismatch: declared={plan.plan_id}, computed={expected}"
        )
    return plan
