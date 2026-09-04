# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Frozen protocol types for a zero-update live scheduler assay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)


Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
GitCommitHex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
PositiveStrictInt = Annotated[int, Field(strict=True, gt=0)]
UnitFloat = Annotated[float, Field(strict=True, ge=1.0, le=1.0)]
ThirtySeconds = Annotated[float, Field(strict=True, ge=30.0, le=30.0)]


class SchedulerAssayPlanError(ValueError):
    """The live scheduler assay plan is malformed or inconsistent."""


class SchedulerAssayPool(BaseModel, extra="forbid", frozen=True):
    """Immutable identity for one previously materialized source ordering."""

    order_seed: Literal[42001, 42002, 42003]
    pool_id: Sha256Hex
    manifest_sha256: Sha256Hex


class SchedulerAssayArm(BaseModel, extra="forbid", frozen=True):
    """One live policy and task-delay mapping."""

    arm_id: Literal[
        "ready_first_aime_delayed",
        "ready_first_gsm8k_delayed",
        "in_order_aime_delayed",
        "in_order_gsm8k_delayed",
    ]
    sampler: Literal["ready_first", "in_order"]
    delayed_task: Literal["gsm8k", "AIME2024"]


class SchedulerAssayThresholds(BaseModel, extra="forbid", frozen=True):
    """Locked progression gates for each four-arm order-seed block."""

    minimum_delayed_hold_seconds: float
    maximum_undelayed_commit_seconds: float
    ready_first_undelayed_share_min: float
    in_order_undelayed_share: float
    ready_first_minus_in_order_min: float


class SchedulerAssayPlan(BaseModel, extra="forbid", frozen=True):
    """Complete, hash-addressed controlled live-scheduler protocol."""

    schema_version: Literal[1]
    analysis_status: Literal["controlled_zero_update_live_scheduler_assay"]
    calibration_only: Literal[True]
    confirmatory_eligible: Literal[False]
    natural_latency_claim_authorized: Literal[False]
    replay_of_natural_workloads_authorized: Literal[False]
    training_authorized: Literal[False]
    analysis_code_commit: GitCommitHex
    expected_base_commit: GitCommitHex
    expected_image_sha256: Sha256Hex
    source_design_id: Literal["ready_bias_v1"]
    pools: tuple[SchedulerAssayPool, ...]
    arms: tuple[SchedulerAssayArm, ...]
    prompt_groups: Literal[48]
    dispatch_cohorts: Literal[12]
    groups_per_cohort: Literal[4]
    groups_per_task_per_cohort: Literal[2]
    completions_per_group: Literal[2]
    generation_study_seed: Literal[51001]
    max_total_sequence_length: Literal[512]
    temperature: UnitFloat
    top_p: UnitFloat
    max_inflight_prompts: Literal[4]
    max_buffered_rollouts: Literal[16]
    sampler_lookahead_versions: Literal[3]
    release_delay_seconds: ThirtySeconds
    primary_horizon: Literal[8]
    replication_order: tuple[Literal[42001, 42002, 42003], ...]
    thresholds: SchedulerAssayThresholds
    plan_id: Sha256Hex

    @model_validator(mode="after")
    def _validate_protocol(self) -> "SchedulerAssayPlan":
        expected_pools = {
            42001: (
                "186215e9b80685ce292408db438319ffb5cf4e281b35ce78e197c0b11d12a88a",
                "340b2cb322e35ebbcbf6c25a320c846bd797765ddfae67be4b480623087a7294",
            ),
            42002: (
                "6825c7ba315a6887708192b46f80e9b88f447063a946eaadab9fb24a1c340cf9",
                "bc0afd679cc0fe8445bbfa27dc1484eabe2c522c12d0a58f8400f875e36d0817",
            ),
            42003: (
                "3cb47d86319e7dae1a8bd00a220ecea912870b80550ea0e4384dfece51c01414",
                "4f0b14cb21963341991470241442cb7ed68fe17288f4348ba9d5c11fd5e0ef42",
            ),
        }
        observed_pools = {
            pool.order_seed: (pool.pool_id, pool.manifest_sha256) for pool in self.pools
        }
        if observed_pools != expected_pools:
            raise ValueError("schema v1 requires the three verified v2 source pools")
        expected_arms = (
            ("ready_first_aime_delayed", "ready_first", "AIME2024"),
            ("ready_first_gsm8k_delayed", "ready_first", "gsm8k"),
            ("in_order_aime_delayed", "in_order", "AIME2024"),
            ("in_order_gsm8k_delayed", "in_order", "gsm8k"),
        )
        observed_arms = tuple(
            (arm.arm_id, arm.sampler, arm.delayed_task) for arm in self.arms
        )
        if observed_arms != expected_arms:
            raise ValueError("schema v1 requires the locked four-arm crossover")
        if self.replication_order != (42001, 42002, 42003):
            raise ValueError("schema v1 requires pilot 42001 before replications")
        if self.thresholds != SchedulerAssayThresholds(
            minimum_delayed_hold_seconds=29.5,
            maximum_undelayed_commit_seconds=5.0,
            ready_first_undelayed_share_min=0.875,
            in_order_undelayed_share=0.5,
            ready_first_minus_in_order_min=0.375,
        ):
            raise ValueError("schema v1 requires the locked progression thresholds")
        return self

    def arm(self, arm_id: str) -> SchedulerAssayArm:
        """Return the named frozen arm or fail loudly."""
        matches = [arm for arm in self.arms if arm.arm_id == arm_id]
        if len(matches) != 1:
            raise SchedulerAssayPlanError(f"unknown scheduler assay arm {arm_id!r}")
        return matches[0]

    def pool(self, order_seed: int) -> SchedulerAssayPool:
        """Return the named frozen source pool or fail loudly."""
        matches = [pool for pool in self.pools if pool.order_seed == order_seed]
        if len(matches) != 1:
            raise SchedulerAssayPlanError(
                f"unknown scheduler assay order seed {order_seed}"
            )
        return matches[0]


def canonical_scheduler_assay_plan(plan: SchedulerAssayPlan) -> bytes:
    """Serialize every protocol field except the self-addressing plan ID."""
    return json.dumps(
        plan.model_dump(mode="json", exclude={"plan_id"}),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def compute_scheduler_assay_plan_id(plan: SchedulerAssayPlan) -> str:
    """Compute the plan's canonical SHA-256 identity."""
    return hashlib.sha256(canonical_scheduler_assay_plan(plan)).hexdigest()


def load_scheduler_assay_plan(path: str | Path) -> SchedulerAssayPlan:
    """Load and validate a frozen live-scheduler assay plan."""
    plan_path = Path(path)
    try:
        raw = plan_path.read_bytes()
    except OSError as error:
        raise SchedulerAssayPlanError(f"cannot read assay plan {plan_path}") from error
    try:
        plan = SchedulerAssayPlan.model_validate_json(raw)
    except ValidationError as error:
        raise SchedulerAssayPlanError(
            f"invalid assay plan {plan_path}: {error}"
        ) from error
    expected_id = compute_scheduler_assay_plan_id(plan)
    if plan.plan_id != expected_id:
        raise SchedulerAssayPlanError(
            f"assay plan ID mismatch: declared={plan.plan_id}, computed={expected_id}"
        )
    return plan
