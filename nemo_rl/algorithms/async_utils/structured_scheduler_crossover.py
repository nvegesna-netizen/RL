# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Frozen protocol types for the structured natural-latency scheduler crossover."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, StringConstraints, ValidationError, model_validator

from nemo_rl.algorithms.async_utils.scheduler_assay import (
    SchedulerAssayArm,
    SchedulerAssayPlan,
    load_scheduler_assay_plan,
)


Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
GitCommitHex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
CrossoverArmId: TypeAlias = Literal["ready_first", "in_order"]

CONFIRMED_CANDIDATE_SHA256 = (
    "ddd7536aa3d67351974629c2c56fbe41456b9e3b48b0d26e47ef7f7e2728f7ef"
)


class StructuredSchedulerCrossoverPlanError(ValueError):
    """The structured scheduler crossover plan is malformed or inconsistent."""


class StructuredSchedulerCrossoverArm(BaseModel, extra="forbid", frozen=True):
    """One production scheduler policy without an artificial release delay."""

    arm_id: CrossoverArmId
    sampler: CrossoverArmId

    @model_validator(mode="after")
    def _arm_matches_sampler(self) -> "StructuredSchedulerCrossoverArm":
        if self.arm_id != self.sampler:
            raise ValueError("crossover arm ID and sampler must match")
        return self


class StructuredSchedulerCrossoverPool(BaseModel, extra="forbid", frozen=True):
    """Immutable identity and execution order for one fresh replication pool."""

    replication_id: str
    order_seed: Literal[46001, 46002, 46003, 46004]
    selection_seed: Literal[2026090801, 2026090802, 2026090803, 2026090804]
    generation_study_seed: Literal[65001, 65002, 65003, 65004]
    arm_execution_order: tuple[CrossoverArmId, CrossoverArmId]
    pool_id: Sha256Hex
    manifest_sha256: Sha256Hex


class StructuredSchedulerCrossoverThresholds(BaseModel, extra="forbid", frozen=True):
    """Locked validity, pilot, and replicated-effect thresholds."""

    backend_length_termination_rate_max_each_stratum_each_arm: float
    in_order_short_share_at_primary_horizon: float
    long_short_generated_token_median_ratio_min_each_arm: float
    long_short_ready_latency_median_ratio_min_each_arm: float
    maximum_concurrent_groups_min_each_arm: int
    primary_contrast_min: float
    ready_first_short_share_min: float
    reward_mean_min_each_stratum_each_arm: float
    contrasts_at_or_above_minimum_required: int
    maximum_negative_contrasts: int
    median_contrast_min: float


class StructuredSchedulerCrossoverPlan(BaseModel, extra="forbid", frozen=True):
    """Complete hash-addressed execution plan for the confirmed crossover."""

    schema_version: Literal[1]
    analysis_status: Literal["controlled_natural_latency_scheduler_crossover"]
    calibration_only: Literal[True]
    confirmatory_eligible: Literal[False]
    natural_benchmark_claim_authorized: Literal[False]
    counterfactual_replay_authorized: Literal[False]
    training_authorized: Literal[False]
    confirmed_candidate_sha256: Sha256Hex
    confirmation_record_sha256: Sha256Hex
    analysis_code_commit: GitCommitHex
    expected_base_commit: GitCommitHex
    expected_image_sha256: Sha256Hex
    source_design_id: Literal["structured_generation_scheduler_crossover_v1"]
    model_revision: Literal["a09a35458c702b33eeacc393d103063234e8bc28"]
    model_weights_sha256: Literal[
        "456f5eff514d78f7b0ef52a057118046acf15d86606655767db068cabf5f49f7"
    ]
    pools: tuple[StructuredSchedulerCrossoverPool, ...]
    arms: tuple[StructuredSchedulerCrossoverArm, ...]
    prompt_groups: Literal[16]
    dispatch_cohorts: Literal[4]
    groups_per_cohort: Literal[4]
    groups_per_task_per_cohort: Literal[2]
    completions_per_group: Literal[2]
    max_total_sequence_length: Literal[768]
    max_new_tokens: Literal[768]
    temperature: Literal[0.7]
    top_p: Literal[0.8]
    top_k: Literal[20]
    repetition_penalty: Literal[1.0]
    max_inflight_prompts: Literal[4]
    max_buffered_rollouts: Literal[8]
    sampler_lookahead_versions: Literal[1]
    release_delay_seconds: Literal[0.0]
    primary_horizon: Literal[8]
    replication_order: tuple[Literal[46001, 46002, 46003, 46004], ...]
    thresholds: StructuredSchedulerCrossoverThresholds
    plan_id: Sha256Hex

    @model_validator(mode="after")
    def _validate_protocol(self) -> "StructuredSchedulerCrossoverPlan":
        if self.confirmed_candidate_sha256 != CONFIRMED_CANDIDATE_SHA256:
            raise ValueError("crossover plan is not bound to the confirmed candidate")
        expected_replications = {
            46001: (2026090801, 65001, ("ready_first", "in_order")),
            46002: (2026090802, 65002, ("in_order", "ready_first")),
            46003: (2026090803, 65003, ("in_order", "ready_first")),
            46004: (2026090804, 65004, ("ready_first", "in_order")),
        }
        observed_replications = {
            pool.order_seed: (
                pool.selection_seed,
                pool.generation_study_seed,
                pool.arm_execution_order,
            )
            for pool in self.pools
        }
        if observed_replications != expected_replications:
            raise ValueError("crossover plan replication bindings mismatch")
        if tuple(pool.order_seed for pool in self.pools) != self.replication_order:
            raise ValueError("crossover pools must follow replication order")
        if self.replication_order != (46001, 46002, 46003, 46004):
            raise ValueError(
                "crossover plan requires pilot 46001 then three replications"
            )
        if tuple((arm.arm_id, arm.sampler) for arm in self.arms) != (
            ("ready_first", "ready_first"),
            ("in_order", "in_order"),
        ):
            raise ValueError("crossover plan requires the locked two-arm comparison")
        expected_thresholds = StructuredSchedulerCrossoverThresholds(
            backend_length_termination_rate_max_each_stratum_each_arm=0.125,
            in_order_short_share_at_primary_horizon=0.5,
            long_short_generated_token_median_ratio_min_each_arm=2.0,
            long_short_ready_latency_median_ratio_min_each_arm=1.5,
            maximum_concurrent_groups_min_each_arm=2,
            primary_contrast_min=0.25,
            ready_first_short_share_min=0.75,
            reward_mean_min_each_stratum_each_arm=0.75,
            contrasts_at_or_above_minimum_required=3,
            maximum_negative_contrasts=0,
            median_contrast_min=0.25,
        )
        if self.thresholds != expected_thresholds:
            raise ValueError("crossover plan progression thresholds mismatch")
        return self

    def arm(self, arm_id: str) -> StructuredSchedulerCrossoverArm:
        """Return the named arm or fail loudly."""
        matches = [arm for arm in self.arms if arm.arm_id == arm_id]
        if len(matches) != 1:
            raise StructuredSchedulerCrossoverPlanError(
                f"unknown crossover arm {arm_id!r}"
            )
        return matches[0]

    def pool(self, order_seed: int) -> StructuredSchedulerCrossoverPool:
        """Return the named replication pool or fail loudly."""
        matches = [pool for pool in self.pools if pool.order_seed == order_seed]
        if len(matches) != 1:
            raise StructuredSchedulerCrossoverPlanError(
                f"unknown crossover order seed {order_seed}"
            )
        return matches[0]


SchedulerProtocolPlan: TypeAlias = SchedulerAssayPlan | StructuredSchedulerCrossoverPlan
SchedulerProtocolArm: TypeAlias = SchedulerAssayArm | StructuredSchedulerCrossoverArm


def canonical_structured_scheduler_crossover_plan(
    plan: StructuredSchedulerCrossoverPlan,
) -> bytes:
    """Serialize every execution field except the self-addressing plan ID."""
    return json.dumps(
        plan.model_dump(mode="json", exclude={"plan_id"}),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def compute_structured_scheduler_crossover_plan_id(
    plan: StructuredSchedulerCrossoverPlan,
) -> str:
    """Compute the final execution plan's canonical SHA-256 identity."""
    return hashlib.sha256(
        canonical_structured_scheduler_crossover_plan(plan)
    ).hexdigest()


def load_structured_scheduler_crossover_plan(
    path: str | Path,
) -> StructuredSchedulerCrossoverPlan:
    """Load and validate a final immutable crossover execution plan."""
    plan_path = Path(path)
    try:
        raw = plan_path.read_bytes()
    except OSError as error:
        raise StructuredSchedulerCrossoverPlanError(
            f"cannot read crossover plan {plan_path}"
        ) from error
    try:
        plan = StructuredSchedulerCrossoverPlan.model_validate_json(raw)
    except ValidationError as error:
        raise StructuredSchedulerCrossoverPlanError(
            f"invalid crossover plan {plan_path}: {error}"
        ) from error
    expected_id = compute_structured_scheduler_crossover_plan_id(plan)
    if plan.plan_id != expected_id:
        raise StructuredSchedulerCrossoverPlanError(
            f"crossover plan ID mismatch: declared={plan.plan_id}, "
            f"computed={expected_id}"
        )
    return plan


def load_scheduler_protocol(path: str | Path) -> SchedulerProtocolPlan:
    """Load either the legacy imposed-delay assay or this crossover protocol."""
    plan_path = Path(path)
    try:
        value = json.loads(plan_path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise StructuredSchedulerCrossoverPlanError(
            f"cannot inspect scheduler protocol {plan_path}"
        ) from error
    if not isinstance(value, dict):
        raise StructuredSchedulerCrossoverPlanError(
            "scheduler protocol must be an object"
        )
    if value.get("analysis_status") == "controlled_zero_update_live_scheduler_assay":
        return load_scheduler_assay_plan(plan_path)
    if value.get("analysis_status") == "controlled_natural_latency_scheduler_crossover":
        return load_structured_scheduler_crossover_plan(plan_path)
    raise StructuredSchedulerCrossoverPlanError("unsupported scheduler protocol type")
