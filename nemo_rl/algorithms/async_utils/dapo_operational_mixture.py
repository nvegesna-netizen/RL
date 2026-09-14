# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Frozen plan types for the confirmed DAPO operational-mixture shadows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, StringConstraints, ValidationError, model_validator


Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
GitCommitHex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
SamplerName: TypeAlias = Literal["ready_first", "in_order"]
PressureLevel: TypeAlias = Literal["l0", "l1", "l3"]
CONFIRMED_PROTOCOL_SHA256 = (
    "e9942c4afbec0b3b622079d1208eaba6a7c83a7759fc307cfd47293e797a9f63"
)


class DapoOperationalMixturePlanError(ValueError):
    """The operational-mixture shadow plan is malformed or inconsistent."""


class DapoOperationalMixtureArm(BaseModel, extra="forbid", frozen=True):
    arm_id: str
    sampler: SamplerName
    pressure_level: PressureLevel
    sampler_lookahead_versions: Literal[0, 1, 3]
    max_buffered_rollouts: Literal[4, 8, 16]

    @model_validator(mode="after")
    def _validate_arm(self) -> "DapoOperationalMixtureArm":
        expected = {"l0": (0, 4), "l1": (1, 8), "l3": (3, 16)}
        if expected[self.pressure_level] != (
            self.sampler_lookahead_versions,
            self.max_buffered_rollouts,
        ):
            raise ValueError("pressure level does not match lookahead and buffer")
        if self.arm_id != f"{self.pressure_level}_{self.sampler}":
            raise ValueError("operational-mixture arm ID mismatch")
        return self


class DapoOperationalMixturePool(BaseModel, extra="forbid", frozen=True):
    replication_id: str
    order_seed: Literal[50001, 50002, 50003]
    scheduler_selection_seed: Literal[2026091301, 2026091302, 2026091303]
    scheduler_generation_seed: Literal[71001, 71002, 71003]
    arm_execution_order: tuple[str, ...]
    pool_id: Sha256Hex
    manifest_sha256: Sha256Hex
    private_reference_manifest_sha256: Sha256Hex


class DapoOperationalMixtureThresholds(BaseModel, extra="forbid", frozen=True):
    backend_length_termination_rate_max_each_arm: float
    maximum_active_generation_groups_required_each_arm: int
    l0_maximum_unreleased_groups_required_each_arm: int
    l1_pressure_median_min: float
    l1_pressure_replications_exceeding_l0_min: int
    l1_median_normalized_selection_step_promotion_min: float
    l1_replications_at_or_above_minimum: int
    l1_negative_replications_max: int


class DapoOperationalMixturePlan(BaseModel, extra="forbid", frozen=True):
    schema_version: Literal[1]
    analysis_status: Literal["controlled_dapo_operational_mixture_zero_update_shadow"]
    calibration_only: Literal[True]
    confirmatory_eligible: Literal[False]
    population_claim_authorized: Literal[False]
    counterfactual_replay_authorized: Literal[False]
    training_authorized: Literal[False]
    confirmed_protocol_sha256: Sha256Hex
    implementation_confirmation_sha256: Sha256Hex
    reference_execution_confirmation_sha256: Sha256Hex
    reference_result_sha256: Sha256Hex
    final_plan_confirmation_sha256: Sha256Hex
    analysis_code_commit: GitCommitHex
    expected_base_commit: GitCommitHex
    expected_image_sha256: Sha256Hex
    source_design_id: Literal["dapo_math_operational_mixture_v1"]
    model_revision: Literal["b101308fe89651ea5ce025f25317fea6fc07e96e"]
    model_weights_sha256: Literal[
        "70a914a3466bf064ca88ae875cb33c366856147e475a2b8da54e7a3d94d8074a"
    ]
    pools: tuple[DapoOperationalMixturePool, ...]
    arms: tuple[DapoOperationalMixtureArm, ...]
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
    release_delay_seconds: float
    replication_order: tuple[Literal[50001, 50002, 50003], ...]
    thresholds: DapoOperationalMixtureThresholds
    plan_id: Sha256Hex

    @model_validator(mode="after")
    def _validate_protocol(self) -> "DapoOperationalMixturePlan":
        if self.confirmed_protocol_sha256 != CONFIRMED_PROTOCOL_SHA256:
            raise ValueError("plan is not bound to the confirmed protocol")
        expected_bindings = {
            50001: (2026091301, 71001),
            50002: (2026091302, 71002),
            50003: (2026091303, 71003),
        }
        observed = {
            pool.order_seed: (
                pool.scheduler_selection_seed,
                pool.scheduler_generation_seed,
            )
            for pool in self.pools
        }
        if observed != expected_bindings or self.replication_order != (
            50001,
            50002,
            50003,
        ):
            raise ValueError("operational-mixture replication bindings mismatch")
        if tuple(pool.order_seed for pool in self.pools) != self.replication_order:
            raise ValueError("operational-mixture pool order mismatch")
        expected_arms = {
            f"{level}_{sampler}"
            for level in ("l0", "l1", "l3")
            for sampler in ("ready_first", "in_order")
        }
        if len(self.arms) != 6 or {arm.arm_id for arm in self.arms} != expected_arms:
            raise ValueError("operational-mixture plan requires the locked six arms")
        if any(
            len(pool.arm_execution_order) != 6
            or set(pool.arm_execution_order) != expected_arms
            for pool in self.pools
        ):
            raise ValueError("each pool must bind every arm exactly once")
        if (self.temperature, self.top_p, self.release_delay_seconds) != (
            1.0,
            0.7,
            0.0,
        ):
            raise ValueError("operational-mixture generation constants mismatch")
        expected_thresholds = DapoOperationalMixtureThresholds(
            backend_length_termination_rate_max_each_arm=0.2,
            maximum_active_generation_groups_required_each_arm=4,
            l0_maximum_unreleased_groups_required_each_arm=4,
            l1_pressure_median_min=0.25,
            l1_pressure_replications_exceeding_l0_min=2,
            l1_median_normalized_selection_step_promotion_min=1 / 28,
            l1_replications_at_or_above_minimum=2,
            l1_negative_replications_max=1,
        )
        if self.thresholds != expected_thresholds:
            raise ValueError("operational-mixture thresholds mismatch")
        return self

    def arm(self, arm_id: str) -> DapoOperationalMixtureArm:
        matches = [arm for arm in self.arms if arm.arm_id == arm_id]
        if len(matches) != 1:
            raise DapoOperationalMixturePlanError(f"unknown arm {arm_id!r}")
        return matches[0]

    def pool(self, order_seed: int) -> DapoOperationalMixturePool:
        matches = [pool for pool in self.pools if pool.order_seed == order_seed]
        if len(matches) != 1:
            raise DapoOperationalMixturePlanError(f"unknown pool {order_seed}")
        return matches[0]


def canonical_dapo_operational_mixture_plan(plan: DapoOperationalMixturePlan) -> bytes:
    return json.dumps(
        plan.model_dump(mode="json", exclude={"plan_id"}),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def compute_dapo_operational_mixture_plan_id(plan: DapoOperationalMixturePlan) -> str:
    return hashlib.sha256(canonical_dapo_operational_mixture_plan(plan)).hexdigest()


def load_dapo_operational_mixture_plan(path: str | Path) -> DapoOperationalMixturePlan:
    plan_path = Path(path)
    try:
        plan = DapoOperationalMixturePlan.model_validate_json(plan_path.read_bytes())
    except (OSError, ValidationError) as error:
        raise DapoOperationalMixturePlanError(
            f"invalid plan {plan_path}: {error}"
        ) from error
    expected = compute_dapo_operational_mixture_plan_id(plan)
    if plan.plan_id != expected:
        raise DapoOperationalMixturePlanError(
            f"plan ID mismatch: declared={plan.plan_id}, computed={expected}"
        )
    return plan
