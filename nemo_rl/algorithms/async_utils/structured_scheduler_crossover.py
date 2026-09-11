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
DAPO_CONFIRMED_CANDIDATE_SHA256 = (
    "c49f9604225db847eded1d298c592938299fb3c3d508cf60d2b598f1e8b604c7"
)
PRESSURE_RESPONSE_CONFIRMED_CANDIDATE_SHA256 = (
    "c35c06797483c83bd1a29f8cf856447387729e0e612480026eb84320127576cc"
)
PRESSURE_RESPONSE_CONCURRENCY_AMENDMENT_SHA256 = (
    "bcd0977439deb1589ee236893000c9e8c2b6867bd86eec0dc240cc1321bf2488"
)
PRESSURE_RESPONSE_CONCURRENCY_AMENDMENT_CONFIRMATION_SHA256 = (
    "892ccdeccc2e91a11fe1a9f692949c7cc9fddcd63b784ae7b43d3191e6a64197"
)
PRESSURE_RESPONSE_SUPERSEDED_PLAN_ID = (
    "1348a5e1d6b26e0329527652a87c6b161b16d0e3a303709edb5501eaf5cb8e61"
)

PressureLevel: TypeAlias = Literal["l0", "l1", "l3"]
PressureLatencyCondition: TypeAlias = Literal["natural", "controlled_positive_control"]
PressureDelayMapping: TypeAlias = Literal["none", "short_delayed", "long_delayed"]


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
    temperature: float
    top_p: float
    top_k: Literal[20]
    repetition_penalty: float
    max_inflight_prompts: Literal[4]
    max_buffered_rollouts: Literal[8]
    sampler_lookahead_versions: Literal[1]
    release_delay_seconds: float
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
        if (
            self.temperature,
            self.top_p,
            self.repetition_penalty,
            self.release_delay_seconds,
        ) != (0.7, 0.8, 1.0, 0.0):
            raise ValueError("crossover plan generation constants mismatch")
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


class DapoSchedulerCrossoverPool(BaseModel, extra="forbid", frozen=True):
    """Exact identity and paired execution order for one DAPO holdout pool."""

    replication_id: str
    order_seed: Literal[48001, 48002, 48003, 48004]
    dispatch_order_seed: Literal[58001, 58002, 58003, 58004]
    generation_study_seed: Literal[68001, 68002, 68003, 68004]
    arm_execution_order: tuple[CrossoverArmId, CrossoverArmId]
    pool_id: Sha256Hex
    manifest_sha256: Sha256Hex


class DapoSchedulerCrossoverThresholds(BaseModel, extra="forbid", frozen=True):
    """Frozen DAPO scheduler-crossover validity and effect thresholds."""

    backend_length_termination_rate_max_each_arm: float
    distinct_in_order_group_reference_loads_min_each_replication: int
    pooled_in_order_within_pool_spearman_group_mean_tokens_ready_latency_min: float
    ready_latency_p90_p10_ratio_min_pooled_in_order: float
    median_lower_load_normalized_selection_rank_advantage_min: float
    replications_at_or_above_minimum_required: int
    replications_below_zero_max: int
    pooled_in_order_reward_variance_group_fraction_min: float


class DapoSchedulerCrossoverPlan(BaseModel, extra="forbid", frozen=True):
    """Complete hash-addressed execution plan for the DAPO crossover."""

    schema_version: Literal[1]
    analysis_status: Literal["controlled_dapo_natural_latency_scheduler_crossover"]
    calibration_only: Literal[True]
    confirmatory_eligible: Literal[False]
    population_claim_authorized: Literal[False]
    counterfactual_replay_authorized: Literal[False]
    training_authorized: Literal[False]
    confirmed_candidate_sha256: Sha256Hex
    confirmation_record_sha256: Sha256Hex
    analysis_code_commit: GitCommitHex
    expected_base_commit: GitCommitHex
    expected_image_sha256: Sha256Hex
    source_design_id: Literal["dapo_math_scheduler_crossover_v1"]
    model_revision: Literal["b101308fe89651ea5ce025f25317fea6fc07e96e"]
    model_weights_sha256: Literal[
        "70a914a3466bf064ca88ae875cb33c366856147e475a2b8da54e7a3d94d8074a"
    ]
    pools: tuple[DapoSchedulerCrossoverPool, ...]
    arms: tuple[StructuredSchedulerCrossoverArm, ...]
    prompt_groups: Literal[16]
    dispatch_cohorts: Literal[4]
    groups_per_cohort: Literal[4]
    completions_per_group: Literal[16]
    max_total_sequence_length: Literal[6144]
    data_max_input_seq_length: Literal[2048]
    hf_config_override_max_position_embeddings: Literal[6144]
    max_new_tokens: Literal[4096]
    temperature: float
    top_p: float
    max_inflight_prompts: Literal[4]
    max_buffered_rollouts: Literal[8]
    sampler_lookahead_versions: Literal[1]
    release_delay_seconds: float
    replication_order: tuple[Literal[48001, 48002, 48003, 48004], ...]
    thresholds: DapoSchedulerCrossoverThresholds
    plan_id: Sha256Hex

    @model_validator(mode="after")
    def _validate_protocol(self) -> "DapoSchedulerCrossoverPlan":
        if self.confirmed_candidate_sha256 != DAPO_CONFIRMED_CANDIDATE_SHA256:
            raise ValueError("DAPO plan is not bound to the confirmed candidate")
        expected_replications = {
            48001: (58001, 68001, ("ready_first", "in_order")),
            48002: (58002, 68002, ("in_order", "ready_first")),
            48003: (58003, 68003, ("in_order", "ready_first")),
            48004: (58004, 68004, ("ready_first", "in_order")),
        }
        observed = {
            pool.order_seed: (
                pool.dispatch_order_seed,
                pool.generation_study_seed,
                pool.arm_execution_order,
            )
            for pool in self.pools
        }
        if observed != expected_replications:
            raise ValueError("DAPO crossover replication bindings mismatch")
        if tuple(pool.order_seed for pool in self.pools) != self.replication_order or (
            self.replication_order != (48001, 48002, 48003, 48004)
        ):
            raise ValueError("DAPO crossover replication order mismatch")
        if tuple((arm.arm_id, arm.sampler) for arm in self.arms) != (
            ("ready_first", "ready_first"),
            ("in_order", "in_order"),
        ):
            raise ValueError("DAPO crossover requires the locked two arms")
        if (self.temperature, self.top_p, self.release_delay_seconds) != (
            1.0,
            0.7,
            0.0,
        ):
            raise ValueError("DAPO crossover generation constants mismatch")
        expected_thresholds = DapoSchedulerCrossoverThresholds(
            backend_length_termination_rate_max_each_arm=0.2,
            distinct_in_order_group_reference_loads_min_each_replication=12,
            pooled_in_order_within_pool_spearman_group_mean_tokens_ready_latency_min=0.5,
            ready_latency_p90_p10_ratio_min_pooled_in_order=1.5,
            median_lower_load_normalized_selection_rank_advantage_min=1 / 15,
            replications_at_or_above_minimum_required=3,
            replications_below_zero_max=0,
            pooled_in_order_reward_variance_group_fraction_min=0.25,
        )
        if self.thresholds != expected_thresholds:
            raise ValueError("DAPO crossover thresholds mismatch")
        return self

    def arm(self, arm_id: str) -> StructuredSchedulerCrossoverArm:
        matches = [arm for arm in self.arms if arm.arm_id == arm_id]
        if len(matches) != 1:
            raise StructuredSchedulerCrossoverPlanError(
                f"unknown DAPO crossover arm {arm_id!r}"
            )
        return matches[0]

    def pool(self, order_seed: int) -> DapoSchedulerCrossoverPool:
        matches = [pool for pool in self.pools if pool.order_seed == order_seed]
        if len(matches) != 1:
            raise StructuredSchedulerCrossoverPlanError(
                f"unknown DAPO crossover order seed {order_seed}"
            )
        return matches[0]


class SchedulerPressureResponseArm(BaseModel, extra="forbid", frozen=True):
    """One immutable scheduler, admission-pressure, and latency-condition arm."""

    arm_id: str
    sampler: CrossoverArmId
    pressure_level: PressureLevel
    sampler_lookahead_versions: Literal[0, 1, 3]
    max_buffered_rollouts: Literal[4, 8, 16]
    latency_condition: PressureLatencyCondition
    delay_mapping: PressureDelayMapping
    delayed_task: Literal["none", "structured_short", "structured_long"]
    release_delay_seconds: float

    @model_validator(mode="after")
    def _validate_arm(self) -> "SchedulerPressureResponseArm":
        pressure = {"l0": (0, 4), "l1": (1, 8), "l3": (3, 16)}
        if pressure[self.pressure_level] != (
            self.sampler_lookahead_versions,
            self.max_buffered_rollouts,
        ):
            raise ValueError("pressure level does not match lookahead and buffer")
        if self.latency_condition == "natural":
            expected = ("none", "none", 0.0)
        elif self.delay_mapping == "short_delayed":
            expected = ("short_delayed", "structured_short", 30.0)
        elif self.delay_mapping == "long_delayed":
            expected = ("long_delayed", "structured_long", 30.0)
        else:
            raise ValueError("controlled arm requires a delay mapping")
        if (
            self.delay_mapping,
            self.delayed_task,
            self.release_delay_seconds,
        ) != expected:
            raise ValueError("latency condition and delay fields mismatch")
        expected_id = "_".join(
            part
            for part in (
                self.latency_condition,
                self.pressure_level,
                self.delay_mapping if self.delay_mapping != "none" else None,
                self.sampler,
            )
            if part is not None
        )
        if self.arm_id != expected_id:
            raise ValueError(f"pressure-response arm ID must be {expected_id!r}")
        return self


class SchedulerPressureResponsePool(BaseModel, extra="forbid", frozen=True):
    """Immutable identity and ten-arm execution order for one replication."""

    replication_id: str
    order_seed: Literal[49001, 49002, 49003, 49004]
    selection_seed: Literal[2026091001, 2026091002, 2026091003, 2026091004]
    generation_study_seed: Literal[69001, 69002, 69003, 69004]
    arm_execution_order: tuple[str, ...]
    pool_id: Sha256Hex
    manifest_sha256: Sha256Hex


class SchedulerPressureResponseThresholds(BaseModel, extra="forbid", frozen=True):
    backend_length_termination_rate_max_each_stratum_each_arm: float
    reward_mean_min_each_stratum_each_arm: float
    maximum_concurrent_groups_required_each_arm: int | None = None
    maximum_active_generation_groups_required_each_arm: int | None = None
    natural_maximum_unreleased_groups_required_each_arm: int | None = None
    natural_long_short_generated_token_median_ratio_min_each_arm: float
    natural_long_short_ready_latency_median_ratio_min_each_arm: float
    positive_control_undelayed_share_ready_first_min: float
    positive_control_undelayed_share_in_order_required: float
    positive_control_contrast_min_each_mapping: float
    pressure_activation_median_min: float
    pressure_activation_replications_exceeding_l0_min: int
    composition_median_normalized_rank_promotion_min: float
    composition_replications_at_or_above_minimum: int
    composition_negative_replications_max: int


class SchedulerPressureResponsePlan(BaseModel, extra="forbid", frozen=True):
    """Hash-addressed final plan for the zero-update scheduler pressure surface."""

    schema_version: Literal[1, 2]
    analysis_status: Literal[
        "controlled_zero_update_scheduler_pressure_response_surface"
    ]
    calibration_only: Literal[True]
    confirmatory_eligible: Literal[False]
    population_claim_authorized: Literal[False]
    counterfactual_replay_authorized: Literal[False]
    training_authorized: Literal[False]
    confirmed_candidate_sha256: Sha256Hex
    confirmation_record_sha256: Sha256Hex
    confirmed_concurrency_amendment_sha256: Sha256Hex | None = None
    concurrency_amendment_confirmation_sha256: Sha256Hex | None = None
    supersedes_plan_id: Sha256Hex | None = None
    excluded_calibration_order_seed: Literal[49001] | None = None
    analysis_code_commit: GitCommitHex
    expected_base_commit: GitCommitHex
    expected_image_sha256: Sha256Hex
    source_design_id: Literal["structured_scheduler_pressure_response_v1"]
    model_revision: Literal["a09a35458c702b33eeacc393d103063234e8bc28"]
    model_weights_sha256: Literal[
        "456f5eff514d78f7b0ef52a057118046acf15d86606655767db068cabf5f49f7"
    ]
    pools: tuple[SchedulerPressureResponsePool, ...]
    arms: tuple[SchedulerPressureResponseArm, ...]
    prompt_groups: Literal[32]
    dispatch_cohorts: Literal[8]
    groups_per_cohort: Literal[4]
    groups_per_stratum_per_cohort: Literal[2]
    completions_per_group: Literal[2]
    selection_groups_per_step: Literal[4]
    max_inflight_prompts: Literal[4]
    max_total_sequence_length: Literal[768]
    max_new_tokens: Literal[768]
    temperature: float
    top_p: float
    top_k: Literal[20]
    repetition_penalty: float
    replication_order: tuple[Literal[49001, 49002, 49003, 49004], ...]
    thresholds: SchedulerPressureResponseThresholds
    plan_id: Sha256Hex

    @model_validator(mode="after")
    def _validate_protocol(self) -> "SchedulerPressureResponsePlan":
        if (
            self.confirmed_candidate_sha256
            != PRESSURE_RESPONSE_CONFIRMED_CANDIDATE_SHA256
        ):
            raise ValueError(
                "pressure-response plan is not bound to the confirmed candidate"
            )
        bindings = {
            pool.order_seed: (pool.selection_seed, pool.generation_study_seed)
            for pool in self.pools
        }
        expected_bindings = (
            {
                49001: (2026091001, 69001),
                49002: (2026091002, 69002),
                49003: (2026091003, 69003),
            }
            if self.schema_version == 1
            else {
                49002: (2026091002, 69002),
                49003: (2026091003, 69003),
                49004: (2026091004, 69004),
            }
        )
        expected_order = (
            (49001, 49002, 49003) if self.schema_version == 1 else (49002, 49003, 49004)
        )
        if bindings != expected_bindings or self.replication_order != expected_order:
            raise ValueError("pressure-response replication bindings mismatch")
        if tuple(pool.order_seed for pool in self.pools) != self.replication_order:
            raise ValueError("pressure-response pools must follow replication order")
        expected_arm_ids = {
            f"natural_{level}_{sampler}"
            for level in ("l0", "l1", "l3")
            for sampler in ("ready_first", "in_order")
        } | {
            f"controlled_positive_control_l3_{mapping}_{sampler}"
            for mapping in ("short_delayed", "long_delayed")
            for sampler in ("ready_first", "in_order")
        }
        if {arm.arm_id for arm in self.arms} != expected_arm_ids or len(
            self.arms
        ) != 10:
            raise ValueError("pressure-response plan requires the locked ten arms")
        if any(
            len(pool.arm_execution_order) != 10
            or set(pool.arm_execution_order) != expected_arm_ids
            for pool in self.pools
        ):
            raise ValueError("each replication must bind every arm exactly once")
        if (self.temperature, self.top_p, self.repetition_penalty) != (0.7, 0.8, 1.0):
            raise ValueError("pressure-response generation constants mismatch")
        expected_thresholds = SchedulerPressureResponseThresholds(
            backend_length_termination_rate_max_each_stratum_each_arm=0.125,
            reward_mean_min_each_stratum_each_arm=0.75,
            maximum_concurrent_groups_required_each_arm=(
                4 if self.schema_version == 1 else None
            ),
            maximum_active_generation_groups_required_each_arm=(
                None if self.schema_version == 1 else 4
            ),
            natural_maximum_unreleased_groups_required_each_arm=(
                None if self.schema_version == 1 else 4
            ),
            natural_long_short_generated_token_median_ratio_min_each_arm=2.0,
            natural_long_short_ready_latency_median_ratio_min_each_arm=1.5,
            positive_control_undelayed_share_ready_first_min=0.75,
            positive_control_undelayed_share_in_order_required=0.5,
            positive_control_contrast_min_each_mapping=0.25,
            pressure_activation_median_min=0.5,
            pressure_activation_replications_exceeding_l0_min=2,
            composition_median_normalized_rank_promotion_min=1 / 31,
            composition_replications_at_or_above_minimum=2,
            composition_negative_replications_max=1,
        )
        if self.schema_version == 1:
            expected_amendment = (None, None, None, None)
        else:
            expected_amendment = (
                PRESSURE_RESPONSE_CONCURRENCY_AMENDMENT_SHA256,
                PRESSURE_RESPONSE_CONCURRENCY_AMENDMENT_CONFIRMATION_SHA256,
                PRESSURE_RESPONSE_SUPERSEDED_PLAN_ID,
                49001,
            )
        if (
            self.confirmed_concurrency_amendment_sha256,
            self.concurrency_amendment_confirmation_sha256,
            self.supersedes_plan_id,
            self.excluded_calibration_order_seed,
        ) != expected_amendment:
            raise ValueError("pressure-response concurrency amendment binding mismatch")
        if self.thresholds != expected_thresholds:
            raise ValueError("pressure-response thresholds mismatch")
        return self

    def arm(self, arm_id: str) -> SchedulerPressureResponseArm:
        matches = [arm for arm in self.arms if arm.arm_id == arm_id]
        if len(matches) != 1:
            raise StructuredSchedulerCrossoverPlanError(
                f"unknown pressure-response arm {arm_id!r}"
            )
        return matches[0]

    def pool(self, order_seed: int) -> SchedulerPressureResponsePool:
        matches = [pool for pool in self.pools if pool.order_seed == order_seed]
        if len(matches) != 1:
            raise StructuredSchedulerCrossoverPlanError(
                f"unknown pressure-response order seed {order_seed}"
            )
        return matches[0]


SchedulerProtocolPlan: TypeAlias = (
    SchedulerAssayPlan
    | StructuredSchedulerCrossoverPlan
    | DapoSchedulerCrossoverPlan
    | SchedulerPressureResponsePlan
)
SchedulerProtocolArm: TypeAlias = (
    SchedulerAssayArm | StructuredSchedulerCrossoverArm | SchedulerPressureResponseArm
)


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


def canonical_dapo_scheduler_crossover_plan(
    plan: DapoSchedulerCrossoverPlan,
) -> bytes:
    return json.dumps(
        plan.model_dump(mode="json", exclude={"plan_id"}),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def compute_dapo_scheduler_crossover_plan_id(
    plan: DapoSchedulerCrossoverPlan,
) -> str:
    return hashlib.sha256(canonical_dapo_scheduler_crossover_plan(plan)).hexdigest()


def load_dapo_scheduler_crossover_plan(
    path: str | Path,
) -> DapoSchedulerCrossoverPlan:
    plan_path = Path(path)
    try:
        plan = DapoSchedulerCrossoverPlan.model_validate_json(plan_path.read_bytes())
    except (OSError, ValidationError) as error:
        raise StructuredSchedulerCrossoverPlanError(
            f"invalid DAPO crossover plan {plan_path}: {error}"
        ) from error
    expected_id = compute_dapo_scheduler_crossover_plan_id(plan)
    if plan.plan_id != expected_id:
        raise StructuredSchedulerCrossoverPlanError(
            f"DAPO crossover plan ID mismatch: declared={plan.plan_id}, computed={expected_id}"
        )
    return plan


def canonical_scheduler_pressure_response_plan(
    plan: SchedulerPressureResponsePlan,
) -> bytes:
    return json.dumps(
        plan.model_dump(mode="json", exclude={"plan_id"}, exclude_none=True),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def compute_scheduler_pressure_response_plan_id(
    plan: SchedulerPressureResponsePlan,
) -> str:
    return hashlib.sha256(canonical_scheduler_pressure_response_plan(plan)).hexdigest()


def load_scheduler_pressure_response_plan(
    path: str | Path,
) -> SchedulerPressureResponsePlan:
    plan_path = Path(path)
    try:
        plan = SchedulerPressureResponsePlan.model_validate_json(plan_path.read_bytes())
    except (OSError, ValidationError) as error:
        raise StructuredSchedulerCrossoverPlanError(
            f"invalid pressure-response plan {plan_path}: {error}"
        ) from error
    expected_id = compute_scheduler_pressure_response_plan_id(plan)
    if plan.plan_id != expected_id:
        raise StructuredSchedulerCrossoverPlanError(
            f"pressure-response plan ID mismatch: declared={plan.plan_id}, computed={expected_id}"
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
    if (
        value.get("analysis_status")
        == "controlled_dapo_natural_latency_scheduler_crossover"
    ):
        return load_dapo_scheduler_crossover_plan(plan_path)
    if (
        value.get("analysis_status")
        == "controlled_zero_update_scheduler_pressure_response_surface"
    ):
        return load_scheduler_pressure_response_plan(plan_path)
    raise StructuredSchedulerCrossoverPlanError("unsupported scheduler protocol type")
