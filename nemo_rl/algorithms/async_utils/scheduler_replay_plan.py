# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Strict, hash-addressed plans for fixed-pool scheduler replay."""

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

from nemo_rl.algorithms.async_utils.scheduler_replay import (
    ReplayGroup,
    ReplayPolicy,
    ReplayTick,
    SchedulerReplayError,
)


Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
GitCommitHex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
RunIdentifier = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
]
PositiveStrictInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeStrictInt = Annotated[int, Field(strict=True, ge=0)]
PositiveFiniteFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
UnitFiniteFloat = Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)]
NonNegativeFiniteFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class ReplayPlanError(SchedulerReplayError):
    """A replay plan is malformed or does not match its source artifacts."""


class ObservedCalibrationResult(BaseModel, extra="forbid", frozen=True):
    """Facts inspected before this explicitly post-observation plan was frozen."""

    run_id: RunIdentifier
    median_latency_ratio: PositiveFiniteFloat
    rank_biserial: UnitFiniteFloat


class ReplaySourceRun(BaseModel, extra="forbid", frozen=True):
    """Immutable identity of one source collection."""

    run_id: RunIdentifier
    pool_id: Sha256Hex
    manifest_sha256: Sha256Hex
    trace_sha256: Sha256Hex
    validation_sha256: Sha256Hex
    order_seed: NonNegativeStrictInt
    generation_seed: NonNegativeStrictInt


class ReplayPolicyPlan(BaseModel, extra="forbid", frozen=True):
    """One fixed-admission scheduler kernel in the closure surface."""

    label: str
    kernel: Literal["in_order", "ready_first", "windowed"]
    max_staleness_versions: NonNegativeStrictInt = 0
    sample_freshest_first: bool = False

    def to_policy(self) -> ReplayPolicy:
        """Convert the serialized policy to the pure replay kernel."""
        return ReplayPolicy(
            name=self.kernel,
            max_staleness_versions=self.max_staleness_versions,
            sample_freshest_first=self.sample_freshest_first,
        )


class ReplayTickPlan(BaseModel, extra="forbid", frozen=True):
    """Fixed formula for dispatch-anchored cohort deadline ticks."""

    anchor: Literal["max_dispatch_in_cohort"]
    min_prompt_groups: PositiveStrictInt
    max_prompt_groups: PositiveStrictInt

    @model_validator(mode="after")
    def _validate_bounds(self) -> "ReplayTickPlan":
        if self.max_prompt_groups < self.min_prompt_groups:
            raise ValueError("max_prompt_groups must be at least min_prompt_groups")
        return self


class ReleaseInterventionPlan(BaseModel, extra="forbid", frozen=True):
    """Version-independent unrestricted blockwise release permutations."""

    method: Literal["sha256_sorted_unrestricted_block_permutation_v1"]
    seed_start: NonNegativeStrictInt
    seed_count: PositiveStrictInt
    block: Literal["dispatch_cohort_and_decorrelation_block"]
    identity_assignments_allowed: Literal[True]

    @property
    def seeds(self) -> range:
        """Return the exact contiguous seed sequence."""
        return range(self.seed_start, self.seed_start + self.seed_count)


class ReplayStopRules(BaseModel, extra="forbid", frozen=True):
    """Precommitted closure thresholds; these are not inferential tests."""

    primary_horizon: PositiveStrictInt
    adjacent_deadline_scenarios_required: PositiveStrictInt
    minimum_absolute_tv_interaction: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ]
    maximum_tv_interaction_mcse: NonNegativeFiniteFloat
    require_same_direction_all_runs: bool


class ExploratoryClosurePlan(BaseModel, extra="forbid", frozen=True):
    """Complete immutable specification for a calibration-only replay."""

    schema_version: Literal[1]
    analysis_status: Literal["exploratory_post_observation"]
    calibration_only: Literal[True]
    confirmatory_eligible: Literal[False]
    semantics: Literal["fixed_tick_target_cohort_deadline_ablation_v1"]
    analysis_code_commit: GitCommitHex
    expected_completions_per_group: Literal[2]
    observed_before_plan: tuple[ObservedCalibrationResult, ...]
    source_runs: tuple[ReplaySourceRun, ...]
    deadlines_ns: tuple[PositiveStrictInt, ...]
    tick_plan: ReplayTickPlan
    policies: tuple[ReplayPolicyPlan, ...]
    release_intervention: ReleaseInterventionPlan
    horizons: tuple[PositiveStrictInt, ...]
    stop_rules: ReplayStopRules
    plan_id: Sha256Hex

    @model_validator(mode="after")
    def _validate_design(self) -> "ExploratoryClosurePlan":
        run_ids = [run.run_id for run in self.source_runs]
        if not run_ids or len(run_ids) != len(set(run_ids)):
            raise ValueError("source run IDs must be nonempty and unique")
        observed_run_ids = [item.run_id for item in self.observed_before_plan]
        if len(observed_run_ids) != len(set(observed_run_ids)) or set(
            observed_run_ids
        ) != set(run_ids):
            raise ValueError("observed calibration results must cover every source run")
        if self.deadlines_ns != (
            250_000_000,
            500_000_000,
            1_000_000_000,
            2_000_000_000,
            4_000_000_000,
        ):
            raise ValueError("schema v1 requires the locked five-deadline surface")
        if self.horizons != (4, 12, 24, 48):
            raise ValueError("schema v1 requires the locked batch-aligned horizons")
        labels = [policy.label for policy in self.policies]
        if len(labels) != len(set(labels)):
            raise ValueError("policy labels must be unique")
        policy_surface = tuple(
            (
                policy.label,
                policy.kernel,
                policy.max_staleness_versions,
                policy.sample_freshest_first,
            )
            for policy in self.policies
        )
        if policy_surface != (
            ("target_cohort_deadline", "in_order", 0, False),
            ("ready_first_fixed_admission", "ready_first", 0, False),
            ("windowed_fifo_w1_fixed_admission", "windowed", 1, False),
        ):
            raise ValueError("schema v1 requires the locked three-policy surface")
        if (
            self.tick_plan.min_prompt_groups != 4
            or self.tick_plan.max_prompt_groups != 4
        ):
            raise ValueError("schema v1 requires fixed four-group deadline ticks")
        if (
            self.release_intervention.seed_start != 0
            or self.release_intervention.seed_count != 1000
        ):
            raise ValueError("schema v1 requires release seeds 0 through 999")
        if self.stop_rules != ReplayStopRules(
            primary_horizon=12,
            adjacent_deadline_scenarios_required=2,
            minimum_absolute_tv_interaction=0.02,
            maximum_tv_interaction_mcse=0.002,
            require_same_direction_all_runs=True,
        ):
            raise ValueError("schema v1 requires the locked closure stop rules")
        return self


def canonical_plan_payload(plan: ExploratoryClosurePlan) -> bytes:
    """Serialize all plan fields except its self-addressing identifier."""
    payload = plan.model_dump(mode="json", exclude={"plan_id"})
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def compute_plan_id(plan: ExploratoryClosurePlan) -> str:
    """Compute the SHA-256 identity of a plan's canonical payload."""
    return hashlib.sha256(canonical_plan_payload(plan)).hexdigest()


def load_exploratory_closure_plan(path: str | Path) -> ExploratoryClosurePlan:
    """Load a strict plan and verify its self-addressing identifier."""
    plan_path = Path(path)
    try:
        raw = plan_path.read_bytes()
    except OSError as error:
        raise ReplayPlanError(
            f"cannot read replay plan {plan_path}: {error}"
        ) from error
    try:
        plan = ExploratoryClosurePlan.model_validate_json(raw)
    except ValidationError as error:
        raise ReplayPlanError(f"invalid replay plan {plan_path}: {error}") from error
    expected = compute_plan_id(plan)
    if plan.plan_id != expected:
        raise ReplayPlanError(
            f"replay plan ID mismatch: declared={plan.plan_id}, computed={expected}"
        )
    return plan


def build_deadline_ticks(
    groups: tuple[ReplayGroup, ...],
    *,
    deadline_ns: int,
    tick_plan: ReplayTickPlan,
) -> tuple[ReplayTick, ...]:
    """Build one exogenous deadline tick from each cohort's last dispatch."""
    if deadline_ns <= 0:
        raise ReplayPlanError("deadline_ns must be positive")
    dispatch_by_cohort: dict[int, list[int]] = {}
    for group in groups:
        cohort = group.nominal_start_version
        dispatch_by_cohort.setdefault(cohort, []).append(group.dispatch_ns)
    cohorts = sorted(dispatch_by_cohort)
    if cohorts != list(range(len(cohorts))):
        raise ReplayPlanError("dispatch cohorts must be contiguous from zero")
    ticks = tuple(
        ReplayTick(
            tick_id=cohort,
            monotonic_ns=max(dispatch_by_cohort[cohort]) + deadline_ns,
            nominal_trainer_version=cohort,
            min_prompt_groups=tick_plan.min_prompt_groups,
            max_prompt_groups=tick_plan.max_prompt_groups,
        )
        for cohort in cohorts
    )
    if any(
        right.monotonic_ns <= left.monotonic_ns for left, right in zip(ticks, ticks[1:])
    ):
        raise ReplayPlanError(
            "derived cohort deadline ticks are not strictly increasing"
        )
    return ticks
