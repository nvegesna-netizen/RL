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

"""Independent, content-addressed contract for the exploratory paced pilot.

This contract validates internal consistency, not external authorization or
artifact provenance. The launcher must verify bound files and prior seed and
identity exclusions before generating anything. No Stage 0 confirmation applies.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, model_validator

Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
GitCommitHex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
Seed = Annotated[int, Field(strict=True, ge=0, le=2147483647)]
PositiveSeconds = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class PacedExposureCalibration(BaseModel, extra="forbid", frozen=True):
    generation_seed: Seed
    trace_sha256: Sha256Hex
    completion_16_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    completion_48_seconds: PositiveSeconds

    @model_validator(mode="after")
    def _ordered_completion_times(self) -> PacedExposureCalibration:
        if self.completion_48_seconds <= self.completion_16_seconds:
            raise ValueError("48th completion must follow 16th completion")
        return self

    @property
    def batch_seconds(self) -> float:
        return (self.completion_48_seconds - self.completion_16_seconds) / 8


class PacedExposureArm(BaseModel, extra="forbid", frozen=True):
    arm_id: str
    sampler: Literal["in_order", "ready_first"]
    cadence_multiplier: PositiveSeconds
    consumer_duration_seconds: PositiveSeconds
    generation_seed: Seed
    sampler_lookahead_versions: Literal[3]
    max_buffered_rollouts: Literal[16]


class PacedExposurePool(BaseModel, extra="forbid", frozen=True):
    order_seed: Seed
    pool_id: Sha256Hex
    manifest_sha256: Sha256Hex
    private_calibration_manifest_sha256: Sha256Hex
    prior_identity_exclusions_sha256: Sha256Hex
    prior_seed_ledger_sha256: Sha256Hex
    # Ordered identities define the common source; halves are calibration-only.
    source_prompt_ids: tuple[Sha256Hex, ...]
    fixed_lower_load_prompt_ids: tuple[Sha256Hex, ...]
    fixed_harder_prompt_ids: tuple[Sha256Hex, ...]

    @model_validator(mode="after")
    def _unique_inventory(self) -> PacedExposurePool:
        inventory = set(self.source_prompt_ids)
        if len(self.source_prompt_ids) != 64 or len(inventory) != 64:
            raise ValueError("paced pilot requires exactly 64 unique prompts")
        for half in (self.fixed_lower_load_prompt_ids, self.fixed_harder_prompt_ids):
            if len(half) != 32 or len(set(half)) != 32 or not set(half) <= inventory:
                raise ValueError("calibration halves must contain 32 pool identities")
        return self


class PacedExposurePlan(BaseModel, extra="forbid", frozen=True):
    schema_version: Literal[1]
    analysis_status: Literal["exploratory_paced_zero_update_exposure"]
    confirmatory_eligible: Literal[False]
    population_claim_authorized: Literal[False]
    training_authorized: Literal[False]
    source_design_id: Literal["dapo_math_paced_exposure_v1"]
    analysis_code_commit: GitCommitHex
    expected_base_commit: Literal["ae07eafe8035b5b2e84efa7234e70e7fd7e493c1"]
    expected_image_sha256: Literal[
        "3df8114a0b3e60ef95ce13f8b982c7cc45d164aca71c63a87388b1e8434ee470"
    ]
    model_revision: Literal["b101308fe89651ea5ce025f25317fea6fc07e96e"]
    model_weights_sha256: Literal[
        "70a914a3466bf064ca88ae875cb33c366856147e475a2b8da54e7a3d94d8074a"
    ]
    dataset_revision: Literal["65877096c24ffa7abc4e4fa5edb95cf3413a5674"]
    pools: tuple[PacedExposurePool]
    calibration_draws: tuple[PacedExposureCalibration, PacedExposureCalibration]
    cadence_seconds: PositiveSeconds
    arms: tuple[PacedExposureArm, ...]
    prompt_groups: Literal[64]
    completions_per_group: Literal[16]
    selection_groups_per_step: Literal[4]
    selection_steps: Literal[16]
    max_inflight_prompts: Literal[4]
    max_total_sequence_length: Literal[6144]
    data_max_input_seq_length: Literal[2048]
    hf_config_override_max_position_embeddings: Literal[6144]
    max_new_tokens: Literal[4096]
    temperature: Annotated[float, Field(ge=1.0, le=1.0, allow_inf_nan=False)]
    top_p: Annotated[float, Field(ge=0.7, le=0.7, allow_inf_nan=False)]
    maximum_completions: Literal[8192]
    scientific_allocation_seconds: Literal[7200]
    allocated_gpus: Literal[2]
    plan_id: Sha256Hex

    @model_validator(mode="after")
    def _validate_design(self) -> PacedExposurePlan:
        cadence = median(draw.batch_seconds for draw in self.calibration_draws)
        if not math.isclose(self.cadence_seconds, cadence, rel_tol=1e-12, abs_tol=0):
            raise ValueError("cadence does not match neutral completion clocks")
        expected = (
            ("in_order_half", "in_order", 0.5),
            ("ready_first_half", "ready_first", 0.5),
            ("ready_first_one", "ready_first", 1.0),
            ("in_order_one", "in_order", 1.0),
            ("in_order_two", "in_order", 2.0),
            ("ready_first_two", "ready_first", 2.0),
        )
        observed = tuple(
            (arm.arm_id, arm.sampler, arm.cadence_multiplier) for arm in self.arms
        )
        if observed != expected:
            raise ValueError("paced pilot arm order or design mismatch")
        for arm in self.arms:
            if not math.isclose(
                arm.consumer_duration_seconds,
                cadence * arm.cadence_multiplier,
                rel_tol=1e-12,
                abs_tol=0,
            ):
                raise ValueError("arm duration does not match frozen cadence")
        calibration_seeds = {draw.generation_seed for draw in self.calibration_draws}
        live_seeds = {arm.generation_seed for arm in self.arms}
        if len(calibration_seeds) != 2 or calibration_seeds & live_seeds:
            raise ValueError(
                "calibration seeds must be distinct and unused by live arms"
            )
        # Paired arms use common generation seeds; cadences use distinct seeds.
        pairs = ((0, 1), (2, 3), (4, 5))
        if len(live_seeds) != 3 or any(
            self.arms[left].generation_seed != self.arms[right].generation_seed
            for left, right in pairs
        ):
            raise ValueError("freeze one distinct generation seed per cadence pair")
        if self.pools[0].order_seed in calibration_seeds | live_seeds:
            raise ValueError("source-order seed must be separate from generation seeds")
        return self

    def arm(self, arm_id: str) -> PacedExposureArm:
        for arm in self.arms:
            if arm.arm_id == arm_id:
                return arm
        raise ValueError("unknown paced-exposure arm")

    def pool(self, order_seed: int) -> PacedExposurePool:
        if self.pools[0].order_seed != order_seed:
            raise ValueError("unknown paced-exposure pool")
        return self.pools[0]


def compute_paced_exposure_plan_id(plan: PacedExposurePlan) -> str:
    """Hash all execution fields except the self-addressing ID."""
    payload = json.dumps(
        plan.model_dump(mode="json", exclude={"plan_id"}),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def load_paced_exposure_plan(path: str | Path) -> PacedExposurePlan:
    """Load a self-consistent plan; launcher verifies external bound artifacts."""
    plan = PacedExposurePlan.model_validate_json(Path(path).read_bytes())
    if plan.plan_id != compute_paced_exposure_plan_id(plan):
        raise ValueError("paced-exposure plan ID mismatch")
    return plan
