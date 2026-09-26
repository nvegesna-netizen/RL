# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Freeze exploratory cadence and prompt classes from two neutral collections."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median

from nemo_rl.algorithms.async_utils.fixed_pool import (
    load_fixed_pool_manifest,
    validate_dapo_paced_exposure_manifest_design,
    validate_fixed_pool_materialization,
    validate_fixed_pool_trace,
)
from nemo_rl.algorithms.async_utils.paced_exposure import (
    PacedExposureCalibration,
    PacedExposureExecutionSpec,
    PacedExposurePlan,
    compute_paced_exposure_plan_id,
    validate_paced_plan_execution_spec,
)
from nemo_rl.algorithms.async_utils.scheduler_trace import (
    SchedulerEventType,
    iter_scheduler_trace,
)


def freeze_calibration(
    *, materialization_root: Path, trace_paths: tuple[Path, Path], output_dir: Path
) -> PacedExposurePlan:
    """Validate complete neutral inputs and publish a new private frozen plan.

    Trace arguments must follow precommitted calibration seed order. Existing
    output is never overwritten. This function performs no sampling or rollout.
    """
    spec_raw = (materialization_root / "execution_spec.v1.json").read_bytes()
    spec = PacedExposureExecutionSpec.model_validate_json(spec_raw)
    spec_sha = hashlib.sha256(spec_raw).hexdigest()
    manifest = load_fixed_pool_manifest(
        materialization_root / f"fixed_pool_manifest.v1.{spec.selection_seed}.json"
    )
    validate_fixed_pool_materialization(manifest)
    validate_dapo_paced_exposure_manifest_design(manifest)
    draws: list[PacedExposureCalibration] = []
    token_totals = [0.0] * 64
    reward_totals = [0.0] * 64
    for trace_path, seed in zip(
        trace_paths, spec.calibration_generation_seeds, strict=True
    ):
        validate_fixed_pool_trace(
            trace_path, manifest, expected_completions_per_group=16
        )
        events = tuple(iter_scheduler_trace(trace_path))
        runtime = events[0].scalar_summaries
        expected_runtime = {
            "fixed_pool_design_id": spec.source_design_id,
            "generation_study_seed": seed,
            "generation_backend": "vllm",
            "configured_max_new_tokens": 4096,
            "max_total_sequence_length": 6144,
            "generation_temperature": 1.0,
            "generation_top_p": 0.7,
            "generation_ignore_eos": False,
            "num_generations_per_prompt": 16,
            "num_prompts_per_step": 4,
            "max_inflight_prompts": 4,
            "max_buffered_rollouts": 16,
        }
        if any(runtime.get(key) != value for key, value in expected_runtime.items()):
            raise ValueError("neutral calibration runtime does not match paced design")
        completed = [
            event
            for event in events
            if event.event_type is SchedulerEventType.ROLLOUT_COMPLETED
        ]
        dispatches = [
            event
            for event in events
            if event.event_type is SchedulerEventType.ATTEMPT_DISPATCHED
        ]
        if len(completed) != 64 or len(dispatches) != 64:
            raise ValueError(
                "calibration requires exactly 64 dispatched/completed groups"
            )
        origin = dispatches[0].monotonic_ns
        clocks = sorted(event.monotonic_ns for event in completed)
        draws.append(
            PacedExposureCalibration(
                generation_seed=seed,
                trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                completion_16_seconds=(clocks[15] - origin) / 1e9,
                completion_48_seconds=(clocks[47] - origin) / 1e9,
            )
        )
        for event in completed:
            ordinal = event.source_pool_ordinal
            if ordinal is None or not 0 <= ordinal < 64:
                raise ValueError("calibration source ordinal missing")
            values = []
            for field in ("mean_gen_tokens_per_sample", "reward_mean"):
                value = event.scalar_summaries.get(field)
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                ):
                    raise ValueError("calibration token/reward summary is invalid")
                values.append(float(value))
            if not 0 <= values[0] <= 4096 or not 0 <= values[1] <= 1:
                raise ValueError(
                    "calibration token/reward summary outside expected range"
                )
            token_totals[ordinal] += values[0]
            reward_totals[ordinal] += values[1]
    lower = sorted(range(64), key=lambda ordinal: (token_totals[ordinal], ordinal))[:32]
    harder = sorted(range(64), key=lambda ordinal: (reward_totals[ordinal], ordinal))[
        :32
    ]
    cadence = median(draw.batch_seconds for draw in draws)
    private_calibration = {
        "schema_version": 1,
        "execution_spec_sha256": spec_sha,
        "manifest_sha256": manifest.manifest_sha256,
        "draws": [draw.model_dump(mode="json") for draw in draws],
        "mean_tokens_by_source_ordinal": [value / 2 for value in token_totals],
        "mean_reward_by_source_ordinal": [value / 2 for value in reward_totals],
        "lower_load_ordinals": lower,
        "harder_ordinals": harder,
        "cadence_seconds": cadence,
    }
    calibration_raw = (
        json.dumps(private_calibration, sort_keys=True, indent=2) + "\n"
    ).encode()
    arm_geometry = (
        ("in_order_half", "in_order", 0.5, 0),
        ("ready_first_half", "ready_first", 0.5, 0),
        ("ready_first_one", "ready_first", 1.0, 1),
        ("in_order_one", "in_order", 1.0, 1),
        ("in_order_two", "in_order", 2.0, 2),
        ("ready_first_two", "ready_first", 2.0, 2),
    )
    plan = PacedExposurePlan.model_validate(
        {
            "schema_version": 1,
            "analysis_status": "exploratory_paced_zero_update_exposure",
            "confirmatory_eligible": False,
            "population_claim_authorized": False,
            "training_authorized": False,
            "source_design_id": spec.source_design_id,
            "execution_spec_sha256": spec_sha,
            "analysis_code_commit": spec.implementation_commit,
            "expected_base_commit": "ae07eafe8035b5b2e84efa7234e70e7fd7e493c1",
            "expected_image_sha256": "3df8114a0b3e60ef95ce13f8b982c7cc45d164aca71c63a87388b1e8434ee470",
            "model_revision": manifest.model_revision,
            "model_weights_sha256": manifest.model_weights_sha256,
            "dataset_revision": "65877096c24ffa7abc4e4fa5edb95cf3413a5674",
            "pools": [
                {
                    "order_seed": spec.selection_seed,
                    "pool_id": manifest.pool_id,
                    "manifest_sha256": manifest.manifest_sha256,
                    "private_calibration_manifest_sha256": hashlib.sha256(
                        calibration_raw
                    ).hexdigest(),
                    "prior_identity_exclusions_sha256": spec.prior_identity_ledger_sha256,
                    "prior_seed_ledger_sha256": spec.prior_seed_ledger_sha256,
                    "source_prompt_ids": [
                        item.source_prompt_id for item in manifest.items
                    ],
                    "fixed_lower_load_prompt_ids": [
                        manifest.items[index].source_prompt_id for index in lower
                    ],
                    "fixed_harder_prompt_ids": [
                        manifest.items[index].source_prompt_id for index in harder
                    ],
                }
            ],
            "calibration_draws": draws,
            "cadence_seconds": cadence,
            "arms": [
                {
                    "arm_id": name,
                    "sampler": sampler,
                    "cadence_multiplier": factor,
                    "consumer_duration_seconds": cadence * factor,
                    "generation_seed": spec.cadence_pair_generation_seeds[pair],
                    "sampler_lookahead_versions": 3,
                    "max_buffered_rollouts": 16,
                }
                for name, sampler, factor, pair in arm_geometry
            ],
            "prompt_groups": 64,
            "completions_per_group": 16,
            "selection_groups_per_step": 4,
            "selection_steps": 16,
            "max_inflight_prompts": 4,
            "max_total_sequence_length": 6144,
            "data_max_input_seq_length": 2048,
            "hf_config_override_max_position_embeddings": 6144,
            "max_new_tokens": 4096,
            "temperature": 1.0,
            "top_p": 0.7,
            "maximum_completions": 8192,
            "scientific_allocation_seconds": 7200,
            "allocated_exclusive_nodes": 1,
            "application_gpus": 2,
            "plan_id": "0" * 64,
        }
    )
    validate_paced_plan_execution_spec(plan, spec)
    plan = plan.model_copy(update={"plan_id": compute_paced_exposure_plan_id(plan)})
    output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    for name, raw in (
        ("private_calibration_manifest.v1.json", calibration_raw),
        (
            "paced_exposure_plan.v1.json",
            (plan.model_dump_json(indent=2) + "\n").encode(),
        ),
    ):
        with os.fdopen(
            os.open(output_dir / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
            "wb",
        ) as stream:
            stream.write(raw)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialization-root", type=Path, required=True)
    parser.add_argument(
        "--calibration-trace", type=Path, action="append", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.calibration_trace) != 2:
        parser.error("exactly two calibration traces are required in frozen seed order")
    freeze_calibration(
        materialization_root=args.materialization_root,
        trace_paths=tuple(args.calibration_trace),
        output_dir=args.output_dir,
    )
    print("paced_calibration_freeze=PASS")


if __name__ == "__main__":
    main()
