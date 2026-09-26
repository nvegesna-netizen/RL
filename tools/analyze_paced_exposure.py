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

"""Aggregate finite-window paced exposure without exporting group identities."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean

from nemo_rl.algorithms.async_utils.paced_exposure import (
    PacedExposurePlan,
    load_paced_exposure_plan,
)
from nemo_rl.algorithms.async_utils.scheduler_trace import (
    SchedulerEventType,
    SchedulerTraceEvent,
    iter_scheduler_trace,
    validate_paced_consumer_trace,
)


@dataclass(frozen=True)
class SelectedGroup:
    """Analysis-only values; identifiers are deliberately absent."""

    selected_seconds: float
    lower_load: bool
    harder: bool
    reward: float
    tokens: float
    nonconstant_reward: bool
    admission_age_steps: int
    ready_wait_seconds: float


def summarize(groups: list[SelectedGroup]) -> dict[str, int | float | None]:
    """Describe an actual selected inventory, with explicit empty checkpoints."""
    if not groups:
        return {
            "selected_groups": 0,
            **{
                key: None
                for key in (
                    "lower_load_share",
                    "harder_share",
                    "mean_reward",
                    "mean_tokens",
                    "nonconstant_reward_share",
                    "mean_admission_age_steps",
                    "mean_ready_wait_seconds",
                )
            },
        }
    return {
        "selected_groups": len(groups),
        "lower_load_share": fmean(group.lower_load for group in groups),
        "harder_share": fmean(group.harder for group in groups),
        "mean_reward": fmean(group.reward for group in groups),
        "mean_tokens": fmean(group.tokens for group in groups),
        "nonconstant_reward_share": fmean(group.nonconstant_reward for group in groups),
        "mean_admission_age_steps": fmean(
            group.admission_age_steps for group in groups
        ),
        "mean_ready_wait_seconds": fmean(group.ready_wait_seconds for group in groups),
    }


def number(event: SchedulerTraceEvent, key: str) -> float:
    value = event.scalar_summaries.get(key)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"invalid numeric trace summary: {key}")
    return float(value)


def analyze_arm(
    plan: PacedExposurePlan, *, arm_id: str, trace_path: Path
) -> dict[str, object]:
    """Analyze one fully reconciled arm; incomplete arms are not scientific zeros."""
    arm = plan.arm(arm_id)
    pool = plan.pools[0]
    validate_paced_consumer_trace(
        trace_path, consumer_seconds=arm.consumer_duration_seconds
    )
    events = tuple(iter_scheduler_trace(trace_path))
    start = events[0]
    if (
        start.pool_id != pool.pool_id
        or start.pool_manifest_sha256 != pool.manifest_sha256
        or start.model_revision != plan.model_revision
        or start.model_weights_sha256 != plan.model_weights_sha256
        or start.run_mode != "scheduler_assay"
        or start.sampler_name != arm.sampler
    ):
        raise ValueError("paced trace source or sampler does not match frozen plan")
    expected = {
        "scheduler_assay_plan_id": plan.plan_id,
        "scheduler_assay_arm_id": arm_id,
        "generation_study_seed": arm.generation_seed,
        "fixed_pool_design_id": plan.source_design_id,
        "max_inflight_prompts": 4,
        "max_buffered_rollouts": 16,
        "num_generations_per_prompt": 16,
        "num_prompts_per_step": 4,
        "configured_max_new_tokens": 4096,
        "max_total_sequence_length": 6144,
        "generation_temperature": 1.0,
        "generation_top_p": 0.7,
        "generation_backend": "vllm",
        "generation_ignore_eos": False,
    }
    if any(start.scalar_summaries.get(key) != value for key, value in expected.items()):
        raise ValueError("paced runtime summaries do not match plan")
    first_dispatch = next(
        event.monotonic_ns
        for event in events
        if event.event_type is SchedulerEventType.ATTEMPT_DISPATCHED
    )
    lower = set(pool.fixed_lower_load_prompt_ids)
    harder = set(pool.fixed_harder_prompt_ids)
    admission_steps: dict[str, int] = {}
    dispatches: dict[str, SchedulerTraceEvent] = {}
    completions: dict[str, SchedulerTraceEvent] = {}
    ready_times: dict[str, int] = {}
    selected: list[SelectedGroup] = []
    source_order: list[str] = []
    step = 0
    consumer_seconds: list[float] = []
    drain_ns = 0
    for event in events:
        kind = event.event_type
        if kind is SchedulerEventType.ADMISSION_GRANTED:
            assert event.admission_id is not None
            admission_steps[event.admission_id] = step
        elif kind is SchedulerEventType.ATTEMPT_DISPATCHED:
            assert (
                event.logical_group_id is not None
                and event.source_prompt_id is not None
            )
            dispatches[event.logical_group_id] = event
            source_order.append(event.source_prompt_id)
        elif kind is SchedulerEventType.ROLLOUT_COMPLETED:
            assert event.logical_group_id is not None
            completions[event.logical_group_id] = event
        elif kind is SchedulerEventType.GROUP_READY:
            assert event.logical_group_id is not None
            ready_times[event.logical_group_id] = event.monotonic_ns
        elif kind is SchedulerEventType.SELECT_DECISION:
            for group_id in event.selected_logical_group_ids:
                dispatch = dispatches[group_id]
                completion = completions[group_id]
                assert dispatch.admission_id is not None
                reward = number(completion, "reward_mean")
                reward_min = number(completion, "reward_min")
                reward_max = number(completion, "reward_max")
                tokens = number(completion, "mean_gen_tokens_per_sample")
                if (
                    number(completion, "completion_count") != 16
                    or not 0 <= reward_min <= reward <= reward_max <= 1
                    or not 0 <= tokens <= 4096
                ):
                    raise ValueError("paced reward/token/count summaries are invalid")
                selected.append(
                    SelectedGroup(
                        selected_seconds=(event.monotonic_ns - first_dispatch) / 1e9,
                        lower_load=dispatch.source_prompt_id in lower,
                        harder=dispatch.source_prompt_id in harder,
                        reward=reward,
                        tokens=tokens,
                        nonconstant_reward=reward_max > reward_min,
                        admission_age_steps=step
                        - admission_steps[dispatch.admission_id],
                        ready_wait_seconds=(event.monotonic_ns - ready_times[group_id])
                        / 1e9,
                    )
                )
        elif kind is SchedulerEventType.CONSUMER_COMPLETED:
            consumer_seconds.append(number(event, "actual_consumer_seconds"))
            step += 1
        elif kind is SchedulerEventType.CONSUMER_DRAINED:
            drain_ns = event.monotonic_ns
    if tuple(source_order) != pool.source_prompt_ids or len(selected) != 64:
        raise ValueError("paced arm did not preserve the full frozen source order")
    elapsed = (drain_ns - first_dispatch) / 1e9
    if elapsed <= 0:
        raise ValueError("paced arm has nonpositive runtime")
    return {
        "arm_id": arm_id,
        "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        "sampler": arm.sampler,
        "cadence_multiplier": arm.cadence_multiplier,
        "requested_consumer_seconds": arm.consumer_duration_seconds,
        "actual_consumer_seconds_by_step": consumer_seconds,
        "first_dispatch_to_drain_seconds": elapsed,
        "groups_per_second_to_drain": 64 / elapsed,
        "cumulative_prefixes": [
            summarize(selected[:count]) for count in range(4, 65, 4)
        ],
        "rolling_16_group_windows": [
            {
                "first_selected_group_index": offset,
                **summarize(selected[offset : offset + 16]),
            }
            for offset in range(0, 49, 4)
        ],
        "disjoint_16_group_blocks": [
            summarize(selected[offset : offset + 16]) for offset in range(0, 64, 16)
        ],
        "wall_checkpoints": [
            {
                "seconds_from_first_dispatch": multiplier * plan.cadence_seconds,
                **summarize(
                    [
                        group
                        for group in selected
                        if group.selected_seconds <= multiplier * plan.cadence_seconds
                    ]
                ),
            }
            for multiplier in (4, 8, 12)
        ],
        "final_drain": summarize(selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--arms-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = load_paced_exposure_plan(args.plan)
    result = {
        "schema_version": 1,
        "analysis_status": "exploratory_paced_exposure_complete_pilot",
        "plan_id": plan.plan_id,
        "confirmatory_eligible": False,
        "learner_effect_measured": False,
        "arms": [
            analyze_arm(
                plan,
                arm_id=arm.arm_id,
                trace_path=args.arms_root / arm.arm_id / "scheduler_trace.v1.jsonl",
            )
            for arm in plan.arms
        ],
        "limitations": [
            "one_pool",
            "runtime_order_confounding",
            "overlapping_windows_not_independent",
            "initial_and_final_16_groups_are_boundaries",
            "no_steady_state_claim",
            "matching_seeds_do_not_imply_identical_live_completions",
        ],
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print("paced_exposure_analysis=PASS")


if __name__ == "__main__":
    main()
