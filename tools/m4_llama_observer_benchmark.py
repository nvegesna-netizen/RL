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

"""Outcome-free benchmark for the final Llama observer repair."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict
from typing import Any

import torch

from nemo_rl.algorithms.advantage_estimator import (
    AdvEstimatorConfig,
    GRPOAdvantageEstimator,
)
from nemo_rl.algorithms.async_utils.gradient_opportunity import (
    GRPOOpportunityInputs,
    GradientOpportunityRecorder,
    GroupOpportunitySummary,
    SiblingOpportunitySummary,
    _validate_prepared_inputs,
    compute_grpo_gradient_opportunity,
)
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.algorithms.single_controller_utils.config import AdvantageConfig
from nemo_rl.algorithms.single_controller_utils.utils import (
    prepare_advantage_inputs_from_data,
)


class _BenchmarkSequencer:
    """Deterministic monotonic sequencer for in-memory ledger timing."""

    def __init__(self) -> None:
        self._sequence = 0

    def __call__(self) -> tuple[int, int]:
        self._sequence += 1
        return self._sequence, self._sequence


def _expanded_reference_summary(
    inputs: GRPOOpportunityInputs,
    *,
    group_id: str,
    sample_ids: tuple[str, ...],
    estimator: GRPOAdvantageEstimator,
) -> GroupOpportunitySummary:
    """Reproduce the pre-repair expanded observer arithmetic exactly."""
    _validate_prepared_inputs(inputs)
    advantages = estimator.compute_advantage(
        prompt_ids=inputs.prompt_ids,
        rewards=inputs.rewards,
        mask=inputs.actor_mask,
        repeated_batch=dict(inputs.repeated_batch),
        **dict(inputs.estimator_kwargs),
    )
    if advantages.shape != inputs.actor_mask.shape:
        raise ValueError("expanded reference advantage shape disagrees")
    scalar_advantages = advantages[:, 0]
    if not bool((advantages == scalar_advantages.unsqueeze(-1)).all().item()):
        raise ValueError("expanded reference advantages are not scalar-constant")

    aligned_advantages = advantages[:, 1:].to(dtype=torch.float64)
    aligned_mask = inputs.actor_mask[:, 1:].to(dtype=torch.float64)
    coefficients = aligned_advantages * aligned_mask
    absolute_coefficients = coefficients.abs()
    siblings = tuple(
        SiblingOpportunitySummary(
            sample_id=sample_id,
            sibling_index=index,
            reward=float(inputs.rewards[index].item()),
            scalar_advantage=float(scalar_advantages[index].item()),
            valid_actor_tokens=int(aligned_mask[index].sum(dtype=torch.float64).item()),
            opportunity=float(
                absolute_coefficients[index].sum(dtype=torch.float64).item()
            ),
            truncated=False,
        )
        for index, sample_id in enumerate(sample_ids)
    )
    positive_mass = coefficients.clamp_min(0).sum(dtype=torch.float64)
    negative_mass = (-coefficients.clamp_max(0)).sum(dtype=torch.float64)
    opportunity = absolute_coefficients.sum(dtype=torch.float64)
    squared_mass = coefficients.square().sum(dtype=torch.float64)
    nonzero_advantage = scalar_advantages != 0
    nonzero_token_mask = aligned_mask.bool() & nonzero_advantage.unsqueeze(-1)
    return GroupOpportunitySummary(
        group_id=group_id,
        sample_ids=sample_ids,
        start_weight_version=3,
        opportunity=float(opportunity.item()),
        l2_coefficient_mass=math.sqrt(float(squared_mass.item())),
        signed_coefficient_mass=float(coefficients.sum(dtype=torch.float64).item()),
        positive_coefficient_mass=float(positive_mass.item()),
        negative_coefficient_mass=float(negative_mass.item()),
        valid_actor_tokens=int(aligned_mask.sum(dtype=torch.float64).item()),
        nonzero_advantage_siblings=int(nonzero_advantage.sum().item()),
        nonzero_advantage_tokens=int(nonzero_token_mask.sum().item()),
        siblings=siblings,
    )


def _fixture(*, sequence_length: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260910)
    prompt = torch.randint(0, 32000, (1, 64), generator=generator)
    token_mask = torch.randint(
        0,
        2,
        (8, sequence_length),
        generator=generator,
        dtype=torch.int64,
    )
    token_mask[:, 0] = 0
    return {
        "prompt_ids_for_adv": prompt.expand(8, -1).clone(),
        "total_reward": torch.tensor(
            [[1.0], [0.0], [1.0], [1.0], [0.0], [0.0], [1.0], [0.0]],
            dtype=torch.float32,
        ),
        "token_mask": token_mask,
        "sample_mask": torch.ones((8, 1), dtype=torch.int64),
    }


def _prepare(
    data: Mapping[str, Any], advantage_config: AdvantageConfig
) -> GRPOOpportunityInputs:
    prepared = prepare_advantage_inputs_from_data(
        data,
        advantage_config=advantage_config,
        policy_logprobs_required=False,
        reference_logprobs_required=False,
    )
    return GRPOOpportunityInputs(
        prompt_ids=prepared.prompt_ids,
        rewards=prepared.rewards,
        actor_mask=prepared.actor_mask,
        repeated_batch=prepared.repeated_batch,
        estimator_kwargs=prepared.estimator_kwargs,
    )


def _measure(
    callback: Callable[[int, GradientOpportunityRecorder], None],
    *,
    iterations: int,
) -> float:
    recorder = GradientOpportunityRecorder(
        run_id="benchmark",
        clock_domain_id="benchmark",
        sequencer=_BenchmarkSequencer(),
    )
    recorder.append_header(
        estimator_name="GRPOAdvantageEstimator",
        estimator_settings={},
        loss_settings={},
    )
    started = time.perf_counter_ns()
    for index in range(iterations):
        callback(index, recorder)
    return (time.perf_counter_ns() - started) / iterations / 1e6


def run_benchmark(
    *, sequence_length: int, iterations: int, trials: int
) -> dict[str, object]:
    """Benchmark candidate and expanded-reference callbacks without training."""
    data = _fixture(sequence_length=sequence_length)
    advantage_config = AdvantageConfig()
    estimator = GRPOAdvantageEstimator(AdvEstimatorConfig(), ClippedPGLossConfig())

    def candidate(index: int, recorder: GradientOpportunityRecorder) -> None:
        group_id = f"candidate-{index}"
        summary = compute_grpo_gradient_opportunity(
            data,
            group_id=group_id,
            sample_ids=tuple(f"{group_id}_g{sibling}" for sibling in range(8)),
            start_weight_version=3,
            truncation=(False,) * 8,
            estimator=estimator,
            prepare_inputs=lambda value: _prepare(value, advantage_config),
        )
        recorder.append_group(summary)

    def reference(index: int, recorder: GradientOpportunityRecorder) -> None:
        group_id = f"reference-{index}"
        summary = _expanded_reference_summary(
            _prepare(data, advantage_config),
            group_id=group_id,
            sample_ids=tuple(f"{group_id}_g{sibling}" for sibling in range(8)),
            estimator=estimator,
        )
        recorder.append_group(summary)

    candidate_summary = compute_grpo_gradient_opportunity(
        data,
        group_id="equivalence",
        sample_ids=tuple(f"equivalence_g{index}" for index in range(8)),
        start_weight_version=3,
        truncation=(False,) * 8,
        estimator=estimator,
        prepare_inputs=lambda value: _prepare(value, advantage_config),
    )
    reference_summary = _expanded_reference_summary(
        _prepare(data, advantage_config),
        group_id="equivalence",
        sample_ids=tuple(f"equivalence_g{index}" for index in range(8)),
        estimator=estimator,
    )
    if asdict(candidate_summary) != asdict(reference_summary):
        raise ValueError("candidate and expanded reference summaries disagree")

    _measure(reference, iterations=16)
    _measure(candidate, iterations=16)
    reference_means: list[float] = []
    candidate_means: list[float] = []
    for trial in range(trials):
        first, second = (
            (reference, candidate) if trial % 2 == 0 else (candidate, reference)
        )
        first_result = _measure(first, iterations=iterations)
        second_result = _measure(second, iterations=iterations)
        if first is reference:
            reference_means.append(first_result)
            candidate_means.append(second_result)
        else:
            candidate_means.append(first_result)
            reference_means.append(second_result)
    reference_ms = statistics.median(reference_means)
    candidate_ms = statistics.median(candidate_means)
    speedup_fraction = 1.0 - candidate_ms / reference_ms
    return {
        "absolute_mean_callback_ms_maximum": 1.0,
        "candidate_trial_mean_ms": candidate_means,
        "candidate_trial_median_mean_ms": candidate_ms,
        "exact_summary_equivalence": True,
        "iterations_per_trial": iterations,
        "minimum_speedup_fraction": 0.25,
        "reference_trial_mean_ms": reference_means,
        "reference_trial_median_mean_ms": reference_ms,
        "schema": "m4-llama-observer-no-training-benchmark-v1",
        "sequence_length": sequence_length,
        "siblings": 8,
        "speedup_fraction": speedup_fraction,
        "trials": trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    if args.sequence_length != 2048 or args.iterations <= 0 or args.trials < 3:
        raise SystemExit("benchmark geometry disagrees with the frozen amendment")
    result = run_benchmark(
        sequence_length=args.sequence_length,
        iterations=args.iterations,
        trials=args.trials,
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    if args.enforce and (
        result["candidate_trial_median_mean_ms"] > 1.0
        or result["speedup_fraction"] < 0.25
    ):
        raise SystemExit("M4_LLAMA_OBSERVER_BENCHMARK_RED")
    print("M4_LLAMA_OBSERVER_BENCHMARK_GREEN")


if __name__ == "__main__":
    main()
