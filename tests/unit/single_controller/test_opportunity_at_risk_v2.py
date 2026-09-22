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

"""CPU-only contracts for the behavior-neutral OARS-v2 shadow observer."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from nemo_rl.algorithms.async_utils.opportunity_at_risk import (
    OPPORTUNITY_GROUP_ID_KEY,
    OPPORTUNITY_L1_KEY,
    OPPORTUNITY_L2_KEY,
    OPPORTUNITY_VALID_TOKENS_KEY,
)
from nemo_rl.algorithms.async_utils.opportunity_at_risk_v2 import (
    OPPORTUNITY_REWARD_MEAN_KEY,
    OPPORTUNITY_REWARD_VARIANCE_KEY,
    OARSV2Candidate,
    OARSV2SelectionFallback,
    OpportunityAtRiskV2ShadowRecorder,
    OpportunityAtRiskV2ShadowSampler,
    compute_reward_moments,
    select_two_sided_oars_v2,
)
from nemo_rl.algorithms.async_utils.rollout_lifecycle import RolloutRemovalReason
from nemo_rl.algorithms.async_utils.staleness_sampler import WeightFifoSampler
from nemo_rl.data_plane import KVBatchMeta


class FakeBuffer:
    """Minimal parallel-list buffer surface for WeightFIFO and OARS-v2."""

    def __init__(self) -> None:
        self.meta_list: list[KVBatchMeta | None] = []
        self.start_weight_list: list[int] = []
        self.end_weight_list: list[int] = []
        self.target_step_list: list[int | None] = []
        self.ready_list: list[bool] = []
        self.ready_timestamp_ns_list: list[int | None] = []
        self.remove_reasons: list[RolloutRemovalReason] = []

    def add(
        self,
        group_id: str,
        *,
        weight: int,
        l1: float,
        l2: float,
        tokens: int,
        reward_mean: float,
        reward_variance: float,
        ready_timestamp_ns: int,
        metadata: bool = True,
        ready: bool = True,
    ) -> None:
        extra_info: dict[str, Any] = {}
        if metadata:
            extra_info = {
                OPPORTUNITY_GROUP_ID_KEY: group_id,
                OPPORTUNITY_L1_KEY: l1,
                OPPORTUNITY_L2_KEY: l2,
                OPPORTUNITY_VALID_TOKENS_KEY: tokens,
                OPPORTUNITY_REWARD_MEAN_KEY: reward_mean,
                OPPORTUNITY_REWARD_VARIANCE_KEY: reward_variance,
            }
        meta = KVBatchMeta(
            partition_id="rollout_data",
            task_name="train",
            sample_ids=[f"{group_id}_g0"],
            extra_info=extra_info,
        )
        self.meta_list.append(meta if ready else None)
        self.start_weight_list.append(weight)
        self.end_weight_list.append(weight)
        self.target_step_list.append(None)
        self.ready_list.append(ready)
        self.ready_timestamp_ns_list.append(ready_timestamp_ns if ready else None)

    async def remove(
        self,
        idxs: list[int],
        remove_in_dp: bool,
        *,
        reason: RolloutRemovalReason,
        learner_weight_version: int | None,
    ) -> int:
        del remove_in_dp, learner_weight_version
        self.remove_reasons.append(reason)
        for index in sorted(idxs, reverse=True):
            del self.meta_list[index]
            del self.start_weight_list[index]
            del self.end_weight_list[index]
            del self.target_step_list[index]
            del self.ready_list[index]
            del self.ready_timestamp_ns_list[index]
        return len(idxs)


def candidate(
    group_id: str,
    *,
    start: int,
    ready: int,
    l1: float,
    variance: float,
    tokens: int = 100,
) -> OARSV2Candidate:
    return OARSV2Candidate(
        group_id=group_id,
        start_weight_version=start,
        ready_timestamp_ns=ready,
        l1=l1,
        l2=l1 / 2,
        valid_actor_tokens=tokens,
        reward_mean=-0.25,
        reward_variance=variance,
    )


def test_reward_moments_match_population_definition() -> None:
    mean, variance = compute_reward_moments((0.0, 1.0, 0.0, 1.0))

    assert mean == 0.5
    assert variance == 0.25

    with pytest.raises(ValueError, match="at least one"):
        compute_reward_moments(())


def test_two_sided_selectors_are_score_specific_and_budget_feasible() -> None:
    candidates = [
        candidate("a", start=1, ready=10, l1=1, variance=10),
        candidate("b", start=1, ready=20, l1=2, variance=9),
        candidate("c", start=1, ready=30, l1=3, variance=8),
        candidate("d", start=1, ready=40, l1=4, variance=7),
        candidate("w", start=1, ready=50, l1=10, variance=4),
        candidate("x", start=1, ready=60, l1=11, variance=3),
        candidate("y", start=1, ready=70, l1=12, variance=2),
        candidate("z", start=1, ready=80, l1=13, variance=1),
    ]
    common = {
        "baseline_group_ids": ("a", "b", "c", "d"),
        "current_train_weight": 2,
        "decision_timestamp_ns": 100,
        "max_staleness_versions": 1,
        "minimum_service_multiplier": 0.98,
        "maximum_service_multiplier": 1.02,
        "exact_search_max_candidates": 25,
    }

    age = select_two_sided_oars_v2(candidates, scorer="age", **common)
    reward = select_two_sided_oars_v2(
        candidates, scorer="reward_variance_risk", **common
    )
    absolute = select_two_sided_oars_v2(candidates, scorer="absolute_m4_risk", **common)

    assert age.proposed_group_ids == ("a", "b", "c", "d")
    assert reward.proposed_group_ids == ("a", "b", "c", "d")
    assert absolute.proposed_group_ids == ("w", "x", "y", "z")
    assert absolute.minimum_tokens == 392
    assert absolute.maximum_tokens == 408
    assert absolute.proposed_valid_actor_tokens == 400


def _natural_choice_buffer() -> FakeBuffer:
    buffer = FakeBuffer()
    for index, group_id in enumerate(("a", "b", "c", "d", "w", "x")):
        buffer.add(
            group_id,
            weight=1,
            l1=1.0 if index < 4 else 20.0,
            l2=1.0 if index < 4 else 10.0,
            tokens=100,
            reward_mean=0.5,
            reward_variance=0.1 if index < 4 else 0.25,
            ready_timestamp_ns=100 + index,
        )
    return buffer


def _shadow(buffer: FakeBuffer, *, max_candidates: int = 64):
    baseline = WeightFifoSampler(
        buffer,
        max_staleness_versions=1,
        selection_candidate_watermark=None,
    )
    events: list[dict[str, Any]] = []
    sampler = OpportunityAtRiskV2ShadowSampler(
        buffer=buffer,
        baseline=baseline,
        candidate_window_policy="natural_eager",
        minimum_service_multiplier=0.98,
        maximum_service_multiplier=1.02,
        max_candidate_groups=max_candidates,
        exact_search_max_candidates=min(25, max_candidates),
        decision_time_budget_ns=100_000_000,
        record=lambda event: events.append(dict(event)),
    )
    return sampler, events


def test_adaptive_shadow_observes_natural_choice_and_executes_fifo() -> None:
    buffer = _natural_choice_buffer()
    sampler, events = _shadow(buffer)

    meta, count = asyncio.run(
        sampler.select(
            current_train_weight=2,
            min_prompt_groups=4,
            max_prompt_groups=4,
        )
    )

    assert meta is not None
    assert count == 4
    assert meta.sample_ids == ["a_g0", "b_g0", "c_g0", "d_g0"]
    assert len(buffer.ready_list) == 2
    assert buffer.remove_reasons == [RolloutRemovalReason.SELECTED]
    assert events[0]["candidate_group_count"] == 6
    assert events[0]["baseline_matches_actual"] is True
    assert events[0]["proposals"]["absolute_m4_risk"]["proposed_group_ids"] == [
        "a",
        "b",
        "w",
        "x",
    ]
    assert len(events[0]["candidates"]) == 6
    assert events[0]["proposals"]["absolute_m4_risk"]["replacement_group_ids"] == [
        "w",
        "x",
    ]
    assert events[0]["proposals"]["absolute_m4_risk"]["deferred_fifo_group_ids"] == [
        "c",
        "d",
    ]
    assert events[0]["proposals"]["absolute_m4_risk"]["deliberate_shed_group_ids"] == [
        "c",
        "d",
    ]


def test_uncontended_shadow_dispatches_without_waiting() -> None:
    buffer = _natural_choice_buffer()
    for index in (5, 4):
        del buffer.meta_list[index]
        del buffer.start_weight_list[index]
        del buffer.end_weight_list[index]
        del buffer.target_step_list[index]
        del buffer.ready_list[index]
        del buffer.ready_timestamp_ns_list[index]
    sampler, events = _shadow(buffer)

    meta, count = asyncio.run(
        sampler.select(
            current_train_weight=2,
            min_prompt_groups=4,
            max_prompt_groups=4,
        )
    )

    assert meta is not None and count == 4
    assert len(events) == 1
    assert events[0]["candidate_group_count"] == 4
    assert all(
        proposal["proposed_group_ids"] == ["a", "b", "c", "d"]
        for proposal in events[0]["proposals"].values()
    )


def test_controlled_frontier_observes_choice_but_executes_fifo() -> None:
    buffer = _natural_choice_buffer()
    baseline = WeightFifoSampler(
        buffer,
        max_staleness_versions=1,
        selection_candidate_watermark=6,
    )
    events: list[dict[str, Any]] = []
    sampler = OpportunityAtRiskV2ShadowSampler(
        buffer=buffer,
        baseline=baseline,
        candidate_window_policy="controlled_frontier",
        minimum_service_multiplier=0.98,
        maximum_service_multiplier=1.02,
        max_candidate_groups=64,
        exact_search_max_candidates=16,
        decision_time_budget_ns=100_000_000,
        record=lambda event: events.append(dict(event)),
    )

    meta, count = asyncio.run(
        sampler.select(
            current_train_weight=2,
            min_prompt_groups=4,
            max_prompt_groups=4,
        )
    )

    assert meta is not None and count == 4
    assert meta.sample_ids == ["a_g0", "b_g0", "c_g0", "d_g0"]
    assert events[0]["candidate_group_count"] == 6
    assert events[0]["baseline_matches_actual"] is True


def test_missing_metadata_falls_back_without_extra_mutation() -> None:
    buffer = _natural_choice_buffer()
    assert buffer.meta_list[-1] is not None
    buffer.meta_list[-1].extra_info = {}
    sampler, events = _shadow(buffer)

    meta, count = asyncio.run(
        sampler.select(
            current_train_weight=2,
            min_prompt_groups=4,
            max_prompt_groups=4,
        )
    )

    assert meta is not None and count == 4
    assert meta.sample_ids == ["a_g0", "b_g0", "c_g0", "d_g0"]
    assert len(buffer.ready_list) == 2
    assert buffer.remove_reasons == [RolloutRemovalReason.SELECTED]
    assert events[0]["skip_reason"] == "missing_or_invalid_opportunity_metadata"
    assert events[0]["baseline_matches_actual"] is True


def test_candidate_cap_falls_back_to_fifo_without_dropping_excess() -> None:
    buffer = _natural_choice_buffer()
    sampler, events = _shadow(buffer, max_candidates=5)

    meta, count = asyncio.run(
        sampler.select(
            current_train_weight=2,
            min_prompt_groups=4,
            max_prompt_groups=4,
        )
    )

    assert meta is not None and count == 4
    assert len(buffer.ready_list) == 2
    assert buffer.remove_reasons == [RolloutRemovalReason.SELECTED]
    assert events[0]["skip_reason"] == "candidate_safety_cap_exceeded"
    assert events[0]["baseline_group_ids"] == ["a", "b", "c", "d"]
    assert events[0]["baseline_matches_actual"] is True


def test_selector_handles_every_candidate_count_through_safety_cap() -> None:
    for candidate_count in range(4, 65):
        candidates = [
            candidate(
                f"g{index:02d}",
                start=1,
                ready=index,
                l1=float(index + 1),
                variance=float(candidate_count - index),
                tokens=100 + (index % 3),
            )
            for index in range(candidate_count)
        ]
        baseline_ids = tuple(candidate.group_id for candidate in candidates[:4])

        result = select_two_sided_oars_v2(
            candidates,
            baseline_group_ids=baseline_ids,
            current_train_weight=1,
            decision_timestamp_ns=1_000,
            max_staleness_versions=1,
            scorer="absolute_m4_risk",
            minimum_service_multiplier=0.98,
            maximum_service_multiplier=1.02,
            exact_search_max_candidates=12,
        )

        assert len(result.proposed_group_ids) == 4
        assert result.minimum_tokens <= result.proposed_valid_actor_tokens
        assert result.proposed_valid_actor_tokens <= result.maximum_tokens
        assert result.candidate_count == candidate_count
        assert result.search_candidate_count <= 12


def test_cross_version_risk_prioritizes_imminent_opportunity() -> None:
    candidates = [
        candidate("old-low", start=0, ready=1, l1=1, variance=1),
        candidate("old-high", start=0, ready=2, l1=10, variance=1),
        candidate("fresh-a", start=1, ready=3, l1=100, variance=1),
        candidate("fresh-b", start=1, ready=4, l1=90, variance=1),
        candidate("fresh-c", start=1, ready=5, l1=80, variance=1),
        candidate("fresh-d", start=1, ready=6, l1=70, variance=1),
    ]

    result = select_two_sided_oars_v2(
        candidates,
        baseline_group_ids=("old-low", "old-high", "fresh-a", "fresh-b"),
        current_train_weight=1,
        decision_timestamp_ns=1_000,
        max_staleness_versions=1,
        scorer="absolute_m4_risk",
        minimum_service_multiplier=0.98,
        maximum_service_multiplier=1.02,
        exact_search_max_candidates=25,
    )

    assert {"old-low", "old-high"}.issubset(result.proposed_group_ids)
    assert result.proposed_imminent_l1 == 11


def test_safe_prune_retains_fifo_and_every_imminent_candidate() -> None:
    candidates = [
        candidate(
            f"fresh-{index:02d}",
            start=1,
            ready=index,
            l1=100 - index,
            variance=100 - index,
        )
        for index in range(18)
    ] + [
        candidate("old-a", start=0, ready=100, l1=1, variance=1),
        candidate("old-b", start=0, ready=101, l1=2, variance=2),
    ]

    result = select_two_sided_oars_v2(
        candidates,
        baseline_group_ids=("fresh-00", "fresh-01", "fresh-02", "fresh-03"),
        current_train_weight=1,
        decision_timestamp_ns=1_000,
        max_staleness_versions=1,
        scorer="absolute_m4_risk",
        minimum_service_multiplier=0.98,
        maximum_service_multiplier=1.02,
        exact_search_max_candidates=8,
    )

    assert result.search_strategy == "mandatory_safe_prune"
    assert result.search_candidate_count == 8
    assert {"old-a", "old-b"}.issubset(result.proposed_group_ids)


def test_selector_timeout_is_an_explicit_fallback() -> None:
    candidates = [
        candidate(f"g{index}", start=1, ready=index, l1=index, variance=index)
        for index in range(8)
    ]

    with pytest.raises(OARSV2SelectionFallback, match="time_budget"):
        select_two_sided_oars_v2(
            candidates,
            baseline_group_ids=("g0", "g1", "g2", "g3"),
            current_train_weight=1,
            decision_timestamp_ns=1_000,
            max_staleness_versions=1,
            scorer="absolute_m4_risk",
            minimum_service_multiplier=0.98,
            maximum_service_multiplier=1.02,
            exact_search_max_candidates=25,
            deadline_ns=0,
        )


def test_prune_refuses_to_drop_mandatory_candidates() -> None:
    candidates = [
        candidate(f"g{index}", start=0, ready=index, l1=index + 1, variance=1)
        for index in range(9)
    ]

    with pytest.raises(OARSV2SelectionFallback, match="mandatory_candidates"):
        select_two_sided_oars_v2(
            candidates,
            baseline_group_ids=("g0", "g1", "g2", "g3"),
            current_train_weight=1,
            decision_timestamp_ns=1_000,
            max_staleness_versions=1,
            scorer="absolute_m4_risk",
            minimum_service_multiplier=0.98,
            maximum_service_multiplier=1.02,
            exact_search_max_candidates=8,
        )


def test_shadow_delegates_stale_eviction_to_fifo() -> None:
    buffer = _natural_choice_buffer()
    buffer.add(
        "stale",
        weight=0,
        l1=1,
        l2=1,
        tokens=100,
        reward_mean=0,
        reward_variance=0,
        ready_timestamp_ns=1,
    )
    sampler, _ = _shadow(buffer)

    removed = asyncio.run(sampler.evict(current_train_weight=2))

    assert removed == 1
    assert buffer.remove_reasons == [RolloutRemovalReason.STALE_EVICTED]


def test_sustained_overload_never_removes_more_than_fifo() -> None:
    buffer = _natural_choice_buffer()
    sampler, events = _shadow(buffer)

    for decision in range(12):
        meta, count = asyncio.run(
            sampler.select(
                current_train_weight=2,
                min_prompt_groups=4,
                max_prompt_groups=4,
            )
        )
        assert meta is not None and count == 4
        for offset in range(4):
            buffer.add(
                f"new-{decision}-{offset}",
                weight=1,
                l1=10 + offset,
                l2=5 + offset,
                tokens=100,
                reward_mean=0.25,
                reward_variance=0.5,
                ready_timestamp_ns=1_000 + decision * 4 + offset,
            )
        assert len(buffer.ready_list) == 6

    assert len(events) == 12
    assert buffer.remove_reasons == [RolloutRemovalReason.SELECTED] * 12
    assert all(event["baseline_matches_actual"] is True for event in events)


def test_recorder_declares_behavior_neutral_adaptive_policy(tmp_path) -> None:
    recorder = OpportunityAtRiskV2ShadowRecorder(
        candidate_window_policy="natural_eager",
        selection_candidate_watermark=None,
        minimum_service_multiplier=0.98,
        maximum_service_multiplier=1.02,
        max_candidate_groups=64,
        exact_search_max_candidates=25,
        decision_time_budget_ns=5_000_000,
    )

    header = recorder.events[0]
    assert header["mode"] == "observe"
    assert header["candidate_window_policy"] == "natural_eager"
    assert header["selection_candidate_watermark"] is None
    assert header["candidate_mutation"] == "none"
    assert header["decision_time_budget_scope"] == "per_scorer"

    output_path = tmp_path / "oars-v2.jsonl"
    recorder.flush_jsonl(str(output_path))
    assert json.loads(output_path.read_text().splitlines()[0]) == header
