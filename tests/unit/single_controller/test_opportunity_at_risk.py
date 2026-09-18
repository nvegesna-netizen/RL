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

"""CPU-only contracts for baseline-budgeted OARS shadow selection."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

import pytest

from nemo_rl.algorithms.async_utils.opportunity_at_risk import (
    OPPORTUNITY_GROUP_ID_KEY,
    OPPORTUNITY_L1_KEY,
    OPPORTUNITY_VALID_TOKENS_KEY,
    OpportunityAtRiskShadowRecorder,
    OpportunityAtRiskShadowSampler,
    OpportunityCandidate,
    select_baseline_budgeted_opportunity,
)
from nemo_rl.algorithms.async_utils.rollout_lifecycle import RolloutRemovalReason
from nemo_rl.algorithms.async_utils.staleness_sampler import WeightFifoSampler
from nemo_rl.data_plane import KVBatchMeta


class FakeBuffer:
    """Minimal parallel-list buffer surface used by both samplers."""

    def __init__(self) -> None:
        self.meta_list: list[KVBatchMeta | None] = []
        self.start_weight_list: list[int] = []
        self.end_weight_list: list[int] = []
        self.target_step_list: list[int | None] = []
        self.ready_list: list[bool] = []
        self.remove_reasons: list[RolloutRemovalReason] = []

    def add(
        self,
        group_id: str,
        *,
        weight: int,
        l1: float,
        tokens: int,
        metadata: bool = True,
        ready: bool = True,
    ) -> None:
        extra_info: dict[str, Any] = {}
        if metadata:
            extra_info = {
                OPPORTUNITY_GROUP_ID_KEY: group_id,
                OPPORTUNITY_L1_KEY: l1,
                OPPORTUNITY_VALID_TOKENS_KEY: tokens,
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
        return len(idxs)


def _run_select(
    buffer: FakeBuffer,
    *,
    observe: bool,
    mode: Literal["observe", "act"] = "observe",
) -> tuple[list[str], list[dict[str, Any]]]:
    baseline = WeightFifoSampler(
        buffer,
        max_staleness_versions=1,
        selection_candidate_watermark=None,
    )
    events: list[dict[str, Any]] = []
    sampler: Any = baseline
    if observe:
        sampler = OpportunityAtRiskShadowSampler(
            buffer=buffer,
            baseline=baseline,
            service_budget_multiplier=1.02,
            max_candidate_groups=25,
            record=lambda event: events.append(dict(event)),
            mode=mode,
        )
    meta, groups = asyncio.run(
        sampler.select(
            current_train_weight=2,
            min_prompt_groups=4,
            max_prompt_groups=4,
        )
    )
    assert meta is not None
    assert groups == 4
    return list(meta.sample_ids), events


def _choice_buffer() -> FakeBuffer:
    buffer = FakeBuffer()
    for group_id in ("a", "b", "c", "d"):
        buffer.add(group_id, weight=1, l1=1.0, tokens=100)
    for group_id in ("w", "x", "y", "z"):
        buffer.add(group_id, weight=1, l1=10.0, tokens=100)
    return buffer


def test_exact_selector_is_deterministic_and_budget_feasible() -> None:
    candidates = [
        OpportunityCandidate("a", 1, 1.0, 100),
        OpportunityCandidate("b", 1, 1.0, 100),
        OpportunityCandidate("c", 1, 1.0, 100),
        OpportunityCandidate("d", 1, 1.0, 100),
        OpportunityCandidate("w", 1, 5.0, 101),
        OpportunityCandidate("x", 1, 5.0, 101),
        OpportunityCandidate("y", 1, 5.0, 101),
        OpportunityCandidate("z", 1, 5.0, 101),
    ]
    first = select_baseline_budgeted_opportunity(
        candidates,
        baseline_group_ids=("a", "b", "c", "d"),
        current_train_weight=2,
        service_budget_multiplier=1.02,
    )
    second = select_baseline_budgeted_opportunity(
        list(reversed(candidates)),
        baseline_group_ids=("a", "b", "c", "d"),
        current_train_weight=2,
        service_budget_multiplier=1.02,
    )

    assert first == second
    assert first.proposed_group_ids == ("w", "x", "y", "z")
    assert first.proposed_valid_actor_tokens == 404
    assert first.token_budget == 408


def test_shadow_header_records_fifo_selection_watermark() -> None:
    recorder = OpportunityAtRiskShadowRecorder(
        service_budget_multiplier=1.02,
        max_candidate_groups=25,
        selection_candidate_watermark=8,
    )

    assert recorder.events[0]["selection_candidate_watermark"] == 8
    assert recorder.events[0]["schema_version"] == 3
    assert (
        recorder.events[0]["candidate_window_policy"]
        == "oldest_ready_exact_watermark_v1"
    )


def test_shadow_proposes_but_executes_exact_weight_fifo_selection() -> None:
    baseline_ids, _ = _run_select(_choice_buffer(), observe=False)
    shadow_ids, events = _run_select(_choice_buffer(), observe=True)

    assert shadow_ids == baseline_ids == ["a_g0", "b_g0", "c_g0", "d_g0"]
    assert len(events) == 1
    assert events[0]["baseline_group_ids"] == ["a", "b", "c", "d"]
    assert events[0]["actual_selected_group_ids"] == ["a", "b", "c", "d"]
    assert events[0]["actual_selected_group_count"] == 4
    assert events[0]["baseline_matches_actual"] is True
    assert events[0]["proposed_group_ids"] == ["w", "x", "y", "z"]
    assert events[0]["skip_reason"] is None


def test_uncontended_shadow_is_baseline_equivalent() -> None:
    buffer = FakeBuffer()
    for group_id in ("a", "b", "c", "d"):
        buffer.add(group_id, weight=1, l1=1.0, tokens=100)

    selected, events = _run_select(buffer, observe=True)

    assert selected == ["a_g0", "b_g0", "c_g0", "d_g0"]
    assert events[0]["proposed_group_ids"] == ["a", "b", "c", "d"]
    assert events[0]["baseline_matches_actual"] is True


def test_act_mode_executes_exact_budgeted_oars_proposal() -> None:
    selected, events = _run_select(_choice_buffer(), observe=True, mode="act")

    assert selected == ["w_g0", "x_g0", "y_g0", "z_g0"]
    assert events[0]["actual_selected_group_ids"] == ["w", "x", "y", "z"]
    assert events[0]["baseline_matches_actual"] is False
    assert events[0]["proposal_matches_actual"] is True
    assert events[0]["mode"] == "act"


def test_act_mode_replenishes_after_stale_eviction_without_expanding_choice_set() -> (
    None
):
    """Reproduce the post-step-2 liveness state from EOS qualification."""
    buffer = FakeBuffer()
    buffer.add("stale", weight=0, l1=0.0, tokens=100)
    for group_id in ("old-a", "old-b", "old-c"):
        buffer.add(group_id, weight=1, l1=1.0, tokens=100)
    for group_id in ("new-a", "new-b", "new-c", "new-d"):
        buffer.add(group_id, weight=2, l1=2.0, tokens=100)

    baseline = WeightFifoSampler(
        buffer,
        max_staleness_versions=1,
        selection_candidate_watermark=8,
    )
    events: list[dict[str, Any]] = []
    sampler = OpportunityAtRiskShadowSampler(
        buffer=buffer,
        baseline=baseline,
        service_budget_multiplier=1.02,
        max_candidate_groups=25,
        record=lambda event: events.append(dict(event)),
        mode="act",
    )

    async def reproduce() -> tuple[KVBatchMeta | None, int]:
        # Two initial batches plus one ordinary batch after each of two updates.
        trainer_version = 0
        await sampler.admit(trainer_version_fn=lambda: trainer_version)
        await sampler.admit(trainer_version_fn=lambda: trainer_version)
        trainer_version = 1
        await sampler.admit(trainer_version_fn=lambda: trainer_version)
        trainer_version = 2
        await sampler.admit(trainer_version_fn=lambda: trainer_version)

        assert await sampler.evict(current_train_weight=trainer_version) == 1
        # The ordinary gate is now closed. The stale eviction must authorize one
        # replacement batch instead of leaving selection at seven of eight.
        await asyncio.wait_for(
            sampler.admit(trainer_version_fn=lambda: trainer_version),
            timeout=0.05,
        )
        for group_id in ("fill-a", "fill-b", "fill-c", "fill-d"):
            buffer.add(group_id, weight=2, l1=3.0, tokens=100)
        return await sampler.select(
            current_train_weight=trainer_version,
            min_prompt_groups=4,
            max_prompt_groups=4,
        )

    meta, count = asyncio.run(reproduce())

    assert meta is not None
    assert count == 4
    assert len(events) == 1
    assert events[0]["candidate_group_count"] == 8
    assert events[0]["combination_count"] == 70
    assert events[0]["baseline_group_ids"] == [
        "old-a",
        "old-b",
        "old-c",
        "new-a",
    ]


def test_act_mode_remains_live_with_exact_choice_sets_across_updates() -> None:
    buffer = FakeBuffer()
    baseline = WeightFifoSampler(
        buffer,
        max_staleness_versions=1,
        selection_candidate_watermark=8,
    )
    events: list[dict[str, Any]] = []
    sampler = OpportunityAtRiskShadowSampler(
        buffer=buffer,
        baseline=baseline,
        service_budget_multiplier=1.02,
        max_candidate_groups=25,
        record=lambda event: events.append(dict(event)),
        mode="act",
    )
    next_group = 0
    maximum_buffered = 0
    replacement_batches = 0

    def add_batch(weight: int) -> None:
        nonlocal next_group, maximum_buffered
        for l1 in (0.0, 1.0, 2.0, 10.0):
            buffer.add(
                f"group-{next_group:03d}",
                weight=weight,
                l1=l1,
                tokens=100,
            )
            next_group += 1
        maximum_buffered = max(maximum_buffered, len(buffer.ready_list))

    async def run_updates() -> None:
        nonlocal replacement_batches, maximum_buffered
        trainer_version = 0
        for _ in range(2):
            await sampler.admit(trainer_version_fn=lambda: trainer_version)
            add_batch(trainer_version)

        for update in range(64):
            evicted = await sampler.evict(current_train_weight=trainer_version)
            if evicted:
                await asyncio.wait_for(
                    sampler.admit(trainer_version_fn=lambda: trainer_version),
                    timeout=0.05,
                )
                add_batch(trainer_version)
                replacement_batches += 1

            meta, count = await sampler.select(
                current_train_weight=trainer_version,
                min_prompt_groups=4,
                max_prompt_groups=4,
            )
            assert meta is not None
            assert count == 4
            maximum_buffered = max(maximum_buffered, len(buffer.ready_list))

            if update < 63:
                trainer_version += 1
                await sampler.admit(trainer_version_fn=lambda: trainer_version)
                add_batch(trainer_version)

    asyncio.run(run_updates())

    assert replacement_batches > 0
    assert RolloutRemovalReason.OARS_CANDIDATE_EXCESS in buffer.remove_reasons
    assert len(events) == 64
    assert all(event["candidate_group_count"] == 8 for event in events)
    assert all(event["combination_count"] == 70 for event in events)
    assert all(8 <= event["eligible_candidate_count"] <= 11 for event in events)
    assert all(
        event["candidate_excess_count"]
        == event["eligible_candidate_count"] - event["candidate_group_count"]
        for event in events
    )
    assert maximum_buffered <= 12


def test_act_mode_is_failure_atomic_on_missing_metadata() -> None:
    buffer = _choice_buffer()
    assert buffer.meta_list[-1] is not None
    buffer.meta_list[-1].extra_info = {}
    original_ids = [meta.sample_ids for meta in buffer.meta_list if meta is not None]

    try:
        _run_select(buffer, observe=True, mode="act")
    except RuntimeError as error:
        assert "missing_or_invalid_opportunity_metadata" in str(error)
    else:
        raise AssertionError("act mode unexpectedly fell back to FIFO")

    assert [
        meta.sample_ids for meta in buffer.meta_list if meta is not None
    ] == original_ids


def test_act_mode_is_failure_atomic_when_candidate_cap_is_exceeded() -> None:
    buffer = _choice_buffer()
    buffer.add("extra", weight=2, l1=100.0, tokens=100)
    baseline = WeightFifoSampler(
        buffer,
        max_staleness_versions=1,
        selection_candidate_watermark=None,
    )
    original_ids = [meta.sample_ids for meta in buffer.meta_list if meta is not None]
    sampler = OpportunityAtRiskShadowSampler(
        buffer=buffer,
        baseline=baseline,
        service_budget_multiplier=1.02,
        max_candidate_groups=8,
        record=lambda event: None,
        mode="act",
    )

    try:
        asyncio.run(
            sampler.select(
                current_train_weight=2,
                min_prompt_groups=4,
                max_prompt_groups=4,
            )
        )
    except RuntimeError as error:
        assert "candidate_safety_cap_exceeded" in str(error)
    else:
        raise AssertionError("act mode unexpectedly mutated an over-cap choice set")

    assert [
        meta.sample_ids for meta in buffer.meta_list if meta is not None
    ] == original_ids


def test_act_mode_rejects_more_than_one_partial_replenishment_batch() -> None:
    buffer = _choice_buffer()
    for index in range(4):
        buffer.add(
            f"excess-{index}",
            weight=2,
            l1=100.0,
            tokens=100,
        )
    baseline = WeightFifoSampler(
        buffer,
        max_staleness_versions=1,
        selection_candidate_watermark=8,
    )
    original_ids = [meta.sample_ids for meta in buffer.meta_list if meta is not None]
    sampler = OpportunityAtRiskShadowSampler(
        buffer=buffer,
        baseline=baseline,
        service_budget_multiplier=1.02,
        max_candidate_groups=25,
        record=lambda event: None,
        mode="act",
    )

    with pytest.raises(RuntimeError, match="replenishment exceeded one partial batch"):
        asyncio.run(
            sampler.select(
                current_train_weight=2,
                min_prompt_groups=4,
                max_prompt_groups=4,
            )
        )

    assert [
        meta.sample_ids for meta in buffer.meta_list if meta is not None
    ] == original_ids


def test_shadow_waits_for_configured_candidate_watermark_before_observing() -> None:
    buffer = FakeBuffer()
    for group_id in ("a", "b", "c", "d"):
        buffer.add(group_id, weight=1, l1=1.0, tokens=100)
    baseline = WeightFifoSampler(
        buffer,
        max_staleness_versions=1,
        selection_candidate_watermark=8,
    )
    events: list[dict[str, Any]] = []
    shadow = OpportunityAtRiskShadowSampler(
        buffer=buffer,
        baseline=baseline,
        service_budget_multiplier=1.02,
        max_candidate_groups=25,
        record=lambda event: events.append(dict(event)),
    )

    assert asyncio.run(
        shadow.select(
            current_train_weight=2,
            min_prompt_groups=4,
            max_prompt_groups=4,
        )
    ) == (None, 0)
    assert events == []

    for group_id in ("w", "x", "y", "z"):
        buffer.add(group_id, weight=2, l1=10.0, tokens=100)
    meta, count = asyncio.run(
        shadow.select(
            current_train_weight=2,
            min_prompt_groups=4,
            max_prompt_groups=4,
        )
    )

    assert meta is not None
    assert count == 4
    assert meta.sample_ids == ["a_g0", "b_g0", "c_g0", "d_g0"]
    assert len(events) == 1
    assert events[0]["candidate_group_count"] == 8
    assert events[0]["baseline_matches_actual"] is True


def test_missing_metadata_skips_observation_and_preserves_baseline() -> None:
    buffer = _choice_buffer()
    assert buffer.meta_list[-1] is not None
    buffer.meta_list[-1].extra_info = {}

    selected, events = _run_select(buffer, observe=True)

    assert selected == ["a_g0", "b_g0", "c_g0", "d_g0"]
    assert events[0]["skip_reason"] == "missing_or_invalid_opportunity_metadata"
    assert events[0]["baseline_matches_actual"] is False


def test_candidate_cap_skips_observation_and_preserves_baseline() -> None:
    buffer = _choice_buffer()
    buffer.add("extra", weight=2, l1=100.0, tokens=100)
    baseline = WeightFifoSampler(
        buffer,
        max_staleness_versions=1,
        selection_candidate_watermark=None,
    )
    events: list[dict[str, Any]] = []
    shadow = OpportunityAtRiskShadowSampler(
        buffer=buffer,
        baseline=baseline,
        service_budget_multiplier=1.02,
        max_candidate_groups=8,
        record=lambda event: events.append(dict(event)),
    )

    meta, groups = asyncio.run(
        shadow.select(
            current_train_weight=2,
            min_prompt_groups=4,
            max_prompt_groups=4,
        )
    )

    assert meta is not None
    assert groups == 4
    assert meta.sample_ids == ["a_g0", "b_g0", "c_g0", "d_g0"]
    assert events[0]["skip_reason"] == "candidate_safety_cap_exceeded"
    assert events[0]["baseline_matches_actual"] is False


def test_older_unready_slot_preserves_weight_fifo_wait_without_observation() -> None:
    buffer = FakeBuffer()
    buffer.add("pending", weight=1, l1=1.0, tokens=100, ready=False)
    for group_id in ("a", "b", "c", "d"):
        buffer.add(group_id, weight=2, l1=10.0, tokens=100)
    baseline = WeightFifoSampler(
        buffer,
        max_staleness_versions=1,
        selection_candidate_watermark=None,
    )
    events: list[dict[str, Any]] = []
    shadow = OpportunityAtRiskShadowSampler(
        buffer=buffer,
        baseline=baseline,
        service_budget_multiplier=1.02,
        max_candidate_groups=25,
        record=lambda event: events.append(dict(event)),
    )

    selected = asyncio.run(
        shadow.select(
            current_train_weight=2,
            min_prompt_groups=4,
            max_prompt_groups=4,
        )
    )

    assert selected == (None, 0)
    assert events == []
