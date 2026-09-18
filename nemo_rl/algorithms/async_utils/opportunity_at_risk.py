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

"""Default-off Opportunity-at-Risk Scheduling (OARS) observer and actuator.

Observe mode proposes a service-budgeted batch from the ready prompt groups and
delegates selection to weight FIFO. Act mode removes exactly the validated OARS
proposal. Both modes record the same decision ledger.
"""

from __future__ import annotations

import itertools
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from nemo_rl.algorithms.async_utils.staleness_sampler import (
    WeightFifoSampler,
)
from nemo_rl.algorithms.async_utils.replay_buffer import TQReplayBuffer
from nemo_rl.algorithms.async_utils.rollout_lifecycle import RolloutRemovalReason
from nemo_rl.data_plane import KVBatchMeta

OPPORTUNITY_GROUP_ID_KEY = "gradient_opportunity_group_id"
OPPORTUNITY_L1_KEY = "gradient_opportunity_l1"
OPPORTUNITY_L2_KEY = "gradient_opportunity_l2"
OPPORTUNITY_VALID_TOKENS_KEY = "gradient_opportunity_valid_actor_tokens"


@dataclass(frozen=True)
class OpportunityCandidate:
    """Pre-decision scalar metadata for one ready prompt group."""

    group_id: str
    start_weight_version: int
    l1: float
    valid_actor_tokens: int


@dataclass(frozen=True)
class BudgetedOpportunitySelection:
    """Deterministic proposal and audit counts for one decision."""

    proposed_group_ids: tuple[str, ...]
    proposed_l1: float
    proposed_valid_actor_tokens: int
    baseline_valid_actor_tokens: int
    token_budget: int
    combination_count: int
    feasible_combination_count: int


def select_baseline_budgeted_opportunity(
    candidates: Sequence[OpportunityCandidate],
    *,
    baseline_group_ids: Sequence[str],
    current_train_weight: int,
    service_budget_multiplier: float,
) -> BudgetedOpportunitySelection:
    """Return the frozen exact OARS proposal for one observable choice set."""
    cardinality = len(baseline_group_ids)
    if cardinality < 1:
        raise ValueError("baseline_group_ids must be nonempty")
    if len(candidates) < cardinality:
        raise ValueError("candidate set is smaller than the baseline batch")
    if not math.isfinite(service_budget_multiplier) or service_budget_multiplier < 1:
        raise ValueError("service_budget_multiplier must be finite and at least one")

    by_id = {candidate.group_id: candidate for candidate in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("candidate group IDs must be unique")
    try:
        baseline = tuple(by_id[group_id] for group_id in baseline_group_ids)
    except KeyError as error:
        raise ValueError("baseline batch must be contained in candidates") from error

    baseline_tokens = sum(group.valid_actor_tokens for group in baseline)
    token_budget = math.floor(baseline_tokens * service_budget_multiplier + 1e-12)
    ordered = sorted(candidates, key=lambda group: group.group_id)
    best: tuple[OpportunityCandidate, ...] | None = None
    best_score: tuple[float, float, int] | None = None
    combination_count = 0
    feasible_count = 0
    for combination in itertools.combinations(ordered, cardinality):
        combination_count += 1
        tokens = sum(group.valid_actor_tokens for group in combination)
        if tokens > token_budget:
            continue
        feasible_count += 1
        imminent_l1 = math.fsum(
            group.l1
            for group in combination
            if group.start_weight_version + 1 <= current_train_weight
        )
        score = (imminent_l1, math.fsum(group.l1 for group in combination), -tokens)
        # Iteration is group-ID ordered. Strict improvement preserves the
        # lexicographically first identity tuple on a complete score tie.
        if best_score is None or score > best_score:
            best = combination
            best_score = score

    if best is None:
        raise RuntimeError("baseline batch was unexpectedly infeasible")
    proposed_tokens = sum(group.valid_actor_tokens for group in best)
    return BudgetedOpportunitySelection(
        proposed_group_ids=tuple(group.group_id for group in best),
        proposed_l1=math.fsum(group.l1 for group in best),
        proposed_valid_actor_tokens=proposed_tokens,
        baseline_valid_actor_tokens=baseline_tokens,
        token_budget=token_budget,
        combination_count=combination_count,
        feasible_combination_count=feasible_count,
    )


class OpportunityAtRiskShadowRecorder:
    """In-memory canonical JSONL ledger for OARS decisions."""

    SCHEMA_VERSION = 3

    def __init__(
        self,
        *,
        service_budget_multiplier: float,
        max_candidate_groups: int,
        selection_candidate_watermark: Optional[int],
        mode: Literal["observe", "act"] = "observe",
    ) -> None:
        self._events: list[dict[str, Any]] = [
            {
                "event_type": "header",
                "candidate_window_policy": "oldest_ready_exact_watermark_v1",
                "max_candidate_groups": max_candidate_groups,
                "mode": mode,
                "policy": "baseline_budgeted_oars_v1",
                "schema_version": self.SCHEMA_VERSION,
                "selection_candidate_watermark": selection_candidate_watermark,
                "service_budget_multiplier": service_budget_multiplier,
                "stale_replenishment_policy": (
                    "one_batch_drop_newest_excess_v1" if mode == "act" else "none"
                ),
            }
        ]

    @property
    def events(self) -> tuple[Mapping[str, Any], ...]:
        """Return immutable views of recorded events for tests and diagnostics."""
        return tuple(self._events)

    def append(self, event: Mapping[str, Any]) -> None:
        """Append one fully materialized decision event."""
        self._events.append({"event_type": "decision", **dict(event)})

    def flush_jsonl(self, output_path: str) -> None:
        """Write the complete ledger canonically at controller shutdown."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(
            json.dumps(
                event,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
            for event in self._events
        )
        path.write_text(payload, encoding="utf-8")


class OpportunityAtRiskShadowSampler:
    """Observe OARS proposals or enact them after complete validation."""

    EXPECTED_BATCH_GROUPS = 4

    def __init__(
        self,
        *,
        buffer: TQReplayBuffer,
        baseline: WeightFifoSampler,
        service_budget_multiplier: float,
        max_candidate_groups: int,
        record: Callable[[Mapping[str, Any]], None],
        mode: Literal["observe", "act"] = "observe",
    ) -> None:
        if max_candidate_groups < self.EXPECTED_BATCH_GROUPS:
            raise ValueError(
                f"max_candidate_groups must be at least {self.EXPECTED_BATCH_GROUPS}"
            )
        self._baseline = baseline
        self._buffer = buffer
        self._service_budget_multiplier = service_budget_multiplier
        self._max_candidate_groups = max_candidate_groups
        self._record = record
        self._mode = mode
        self._replenishment_credits = 0
        self._pending_excess_sample_ids: set[tuple[str, ...]] = set()

    def _consume_replenishment_credit(self) -> bool:
        if self._replenishment_credits == 0:
            return False
        self._replenishment_credits -= 1
        return True

    async def admit(self, *, trainer_version_fn: Callable[[], int]) -> Optional[int]:
        """Admit ordinary cadence or replace one batch after stale eviction."""
        return await self._baseline.admit_with_replenishment(
            trainer_version_fn=trainer_version_fn,
            consume_replenishment=self._consume_replenishment_credit,
        )

    @property
    def is_on_policy(self) -> bool:
        """Delegate the baseline off-policyness fact unchanged."""
        return self._baseline.is_on_policy

    def required_buffer_capacity(self, groups_per_step: int) -> Optional[int]:
        """Delegate baseline capacity requirements unchanged."""
        return self._baseline.required_buffer_capacity(groups_per_step)

    async def evict(self, *, current_train_weight: int) -> int:
        """Drop prior candidate excess, then replace any newly stale batch."""
        excess_indices = [
            index
            for index, meta in enumerate(self._buffer.meta_list)
            if meta is not None
            and tuple(meta.sample_ids) in self._pending_excess_sample_ids
            and self._buffer.ready_list[index]
        ]
        excess_removed = 0
        if excess_indices:
            excess_removed = await self._buffer.remove(
                excess_indices,
                remove_in_dp=True,
                reason=RolloutRemovalReason.OARS_CANDIDATE_EXCESS,
                learner_weight_version=current_train_weight,
            )
        self._pending_excess_sample_ids.clear()

        stale_evicted = await self._baseline.evict(
            current_train_weight=current_train_weight
        )
        if self._mode == "act" and stale_evicted:
            if stale_evicted > self.EXPECTED_BATCH_GROUPS:
                raise RuntimeError(
                    "OARS actuation liveness invariant violated: one decision "
                    "evicted more than one batch"
                )
            self._replenishment_credits += 1
        return excess_removed + stale_evicted

    @staticmethod
    def _candidate(meta: KVBatchMeta, *, start_weight: int) -> OpportunityCandidate:
        info = meta.extra_info
        group_id = info[OPPORTUNITY_GROUP_ID_KEY]
        l1 = info[OPPORTUNITY_L1_KEY]
        valid_tokens = info[OPPORTUNITY_VALID_TOKENS_KEY]
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("opportunity group ID must be a nonempty string")
        if not isinstance(l1, (int, float)) or not math.isfinite(float(l1)) or l1 < 0:
            raise ValueError("group L1 opportunity must be finite and nonnegative")
        if not isinstance(valid_tokens, int) or isinstance(valid_tokens, bool):
            raise ValueError("valid actor tokens must be an integer")
        if valid_tokens < 1:
            raise ValueError("valid actor tokens must be positive")
        return OpportunityCandidate(
            group_id=group_id,
            start_weight_version=start_weight,
            l1=float(l1),
            valid_actor_tokens=valid_tokens,
        )

    def _observe(
        self,
        *,
        current_train_weight: int,
        min_prompt_groups: int,
        max_prompt_groups: int,
    ) -> Optional[dict[str, Any]]:
        started_ns = time.perf_counter_ns()
        minimum_version = max(
            0, current_train_weight - self._baseline.max_staleness_versions
        )
        in_window_indices = [
            index
            for index, start_weight in enumerate(self._buffer.start_weight_list)
            if minimum_version <= start_weight <= current_train_weight
        ]
        if not in_window_indices:
            return None
        eligible_candidate_indices = [
            index for index in in_window_indices if self._buffer.ready_list[index]
        ]
        eligible_candidate_indices.sort(
            key=lambda index: (self._buffer.start_weight_list[index], index)
        )
        watermark = self._baseline.selection_candidate_watermark
        if watermark is not None and len(eligible_candidate_indices) < watermark:
            return None
        if (
            self._mode == "act"
            and watermark is not None
            and len(eligible_candidate_indices)
            > watermark + self.EXPECTED_BATCH_GROUPS - 1
        ):
            raise RuntimeError(
                "OARS actuation refused: candidate replenishment exceeded one "
                "partial batch"
            )
        candidate_indices = eligible_candidate_indices[:watermark]
        if self._mode == "act":
            pending_excess_sample_ids: set[tuple[str, ...]] = set()
            for index in eligible_candidate_indices[len(candidate_indices) :]:
                meta = self._buffer.meta_list[index]
                if meta is None:
                    raise RuntimeError(
                        "OARS actuation refused: ready candidate metadata disappeared"
                    )
                pending_excess_sample_ids.add(tuple(meta.sample_ids))
            self._pending_excess_sample_ids = pending_excess_sample_ids
        if self._mode == "observe":
            target_version = min(
                self._buffer.start_weight_list[index] for index in in_window_indices
            )
            baseline_indices = [
                index
                for index in candidate_indices
                if self._buffer.start_weight_list[index] == target_version
            ]
            if len(baseline_indices) < min_prompt_groups:
                return None
        else:
            # Enacted OARS can leave a partial oldest weight cohort, a state
            # strict batch-FIFO cannot itself create. Define the next
            # counterfactual FIFO batch as the oldest ready groups in the fixed
            # candidate window so actuation remains live and auditable.
            baseline_indices = candidate_indices
        requested = min(len(baseline_indices), max_prompt_groups)
        baseline_indices = baseline_indices[:requested]
        event: dict[str, Any] = {
            "baseline_group_ids": [],
            "candidate_excess_count": (
                len(eligible_candidate_indices) - len(candidate_indices)
            ),
            "candidate_group_count": len(candidate_indices),
            "current_learner_version": current_train_weight,
            "decision_latency_ns": 0,
            "eligible_candidate_count": len(eligible_candidate_indices),
            "proposed_group_ids": [],
            "skip_reason": None,
        }
        if requested != self.EXPECTED_BATCH_GROUPS:
            event["skip_reason"] = "baseline_cardinality_not_four"
        elif len(candidate_indices) > self._max_candidate_groups:
            event["skip_reason"] = "candidate_safety_cap_exceeded"
        else:
            try:
                candidates = [
                    self._candidate(
                        self._buffer.meta_list[index],  # type: ignore[arg-type]
                        start_weight=self._buffer.start_weight_list[index],
                    )
                    for index in candidate_indices
                ]
                by_index = dict(zip(candidate_indices, candidates))
                baseline = [by_index[index] for index in baseline_indices]
                event["baseline_group_ids"] = [group.group_id for group in baseline]
                event["baseline_l1"] = math.fsum(group.l1 for group in baseline)
                proposal = select_baseline_budgeted_opportunity(
                    candidates,
                    baseline_group_ids=event["baseline_group_ids"],
                    current_train_weight=current_train_weight,
                    service_budget_multiplier=self._service_budget_multiplier,
                )
                event.update(
                    {
                        "baseline_valid_actor_tokens": (
                            proposal.baseline_valid_actor_tokens
                        ),
                        "combination_count": proposal.combination_count,
                        "feasible_combination_count": (
                            proposal.feasible_combination_count
                        ),
                        "proposed_group_ids": list(proposal.proposed_group_ids),
                        "proposed_l1": proposal.proposed_l1,
                        "proposed_valid_actor_tokens": (
                            proposal.proposed_valid_actor_tokens
                        ),
                        "token_budget": proposal.token_budget,
                    }
                )
            except (KeyError, RuntimeError, TypeError, ValueError):
                event["skip_reason"] = "missing_or_invalid_opportunity_metadata"
        event["decision_latency_ns"] = time.perf_counter_ns() - started_ns
        return event

    @staticmethod
    def _actual_group_ids(
        selected_meta: Optional[KVBatchMeta], *, selected_group_count: int
    ) -> Optional[tuple[str, ...]]:
        """Recover ordered group IDs from the replay payload's stable key format."""
        if selected_meta is None:
            return None
        group_ids: list[str] = []
        for sample_id in selected_meta.sample_ids:
            group_id, separator, sibling_index = sample_id.rpartition("_g")
            if separator and group_id and sibling_index == "0":
                group_ids.append(group_id)
        if len(group_ids) != selected_group_count:
            return None
        return tuple(group_ids)

    async def select(
        self,
        *,
        current_train_weight: int,
        min_prompt_groups: int,
        max_prompt_groups: int,
    ) -> tuple[Optional[KVBatchMeta], int]:
        """Observe the choice set, then execute FIFO or the validated proposal."""
        event = self._observe(
            current_train_weight=current_train_weight,
            min_prompt_groups=min_prompt_groups,
            max_prompt_groups=max_prompt_groups,
        )
        if self._mode == "act" and event is not None:
            skip_reason = event["skip_reason"]
            if skip_reason is not None:
                # Fail before mutating the replay buffer. Silent FIFO fallback
                # would dilute the randomized intervention and make compliance
                # depend on runtime failures.
                raise RuntimeError(f"OARS actuation refused: {skip_reason}")
            proposed_group_ids = tuple(event["proposed_group_ids"])
            candidate_indices_by_id: dict[str, int] = {}
            for index, ready in enumerate(self._buffer.ready_list):
                if not ready:
                    continue
                meta = self._buffer.meta_list[index]
                if meta is None:
                    continue
                group_id = meta.extra_info.get(OPPORTUNITY_GROUP_ID_KEY)
                if isinstance(group_id, str):
                    if group_id in candidate_indices_by_id:
                        raise RuntimeError(
                            "OARS actuation refused: duplicate ready group ID"
                        )
                    candidate_indices_by_id[group_id] = index
            if (
                len(proposed_group_ids) != self.EXPECTED_BATCH_GROUPS
                or len(set(proposed_group_ids)) != self.EXPECTED_BATCH_GROUPS
            ):
                raise RuntimeError(
                    "OARS actuation refused: proposal cardinality or identity invalid"
                )
            try:
                selected_indices = [
                    candidate_indices_by_id[group_id] for group_id in proposed_group_ids
                ]
            except KeyError as error:
                raise RuntimeError(
                    "OARS actuation refused: proposed group is no longer ready"
                ) from error
            selected_metas = [
                self._buffer.meta_list[index] for index in selected_indices
            ]
            if any(meta is None for meta in selected_metas):
                raise RuntimeError(
                    "OARS actuation refused: proposed metadata disappeared"
                )
            await self._buffer.remove(
                selected_indices,
                remove_in_dp=False,
                reason=RolloutRemovalReason.SELECTED,
                learner_weight_version=current_train_weight,
            )
            concrete_metas = [meta for meta in selected_metas if meta is not None]
            selected_meta = concrete_metas[0].concat(*concrete_metas[1:])
            selected_group_count = len(selected_indices)
        else:
            selected_meta, selected_group_count = await self._baseline.select(
                current_train_weight=current_train_weight,
                min_prompt_groups=min_prompt_groups,
                max_prompt_groups=max_prompt_groups,
            )
        if event is not None:
            actual_group_ids = self._actual_group_ids(
                selected_meta, selected_group_count=selected_group_count
            )
            event.update(
                {
                    "actual_selected_group_count": selected_group_count,
                    "actual_selected_group_ids": (
                        list(actual_group_ids) if actual_group_ids is not None else None
                    ),
                    "baseline_matches_actual": actual_group_ids
                    == tuple(event["baseline_group_ids"]),
                    "mode": self._mode,
                    "proposal_matches_actual": actual_group_ids
                    == tuple(event["proposed_group_ids"]),
                }
            )
            self._record(event)
        return selected_meta, selected_group_count
