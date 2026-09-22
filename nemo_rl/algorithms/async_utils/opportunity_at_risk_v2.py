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

"""Default-off multi-scorer OARS-v2 observer and fail-closed actuator."""

from __future__ import annotations

import itertools
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from nemo_rl.algorithms.async_utils.opportunity_at_risk import (
    OPPORTUNITY_GROUP_ID_KEY,
    OPPORTUNITY_L1_KEY,
    OPPORTUNITY_L2_KEY,
    OPPORTUNITY_VALID_TOKENS_KEY,
)
from nemo_rl.algorithms.async_utils.replay_buffer import TQReplayBuffer
from nemo_rl.algorithms.async_utils.rollout_lifecycle import RolloutRemovalReason
from nemo_rl.algorithms.async_utils.staleness_sampler import WeightFifoSampler
from nemo_rl.data_plane import KVBatchMeta


OPPORTUNITY_REWARD_MEAN_KEY = "gradient_opportunity_reward_mean"
OPPORTUNITY_REWARD_VARIANCE_KEY = "gradient_opportunity_reward_variance"

OARSV2_SCORERS = (
    "age",
    "reward_variance_risk",
    "token_normalized_m4_risk",
    "absolute_m4_risk",
)
OARSV2_ACTUATION_SCORERS = ("reward_variance_risk", "absolute_m4_risk")


class OARSV2SelectionFallback(RuntimeError):
    """Raised internally when a scorer must fall back to FIFO."""


@dataclass(frozen=True)
class OARSV2Candidate:
    """Scalar-only candidate metadata available before learner selection."""

    group_id: str
    start_weight_version: int
    ready_timestamp_ns: int
    l1: float
    l2: float
    valid_actor_tokens: int
    reward_mean: float
    reward_variance: float


@dataclass(frozen=True)
class OARSV2Selection:
    """One deterministic scorer proposal and its audit facts."""

    scorer: str
    proposed_group_ids: tuple[str, ...]
    proposed_l1: float
    proposed_l2: float
    proposed_valid_actor_tokens: int
    proposed_imminent_l1: float
    baseline_valid_actor_tokens: int
    minimum_tokens: int
    maximum_tokens: int
    combination_count: int
    feasible_combination_count: int
    candidate_count: int
    search_candidate_count: int
    search_strategy: Literal["exact", "mandatory_safe_prune"]


def _finite_nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return converted


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def compute_reward_moments(rewards: Sequence[float]) -> tuple[float, float]:
    """Return finite population mean and variance for one sibling group."""
    values = tuple(_finite(reward, name="reward") for reward in rewards)
    if not values:
        raise ValueError("reward moments require at least one sibling")
    mean = math.fsum(values) / len(values)
    variance = math.fsum((reward - mean) ** 2 for reward in values) / len(values)
    return mean, variance


def _score_candidate(
    candidate: OARSV2Candidate,
    *,
    scorer: str,
    current_train_weight: int,
    decision_timestamp_ns: int,
    max_staleness_versions: int,
) -> tuple[float, ...]:
    lag = current_train_weight - candidate.start_weight_version
    imminent = (
        candidate.start_weight_version + max_staleness_versions <= current_train_weight
    )
    if scorer == "age":
        return (
            float(imminent),
            float(lag),
            float(decision_timestamp_ns - candidate.ready_timestamp_ns),
            float(-candidate.valid_actor_tokens),
        )
    if scorer == "reward_variance_risk":
        value = candidate.reward_variance
    elif scorer == "token_normalized_m4_risk":
        value = candidate.l1 / candidate.valid_actor_tokens
    elif scorer == "absolute_m4_risk":
        value = candidate.l1
    else:
        raise ValueError(f"unknown OARS-v2 scorer: {scorer}")
    return (value if imminent else 0.0, value, float(-candidate.valid_actor_tokens))


def _score_combination(
    combination: Sequence[OARSV2Candidate],
    *,
    scorer: str,
    current_train_weight: int,
    decision_timestamp_ns: int,
    max_staleness_versions: int,
) -> tuple[float, ...]:
    individual = [
        _score_candidate(
            candidate,
            scorer=scorer,
            current_train_weight=current_train_weight,
            decision_timestamp_ns=decision_timestamp_ns,
            max_staleness_versions=max_staleness_versions,
        )
        for candidate in combination
    ]
    return tuple(
        math.fsum(score[dimension] for score in individual)
        for dimension in range(len(individual[0]))
    )


def _search_candidates(
    candidates: Sequence[OARSV2Candidate],
    *,
    baseline_group_ids: Sequence[str],
    scorer: str,
    current_train_weight: int,
    decision_timestamp_ns: int,
    max_staleness_versions: int,
    exact_search_max_candidates: int,
) -> tuple[list[OARSV2Candidate], Literal["exact", "mandatory_safe_prune"]]:
    ordered = sorted(candidates, key=lambda candidate: candidate.group_id)
    if len(ordered) <= exact_search_max_candidates:
        return ordered, "exact"
    baseline_ids = set(baseline_group_ids)
    mandatory_ids = baseline_ids | {
        candidate.group_id
        for candidate in ordered
        if candidate.start_weight_version + max_staleness_versions
        <= current_train_weight
    }
    if len(mandatory_ids) > exact_search_max_candidates:
        raise OARSV2SelectionFallback("mandatory_candidates_exceed_exact_search_cap")
    mandatory = [
        candidate for candidate in ordered if candidate.group_id in mandatory_ids
    ]
    optional = [
        candidate for candidate in ordered if candidate.group_id not in mandatory_ids
    ]
    optional.sort(
        key=lambda candidate: _score_candidate(
            candidate,
            scorer=scorer,
            current_train_weight=current_train_weight,
            decision_timestamp_ns=decision_timestamp_ns,
            max_staleness_versions=max_staleness_versions,
        ),
        reverse=True,
    )
    pruned = mandatory + optional[: exact_search_max_candidates - len(mandatory)]
    pruned.sort(key=lambda candidate: candidate.group_id)
    return pruned, "mandatory_safe_prune"


def select_two_sided_oars_v2(
    candidates: Sequence[OARSV2Candidate],
    *,
    baseline_group_ids: Sequence[str],
    current_train_weight: int,
    decision_timestamp_ns: int,
    max_staleness_versions: int,
    scorer: str,
    minimum_service_multiplier: float,
    maximum_service_multiplier: float,
    exact_search_max_candidates: int,
    deadline_ns: Optional[int] = None,
) -> OARSV2Selection:
    """Select one deterministic fixed-cardinality batch within a FIFO token band."""
    if scorer not in OARSV2_SCORERS:
        raise ValueError(f"unknown OARS-v2 scorer: {scorer}")
    cardinality = len(baseline_group_ids)
    if cardinality < 1:
        raise ValueError("baseline_group_ids must be nonempty")
    if len(candidates) < cardinality:
        raise ValueError("candidate set is smaller than the baseline batch")
    if (
        not math.isfinite(minimum_service_multiplier)
        or not math.isfinite(maximum_service_multiplier)
        or minimum_service_multiplier <= 0
        or minimum_service_multiplier > 1
        or maximum_service_multiplier < 1
        or minimum_service_multiplier > maximum_service_multiplier
    ):
        raise ValueError("invalid two-sided service multipliers")
    if exact_search_max_candidates < cardinality:
        raise ValueError("exact_search_max_candidates is below batch cardinality")
    if max_staleness_versions < 0:
        raise ValueError("max_staleness_versions must be nonnegative")
    by_id = {candidate.group_id: candidate for candidate in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("candidate group IDs must be unique")
    try:
        baseline = tuple(by_id[group_id] for group_id in baseline_group_ids)
    except KeyError as error:
        raise ValueError("baseline batch must be contained in candidates") from error
    baseline_tokens = sum(candidate.valid_actor_tokens for candidate in baseline)
    minimum_tokens = math.ceil(baseline_tokens * minimum_service_multiplier - 1e-12)
    maximum_tokens = math.floor(baseline_tokens * maximum_service_multiplier + 1e-12)
    search, search_strategy = _search_candidates(
        candidates,
        baseline_group_ids=baseline_group_ids,
        scorer=scorer,
        current_train_weight=current_train_weight,
        decision_timestamp_ns=decision_timestamp_ns,
        max_staleness_versions=max_staleness_versions,
        exact_search_max_candidates=exact_search_max_candidates,
    )
    best: tuple[OARSV2Candidate, ...] | None = None
    best_score: tuple[float, ...] | None = None
    combination_count = 0
    feasible_count = 0
    for combination in itertools.combinations(search, cardinality):
        if deadline_ns is not None and time.perf_counter_ns() > deadline_ns:
            raise OARSV2SelectionFallback("decision_time_budget_exceeded")
        combination_count += 1
        tokens = sum(candidate.valid_actor_tokens for candidate in combination)
        if tokens < minimum_tokens or tokens > maximum_tokens:
            continue
        feasible_count += 1
        score = _score_combination(
            combination,
            scorer=scorer,
            current_train_weight=current_train_weight,
            decision_timestamp_ns=decision_timestamp_ns,
            max_staleness_versions=max_staleness_versions,
        )
        if best_score is None or score > best_score:
            best = combination
            best_score = score
    if best is None:
        raise RuntimeError("FIFO baseline was unexpectedly absent from feasible search")
    return OARSV2Selection(
        scorer=scorer,
        proposed_group_ids=tuple(candidate.group_id for candidate in best),
        proposed_l1=math.fsum(candidate.l1 for candidate in best),
        proposed_l2=math.fsum(candidate.l2 for candidate in best),
        proposed_valid_actor_tokens=sum(
            candidate.valid_actor_tokens for candidate in best
        ),
        proposed_imminent_l1=math.fsum(
            candidate.l1
            for candidate in best
            if candidate.start_weight_version + max_staleness_versions
            <= current_train_weight
        ),
        baseline_valid_actor_tokens=baseline_tokens,
        minimum_tokens=minimum_tokens,
        maximum_tokens=maximum_tokens,
        combination_count=combination_count,
        feasible_combination_count=feasible_count,
        candidate_count=len(candidates),
        search_candidate_count=len(search),
        search_strategy=search_strategy,
    )


class OpportunityAtRiskV2ShadowRecorder:
    """In-memory canonical JSONL ledger for OARS-v2 decisions."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        mode: Literal["observe", "act"] = "observe",
        actuation_scorer: Optional[str] = None,
        candidate_window_policy: Literal["natural_eager", "controlled_frontier"],
        selection_candidate_watermark: Optional[int],
        minimum_service_multiplier: float,
        maximum_service_multiplier: float,
        max_candidate_groups: int,
        exact_search_max_candidates: int,
        decision_time_budget_ns: int,
    ) -> None:
        if mode == "observe" and actuation_scorer is not None:
            raise ValueError("observe mode forbids an actuation scorer")
        if mode == "act" and actuation_scorer not in OARSV2_ACTUATION_SCORERS:
            raise ValueError("act mode requires a supported actuation scorer")
        header: dict[str, Any] = {
            "event_type": "header",
            "schema_version": self.SCHEMA_VERSION if mode == "observe" else 2,
            "mode": mode,
            "policy": (
                "multi_scorer_oars_v2_shadow"
                if mode == "observe"
                else "multi_scorer_oars_v2_actuator"
            ),
            "candidate_window_policy": candidate_window_policy,
            "selection_candidate_watermark": selection_candidate_watermark,
            "scorers": list(OARSV2_SCORERS),
            "minimum_service_multiplier": minimum_service_multiplier,
            "maximum_service_multiplier": maximum_service_multiplier,
            "max_candidate_groups": max_candidate_groups,
            "exact_search_max_candidates": exact_search_max_candidates,
            "decision_time_budget_ns": decision_time_budget_ns,
            "decision_time_budget_scope": "per_scorer",
            "candidate_mutation": (
                "none" if mode == "observe" else "configured_scorer_exact_removal"
            ),
        }
        if mode == "act":
            header.update(
                {
                    "actuation_scorer": actuation_scorer,
                    "stale_replenishment_policy": "one_batch_drop_newest_excess_v1",
                }
            )
        self._events: list[dict[str, Any]] = [header]

    @property
    def events(self) -> tuple[Mapping[str, Any], ...]:
        """Return immutable views of recorded events for tests and diagnostics."""
        return tuple(self._events)

    def append(self, event: Mapping[str, Any]) -> None:
        """Append one fully materialized shadow-decision event."""
        self._events.append({"event_type": "decision", **dict(event)})

    def flush_jsonl(self, output_path: str) -> None:
        """Write the complete canonical ledger at controller shutdown."""
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


class OpportunityAtRiskV2ShadowSampler:
    """Observe all v2 proposals and optionally enact one validated scorer."""

    EXPECTED_BATCH_GROUPS = 4

    def __init__(
        self,
        *,
        buffer: TQReplayBuffer,
        baseline: WeightFifoSampler,
        mode: Literal["observe", "act"] = "observe",
        actuation_scorer: Optional[str] = None,
        candidate_window_policy: Literal["natural_eager", "controlled_frontier"],
        minimum_service_multiplier: float,
        maximum_service_multiplier: float,
        max_candidate_groups: int,
        exact_search_max_candidates: int,
        decision_time_budget_ns: int,
        record: Callable[[Mapping[str, Any]], None],
    ) -> None:
        if (
            candidate_window_policy == "natural_eager"
            and baseline.selection_candidate_watermark is not None
        ):
            raise ValueError(
                "natural_eager OARS-v2 requires WeightFIFO without a watermark"
            )
        if candidate_window_policy == "controlled_frontier" and (
            baseline.selection_candidate_watermark is None
            or baseline.selection_candidate_watermark <= self.EXPECTED_BATCH_GROUPS
        ):
            raise ValueError(
                "controlled_frontier OARS-v2 requires a WeightFIFO watermark "
                "greater than the batch cardinality"
            )
        if mode == "observe" and actuation_scorer is not None:
            raise ValueError("observe mode forbids an actuation scorer")
        if mode == "act" and actuation_scorer not in OARSV2_ACTUATION_SCORERS:
            raise ValueError("act mode requires a supported actuation scorer")
        if mode == "act" and candidate_window_policy != "controlled_frontier":
            raise ValueError("act mode requires a controlled candidate frontier")
        if max_candidate_groups < self.EXPECTED_BATCH_GROUPS:
            raise ValueError("max_candidate_groups must be at least four")
        if not (
            self.EXPECTED_BATCH_GROUPS
            <= exact_search_max_candidates
            <= max_candidate_groups
        ):
            raise ValueError("exact-search cap must be between four and max candidates")
        if decision_time_budget_ns < 1:
            raise ValueError("decision_time_budget_ns must be positive")
        if (
            not math.isfinite(minimum_service_multiplier)
            or not math.isfinite(maximum_service_multiplier)
            or minimum_service_multiplier <= 0
            or minimum_service_multiplier > 1
            or maximum_service_multiplier < 1
            or minimum_service_multiplier > maximum_service_multiplier
        ):
            raise ValueError("invalid two-sided service multipliers")
        self._buffer = buffer
        self._baseline = baseline
        self._minimum_service_multiplier = minimum_service_multiplier
        self._maximum_service_multiplier = maximum_service_multiplier
        self._max_candidate_groups = max_candidate_groups
        self._exact_search_max_candidates = exact_search_max_candidates
        self._decision_time_budget_ns = decision_time_budget_ns
        self._record = record
        self._mode = mode
        self._actuation_scorer = actuation_scorer
        self._replenishment_credits = 0
        self._pending_excess_sample_ids: set[tuple[str, ...]] = set()
        self._candidate_excess_removed_groups_total = 0
        self._stale_evicted_groups_total = 0
        self._replenishment_batches_earned_total = 0
        self._replenishment_batches_consumed_total = 0

    def _consume_replenishment_credit(self) -> bool:
        if self._replenishment_credits == 0:
            return False
        self._replenishment_credits -= 1
        self._replenishment_batches_consumed_total += 1
        return True

    async def admit(self, *, trainer_version_fn: Callable[[], int]) -> Optional[int]:
        """Admit ordinary cadence or one replacement after stale eviction."""
        if self._mode == "observe":
            return await self._baseline.admit(trainer_version_fn=trainer_version_fn)
        return await self._baseline.admit_with_replenishment(
            trainer_version_fn=trainer_version_fn,
            consume_replenishment=self._consume_replenishment_credit,
        )

    async def evict(self, *, current_train_weight: int) -> int:
        """Drop bounded candidate excess, then replace a newly stale batch."""
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
            self._candidate_excess_removed_groups_total += excess_removed
        self._pending_excess_sample_ids.clear()
        stale_evicted = await self._baseline.evict(
            current_train_weight=current_train_weight
        )
        if self._mode == "act" and stale_evicted:
            if stale_evicted > self.EXPECTED_BATCH_GROUPS:
                raise RuntimeError(
                    "OARS-v2 actuation liveness invariant violated: one decision "
                    "evicted more than one batch"
                )
            self._replenishment_credits += 1
            self._stale_evicted_groups_total += stale_evicted
            self._replenishment_batches_earned_total += 1
        return excess_removed + stale_evicted

    @property
    def is_on_policy(self) -> bool:
        """Delegate the baseline off-policyness fact unchanged."""
        return self._baseline.is_on_policy

    def required_buffer_capacity(self, groups_per_step: int) -> Optional[int]:
        """Delegate the baseline buffer-capacity requirement unchanged."""
        return self._baseline.required_buffer_capacity(groups_per_step)

    @staticmethod
    def _candidate(
        meta: KVBatchMeta,
        *,
        start_weight_version: int,
        ready_timestamp_ns: int,
    ) -> OARSV2Candidate:
        info = meta.extra_info
        group_id = info[OPPORTUNITY_GROUP_ID_KEY]
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("opportunity group ID must be a nonempty string")
        valid_tokens = info[OPPORTUNITY_VALID_TOKENS_KEY]
        if not isinstance(valid_tokens, int) or isinstance(valid_tokens, bool):
            raise ValueError("valid actor tokens must be an integer")
        if valid_tokens < 1:
            raise ValueError("valid actor tokens must be positive")
        if not isinstance(ready_timestamp_ns, int) or ready_timestamp_ns < 0:
            raise ValueError("ready timestamp must be a nonnegative integer")
        return OARSV2Candidate(
            group_id=group_id,
            start_weight_version=start_weight_version,
            ready_timestamp_ns=ready_timestamp_ns,
            l1=_finite_nonnegative(info[OPPORTUNITY_L1_KEY], name="L1 opportunity"),
            l2=_finite_nonnegative(info[OPPORTUNITY_L2_KEY], name="L2 opportunity"),
            valid_actor_tokens=valid_tokens,
            reward_mean=_finite(info[OPPORTUNITY_REWARD_MEAN_KEY], name="reward mean"),
            reward_variance=_finite_nonnegative(
                info[OPPORTUNITY_REWARD_VARIANCE_KEY], name="reward variance"
            ),
        )

    @staticmethod
    def _group_id_from_meta(meta: KVBatchMeta) -> str:
        group_ids: list[str] = []
        for sample_id in meta.sample_ids:
            group_id, separator, sibling_index = sample_id.rpartition("_g")
            if separator and group_id and sibling_index == "0":
                group_ids.append(group_id)
        if len(group_ids) != 1:
            raise ValueError("candidate metadata must contain exactly one _g0 ID")
        return group_ids[0]

    def _proposal_record(
        self,
        *,
        selected_group_ids: Sequence[str],
        candidates: Sequence[OARSV2Candidate],
        baseline_group_ids: Sequence[str],
        current_train_weight: int,
    ) -> dict[str, Any]:
        selected_ids = set(selected_group_ids)
        baseline_ids = set(baseline_group_ids)
        selected = [
            candidate for candidate in candidates if candidate.group_id in selected_ids
        ]
        unselected = [
            candidate
            for candidate in candidates
            if candidate.group_id not in selected_ids
        ]

        def is_imminent(candidate: OARSV2Candidate) -> bool:
            """Return whether deferral across the update would expire a group."""
            return (
                candidate.start_weight_version + self._baseline.max_staleness_versions
                <= current_train_weight
            )

        return {
            "proposed_group_ids": list(selected_group_ids),
            "proposed_l1": math.fsum(candidate.l1 for candidate in selected),
            "proposed_l2": math.fsum(candidate.l2 for candidate in selected),
            "proposed_valid_actor_tokens": sum(
                candidate.valid_actor_tokens for candidate in selected
            ),
            "proposed_imminent_l1": math.fsum(
                candidate.l1 for candidate in selected if is_imminent(candidate)
            ),
            "proposed_reward_mean": (
                math.fsum(candidate.reward_mean for candidate in selected)
                / len(selected)
            ),
            "proposed_reward_variance_sum": math.fsum(
                candidate.reward_variance for candidate in selected
            ),
            "fifo_overlap_count": len(selected_ids & baseline_ids),
            "retained_fifo_group_ids": sorted(selected_ids & baseline_ids),
            "replacement_group_ids": sorted(selected_ids - baseline_ids),
            "deferred_fifo_group_ids": sorted(baseline_ids - selected_ids),
            "deliberate_shed_group_ids": sorted(
                candidate.group_id for candidate in unselected if is_imminent(candidate)
            ),
            "deferred_group_ids": sorted(
                candidate.group_id
                for candidate in unselected
                if not is_imminent(candidate)
            ),
        }

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
        eligible_indices = [
            index for index in in_window_indices if self._buffer.ready_list[index]
        ]
        eligible_indices.sort(
            key=lambda index: (self._buffer.start_weight_list[index], index)
        )
        watermark = self._baseline.selection_candidate_watermark
        if watermark is not None and len(eligible_indices) < watermark:
            return None
        if (
            self._mode == "act"
            and watermark is not None
            and len(eligible_indices) > watermark + self.EXPECTED_BATCH_GROUPS - 1
        ):
            raise RuntimeError(
                "OARS-v2 actuation refused: candidate replenishment exceeded one "
                "partial batch"
            )
        candidate_indices = (
            eligible_indices[:watermark] if watermark is not None else eligible_indices
        )
        if self._mode == "act":
            pending_excess_sample_ids: set[tuple[str, ...]] = set()
            for index in eligible_indices[len(candidate_indices) :]:
                meta = self._buffer.meta_list[index]
                if meta is None:
                    raise RuntimeError(
                        "OARS-v2 actuation refused: ready candidate metadata disappeared"
                    )
                pending_excess_sample_ids.add(tuple(meta.sample_ids))
            self._pending_excess_sample_ids = pending_excess_sample_ids
            baseline_indices = candidate_indices
        else:
            target_version = min(
                self._buffer.start_weight_list[index] for index in in_window_indices
            )
            baseline_indices = [
                index
                for index in candidate_indices
                if self._buffer.start_weight_list[index] == target_version
            ]
        requested = min(len(baseline_indices), max_prompt_groups)
        if requested < min_prompt_groups:
            return None
        baseline_indices = baseline_indices[:requested]
        baseline_group_ids: list[str] = []
        try:
            for index in baseline_indices:
                baseline_meta = self._buffer.meta_list[index]
                if baseline_meta is None:
                    raise ValueError("ready baseline metadata disappeared")
                baseline_group_ids.append(self._group_id_from_meta(baseline_meta))
        except (TypeError, ValueError):
            baseline_group_ids = []
        event: dict[str, Any] = {
            "actual_selected_group_count": None,
            "actual_selected_group_ids": None,
            "baseline_group_ids": baseline_group_ids,
            "baseline_matches_actual": None,
            "candidate_excess_count": len(eligible_indices) - len(candidate_indices),
            "candidate_group_count": len(candidate_indices),
            "current_learner_version": current_train_weight,
            "decision_latency_ns": 0,
            "proposals": {},
            "skip_reason": None,
            "eligible_candidate_count": len(eligible_indices),
            "liveness_accounting": {
                "candidate_excess_pending_groups": len(self._pending_excess_sample_ids),
                "candidate_excess_removed_groups_total": (
                    self._candidate_excess_removed_groups_total
                ),
                "replenishment_batches_consumed_total": (
                    self._replenishment_batches_consumed_total
                ),
                "replenishment_batches_earned_total": (
                    self._replenishment_batches_earned_total
                ),
                "replenishment_credits_outstanding": self._replenishment_credits,
                "stale_evicted_groups_total": self._stale_evicted_groups_total,
            },
        }
        if requested != self.EXPECTED_BATCH_GROUPS:
            event["skip_reason"] = "baseline_cardinality_not_four"
        elif len(candidate_indices) > self._max_candidate_groups:
            event["skip_reason"] = "candidate_safety_cap_exceeded"
        else:
            try:
                candidates: list[OARSV2Candidate] = []
                by_index: dict[int, OARSV2Candidate] = {}
                for index in candidate_indices:
                    meta = self._buffer.meta_list[index]
                    ready_timestamp = self._buffer.ready_timestamp_ns_list[index]
                    if meta is None or ready_timestamp is None:
                        raise ValueError("ready candidate metadata disappeared")
                    candidate = self._candidate(
                        meta,
                        start_weight_version=self._buffer.start_weight_list[index],
                        ready_timestamp_ns=ready_timestamp,
                    )
                    if candidate.group_id != self._group_id_from_meta(meta):
                        raise ValueError("opportunity and sample group IDs disagree")
                    if candidate.ready_timestamp_ns > started_ns:
                        raise ValueError("ready timestamp is later than the decision")
                    candidates.append(candidate)
                    by_index[index] = candidate
                baseline = [by_index[index] for index in baseline_indices]
                baseline_ids = [candidate.group_id for candidate in baseline]
                event["baseline_group_ids"] = baseline_ids
                event["candidates"] = [
                    {
                        "group_id": candidate.group_id,
                        "start_weight_version": candidate.start_weight_version,
                        "learner_version_lag": (
                            current_train_weight - candidate.start_weight_version
                        ),
                        "ready_timestamp_ns": candidate.ready_timestamp_ns,
                        "ready_age_ns": started_ns - candidate.ready_timestamp_ns,
                        "l1": candidate.l1,
                        "l2": candidate.l2,
                        "valid_actor_tokens": candidate.valid_actor_tokens,
                        "reward_mean": candidate.reward_mean,
                        "reward_variance": candidate.reward_variance,
                        "becomes_stale_next_update": (
                            candidate.start_weight_version
                            + self._baseline.max_staleness_versions
                            <= current_train_weight
                        ),
                        "fifo_selected": candidate.group_id in set(baseline_ids),
                    }
                    for candidate in candidates
                ]
                event["baseline_l1"] = math.fsum(candidate.l1 for candidate in baseline)
                event["baseline_l2"] = math.fsum(candidate.l2 for candidate in baseline)
                event["baseline_valid_actor_tokens"] = sum(
                    candidate.valid_actor_tokens for candidate in baseline
                )
                event["baseline_reward_mean"] = math.fsum(
                    candidate.reward_mean for candidate in baseline
                ) / len(baseline)
                event["baseline_reward_variance_sum"] = math.fsum(
                    candidate.reward_variance for candidate in baseline
                )
                proposals: dict[str, Any] = {}
                for scorer in OARSV2_SCORERS:
                    scorer_started_ns = time.perf_counter_ns()
                    try:
                        proposal = select_two_sided_oars_v2(
                            candidates,
                            baseline_group_ids=baseline_ids,
                            current_train_weight=current_train_weight,
                            decision_timestamp_ns=started_ns,
                            max_staleness_versions=(
                                self._baseline.max_staleness_versions
                            ),
                            scorer=scorer,
                            minimum_service_multiplier=(
                                self._minimum_service_multiplier
                            ),
                            maximum_service_multiplier=(
                                self._maximum_service_multiplier
                            ),
                            exact_search_max_candidates=(
                                self._exact_search_max_candidates
                            ),
                            deadline_ns=(
                                scorer_started_ns + self._decision_time_budget_ns
                            ),
                        )
                        proposals[scorer] = self._proposal_record(
                            selected_group_ids=proposal.proposed_group_ids,
                            candidates=candidates,
                            baseline_group_ids=baseline_ids,
                            current_train_weight=current_train_weight,
                        ) | {
                            "status": "proposed",
                            "minimum_tokens": proposal.minimum_tokens,
                            "maximum_tokens": proposal.maximum_tokens,
                            "combination_count": proposal.combination_count,
                            "feasible_combination_count": (
                                proposal.feasible_combination_count
                            ),
                            "search_candidate_count": (proposal.search_candidate_count),
                            "search_strategy": proposal.search_strategy,
                        }
                    except OARSV2SelectionFallback as error:
                        proposals[scorer] = self._proposal_record(
                            selected_group_ids=baseline_ids,
                            candidates=candidates,
                            baseline_group_ids=baseline_ids,
                            current_train_weight=current_train_weight,
                        ) | {
                            "status": "fallback",
                            "fallback_reason": str(error),
                        }
                    proposals[scorer]["decision_latency_ns"] = (
                        time.perf_counter_ns() - scorer_started_ns
                    )
                event["proposals"] = proposals
            except (KeyError, TypeError, ValueError):
                event["skip_reason"] = "missing_or_invalid_opportunity_metadata"
            except RuntimeError as error:
                event["skip_reason"] = "selector_invariant_failure"
                event["selector_error"] = str(error)
        event["decision_latency_ns"] = time.perf_counter_ns() - started_ns
        return event

    @staticmethod
    def _actual_group_ids(
        selected_meta: Optional[KVBatchMeta], *, selected_group_count: int
    ) -> Optional[tuple[str, ...]]:
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
        """Observe every scorer proposal, then execute FIFO or one frozen scorer."""
        event = self._observe(
            current_train_weight=current_train_weight,
            min_prompt_groups=min_prompt_groups,
            max_prompt_groups=max_prompt_groups,
        )
        if self._mode == "act" and event is not None:
            if event["skip_reason"] is not None:
                raise RuntimeError(f"OARS-v2 actuation refused: {event['skip_reason']}")
            assert self._actuation_scorer is not None
            proposal = event["proposals"][self._actuation_scorer]
            if proposal["status"] != "proposed":
                raise RuntimeError(
                    "OARS-v2 actuation refused: configured scorer fell back: "
                    f"{proposal.get('fallback_reason', 'unknown')}"
                )
            proposed_group_ids = tuple(proposal["proposed_group_ids"])
            if (
                len(proposed_group_ids) != self.EXPECTED_BATCH_GROUPS
                or len(set(proposed_group_ids)) != self.EXPECTED_BATCH_GROUPS
            ):
                raise RuntimeError(
                    "OARS-v2 actuation refused: proposal cardinality or identity invalid"
                )
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
                            "OARS-v2 actuation refused: duplicate ready group ID"
                        )
                    candidate_indices_by_id[group_id] = index
            try:
                selected_indices = [
                    candidate_indices_by_id[group_id] for group_id in proposed_group_ids
                ]
            except KeyError as error:
                raise RuntimeError(
                    "OARS-v2 actuation refused: proposed group is no longer ready"
                ) from error
            selected_metas = [
                self._buffer.meta_list[index] for index in selected_indices
            ]
            if any(meta is None for meta in selected_metas):
                raise RuntimeError(
                    "OARS-v2 actuation refused: proposed metadata disappeared"
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
            actual_ids = self._actual_group_ids(
                selected_meta, selected_group_count=selected_group_count
            )
            event["actual_selected_group_count"] = selected_group_count
            event["actual_selected_group_ids"] = (
                list(actual_ids) if actual_ids is not None else None
            )
            event["baseline_matches_actual"] = actual_ids == tuple(
                event["baseline_group_ids"]
            )
            event["mode"] = self._mode
            if self._actuation_scorer is not None:
                event["actuation_scorer"] = self._actuation_scorer
                event["proposal_matches_actual"] = actual_ids == tuple(
                    event["proposals"][self._actuation_scorer]["proposed_group_ids"]
                )
            self._record(event)
        return selected_meta, selected_group_count
