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

"""Strictly join controlled-release lifecycle and opportunity audit ledgers."""

from __future__ import annotations

import hashlib
import math
import struct
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from tools.opportunity_loss_analysis import OpportunityAssignment


class OpportunityLedgerJoinError(ValueError):
    """Raised when raw audit ledgers cannot support an all-assignment join."""


@dataclass(frozen=True)
class ReleaseArm:
    """One registered exact-mass release arm."""

    label: str
    delay_seconds: float
    mass: int


@dataclass(frozen=True)
class OpportunityAuditContract:
    """Frozen estimator/loss contract used to recompute each opportunity row."""

    estimator_name: str = "GRPOAdvantageEstimator"
    baseline_algorithm: str = "nemo_rl_grpo_v1"
    normalize_rewards: bool = True
    normalization_epsilon: float = 1e-6
    reduction_dtype: str = "float32"
    reward_dtype: str = "float32"
    use_leave_one_out_baseline: bool = True
    allowed_reward_values: tuple[float, ...] = (0.0, 1.0)
    disable_ppo_ratio: bool = False
    positive_example_nll_weight: float = 0.0
    sequence_level_importance_ratios: bool = False
    token_level_loss: bool = True
    use_cispo: bool = False
    opportunity_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if not self.estimator_name or self.baseline_algorithm != "nemo_rl_grpo_v1":
            raise OpportunityLedgerJoinError("unsupported opportunity estimator")
        if self.reduction_dtype != "float32" or self.reward_dtype != "float32":
            raise OpportunityLedgerJoinError("opportunity replay requires float32")
        if (
            not math.isfinite(self.normalization_epsilon)
            or self.normalization_epsilon <= 0.0
            or not math.isfinite(self.opportunity_tolerance)
            or self.opportunity_tolerance < 0.0
            or len(self.allowed_reward_values) == 0
        ):
            raise OpportunityLedgerJoinError("invalid opportunity replay contract")
        if any(not math.isfinite(value) for value in self.allowed_reward_values):
            raise OpportunityLedgerJoinError("allowed rewards must be finite")


@dataclass(frozen=True)
class LedgerJoinProtocol:
    """Protocol fields needed to reconstruct and join raw audit ledgers."""

    assignment_domain: str
    assignment_seed: int
    arms: tuple[ReleaseArm, ...]
    primary_start_version: int
    primary_end_version: int
    siblings_per_group: int
    train_batch_size: int
    opportunity_audit: OpportunityAuditContract = OpportunityAuditContract()

    def __post_init__(self) -> None:
        if (
            not self.assignment_domain
            or not self.assignment_domain.isascii()
            or "\0" in self.assignment_domain
        ):
            raise OpportunityLedgerJoinError(
                "assignment_domain must be nonempty ASCII without NUL"
            )
        if isinstance(self.assignment_seed, bool) or not isinstance(
            self.assignment_seed, int
        ):
            raise OpportunityLedgerJoinError("assignment_seed must be an integer")
        if not self.arms or len({arm.label for arm in self.arms}) != len(self.arms):
            raise OpportunityLedgerJoinError("release arms must be nonempty and unique")
        for arm in self.arms:
            if (
                not arm.label
                or not arm.label.isascii()
                or not math.isfinite(arm.delay_seconds)
                or arm.delay_seconds < 0.0
                or isinstance(arm.mass, bool)
                or not isinstance(arm.mass, int)
                or arm.mass <= 0
            ):
                raise OpportunityLedgerJoinError("invalid registered release arm")
        if (
            self.primary_start_version < 0
            or self.primary_end_version < self.primary_start_version
            or self.siblings_per_group <= 0
            or self.train_batch_size <= 0
        ):
            raise OpportunityLedgerJoinError("invalid registered window or batch size")


@dataclass(frozen=True)
class JoinedOpportunityAssignment:
    """One joined primary assignment with randomized/cohort identity retained."""

    assignment_id: str
    ordinal: int
    start_version: int
    arm: str
    opportunity: float
    delivered: bool | None

    def analysis_assignment(self) -> OpportunityAssignment:
        """Project to the terminal-missingness analysis row."""
        return OpportunityAssignment(
            assignment_id=self.assignment_id,
            arm=self.arm,
            opportunity=self.opportunity,
            delivered=self.delivered,
        )


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OpportunityLedgerJoinError(f"{name} must be an integer")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpportunityLedgerJoinError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise OpportunityLedgerJoinError(f"{name} must be finite")
    return result


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise OpportunityLedgerJoinError(f"{name} must be a nonempty string")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise OpportunityLedgerJoinError(f"{name} must be boolean")
    return value


def _strings(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise OpportunityLedgerJoinError(f"{name} must be an array")
    result = tuple(_string(item, name=name) for item in value)
    if len(result) != len(set(result)):
        raise OpportunityLedgerJoinError(f"{name} contains duplicates")
    return result


def _assignment(
    protocol: LedgerJoinProtocol, ordinal: int
) -> tuple[int, int, ReleaseArm]:
    total_mass = sum(arm.mass for arm in protocol.arms)
    cardinality = 1 << 256
    limit = cardinality - cardinality % total_mass
    nonce = 0
    while True:
        payload = (
            protocol.assignment_domain.encode("ascii")
            + b"\0"
            + f"{protocol.assignment_seed}|{ordinal}|{nonce}".encode("ascii")
        )
        value = int.from_bytes(hashlib.sha256(payload).digest(), "big")
        if value < limit:
            break
        nonce += 1
    draw = value % total_mass
    cumulative = 0
    for arm in protocol.arms:
        cumulative += arm.mass
        if draw < cumulative:
            return nonce, draw, arm
    raise AssertionError("registered arm masses do not cover the draw")


def _one_or_none(
    stage_rows: Mapping[str, list[Mapping[str, Any]]], stage: str, *, group_id: str
) -> Mapping[str, Any] | None:
    rows = stage_rows.get(stage, [])
    if len(rows) > 1:
        raise OpportunityLedgerJoinError(f"{group_id}: duplicate {stage} stage")
    return rows[0] if rows else None


def _require_one(
    stage_rows: Mapping[str, list[Mapping[str, Any]]], stage: str, *, group_id: str
) -> Mapping[str, Any]:
    row = _one_or_none(stage_rows, stage, group_id=group_id)
    if row is None:
        raise OpportunityLedgerJoinError(f"{group_id}: missing {stage} stage")
    return row


def _validate_assignment_event(
    row: Mapping[str, Any],
    *,
    protocol: LedgerJoinProtocol,
    ordinal: int,
    expected_nonce: int,
    expected_draw: int,
    expected_arm: ReleaseArm,
    group_id: str,
) -> None:
    if (
        _integer(row.get("release_global_ordinal"), name="release_global_ordinal")
        != ordinal
        or _integer(row.get("release_nonce"), name="release_nonce") != expected_nonce
        or _integer(row.get("release_draw"), name="release_draw") != expected_draw
        or _string(row.get("release_arm"), name="release_arm") != expected_arm.label
        or _number(row.get("release_delay_seconds"), name="release_delay_seconds")
        != expected_arm.delay_seconds
        or _integer(row.get("release_arm_mass"), name="release_arm_mass")
        != expected_arm.mass
        or _integer(row.get("release_total_mass"), name="release_total_mass")
        != sum(arm.mass for arm in protocol.arms)
    ):
        raise OpportunityLedgerJoinError(
            f"{group_id}: controlled-release assignment reconstruction failed"
        )


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OpportunityLedgerJoinError(f"{name} must be an object")
    return value


def _f32(value: float) -> float:
    try:
        result = struct.unpack("!f", struct.pack("!f", value))[0]
    except (OverflowError, struct.error) as error:
        raise OpportunityLedgerJoinError(
            f"value is not representable as binary32: {value!r}"
        ) from error
    if not math.isfinite(result):
        raise OpportunityLedgerJoinError(f"value is not finite binary32: {value!r}")
    return result


def _f32_bits(value: float) -> bytes:
    return struct.pack("!f", _f32(value))


def _f32_add(left: float, right: float) -> float:
    return _f32(_f32(left) + _f32(right))


def _f32_subtract(left: float, right: float) -> float:
    return _f32(_f32(left) - _f32(right))


def _f32_multiply(left: float, right: float) -> float:
    return _f32(_f32(left) * _f32(right))


def _f32_divide(numerator: float, denominator: float) -> float:
    divisor = _f32(denominator)
    if divisor == 0.0:
        raise OpportunityLedgerJoinError("binary32 replay attempted division by zero")
    return _f32(_f32(numerator) / divisor)


def _replay_grpo_scalar_advantages(
    rewards: Sequence[float], contract: OpportunityAuditContract
) -> tuple[float, ...]:
    if len(rewards) < 2:
        raise OpportunityLedgerJoinError("GRPO replay requires at least two siblings")
    rounded_rewards = tuple(_f32(reward) for reward in rewards)
    epsilon = _f32(contract.normalization_epsilon)
    replayed: list[float] = []
    for focal_index, reward in enumerate(rounded_rewards):
        population = tuple(
            value
            for sibling_index, value in enumerate(rounded_rewards)
            if not contract.use_leave_one_out_baseline or sibling_index != focal_index
        )
        if len(population) < 2:
            raise OpportunityLedgerJoinError(
                "GRPO variance replay requires at least two values"
            )
        total = _f32(0.0)
        squared_total = _f32(0.0)
        for value in population:
            total = _f32_add(total, value)
            squared_total = _f32_add(squared_total, _f32_multiply(value, value))
        count = _f32(float(len(population)))
        baseline = _f32_divide(total, count)
        second_moment = _f32_divide(squared_total, count)
        population_variance = _f32_subtract(
            second_moment, _f32_multiply(baseline, baseline)
        )
        correction = _f32_divide(count, _f32(float(len(population) - 1)))
        variance = _f32_multiply(population_variance, correction)
        standard_deviation = _f32(math.sqrt(variance)) if variance > 0.0 else _f32(0.0)
        advantage = _f32_subtract(reward, baseline)
        if contract.normalize_rewards and standard_deviation > 0.0:
            advantage = _f32_divide(advantage, _f32_add(standard_deviation, epsilon))
        replayed.append(advantage)
    return tuple(replayed)


def _validate_opportunity_header(
    header: Mapping[str, Any], contract: OpportunityAuditContract
) -> None:
    if _string(header.get("estimator_name"), name="estimator_name") != (
        contract.estimator_name
    ):
        raise OpportunityLedgerJoinError("opportunity estimator disagrees")
    estimator = _mapping(header.get("estimator_settings"), name="estimator_settings")
    expected_estimator: Mapping[str, object] = {
        "baseline_algorithm": contract.baseline_algorithm,
        "normalization_epsilon": contract.normalization_epsilon,
        "normalize_rewards": contract.normalize_rewards,
        "reduction_dtype": contract.reduction_dtype,
        "reward_dtype": contract.reward_dtype,
        "use_leave_one_out_baseline": contract.use_leave_one_out_baseline,
    }
    if set(estimator) != set(expected_estimator) or any(
        estimator[key] != expected for key, expected in expected_estimator.items()
    ):
        raise OpportunityLedgerJoinError("opportunity estimator settings disagree")
    loss = _mapping(header.get("loss_settings"), name="loss_settings")
    expected_loss: Mapping[str, object] = {
        "disable_ppo_ratio": contract.disable_ppo_ratio,
        "positive_example_nll_weight": contract.positive_example_nll_weight,
        "sequence_level_importance_ratios": (contract.sequence_level_importance_ratios),
        "token_level_loss": contract.token_level_loss,
        "use_cispo": contract.use_cispo,
    }
    if set(loss) != set(expected_loss) or any(
        loss[key] != expected for key, expected in expected_loss.items()
    ):
        raise OpportunityLedgerJoinError("opportunity loss settings disagree")


def _validate_opportunity_group(
    row: Mapping[str, Any],
    *,
    lifecycle_siblings: Mapping[int, Mapping[str, Any]],
    protocol: LedgerJoinProtocol,
    group_id: str,
) -> tuple[tuple[str, ...], float]:
    contract = protocol.opportunity_audit
    sample_ids = _strings(row.get("sample_ids"), name="sample_ids")
    siblings = row.get("siblings")
    if (
        len(sample_ids) != protocol.siblings_per_group
        or not isinstance(siblings, list)
        or len(siblings) != len(sample_ids)
    ):
        raise OpportunityLedgerJoinError(
            f"{group_id}: malformed opportunity sibling summaries"
        )
    allowed_reward_bits = {_f32_bits(value) for value in contract.allowed_reward_values}
    rewards: list[float | None] = [None] * len(sample_ids)
    advantages: list[float | None] = [None] * len(sample_ids)
    tokens: list[int | None] = [None] * len(sample_ids)
    sibling_q: list[float | None] = [None] * len(sample_ids)
    seen: set[int] = set()
    for raw_sibling in siblings:
        sibling = _mapping(raw_sibling, name="sibling")
        index = _integer(sibling.get("sibling_index"), name="sibling_index")
        if index in seen or not 0 <= index < len(sample_ids):
            raise OpportunityLedgerJoinError(
                f"{group_id}: invalid or duplicate opportunity sibling index"
            )
        seen.add(index)
        lifecycle = lifecycle_siblings[index]
        sample_id = _string(sibling.get("sample_id"), name="sample_id")
        if sample_id != sample_ids[index] or sample_id != _string(
            lifecycle.get("trajectory_id"), name="trajectory_id"
        ):
            raise OpportunityLedgerJoinError(
                f"{group_id}: opportunity sibling identity disagrees"
            )
        reward = _number(sibling.get("reward"), name="reward")
        advantage = _number(sibling.get("scalar_advantage"), name="scalar_advantage")
        if _f32_bits(reward) not in allowed_reward_bits:
            raise OpportunityLedgerJoinError(
                f"{group_id}: reward outside frozen replay support"
            )
        lifecycle_reward = _number(lifecycle.get("reward"), name="reward")
        if _f32_bits(reward) != _f32_bits(lifecycle_reward) or _boolean(
            sibling.get("truncated"), name="truncated"
        ) != _boolean(lifecycle.get("truncated"), name="truncated"):
            raise OpportunityLedgerJoinError(
                f"{group_id}: opportunity and lifecycle sibling facts disagree"
            )
        token_count = _integer(
            sibling.get("valid_actor_tokens"), name="valid_actor_tokens"
        )
        if token_count < 0:
            raise OpportunityLedgerJoinError(
                f"{group_id}: valid_actor_tokens must be nonnegative"
            )
        observed_q = _number(sibling.get("opportunity"), name="sibling opportunity")
        expected_q = abs(advantage) * token_count
        if observed_q < 0.0 or not math.isclose(
            observed_q,
            expected_q,
            rel_tol=0.0,
            abs_tol=contract.opportunity_tolerance,
        ):
            raise OpportunityLedgerJoinError(
                f"{group_id}: sibling opportunity recomputation failed"
            )
        rewards[index] = reward
        advantages[index] = advantage
        tokens[index] = token_count
        sibling_q[index] = expected_q
    if any(value is None for value in (*rewards, *advantages, *tokens, *sibling_q)):
        raise OpportunityLedgerJoinError(f"{group_id}: missing opportunity sibling")
    typed_rewards = tuple(float(value) for value in rewards)
    typed_advantages = tuple(float(value) for value in advantages)
    typed_tokens = tuple(int(value) for value in tokens)
    replayed = _replay_grpo_scalar_advantages(typed_rewards, contract)
    for index, (expected, observed) in enumerate(
        zip(replayed, typed_advantages, strict=True)
    ):
        if _f32_bits(observed) != _f32_bits(expected):
            raise OpportunityLedgerJoinError(
                f"{group_id}: scalar advantage replay failed for sibling {index}"
            )
    expected_group: Mapping[str, float] = {
        "opportunity": math.fsum(float(value) for value in sibling_q),
        "l2_coefficient_mass": math.sqrt(
            math.fsum(
                advantage * advantage * token_count
                for advantage, token_count in zip(
                    typed_advantages, typed_tokens, strict=True
                )
            )
        ),
        "signed_coefficient_mass": math.fsum(
            advantage * token_count
            for advantage, token_count in zip(
                typed_advantages, typed_tokens, strict=True
            )
        ),
        "positive_coefficient_mass": math.fsum(
            max(advantage, 0.0) * token_count
            for advantage, token_count in zip(
                typed_advantages, typed_tokens, strict=True
            )
        ),
        "negative_coefficient_mass": math.fsum(
            max(-advantage, 0.0) * token_count
            for advantage, token_count in zip(
                typed_advantages, typed_tokens, strict=True
            )
        ),
    }
    for key, expected in expected_group.items():
        if not math.isclose(
            _number(row.get(key), name=key),
            expected,
            rel_tol=0.0,
            abs_tol=contract.opportunity_tolerance,
        ):
            raise OpportunityLedgerJoinError(
                f"{group_id}: group {key} recomputation failed"
            )
    nonzero_siblings = sum(advantage != 0.0 for advantage in typed_advantages)
    nonzero_tokens = sum(
        token_count
        for advantage, token_count in zip(typed_advantages, typed_tokens, strict=True)
        if advantage != 0.0
    )
    if (
        _integer(row.get("valid_actor_tokens"), name="valid_actor_tokens")
        != sum(typed_tokens)
        or _integer(
            row.get("nonzero_advantage_siblings"),
            name="nonzero_advantage_siblings",
        )
        != nonzero_siblings
        or _integer(
            row.get("nonzero_advantage_tokens"), name="nonzero_advantage_tokens"
        )
        != nonzero_tokens
    ):
        raise OpportunityLedgerJoinError(
            f"{group_id}: opportunity aggregate counts disagree"
        )
    return sample_ids, expected_group["opportunity"]


def join_opportunity_ledgers(
    *,
    protocol: LedgerJoinProtocol,
    lifecycle_rows: Sequence[Mapping[str, Any]],
    opportunity_rows: Sequence[Mapping[str, Any]],
) -> tuple[JoinedOpportunityAssignment, ...]:
    """Return strict all-assignment primary rows from two raw audit streams.

    Every primary assignment must have pre-release opportunity. A faithfully
    observed non-completion is ``delivered=False``. An absent or incomplete later
    disposition is ``delivered=None`` and is therefore handled by observed-Q bounds.
    Contradictory identities, ordering, assignments, or partial learner-step joins
    fail closed.
    """
    if not lifecycle_rows or not opportunity_rows:
        raise OpportunityLedgerJoinError("both audit ledgers must be nonempty")
    for sequence, row in enumerate(lifecycle_rows):
        if _integer(row.get("schema_version"), name="lifecycle schema_version") != 4:
            raise OpportunityLedgerJoinError("lifecycle ledger must use schema v4")
        if _integer(row.get("sequence"), name="lifecycle sequence") != sequence:
            raise OpportunityLedgerJoinError("lifecycle sequence must be contiguous")

    run_ids = {_string(row.get("run_id"), name="run_id") for row in lifecycle_rows}
    clock_ids = {
        _string(row.get("clock_domain_id"), name="clock_domain_id")
        for row in lifecycle_rows
    }
    if len(run_ids) != 1 or len(clock_ids) != 1:
        raise OpportunityLedgerJoinError(
            "lifecycle ledger must have one run and clock domain"
        )

    header_rows = [row for row in opportunity_rows if row.get("event_type") == "header"]
    if len(header_rows) != 1 or opportunity_rows[0] is not header_rows[0]:
        raise OpportunityLedgerJoinError(
            "opportunity ledger must begin with exactly one header"
        )
    _validate_opportunity_header(header_rows[0], protocol.opportunity_audit)
    for row in opportunity_rows:
        if _integer(row.get("schema_version"), name="opportunity schema_version") != 1:
            raise OpportunityLedgerJoinError("opportunity ledger must use schema v1")
        if (
            _string(row.get("run_id"), name="run_id") not in run_ids
            or _string(row.get("clock_domain_id"), name="clock_domain_id")
            not in clock_ids
        ):
            raise OpportunityLedgerJoinError("audit run or clock domains disagree")
    opportunity_sequences = [
        _integer(row.get("controller_sequence"), name="controller_sequence")
        for row in opportunity_rows
    ]
    if any(
        right <= left
        for left, right in zip(opportunity_sequences, opportunity_sequences[1:])
    ):
        raise OpportunityLedgerJoinError(
            "opportunity file order must follow controller sequence"
        )

    combined = (*lifecycle_rows, *opportunity_rows)
    controller_sequences = [
        _integer(row.get("controller_sequence"), name="controller_sequence")
        for row in combined
    ]
    if len(controller_sequences) != len(set(controller_sequences)):
        raise OpportunityLedgerJoinError(
            "shared controller sequence contains duplicates"
        )
    ordered = sorted(
        combined,
        key=lambda row: _integer(
            row.get("controller_sequence"), name="controller_sequence"
        ),
    )
    timestamps = [
        _integer(row.get("timestamp_ns"), name="timestamp_ns") for row in ordered
    ]
    if any(right < left for left, right in zip(timestamps, timestamps[1:])):
        raise OpportunityLedgerJoinError(
            "shared controller timestamps are nonmonotonic"
        )

    lifecycle_by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    learner_events: dict[int, Mapping[str, Any]] = {}
    for row in lifecycle_rows:
        if row.get("stage") == "learner_version_advanced":
            learner_version = _integer(
                row.get("learner_weight_version"), name="learner_weight_version"
            )
            if learner_version in learner_events:
                raise OpportunityLedgerJoinError("duplicate learner-version advance")
            learner_events[learner_version] = row
        else:
            lifecycle_by_group[_string(row.get("group_id"), name="group_id")].append(
                row
            )

    opportunity_by_group: dict[str, Mapping[str, Any]] = {}
    opportunity_sample_owner: dict[str, str] = {}
    completed_step_for_sample: dict[str, Mapping[str, Any]] = {}
    for row in opportunity_rows[1:]:
        event_type = row.get("event_type")
        if event_type == "group":
            group_id = _string(row.get("group_id"), name="group_id")
            if group_id in opportunity_by_group:
                raise OpportunityLedgerJoinError(
                    f"duplicate opportunity group {group_id}"
                )
            sample_ids = _strings(row.get("sample_ids"), name="sample_ids")
            if len(sample_ids) != protocol.siblings_per_group:
                raise OpportunityLedgerJoinError(
                    f"{group_id}: wrong opportunity sibling count"
                )
            q = _number(row.get("opportunity"), name="opportunity")
            if q < 0.0:
                raise OpportunityLedgerJoinError("opportunity must be nonnegative")
            opportunity_by_group[group_id] = row
            for sample_id in sample_ids:
                if sample_id in opportunity_sample_owner:
                    raise OpportunityLedgerJoinError(
                        f"duplicate opportunity sample {sample_id}"
                    )
                opportunity_sample_owner[sample_id] = group_id
        elif event_type == "train_step_completed":
            previous = _integer(
                row.get("previous_learner_version"),
                name="previous_learner_version",
            )
            learner = _integer(row.get("learner_version"), name="learner_version")
            if learner != previous + 1:
                raise OpportunityLedgerJoinError(
                    "completed learner step must advance exactly one version"
                )
            learner_event = learner_events.get(learner)
            if learner_event is None:
                raise OpportunityLedgerJoinError(
                    "completed learner step lacks lifecycle version advance"
                )
            if _integer(
                learner_event.get("start_weight_version"),
                name="start_weight_version",
            ) != previous or _integer(
                learner_event.get("controller_sequence"),
                name="controller_sequence",
            ) >= _integer(row.get("controller_sequence"), name="controller_sequence"):
                raise OpportunityLedgerJoinError(
                    "completed-step and learner-advance ordering disagree"
                )
            sample_ids = _strings(row.get("sample_ids"), name="completed sample_ids")
            if len(sample_ids) != protocol.train_batch_size:
                raise OpportunityLedgerJoinError(
                    "completed learner step has wrong batch size"
                )
            for sample_id in sample_ids:
                if sample_id not in opportunity_sample_owner:
                    raise OpportunityLedgerJoinError(
                        "completed sample lacks pre-release opportunity"
                    )
                if sample_id in completed_step_for_sample:
                    raise OpportunityLedgerJoinError(
                        "sample appears in multiple completed learner steps"
                    )
                completed_step_for_sample[sample_id] = row
        else:
            raise OpportunityLedgerJoinError(
                f"unsupported opportunity event_type={event_type!r}"
            )

    unknown_opportunity = set(opportunity_by_group).difference(lifecycle_by_group)
    if unknown_opportunity:
        raise OpportunityLedgerJoinError(
            "opportunity group lacks lifecycle: "
            + ", ".join(sorted(unknown_opportunity))
        )

    assigned: list[tuple[int, int, JoinedOpportunityAssignment]] = []
    all_ordinals: list[int] = []
    for group_id, group_rows in lifecycle_by_group.items():
        rows = sorted(
            group_rows,
            key=lambda row: _integer(
                row.get("controller_sequence"), name="controller_sequence"
            ),
        )
        stages: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            stages[_string(row.get("stage"), name="stage")].append(row)
        allowed_stages = {
            "reserved",
            "release_delay_assigned",
            "sibling_done",
            "release_delay_started",
            "release_delay_completed",
            "group_completed",
            "group_ready",
            "removed",
        }
        unknown_stages = set(stages).difference(allowed_stages)
        if unknown_stages:
            raise OpportunityLedgerJoinError(
                f"{group_id}: unsupported lifecycle stage(s) "
                + ", ".join(sorted(unknown_stages))
            )
        for unique_stage in allowed_stages.difference({"sibling_done"}):
            _one_or_none(stages, unique_stage, group_id=group_id)
        reserved = _require_one(stages, "reserved", group_id=group_id)
        assignment = _require_one(stages, "release_delay_assigned", group_id=group_id)
        if _integer(
            reserved.get("controller_sequence"), name="controller_sequence"
        ) >= _integer(
            assignment.get("controller_sequence"), name="controller_sequence"
        ):
            raise OpportunityLedgerJoinError(
                f"{group_id}: reservation must precede assignment"
            )
        ordinal = _integer(
            assignment.get("release_global_ordinal"), name="release_global_ordinal"
        )
        if ordinal < 0:
            raise OpportunityLedgerJoinError("assignment ordinal must be nonnegative")
        all_ordinals.append(ordinal)
        expected_nonce, expected_draw, expected_arm = _assignment(protocol, ordinal)
        for stage in (
            "release_delay_assigned",
            "release_delay_started",
            "release_delay_completed",
        ):
            for row in stages.get(stage, []):
                _validate_assignment_event(
                    row,
                    protocol=protocol,
                    ordinal=ordinal,
                    expected_nonce=expected_nonce,
                    expected_draw=expected_draw,
                    expected_arm=expected_arm,
                    group_id=group_id,
                )
        start_version = _integer(
            assignment.get("start_weight_version"), name="start_weight_version"
        )
        if any(
            _integer(row.get("start_weight_version"), name="start_weight_version")
            != start_version
            for row in rows
        ):
            raise OpportunityLedgerJoinError(
                f"{group_id}: inconsistent start_weight_version"
            )
        if not (
            protocol.primary_start_version
            <= start_version
            <= protocol.primary_end_version
        ):
            continue

        opportunity = opportunity_by_group.get(group_id)
        if opportunity is None:
            raise OpportunityLedgerJoinError(
                f"{group_id}: primary assignment lacks opportunity"
            )
        if (
            _integer(
                opportunity.get("start_weight_version"), name="start_weight_version"
            )
            != start_version
        ):
            raise OpportunityLedgerJoinError(
                f"{group_id}: opportunity start version disagrees"
            )
        sibling_rows = stages.get("sibling_done", [])
        if len(sibling_rows) != protocol.siblings_per_group:
            raise OpportunityLedgerJoinError(
                f"{group_id}: primary opportunity lacks complete siblings"
            )
        lifecycle_siblings: dict[int, Mapping[str, Any]] = {}
        for row in sibling_rows:
            index = _integer(row.get("sibling_idx"), name="sibling_idx")
            if (
                index in lifecycle_siblings
                or not 0 <= index < protocol.siblings_per_group
            ):
                raise OpportunityLedgerJoinError(f"{group_id}: invalid sibling index")
            lifecycle_siblings[index] = row
        sample_ids, q = _validate_opportunity_group(
            opportunity,
            lifecycle_siblings=lifecycle_siblings,
            protocol=protocol,
            group_id=group_id,
        )
        delay_started = _require_one(stages, "release_delay_started", group_id=group_id)
        last_sibling_sequence = max(
            _integer(row.get("controller_sequence"), name="controller_sequence")
            for row in sibling_rows
        )
        opportunity_sequence = _integer(
            opportunity.get("controller_sequence"), name="controller_sequence"
        )
        delay_sequence = _integer(
            delay_started.get("controller_sequence"), name="controller_sequence"
        )
        if not last_sibling_sequence < opportunity_sequence < delay_sequence:
            raise OpportunityLedgerJoinError(
                f"{group_id}: opportunity is not strictly pre-delay"
            )

        completed_rows = {
            id(completed_step_for_sample[sample_id]): completed_step_for_sample[
                sample_id
            ]
            for sample_id in sample_ids
            if sample_id in completed_step_for_sample
        }
        completed_count = sum(
            sample_id in completed_step_for_sample for sample_id in sample_ids
        )
        if completed_count not in (0, len(sample_ids)) or len(completed_rows) > 1:
            raise OpportunityLedgerJoinError(
                f"{group_id}: partial or split completed-step identity"
            )
        completed = completed_count == len(sample_ids)
        removed = _one_or_none(stages, "removed", group_id=group_id)
        for identity_stage in ("group_ready", "removed"):
            identity_row = _one_or_none(stages, identity_stage, group_id=group_id)
            if identity_row is not None:
                lifecycle_ids = _strings(
                    identity_row.get("sample_ids"),
                    name=f"{identity_stage} sample_ids",
                )
                if lifecycle_ids and lifecycle_ids != sample_ids:
                    raise OpportunityLedgerJoinError(
                        f"{group_id}: {identity_stage} sample IDs disagree"
                    )
        if removed is None:
            delivered: bool | None = True if completed else None
        else:
            if _integer(
                removed.get("controller_sequence"), name="controller_sequence"
            ) != max(
                _integer(row.get("controller_sequence"), name="controller_sequence")
                for row in rows
            ):
                raise OpportunityLedgerJoinError(
                    f"{group_id}: lifecycle continues after removal"
                )
            reason = _string(removed.get("removal_reason"), name="removal_reason")
            if reason == "unknown":
                raise OpportunityLedgerJoinError(
                    f"{group_id}: unknown terminal disposition"
                )
            if reason == "selected":
                delivered = True if completed else None
            elif reason in {
                "stale_evicted",
                "failed",
                "cancelled",
                "bounded_shutdown",
            }:
                if completed:
                    raise OpportunityLedgerJoinError(
                        f"{group_id}: nonselected group entered completed step"
                    )
                delivered = False
            else:
                raise OpportunityLedgerJoinError(
                    f"{group_id}: unsupported removal_reason={reason!r}"
                )
        if completed:
            completed_step = next(iter(completed_rows.values()))
            if _integer(
                completed_step.get("controller_sequence"), name="controller_sequence"
            ) <= max(
                _integer(row.get("controller_sequence"), name="controller_sequence")
                for row in rows
            ):
                raise OpportunityLedgerJoinError(
                    f"{group_id}: completed step does not follow group lifecycle"
                )
        assigned.append(
            (
                ordinal,
                _integer(
                    reserved.get("controller_sequence"), name="controller_sequence"
                ),
                JoinedOpportunityAssignment(
                    assignment_id=group_id,
                    ordinal=ordinal,
                    start_version=start_version,
                    arm=expected_arm.label,
                    opportunity=q,
                    delivered=delivered,
                ),
            )
        )

    if sorted(all_ordinals) != list(range(len(all_ordinals))):
        raise OpportunityLedgerJoinError(
            "assignment ordinals must be unique and contiguous"
        )
    if [
        ordinal
        for ordinal, _sequence, _row in sorted(assigned, key=lambda item: item[1])
    ] != sorted(ordinal for ordinal, _sequence, _row in assigned):
        raise OpportunityLedgerJoinError(
            "primary reservation order disagrees with assignment ordinals"
        )
    return tuple(row for _ordinal, _sequence, row in sorted(assigned))
