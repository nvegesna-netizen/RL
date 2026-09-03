# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Analyze the frozen, calibration-only sliding-puzzle feasibility screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from nemo_rl.algorithms.async_utils.fixed_pool import (
    load_fixed_pool_manifest,
    validate_fixed_pool_manifest_design,
    validate_fixed_pool_materialization,
    validate_fixed_pool_trace,
)
from nemo_rl.algorithms.async_utils.scheduler_trace import (
    SchedulerEventType,
    SchedulerTraceEvent,
    iter_scheduler_trace,
)


EXPECTED_GROUPS = 16
EXPECTED_PAIRS = 8
EXPECTED_COMPLETIONS = 2
BOOTSTRAP_SEED = 20260905
BOOTSTRAP_REPLICATES = 10_000
EXPECTED_LOCKED_GATES = {
    "action_format_valid_rate_min": 0.9,
    "truncation_rate_max_exclusive": 0.1,
    "easy_max_turn_rate_max": 0.25,
    "turn_median_ratio_min": 2.0,
    "generated_token_median_ratio_min": 1.5,
    "pair_turn_sign_count_min": 6,
    "ready_latency_median_ratio_min": 1.5,
    "ready_latency_rank_biserial_min": 0.3,
    "pair_ready_sign_count_min": 6,
}
EXPECTED_RUNTIME = {
    "fixed_pool_design_id": "sliding_puzzle_latency_feasibility_v1",
    "generation_backend": "vllm",
    "max_total_sequence_length": 2048,
    "configured_max_new_tokens": 128,
    "generation_context_length": 2048,
    "generation_temperature": 1.0,
    "generation_top_p": 0.999,
    "generation_top_k": 10_000,
    "generation_study_seed": 53001,
    "grpo_seed": 20260901,
    "num_generations_per_prompt": 2,
    "max_rollout_turns": 12,
    "num_prompts_per_step": 4,
    "max_inflight_prompts": 4,
    "max_buffered_rollouts": 8,
    "generation_use_async_rollouts": True,
    "tokenizer_eos_token_present": True,
    "effective_stop_token_count": 1,
    "effective_stop_string_count": 0,
    "generation_ignore_eos": False,
    "vllm_skip_tokenizer_init": True,
    "vllm_include_stop_str_in_output": True,
    "finish_reason_code_schema_version": 1,
}


class PuzzleFeasibilityAnalysisError(ValueError):
    """The frozen design or collected trace is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class DesignItem:
    source_pool_ordinal: int
    source_prompt_id: str
    task_name: str
    pair_id: str
    stratum: Literal["easy", "hard"]
    optimal_distance: int
    rendered_prompt_tokens: int
    board_state_sha256: str


@dataclass(frozen=True, slots=True)
class GroupObservation:
    item: DesignItem
    dispatch_ns: int
    completed_ns: int
    ready_ns: int
    archived_ns: int
    mean_turns: float
    min_turns: int
    max_turns: int
    mean_generated_tokens: float
    min_generated_tokens: int
    max_generated_tokens: int
    solved_completions: int
    truncations: int
    max_turns_reached: int
    action_turns: int
    format_valid_actions: int
    legal_moves: int
    invalid_format_actions: int
    invalid_moves: int
    view_actions: int

    @property
    def ready_latency_ns(self) -> int:
        return self.ready_ns - self.dispatch_ns

    @property
    def rollout_latency_ns(self) -> int:
        return self.completed_ns - self.dispatch_ns


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_number(summaries: Mapping[str, object], key: str) -> float:
    value = summaries.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PuzzleFeasibilityAnalysisError(f"missing numeric rollout summary {key}")
    result = float(value)
    if not math.isfinite(result):
        raise PuzzleFeasibilityAnalysisError(f"non-finite rollout summary {key}")
    return result


def _integer(summaries: Mapping[str, object], key: str) -> int:
    value = _required_number(summaries, key)
    if value < 0 or not value.is_integer():
        raise PuzzleFeasibilityAnalysisError(f"{key} must be a nonnegative integer")
    return int(value)


def _count_from_rate(summaries: Mapping[str, object], key: str) -> int:
    rate = _required_number(summaries, key)
    count = rate * EXPECTED_COMPLETIONS
    if not 0 <= rate <= 1 or not math.isclose(count, round(count), abs_tol=1e-9):
        raise PuzzleFeasibilityAnalysisError(f"invalid two-completion rate {key}")
    return round(count)


def load_design(path: str | Path) -> tuple[str, str, tuple[DesignItem, ...]]:
    """Load and validate the exact pre-generation design artifact."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PuzzleFeasibilityAnalysisError(
            f"cannot read selection design: {error}"
        ) from error
    required = {
        "schema_version",
        "analysis_status",
        "calibration_only",
        "confirmatory_eligible",
        "fixed_pool_id",
        "fixed_pool_manifest_sha256",
        "selection_seed",
        "order_seed",
        "generation_study_seed",
        "bootstrap_seed",
        "bootstrap_replicates",
        "max_moves",
        "max_rollout_turns",
        "max_total_sequence_length",
        "max_new_tokens_per_turn",
        "num_generations_per_prompt",
        "temperature",
        "top_p",
        "top_k",
        "locked_gates",
        "items",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise PuzzleFeasibilityAnalysisError("selection design has unexpected fields")
    if (
        raw["schema_version"] != 1
        or raw["analysis_status"] != "preregistered_pre_generation"
        or raw["calibration_only"] is not True
        or raw["confirmatory_eligible"] is not False
        or raw["selection_seed"] != 20260903
        or raw["order_seed"] != 44001
        or raw["generation_study_seed"] != 53001
        or raw["bootstrap_seed"] != BOOTSTRAP_SEED
        or raw["bootstrap_replicates"] != BOOTSTRAP_REPLICATES
        or raw["max_moves"] != 12
        or raw["max_rollout_turns"] != 12
        or raw["max_total_sequence_length"] != 2048
        or raw["max_new_tokens_per_turn"] != 128
        or raw["num_generations_per_prompt"] != EXPECTED_COMPLETIONS
        or raw["temperature"] != 1.0
        or raw["top_p"] != 0.999
        or raw["top_k"] != 10_000
        or raw["locked_gates"] != EXPECTED_LOCKED_GATES
    ):
        raise PuzzleFeasibilityAnalysisError(
            "selection design violates locked v1 semantics"
        )
    item_fields = set(DesignItem.__dataclass_fields__)
    items: list[DesignItem] = []
    if not isinstance(raw["items"], list):
        raise PuzzleFeasibilityAnalysisError("selection design items must be a list")
    for value in raw["items"]:
        if not isinstance(value, dict) or set(value) != item_fields:
            raise PuzzleFeasibilityAnalysisError("selection item has unexpected fields")
        try:
            items.append(DesignItem(**value))
        except TypeError as error:
            raise PuzzleFeasibilityAnalysisError(
                f"invalid selection item: {error}"
            ) from error
    _validate_design_items(items)
    return (
        str(raw["fixed_pool_id"]),
        str(raw["fixed_pool_manifest_sha256"]),
        tuple(items),
    )


def _validate_design_items(items: Sequence[DesignItem]) -> None:
    if len(items) != EXPECTED_GROUPS:
        raise PuzzleFeasibilityAnalysisError("design requires exactly sixteen boards")
    if [item.source_pool_ordinal for item in items] != list(range(EXPECTED_GROUPS)):
        raise PuzzleFeasibilityAnalysisError("selection ordinals must match pool order")
    if (
        len({item.source_prompt_id for item in items}) != EXPECTED_GROUPS
        or len({item.board_state_sha256 for item in items}) != EXPECTED_GROUPS
    ):
        raise PuzzleFeasibilityAnalysisError("all selected boards must be distinct")
    by_pair: dict[str, list[DesignItem]] = defaultdict(list)
    for item in items:
        if (
            item.stratum not in {"easy", "hard"}
            or item.task_name != f"sliding_puzzle_{item.stratum}"
            or item.rendered_prompt_tokens < 1
            or item.rendered_prompt_tokens > 512
            or item.optimal_distance
            not in ({1, 2, 3} if item.stratum == "easy" else {10, 11, 12})
            or len(item.board_state_sha256) != 64
        ):
            raise PuzzleFeasibilityAnalysisError("invalid selection item value")
        by_pair[item.pair_id].append(item)
    if len(by_pair) != EXPECTED_PAIRS:
        raise PuzzleFeasibilityAnalysisError("design requires eight complete pairs")
    for pair_id, pair in by_pair.items():
        if len(pair) != 2 or {item.stratum for item in pair} != {"easy", "hard"}:
            raise PuzzleFeasibilityAnalysisError(f"invalid pair {pair_id}")
        if abs(pair[0].rendered_prompt_tokens - pair[1].rendered_prompt_tokens) > 8:
            raise PuzzleFeasibilityAnalysisError(
                f"input-token caliper failed in {pair_id}"
            )


def validate_runtime(
    start: SchedulerTraceEvent, *, expected: Mapping[str, object]
) -> Mapping[str, object]:
    """Validate the exact runtime map and EOS/stop provenance."""
    values = start.scalar_summaries
    if any(values.get(key) != value for key, value in expected.items()):
        raise PuzzleFeasibilityAnalysisError(
            "runtime does not match frozen puzzle design"
        )
    eos_id = values.get("tokenizer_eos_token_id")
    if (
        not isinstance(eos_id, int)
        or values.get("effective_single_stop_token_id") != eos_id
        or values.get("effective_stop_token_ids_sha256")
        != hashlib.sha256(
            json.dumps([eos_id], separators=(",", ":")).encode()
        ).hexdigest()
        or values.get("effective_stop_strings_sha256")
        != hashlib.sha256(b"[]").hexdigest()
    ):
        raise PuzzleFeasibilityAnalysisError("runtime EOS/stop provenance mismatch")
    return values


def observations_from_events(
    events: Iterable[SchedulerTraceEvent],
    items: Sequence[DesignItem],
    *,
    expected_runtime: Mapping[str, object],
    generation_seed: int,
) -> tuple[GroupObservation, ...]:
    """Join complete lifecycle events and validate puzzle telemetry."""
    events = tuple(events)
    starts = [
        event for event in events if event.event_type is SchedulerEventType.RUN_STARTED
    ]
    if len(starts) != 1:
        raise PuzzleFeasibilityAnalysisError("trace requires exactly one run_started")
    validate_runtime(starts[0], expected=expected_runtime)
    by_ordinal = {item.source_pool_ordinal: item for item in items}
    event_maps = {
        kind: {}
        for kind in (
            SchedulerEventType.ATTEMPT_DISPATCHED,
            SchedulerEventType.ROLLOUT_COMPLETED,
            SchedulerEventType.GROUP_READY,
            SchedulerEventType.GROUP_ARCHIVED,
        )
    }
    forbidden = {
        SchedulerEventType.ATTEMPT_FAILED,
        SchedulerEventType.ATTEMPT_REMOVED,
        SchedulerEventType.GROUP_EVICTED,
    }
    for event in events:
        if event.event_type in forbidden:
            raise PuzzleFeasibilityAnalysisError(
                f"forbidden lifecycle {event.event_type.value}"
            )
        if event.event_type not in event_maps:
            continue
        ordinal = event.source_pool_ordinal
        if ordinal not in by_ordinal or ordinal in event_maps[event.event_type]:
            raise PuzzleFeasibilityAnalysisError("unknown or duplicate source ordinal")
        event_maps[event.event_type][ordinal] = event
    expected_ordinals = set(by_ordinal)
    if any(set(values) != expected_ordinals for values in event_maps.values()):
        raise PuzzleFeasibilityAnalysisError(
            "every board requires a complete lifecycle"
        )

    observations: list[GroupObservation] = []
    for ordinal, item in sorted(by_ordinal.items()):
        dispatch = event_maps[SchedulerEventType.ATTEMPT_DISPATCHED][ordinal]
        completed = event_maps[SchedulerEventType.ROLLOUT_COMPLETED][ordinal]
        ready = event_maps[SchedulerEventType.GROUP_READY][ordinal]
        archived = event_maps[SchedulerEventType.GROUP_ARCHIVED][ordinal]
        lifecycle = (dispatch, completed, ready, archived)
        if any(
            event.source_prompt_id != item.source_prompt_id
            or event.task_name != item.task_name
            or event.logical_group_id != dispatch.logical_group_id
            for event in lifecycle
        ) or not (
            dispatch.monotonic_ns
            < completed.monotonic_ns
            <= ready.monotonic_ns
            <= archived.monotonic_ns
        ):
            raise PuzzleFeasibilityAnalysisError(
                f"invalid lifecycle identity at {ordinal}"
            )
        summaries = completed.scalar_summaries
        if _integer(summaries, "completion_count") != EXPECTED_COMPLETIONS:
            raise PuzzleFeasibilityAnalysisError("each board requires two completions")
        mean_turns = _required_number(summaries, "turns_per_sample/mean")
        min_turns = _integer(summaries, "turns_per_sample/min")
        max_turns = _integer(summaries, "turns_per_sample/max")
        mean_tokens = _required_number(summaries, "mean_gen_tokens_per_sample")
        min_tokens = _integer(summaries, "gen_tokens_per_sample/min")
        max_tokens = _integer(summaries, "gen_tokens_per_sample/max")
        reward_mean = _required_number(summaries, "reward_mean")
        reward_min = _required_number(summaries, "reward_min")
        reward_max = _required_number(summaries, "reward_max")
        solved = reward_mean * EXPECTED_COMPLETIONS
        if (
            min_turns < 1
            or not math.isclose(mean_turns * 2, min_turns + max_turns, abs_tol=1e-9)
            or not math.isclose(mean_tokens * 2, min_tokens + max_tokens, abs_tol=1e-9)
            or _required_number(summaries, "backend_finish_reason_availability_rate")
            != 1
            or _required_number(summaries, "action_status_availability_rate") != 1
            or _count_from_rate(summaries, "backend_abort_termination_rate")
            or _count_from_rate(summaries, "backend_other_termination_rate")
            or _count_from_rate(summaries, "backend_context_exhausted_rate")
            or _integer(summaries, "effective_engine_seed/min") != generation_seed
            or _integer(summaries, "effective_engine_seed/max") != generation_seed
            or reward_min not in {0.0, 1.0}
            or reward_max not in {0.0, 1.0}
            or not math.isclose(solved, round(solved), abs_tol=1e-9)
        ):
            raise PuzzleFeasibilityAnalysisError(
                "inconsistent puzzle rollout diagnostics"
            )
        action_turns = _integer(summaries, "action_turn_count")
        format_valid = _integer(summaries, "action_format_valid_count")
        legal_moves = _integer(summaries, "action_legal_move_count")
        invalid_format = _integer(summaries, "action_invalid_format_count")
        invalid_moves = _integer(summaries, "action_invalid_move_count")
        views = _integer(summaries, "action_view_count")
        if (
            format_valid + invalid_format != action_turns
            or legal_moves + invalid_moves + views != format_valid
            or action_turns > min_turns + max_turns
        ):
            raise PuzzleFeasibilityAnalysisError("action counters do not reconcile")
        observations.append(
            GroupObservation(
                item=item,
                dispatch_ns=dispatch.monotonic_ns,
                completed_ns=completed.monotonic_ns,
                ready_ns=ready.monotonic_ns,
                archived_ns=archived.monotonic_ns,
                mean_turns=mean_turns,
                min_turns=min_turns,
                max_turns=max_turns,
                mean_generated_tokens=mean_tokens,
                min_generated_tokens=min_tokens,
                max_generated_tokens=max_tokens,
                solved_completions=round(solved),
                truncations=_count_from_rate(summaries, "truncation_rate"),
                max_turns_reached=_count_from_rate(summaries, "max_turns_reached_rate"),
                action_turns=action_turns,
                format_valid_actions=format_valid,
                legal_moves=legal_moves,
                invalid_format_actions=invalid_format,
                invalid_moves=invalid_moves,
                view_actions=views,
            )
        )
    return tuple(observations)


def _median_ratio(numerator: Sequence[float], denominator: Sequence[float]) -> float:
    value = statistics.median(denominator)
    if value <= 0:
        raise PuzzleFeasibilityAnalysisError("ratio denominator must be positive")
    return statistics.median(numerator) / value


def _matched_rank_biserial(differences: Sequence[float]) -> float:
    nonzero = [(abs(value), 1 if value > 0 else -1) for value in differences if value]
    if not nonzero:
        return 0.0
    ordered = sorted(enumerate(nonzero), key=lambda item: item[1][0])
    ranks = [0.0] * len(nonzero)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1][0] == ordered[start][1][0]:
            end += 1
        average = ((start + 1) + end) / 2
        for position in range(start, end):
            ranks[ordered[position][0]] = average
        start = end
    return sum(
        rank * sign for rank, (_, sign) in zip(ranks, nonzero, strict=True)
    ) / sum(ranks)


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _bootstrap(
    pairs: Sequence[tuple[GroupObservation, GroupObservation]], value
) -> dict[str, list[float]]:
    rng = random.Random(BOOTSTRAP_SEED)
    samples = []
    for _ in range(BOOTSTRAP_REPLICATES):
        draw = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        samples.append(value(draw))
    return {
        "ci90": [_quantile(samples, 0.05), _quantile(samples, 0.95)],
        "ci95": [_quantile(samples, 0.025), _quantile(samples, 0.975)],
    }


def _max_concurrency(observations: Sequence[GroupObservation]) -> int:
    changes = [
        change
        for item in observations
        for change in ((item.dispatch_ns, 1), (item.ready_ns, -1))
    ]
    active = maximum = 0
    for _, change in sorted(changes, key=lambda value: (value[0], value[1])):
        active += change
        maximum = max(maximum, active)
    return maximum


def analyze_observations(observations: Sequence[GroupObservation]) -> dict[str, object]:
    """Compute the frozen feasibility metrics and stop/progression decision."""
    if len(observations) != EXPECTED_GROUPS:
        raise PuzzleFeasibilityAnalysisError("analysis requires sixteen observations")
    _validate_design_items([observation.item for observation in observations])
    by_stratum: dict[str, list[GroupObservation]] = defaultdict(list)
    by_pair: dict[str, dict[str, GroupObservation]] = defaultdict(dict)
    for observation in observations:
        by_stratum[observation.item.stratum].append(observation)
        by_pair[observation.item.pair_id][observation.item.stratum] = observation
    if len(by_pair) != 8 or any(
        set(pair) != {"easy", "hard"} for pair in by_pair.values()
    ):
        raise PuzzleFeasibilityAnalysisError("observations do not cover eight pairs")
    pairs = [(pair["easy"], pair["hard"]) for _, pair in sorted(by_pair.items())]
    easy, hard = by_stratum["easy"], by_stratum["hard"]
    turn_ratio = _median_ratio(
        [item.mean_turns for item in hard], [item.mean_turns for item in easy]
    )
    token_ratio = _median_ratio(
        [
            token
            for item in hard
            for token in (item.min_generated_tokens, item.max_generated_tokens)
        ],
        [
            token
            for item in easy
            for token in (item.min_generated_tokens, item.max_generated_tokens)
        ],
    )
    ready_ratio = _median_ratio(
        [item.ready_latency_ns for item in hard],
        [item.ready_latency_ns for item in easy],
    )
    ready_differences = [
        hard_item.ready_latency_ns - easy_item.ready_latency_ns
        for easy_item, hard_item in pairs
    ]
    ready_rank = _matched_rank_biserial(ready_differences)
    pair_turn_ratios = [
        hard_item.mean_turns / easy_item.mean_turns for easy_item, hard_item in pairs
    ]
    pair_ready_ratios = [
        hard_item.ready_latency_ns / easy_item.ready_latency_ns
        for easy_item, hard_item in pairs
    ]
    truncation_rates = {
        stratum: sum(item.truncations for item in values) / (len(values) * 2)
        for stratum, values in by_stratum.items()
    }
    max_turn_rates = {
        stratum: sum(item.max_turns_reached for item in values) / (len(values) * 2)
        for stratum, values in by_stratum.items()
    }
    action_turns = sum(item.action_turns for item in observations)
    format_valid_rate = (
        sum(item.format_valid_actions for item in observations) / action_turns
        if action_turns
        else 0.0
    )
    maximum_concurrency = _max_concurrency(observations)
    checks = {
        "action_format_valid_rate_ge_0_9": format_valid_rate >= 0.9,
        "truncation_rate_lt_0_1_each": all(
            value < 0.1 for value in truncation_rates.values()
        ),
        "easy_max_turn_rate_le_0_25": max_turn_rates["easy"] <= 0.25,
        "turn_median_ratio_ge_2": turn_ratio >= 2,
        "generated_token_median_ratio_ge_1_5": token_ratio >= 1.5,
        "pair_turn_sign_count_ge_6": sum(value > 1 for value in pair_turn_ratios) >= 6,
        "ready_latency_median_ratio_ge_1_5": ready_ratio >= 1.5,
        "ready_latency_rank_biserial_ge_0_3": ready_rank >= 0.3,
        "pair_ready_sign_count_ge_6": sum(value > 1 for value in pair_ready_ratios)
        >= 6,
        "maximum_concurrency_gt_1": maximum_concurrency > 1,
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "analysis_status": "calibration_only_sliding_puzzle_exact_distance",
        "calibration_only": True,
        "confirmatory_eligible": False,
        "replay_authorized": False,
        "training_authorized": False,
        "unit_of_analysis": "matched_board_pair",
        "pairs": 8,
        "completions_per_group": 2,
        "turns": {
            "median_hard_easy_ratio": turn_ratio,
            "pair_hard_easy_ratios": pair_turn_ratios,
        },
        "generated_tokens": {"median_hard_easy_ratio": token_ratio},
        "actions": {
            "format_valid_rate": format_valid_rate,
            "turn_count": action_turns,
            "legal_move_count": sum(item.legal_moves for item in observations),
            "invalid_format_count": sum(
                item.invalid_format_actions for item in observations
            ),
            "invalid_move_count": sum(item.invalid_moves for item in observations),
            "view_count": sum(item.view_actions for item in observations),
        },
        "truncation_rate_by_stratum": truncation_rates,
        "max_turn_rate_by_stratum": max_turn_rates,
        "solve_rate_by_stratum": {
            stratum: sum(item.solved_completions for item in values) / (len(values) * 2)
            for stratum, values in by_stratum.items()
        },
        "ready_latency": {
            "median_hard_easy_ratio": ready_ratio,
            "matched_pairs_rank_biserial": ready_rank,
            "pair_hard_easy_ratios": pair_ready_ratios,
            "pair_sign_count": sum(value > 1 for value in pair_ready_ratios),
            "median_ratio_bootstrap": _bootstrap(
                pairs,
                lambda draw: _median_ratio(
                    [hard_item.ready_latency_ns for _, hard_item in draw],
                    [easy_item.ready_latency_ns for easy_item, _ in draw],
                ),
            ),
        },
        "rollout_latency_median_hard_easy_ratio": _median_ratio(
            [item.rollout_latency_ns for item in hard],
            [item.rollout_latency_ns for item in easy],
        ),
        "maximum_concurrent_groups": maximum_concurrency,
        "locked_checks": checks,
        "decision": "pass_to_fresh_disjoint_replicated_calibration_design"
        if passed
        else "stop_no_replay",
        "interpretation": "exact-distance multi-turn feasibility calibration; this single exposed pool is not confirmatory",
    }


def analyze(
    *, trace_path: Path, manifest_path: Path, design_path: Path
) -> dict[str, object]:
    pool_id, manifest_sha, items = load_design(design_path)
    manifest = load_fixed_pool_manifest(manifest_path)
    validate_fixed_pool_materialization(manifest)
    validate_fixed_pool_manifest_design(
        manifest, "sliding_puzzle_latency_feasibility_v1"
    )
    if manifest.pool_id != pool_id or manifest.manifest_sha256 != manifest_sha:
        raise PuzzleFeasibilityAnalysisError(
            "selection design/fixed-pool identity mismatch"
        )
    by_ordinal = {item.source_pool_ordinal: item for item in items}
    for manifest_item in manifest.items:
        design_item = by_ordinal.get(manifest_item.ordinal)
        if design_item is None or (
            manifest_item.source_prompt_id != design_item.source_prompt_id
            or manifest_item.task_name != design_item.task_name
            or manifest_item.matching_pair_id != design_item.pair_id
            or manifest_item.input_token_count != design_item.rendered_prompt_tokens
        ):
            raise PuzzleFeasibilityAnalysisError(
                "selection design/fixed-pool item mismatch"
            )
    report = validate_fixed_pool_trace(
        trace_path, manifest, expected_completions_per_group=2
    )
    if report.physical_weight_version != 0:
        raise PuzzleFeasibilityAnalysisError(
            "calibration requires physical weight version zero"
        )
    events = tuple(iter_scheduler_trace(trace_path))
    starts = [
        event for event in events if event.event_type is SchedulerEventType.RUN_STARTED
    ]
    if len(starts) != 1:
        raise PuzzleFeasibilityAnalysisError("trace requires exactly one run_started")
    runtime = validate_runtime(starts[0], expected=EXPECTED_RUNTIME)
    expected_path = str(manifest_path.resolve().parent / "model_snapshot")
    expected_path_sha = hashlib.sha256(
        json.dumps(expected_path, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        runtime.get("policy_model_name_sha256") != expected_path_sha
        or runtime.get("policy_tokenizer_name_sha256") != expected_path_sha
    ):
        raise PuzzleFeasibilityAnalysisError(
            "runtime model/tokenizer do not match the manifest snapshot"
        )
    result = analyze_observations(
        observations_from_events(
            events,
            items,
            expected_runtime=EXPECTED_RUNTIME,
            generation_seed=53001,
        )
    )
    result["runtime_binding"] = dict(runtime)
    result["source_artifacts"] = {
        "trace_sha256": _sha256(trace_path),
        "manifest_sha256": _sha256(manifest_path),
        "selection_design_sha256": _sha256(design_path),
        "pool_id": manifest.pool_id,
        "physical_weight_version": report.physical_weight_version,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection-design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = analyze(
        trace_path=args.trace,
        manifest_path=args.manifest,
        design_path=args.selection_design,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
