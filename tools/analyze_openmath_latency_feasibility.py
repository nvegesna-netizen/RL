# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Analyze the frozen, calibration-only OpenMath latency feasibility screen."""

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


BOOTSTRAP_SEED = 20260904
BOOTSTRAP_REPLICATES = 10_000
EXPECTED_GROUPS = 16
EXPECTED_PAIRS = 8
EXPECTED_COMPLETIONS = 2


class FeasibilityAnalysisError(ValueError):
    """A frozen design or collected trace is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class DesignItem:
    source_pool_ordinal: int
    source_prompt_id: str
    task_name: str
    pair_id: str
    stratum: Literal["short", "long"]
    problem_source: str
    rendered_prompt_tokens: int
    reference_solution_tokens: int
    reference_answer_token_count: int


@dataclass(frozen=True, slots=True)
class GroupObservation:
    item: DesignItem
    dispatch_ns: int
    completed_ns: int
    ready_ns: int
    archived_ns: int
    mean_generated_tokens: float
    min_generated_tokens: int
    max_generated_tokens: int
    cap_hits: int
    natural_terminations: int
    effective_output_cap: int

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


def load_design(path: str | Path) -> tuple[str, str, tuple[DesignItem, ...]]:
    """Load the strict pre-generation selection manifest."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FeasibilityAnalysisError(
            f"cannot read selection design: {error}"
        ) from error
    required = {
        "schema_version",
        "analysis_status",
        "calibration_only",
        "confirmatory_eligible",
        "fixed_pool_id",
        "fixed_pool_manifest_sha256",
        "bootstrap_seed",
        "bootstrap_reps",
        "items",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise FeasibilityAnalysisError("selection design has unexpected fields")
    if (
        raw["schema_version"] != 1
        or raw["analysis_status"] != "preregistered_pre_generation"
        or raw["calibration_only"] is not True
        or raw["confirmatory_eligible"] is not False
        or raw["bootstrap_seed"] != BOOTSTRAP_SEED
        or raw["bootstrap_reps"] != BOOTSTRAP_REPLICATES
    ):
        raise FeasibilityAnalysisError(
            "selection design does not match locked v1 semantics"
        )
    item_fields = set(DesignItem.__dataclass_fields__)
    items = []
    for value in raw["items"]:
        if not isinstance(value, dict) or set(value) != item_fields:
            raise FeasibilityAnalysisError("selection item has unexpected fields")
        try:
            item = DesignItem(**value)
        except TypeError as error:
            raise FeasibilityAnalysisError(
                f"invalid selection item: {error}"
            ) from error
        if (
            item.stratum not in {"short", "long"}
            or item.source_pool_ordinal < 0
            or item.rendered_prompt_tokens < 1
            or item.reference_solution_tokens < 1
            or item.reference_answer_token_count < 1
            or not all(
                isinstance(field, str) and field
                for field in (
                    item.source_prompt_id,
                    item.task_name,
                    item.pair_id,
                    item.problem_source,
                )
            )
        ):
            raise FeasibilityAnalysisError("invalid selection item value")
        items.append(item)
    _validate_design_items(items)
    return (
        str(raw["fixed_pool_id"]),
        str(raw["fixed_pool_manifest_sha256"]),
        tuple(items),
    )


def _validate_design_items(items: Sequence[DesignItem]) -> None:
    if len(items) != EXPECTED_GROUPS:
        raise FeasibilityAnalysisError("design requires exactly 16 disjoint problems")
    if [item.source_pool_ordinal for item in items] != list(range(EXPECTED_GROUPS)):
        raise FeasibilityAnalysisError(
            "selection ordinals must be contiguous in pool order"
        )
    if len({item.source_prompt_id for item in items}) != EXPECTED_GROUPS:
        raise FeasibilityAnalysisError("OpenMath source problems must be disjoint")
    by_pair: dict[str, list[DesignItem]] = defaultdict(list)
    for item in items:
        by_pair[item.pair_id].append(item)
    if len(by_pair) != EXPECTED_PAIRS:
        raise FeasibilityAnalysisError("design requires exactly eight matched pairs")
    for pair_id, pair in by_pair.items():
        if len(pair) != 2 or {item.stratum for item in pair} != {"short", "long"}:
            raise FeasibilityAnalysisError(f"invalid strata in pair {pair_id}")
        short = next(item for item in pair if item.stratum == "short")
        long = next(item for item in pair if item.stratum == "long")
        difference = abs(long.rendered_prompt_tokens - short.rendered_prompt_tokens)
        if long.problem_source != short.problem_source:
            raise FeasibilityAnalysisError(f"problem_source mismatch in pair {pair_id}")
        if difference > 8:
            raise FeasibilityAnalysisError(
                f"rendered-prompt token caliper failed in {pair_id}"
            )
        if long.reference_solution_tokens <= short.reference_solution_tokens:
            raise FeasibilityAnalysisError(
                f"reference strata reversed in pair {pair_id}"
            )
        if not (
            32 <= short.reference_solution_tokens <= 96
            and 256 <= long.reference_solution_tokens <= 384
        ):
            raise FeasibilityAnalysisError(
                f"reference-solution stratum band failed in {pair_id}"
            )
        if (
            abs(long.reference_answer_token_count - short.reference_answer_token_count)
            > 4
        ):
            raise FeasibilityAnalysisError(
                f"reference-answer token caliper failed in {pair_id}"
            )


def _required_number(summaries: Mapping[str, object], key: str) -> float:
    value = summaries.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FeasibilityAnalysisError(f"rollout completion lacks numeric {key}")
    number = float(value)
    if not math.isfinite(number):
        raise FeasibilityAnalysisError(f"rollout completion has non-finite {key}")
    return number


def _count_from_rate(rate: float, *, name: str) -> int:
    if not 0 <= rate <= 1:
        raise FeasibilityAnalysisError(f"{name} must lie in [0, 1]")
    count = rate * EXPECTED_COMPLETIONS
    rounded = round(count)
    if not math.isclose(count, rounded, abs_tol=1e-9):
        raise FeasibilityAnalysisError(f"{name} is incompatible with two completions")
    return int(rounded)


def observations_from_events(
    events: Iterable[SchedulerTraceEvent], items: Sequence[DesignItem]
) -> tuple[GroupObservation, ...]:
    """Join the exact lifecycle and fail closed on missing token diagnostics."""
    events = tuple(events)
    starts = [
        event for event in events if event.event_type is SchedulerEventType.RUN_STARTED
    ]
    if len(starts) != 1:
        raise FeasibilityAnalysisError("trace requires exactly one run_started event")
    runtime = starts[0].scalar_summaries
    if (
        runtime.get("generation_backend") != "vllm"
        or runtime.get("max_total_sequence_length") != 512
        or runtime.get("configured_max_new_tokens") != 512
        or runtime.get("generation_context_length") != 512
    ):
        raise FeasibilityAnalysisError(
            "exact cap semantics require trace-bound vllm, "
            "max_total_sequence_length=512, configured_max_new_tokens=512, "
            "and generation_context_length=512"
        )
    by_ordinal = {item.source_pool_ordinal: item for item in items}
    event_maps: dict[SchedulerEventType, dict[int, SchedulerTraceEvent]] = {
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
            raise FeasibilityAnalysisError(
                f"non-complete lifecycle: {event.event_type.value}"
            )
        if event.event_type not in event_maps:
            continue
        ordinal = event.source_pool_ordinal
        if ordinal not in by_ordinal or ordinal in event_maps[event.event_type]:
            raise FeasibilityAnalysisError("unknown or duplicate source pool ordinal")
        event_maps[event.event_type][ordinal] = event
    expected = set(by_ordinal)
    if any(set(values) != expected for values in event_maps.values()):
        raise FeasibilityAnalysisError(
            "every group requires dispatch/completion/ready/archive"
        )

    output = []
    for ordinal, item in sorted(by_ordinal.items()):
        dispatch = event_maps[SchedulerEventType.ATTEMPT_DISPATCHED][ordinal]
        completed = event_maps[SchedulerEventType.ROLLOUT_COMPLETED][ordinal]
        ready = event_maps[SchedulerEventType.GROUP_READY][ordinal]
        archived = event_maps[SchedulerEventType.GROUP_ARCHIVED][ordinal]
        identities = (dispatch, completed, ready, archived)
        if any(
            event.source_prompt_id != item.source_prompt_id
            or event.task_name != item.task_name
            or event.logical_group_id != dispatch.logical_group_id
            for event in identities
        ):
            raise FeasibilityAnalysisError(
                f"trace/design identity mismatch at {ordinal}"
            )
        if not (
            dispatch.monotonic_ns
            < completed.monotonic_ns
            <= ready.monotonic_ns
            <= archived.monotonic_ns
        ):
            raise FeasibilityAnalysisError(f"non-monotonic lifecycle at {ordinal}")
        summaries = completed.scalar_summaries
        if _required_number(summaries, "completion_count") != EXPECTED_COMPLETIONS:
            raise FeasibilityAnalysisError("each prompt group requires two completions")
        mean_tokens = _required_number(summaries, "mean_gen_tokens_per_sample")
        min_tokens = _required_number(summaries, "gen_tokens_per_sample/min")
        max_tokens = _required_number(summaries, "gen_tokens_per_sample/max")
        finish_reason_availability = _required_number(
            summaries, "backend_finish_reason_availability_rate"
        )
        length_termination_rate = _required_number(
            summaries, "backend_length_termination_rate"
        )
        natural_rate = _required_number(summaries, "natural_termination_rate")
        effective_cap = 512 - item.rendered_prompt_tokens
        if (
            min_tokens < 0
            or min_tokens > mean_tokens
            or mean_tokens > max_tokens
            or not min_tokens.is_integer()
            or not max_tokens.is_integer()
            or not math.isclose(
                EXPECTED_COMPLETIONS * mean_tokens,
                min_tokens + max_tokens,
                abs_tol=1e-9,
            )
            or finish_reason_availability != 1.0
            or effective_cap < 1
            or max_tokens > effective_cap
        ):
            raise FeasibilityAnalysisError(
                "inconsistent token/termination/effective-cap diagnostics"
            )
        output.append(
            GroupObservation(
                item=item,
                dispatch_ns=dispatch.monotonic_ns,
                completed_ns=completed.monotonic_ns,
                ready_ns=ready.monotonic_ns,
                archived_ns=archived.monotonic_ns,
                mean_generated_tokens=mean_tokens,
                min_generated_tokens=int(min_tokens),
                max_generated_tokens=int(max_tokens),
                cap_hits=_count_from_rate(
                    length_termination_rate,
                    name="backend_length_termination_rate",
                ),
                natural_terminations=_count_from_rate(
                    natural_rate, name="natural_termination_rate"
                ),
                effective_output_cap=effective_cap,
            )
        )
    return tuple(output)


def _median_ratio(long_values: Sequence[float], short_values: Sequence[float]) -> float:
    denominator = statistics.median(short_values)
    if denominator <= 0:
        raise FeasibilityAnalysisError("ratio denominator must be positive")
    return statistics.median(long_values) / denominator


def _positive_ratio(numerator: float, denominator: float) -> float:
    if numerator < 0 or denominator <= 0:
        raise FeasibilityAnalysisError("paired ratio denominator must be positive")
    return numerator / denominator


def _matched_rank_biserial(differences: Sequence[float]) -> float:
    """Wilcoxon matched-pairs rank-biserial effect, oriented long minus short."""
    nonzero = [
        (abs(value), (value > 0) - (value < 0)) for value in differences if value
    ]
    if not nonzero:
        return 0.0
    ordered = sorted(enumerate(nonzero), key=lambda item: item[1][0])
    ranks = [0.0] * len(nonzero)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1][0] == ordered[start][1][0]:
            end += 1
        average_rank = ((start + 1) + end) / 2
        for position in range(start, end):
            ranks[ordered[position][0]] = average_rank
        start = end
    signed_rank_sum = sum(
        rank * sign for rank, (_, sign) in zip(ranks, nonzero, strict=True)
    )
    return signed_rank_sum / sum(ranks)


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _bootstrap(
    pairs: Sequence[tuple[GroupObservation, GroupObservation]],
    value,
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
    changes = []
    for item in observations:
        changes.append((item.dispatch_ns, 1))
        changes.append((item.ready_ns, -1))
    active = maximum = 0
    # A group whose ready event is simultaneous with another dispatch no
    # longer occupies a slot at that instant.
    for _, change in sorted(changes, key=lambda value: (value[0], value[1])):
        active += change
        maximum = max(maximum, active)
    return maximum


def analyze_observations(observations: Sequence[GroupObservation]) -> dict[str, object]:
    """Compute locked Stage A gates and separately labeled robustness screens."""
    if len(observations) != EXPECTED_GROUPS:
        raise FeasibilityAnalysisError("analysis requires 16 observations")
    _validate_design_items([observation.item for observation in observations])
    for observation in observations:
        if not (
            observation.dispatch_ns
            < observation.completed_ns
            <= observation.ready_ns
            <= observation.archived_ns
        ):
            raise FeasibilityAnalysisError("observation has a non-monotonic lifecycle")
        if (
            observation.min_generated_tokens < 0
            or observation.min_generated_tokens > observation.mean_generated_tokens
            or observation.mean_generated_tokens > observation.max_generated_tokens
            or not math.isclose(
                EXPECTED_COMPLETIONS * observation.mean_generated_tokens,
                observation.min_generated_tokens + observation.max_generated_tokens,
                abs_tol=1e-9,
            )
            or observation.cap_hits < 0
            or observation.cap_hits > EXPECTED_COMPLETIONS
            or observation.natural_terminations < 0
            or observation.natural_terminations > EXPECTED_COMPLETIONS
            or observation.effective_output_cap
            != 512 - observation.item.rendered_prompt_tokens
            or observation.max_generated_tokens > observation.effective_output_cap
        ):
            raise FeasibilityAnalysisError("observation has inconsistent diagnostics")
    by_pair: dict[str, dict[str, GroupObservation]] = defaultdict(dict)
    by_stratum: dict[str, list[GroupObservation]] = defaultdict(list)
    for observation in observations:
        by_pair[observation.item.pair_id][observation.item.stratum] = observation
        by_stratum[observation.item.stratum].append(observation)
    if len(by_pair) != EXPECTED_PAIRS or any(
        set(pair) != {"short", "long"} for pair in by_pair.values()
    ):
        raise FeasibilityAnalysisError("observations do not cover eight complete pairs")
    pairs = [(pair["short"], pair["long"]) for _, pair in sorted(by_pair.items())]
    short = by_stratum["short"]
    long = by_stratum["long"]

    reference_ratio = _median_ratio(
        [item.item.reference_solution_tokens for item in long],
        [item.item.reference_solution_tokens for item in short],
    )
    long_output_tokens = [
        token
        for item in long
        for token in (item.min_generated_tokens, item.max_generated_tokens)
    ]
    short_output_tokens = [
        token
        for item in short
        for token in (item.min_generated_tokens, item.max_generated_tokens)
    ]
    output_ratio = _median_ratio(
        long_output_tokens,
        short_output_tokens,
    )
    ready_ratio = _median_ratio(
        [item.ready_latency_ns for item in long],
        [item.ready_latency_ns for item in short],
    )
    rollout_ratio = _median_ratio(
        [item.rollout_latency_ns for item in long],
        [item.rollout_latency_ns for item in short],
    )
    ready_rank_biserial = _matched_rank_biserial(
        [
            long_item.ready_latency_ns - short_item.ready_latency_ns
            for short_item, long_item in pairs
        ]
    )
    rollout_rank_biserial = _matched_rank_biserial(
        [
            long_item.rollout_latency_ns - short_item.rollout_latency_ns
            for short_item, long_item in pairs
        ]
    )
    pair_ready_ratios = [
        _positive_ratio(long_item.ready_latency_ns, short_item.ready_latency_ns)
        for short_item, long_item in pairs
    ]
    pair_rollout_ratios = [
        _positive_ratio(long_item.rollout_latency_ns, short_item.rollout_latency_ns)
        for short_item, long_item in pairs
    ]
    pair_output_ratios = [
        _positive_ratio(
            long_item.mean_generated_tokens, short_item.mean_generated_tokens
        )
        for short_item, long_item in pairs
    ]
    ready_sign_count = sum(value > 1 for value in pair_ready_ratios)
    rollout_sign_count = sum(value > 1 for value in pair_rollout_ratios)
    output_sign_count = sum(value > 1 for value in pair_output_ratios)
    cap_rates = {
        stratum: sum(item.cap_hits for item in values)
        / (len(values) * EXPECTED_COMPLETIONS)
        for stratum, values in by_stratum.items()
    }
    near_cap_rates = {
        stratum: sum(
            token >= 0.90 * item.effective_output_cap
            for item in values
            for token in (item.min_generated_tokens, item.max_generated_tokens)
        )
        / (len(values) * EXPECTED_COMPLETIONS)
        for stratum, values in by_stratum.items()
    }
    maximum_concurrency = _max_concurrency(observations)

    def ratio_for(draw, attribute):
        return _median_ratio(
            [getattr(long_item, attribute) for _, long_item in draw],
            [getattr(short_item, attribute) for short_item, _ in draw],
        )

    ready_bootstrap = _bootstrap(
        pairs, lambda draw: ratio_for(draw, "ready_latency_ns")
    )
    rollout_bootstrap = _bootstrap(
        pairs, lambda draw: ratio_for(draw, "rollout_latency_ns")
    )
    output_bootstrap = _bootstrap(
        pairs,
        lambda draw: _median_ratio(
            [
                token
                for _, long_item in draw
                for token in (
                    long_item.min_generated_tokens,
                    long_item.max_generated_tokens,
                )
            ],
            [
                token
                for short_item, _ in draw
                for token in (
                    short_item.min_generated_tokens,
                    short_item.max_generated_tokens,
                )
            ],
        ),
    )
    rank_bootstrap = _bootstrap(
        pairs,
        lambda draw: _matched_rank_biserial(
            [
                long_item.ready_latency_ns - short_item.ready_latency_ns
                for short_item, long_item in draw
            ]
        ),
    )

    locked_checks = {
        "reference_median_ratio_ge_3": reference_ratio >= 3.0,
        "output_median_ratio_ge_1_5": output_ratio >= 1.5,
        "cap_hit_rate_lt_0_2_each": all(value < 0.20 for value in cap_rates.values()),
        "cap_hit_difference_le_0_1": abs(cap_rates["long"] - cap_rates["short"])
        <= 0.10,
        "ready_latency_median_ratio_ge_1_25": ready_ratio >= 1.25,
        "ready_latency_matched_rank_biserial_ge_0_2": ready_rank_biserial >= 0.20,
        "ready_pair_sign_count_ge_6": ready_sign_count >= 6,
        "max_concurrency_gt_1": maximum_concurrency > 1,
    }
    passed = all(locked_checks.values())
    return {
        "schema_version": 1,
        "analysis_status": "calibration_only_openmath_natural_solution_length",
        "calibration_only": True,
        "confirmatory_eligible": False,
        "unit_of_analysis": "matched_problem_pair",
        "pairs": EXPECTED_PAIRS,
        "completions_per_group": EXPECTED_COMPLETIONS,
        "reference_solution_tokens": {"median_long_short_ratio": reference_ratio},
        "generated_output_tokens": {
            "median_long_short_ratio": output_ratio,
            "completion_count_per_stratum": len(long_output_tokens),
            "pair_long_short_ratios": pair_output_ratios,
            "pair_sign_count": output_sign_count,
            **output_bootstrap,
        },
        "backend_length_termination": {
            "rate_by_stratum": cap_rates,
            "count_by_stratum": {
                stratum: sum(item.cap_hits for item in values)
                for stratum, values in by_stratum.items()
            },
            "runtime_binding": {
                "generation_backend": "vllm",
                "max_total_sequence_length": 512,
                "configured_max_new_tokens": 512,
                "generation_context_length": 512,
            },
            "effective_output_cap_by_ordinal": {
                str(item.item.source_pool_ordinal): item.effective_output_cap
                for item in observations
            },
            "maximum_generated_to_effective_cap_fraction": max(
                item.max_generated_tokens / item.effective_output_cap
                for item in observations
            ),
            "near_cap_rate_by_stratum": near_cap_rates,
            "near_cap_threshold_fraction": 0.90,
            "near_cap_is_diagnostic_not_a_locked_gate": True,
        },
        "environment_natural_termination_rate": {
            stratum: sum(item.natural_terminations for item in values)
            / (len(values) * EXPECTED_COMPLETIONS)
            for stratum, values in by_stratum.items()
        },
        "ready_latency": {
            "median_long_short_ratio": ready_ratio,
            "matched_pairs_rank_biserial": ready_rank_biserial,
            "pair_long_short_ratios": pair_ready_ratios,
            "pair_sign_count": ready_sign_count,
            "median_ratio_bootstrap": ready_bootstrap,
            "matched_pairs_rank_biserial_bootstrap": rank_bootstrap,
        },
        "rollout_latency": {
            "median_long_short_ratio": rollout_ratio,
            "matched_pairs_rank_biserial": rollout_rank_biserial,
            "pair_long_short_ratios": pair_rollout_ratios,
            "pair_sign_count": rollout_sign_count,
            **rollout_bootstrap,
        },
        "maximum_concurrent_groups": maximum_concurrency,
        "locked_stage_a_checks": locked_checks,
        "stronger_descriptive_robustness": {
            "output_median_ratio_ge_1_75": output_ratio >= 1.75,
            "ready_latency_median_ratio_ge_1_5": ready_ratio >= 1.50,
            "ready_pair_sign_count_ge_7": ready_sign_count >= 7,
        },
        "decision": "pass_to_fresh_replicated_design" if passed else "stop_no_replay",
        "interpretation": (
            "matched natural-stratum calibration; association is not a causal effect "
            "of reference-solution length and this run is not confirmatory"
        ),
    }


def analyze(
    *, trace_path: Path, manifest_path: Path, design_path: Path
) -> dict[str, object]:
    pool_id, manifest_sha, items = load_design(design_path)
    manifest = load_fixed_pool_manifest(manifest_path)
    validate_fixed_pool_materialization(manifest)
    validate_fixed_pool_manifest_design(manifest, "openmath_latency_feasibility_v1")
    if manifest.pool_id != pool_id or manifest.manifest_sha256 != manifest_sha:
        raise FeasibilityAnalysisError("selection design/fixed-pool identity mismatch")
    by_ordinal = {item.source_pool_ordinal: item for item in items}
    for manifest_item in manifest.items:
        design_item = by_ordinal.get(manifest_item.ordinal)
        if design_item is None or (
            manifest_item.source_prompt_id != design_item.source_prompt_id
            or manifest_item.task_name != design_item.task_name
            or manifest_item.matching_pair_id != design_item.pair_id
            or manifest_item.input_token_count != design_item.rendered_prompt_tokens
        ):
            raise FeasibilityAnalysisError("selection design/fixed-pool item mismatch")
    trace_report = validate_fixed_pool_trace(
        trace_path,
        manifest,
        expected_completions_per_group=EXPECTED_COMPLETIONS,
    )
    if trace_report.physical_weight_version != 0:
        raise FeasibilityAnalysisError(
            "calibration requires physical_weight_version=0 throughout collection"
        )
    result = analyze_observations(
        observations_from_events(iter_scheduler_trace(trace_path), items)
    )
    result["source_artifacts"] = {
        "trace_sha256": _sha256(trace_path),
        "manifest_sha256": _sha256(manifest_path),
        "selection_design_sha256": _sha256(design_path),
        "pool_id": manifest.pool_id,
        "physical_weight_version": trace_report.physical_weight_version,
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
