# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Analyze the frozen structured-generation latency feasibility collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from tools import materialize_structured_generation_latency_pool as materializer


EXPECTED_GROUPS = 16
EXPECTED_PAIRS = 8
EXPECTED_COMPLETIONS = 2


class StructuredGenerationAnalysisError(ValueError):
    """The design, trace, or frozen runtime contract is inconsistent."""


@dataclass(frozen=True, slots=True)
class DesignItem:
    source_pool_ordinal: int
    source_prompt_id: str
    task_name: str
    pair_id: str
    stratum: Literal["short", "long"]
    left_operand: int
    right_operand: int
    expected_answer: int
    required_check_lines: int
    rendered_prompt_tokens: int


@dataclass(frozen=True, slots=True)
class Observation:
    item: DesignItem
    dispatch_ns: int
    completed_ns: int
    ready_ns: int
    archived_ns: int
    reward_mean: float
    mean_generated_tokens: float
    min_generated_tokens: int
    max_generated_tokens: int
    length_terminations: int

    @property
    def ready_latency_ns(self) -> int:
        return self.ready_ns - self.dispatch_ns


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _number(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StructuredGenerationAnalysisError(f"missing numeric summary {key}")
    number = float(value)
    if not math.isfinite(number):
        raise StructuredGenerationAnalysisError(f"non-finite summary {key}")
    return number


def load_design(path: Path) -> tuple[str, tuple[DesignItem, ...]]:
    raw = json.loads(path.read_text())
    required = {
        "schema_version",
        "analysis_status",
        "calibration_only",
        "confirmatory_eligible",
        "plan_id",
        "fixed_pool_id",
        "items",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise StructuredGenerationAnalysisError("selection design fields mismatch")
    if (
        raw["schema_version"] != 1
        or raw["analysis_status"]
        != "preregistered_controlled_generative_demand_feasibility"
        or raw["calibration_only"] is not True
        or raw["confirmatory_eligible"] is not False
        or raw["plan_id"] != materializer.PLAN_ID
    ):
        raise StructuredGenerationAnalysisError("selection design labels mismatch")
    fields = set(DesignItem.__dataclass_fields__)
    items = []
    for value in raw["items"]:
        if not isinstance(value, dict) or set(value) != fields:
            raise StructuredGenerationAnalysisError("selection item fields mismatch")
        items.append(DesignItem(**value))
    if len(items) != EXPECTED_GROUPS:
        raise StructuredGenerationAnalysisError("expected exactly 16 prompt groups")
    if [item.source_pool_ordinal for item in items] != list(range(EXPECTED_GROUPS)):
        raise StructuredGenerationAnalysisError("ordinals are not contiguous")
    by_pair: dict[str, list[DesignItem]] = defaultdict(list)
    for item in items:
        by_pair[item.pair_id].append(item)
        expected_lines = 2 if item.stratum == "short" else 16
        if (
            item.task_name != f"structured_{item.stratum}"
            or item.required_check_lines != expected_lines
            or item.expected_answer != item.left_operand + item.right_operand
            or item.rendered_prompt_tokens < 1
        ):
            raise StructuredGenerationAnalysisError("selection item value mismatch")
    if len(by_pair) != EXPECTED_PAIRS or any(
        len(pair) != 2 or {item.stratum for item in pair} != {"short", "long"}
        for pair in by_pair.values()
    ):
        raise StructuredGenerationAnalysisError("pairing is incomplete")
    return str(raw["fixed_pool_id"]), tuple(items)


def observations_from_events(
    events: Iterable[SchedulerTraceEvent], items: Sequence[DesignItem]
) -> tuple[Observation, ...]:
    events = tuple(events)
    starts = [event for event in events if event.event_type is SchedulerEventType.RUN_STARTED]
    if len(starts) != 1:
        raise StructuredGenerationAnalysisError("trace requires one run_started")
    runtime = starts[0].scalar_summaries
    if (
        runtime.get("generation_backend") != "vllm"
        or runtime.get("max_total_sequence_length") != 768
        or runtime.get("configured_max_new_tokens") != 768
        or runtime.get("generation_context_length") != 768
    ):
        raise StructuredGenerationAnalysisError("runtime generation bounds mismatch")
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
            raise StructuredGenerationAnalysisError("trace has an incomplete lifecycle")
        if event.event_type not in event_maps:
            continue
        ordinal = event.source_pool_ordinal
        if ordinal not in by_ordinal or ordinal in event_maps[event.event_type]:
            raise StructuredGenerationAnalysisError("unknown or duplicate ordinal")
        event_maps[event.event_type][ordinal] = event
    expected = set(by_ordinal)
    if any(set(values) != expected for values in event_maps.values()):
        raise StructuredGenerationAnalysisError("lifecycle coverage is incomplete")
    output = []
    for ordinal, item in sorted(by_ordinal.items()):
        dispatch = event_maps[SchedulerEventType.ATTEMPT_DISPATCHED][ordinal]
        completed = event_maps[SchedulerEventType.ROLLOUT_COMPLETED][ordinal]
        ready = event_maps[SchedulerEventType.GROUP_READY][ordinal]
        archived = event_maps[SchedulerEventType.GROUP_ARCHIVED][ordinal]
        identity_events = (dispatch, completed, ready, archived)
        if any(
            event.source_prompt_id != item.source_prompt_id
            or event.task_name != item.task_name
            or event.logical_group_id != dispatch.logical_group_id
            for event in identity_events
        ):
            raise StructuredGenerationAnalysisError("trace/design identity mismatch")
        if not (
            dispatch.monotonic_ns
            < completed.monotonic_ns
            <= ready.monotonic_ns
            <= archived.monotonic_ns
        ):
            raise StructuredGenerationAnalysisError("non-monotonic lifecycle")
        summaries = completed.scalar_summaries
        if _number(summaries, "completion_count") != EXPECTED_COMPLETIONS:
            raise StructuredGenerationAnalysisError("completion count mismatch")
        mean_tokens = _number(summaries, "mean_gen_tokens_per_sample")
        min_tokens = _number(summaries, "gen_tokens_per_sample/min")
        max_tokens = _number(summaries, "gen_tokens_per_sample/max")
        length_rate = _number(summaries, "backend_length_termination_rate")
        finish_available = _number(
            summaries, "backend_finish_reason_availability_rate"
        )
        length_count = round(length_rate * EXPECTED_COMPLETIONS)
        if (
            not min_tokens.is_integer()
            or not max_tokens.is_integer()
            or not 0 <= min_tokens <= mean_tokens <= max_tokens
            or not math.isclose(
                EXPECTED_COMPLETIONS * mean_tokens,
                min_tokens + max_tokens,
                abs_tol=1e-9,
            )
            or not math.isclose(
                length_count, length_rate * EXPECTED_COMPLETIONS, abs_tol=1e-9
            )
            or finish_available != 1.0
        ):
            raise StructuredGenerationAnalysisError("completion summaries mismatch")
        output.append(
            Observation(
                item=item,
                dispatch_ns=dispatch.monotonic_ns,
                completed_ns=completed.monotonic_ns,
                ready_ns=ready.monotonic_ns,
                archived_ns=archived.monotonic_ns,
                reward_mean=_number(summaries, "reward_mean"),
                mean_generated_tokens=mean_tokens,
                min_generated_tokens=int(min_tokens),
                max_generated_tokens=int(max_tokens),
                length_terminations=length_count,
            )
        )
    return tuple(output)


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
        rank = ((start + 1) + end) / 2
        for position in range(start, end):
            ranks[ordered[position][0]] = rank
        start = end
    return sum(
        rank * sign for rank, (_, sign) in zip(ranks, nonzero, strict=True)
    ) / sum(ranks)


def _maximum_concurrency(observations: Sequence[Observation]) -> int:
    changes = [
        item
        for observation in observations
        for item in ((observation.dispatch_ns, 1), (observation.ready_ns, -1))
    ]
    active = maximum = 0
    for _, change in sorted(changes, key=lambda value: (value[0], value[1])):
        active += change
        maximum = max(maximum, active)
    return maximum


def summarize(observations: Sequence[Observation]) -> dict[str, object]:
    by_stratum: dict[str, list[Observation]] = defaultdict(list)
    by_pair: dict[str, dict[str, Observation]] = defaultdict(dict)
    for observation in observations:
        by_stratum[observation.item.stratum].append(observation)
        by_pair[observation.item.pair_id][observation.item.stratum] = observation
    if set(by_stratum) != {"short", "long"} or any(
        set(pair) != {"short", "long"} for pair in by_pair.values()
    ):
        raise StructuredGenerationAnalysisError("observations are not fully paired")
    short = by_stratum["short"]
    long = by_stratum["long"]
    generated_ratio = statistics.median(
        [value for item in long for value in (item.min_generated_tokens, item.max_generated_tokens)]
    ) / statistics.median(
        [value for item in short for value in (item.min_generated_tokens, item.max_generated_tokens)]
    )
    ready_ratio = statistics.median([item.ready_latency_ns for item in long]) / statistics.median(
        [item.ready_latency_ns for item in short]
    )
    differences = [
        pair["long"].ready_latency_ns - pair["short"].ready_latency_ns
        for _, pair in sorted(by_pair.items())
    ]
    ready_sign_count = sum(value > 0 for value in differences)
    rank_biserial = _matched_rank_biserial(differences)
    reward_means = {
        stratum: statistics.fmean(item.reward_mean for item in values)
        for stratum, values in by_stratum.items()
    }
    length_rates = {
        stratum: sum(item.length_terminations for item in values)
        / (len(values) * EXPECTED_COMPLETIONS)
        for stratum, values in by_stratum.items()
    }
    maximum_concurrency = _maximum_concurrency(observations)
    checks = {
        "all_expected_completions": len(observations) == EXPECTED_GROUPS,
        "reward_mean_ge_0_75_each": all(value >= 0.75 for value in reward_means.values()),
        "length_termination_rate_le_0_125_each": all(
            value <= 0.125 for value in length_rates.values()
        ),
        "generated_token_median_ratio_ge_2": generated_ratio >= 2.0,
        "ready_latency_median_ratio_ge_1_5": ready_ratio >= 1.5,
        "paired_ready_longer_count_ge_7": ready_sign_count >= 7,
        "ready_latency_rank_biserial_ge_0_5": rank_biserial >= 0.5,
        "maximum_concurrent_groups_ge_2": maximum_concurrency >= 2,
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "analysis_status": "calibration_only_controlled_generative_demand",
        "calibration_only": True,
        "confirmatory_eligible": False,
        "natural_benchmark_claim_authorized": False,
        "scheduler_comparison_authorized": False,
        "replay_authorized": False,
        "training_authorized": False,
        "unit_of_analysis": "paired_structured_addition_prompt_group",
        "pairs": EXPECTED_PAIRS,
        "completions_per_group": EXPECTED_COMPLETIONS,
        "reward_mean_by_stratum": reward_means,
        "backend_length_termination_rate_by_stratum": length_rates,
        "generated_output_tokens": {"median_long_short_ratio": generated_ratio},
        "ready_latency": {
            "median_long_short_ratio": ready_ratio,
            "paired_longer_count": ready_sign_count,
            "matched_pairs_rank_biserial": rank_biserial,
        },
        "maximum_concurrent_groups": maximum_concurrency,
        "locked_checks": checks,
        "decision": (
            "pass_to_fresh_replicated_scheduler_design"
            if passed
            else "stop_no_scheduler_comparison"
        ),
        "interpretation": (
            "controlled generative-demand feasibility without injected delays; "
            "not a natural benchmark or training result"
        ),
    }


def analyze(*, trace_path: Path, manifest_path: Path, design_path: Path) -> dict[str, object]:
    pool_id, items = load_design(design_path)
    manifest = load_fixed_pool_manifest(manifest_path)
    validate_fixed_pool_materialization(manifest)
    validate_fixed_pool_manifest_design(manifest, materializer.DESIGN_ID)
    if manifest.pool_id != pool_id or manifest.design_protocol_sha256 != materializer.PLAN_SHA256:
        raise StructuredGenerationAnalysisError("manifest/design/plan binding mismatch")
    by_ordinal = {item.source_pool_ordinal: item for item in items}
    for manifest_item in manifest.items:
        item = by_ordinal.get(manifest_item.ordinal)
        if item is None or (
            manifest_item.source_prompt_id != item.source_prompt_id
            or manifest_item.task_name != item.task_name
            or manifest_item.matching_pair_id != item.pair_id
            or manifest_item.input_token_count != item.rendered_prompt_tokens
        ):
            raise StructuredGenerationAnalysisError("manifest item binding mismatch")
    report = validate_fixed_pool_trace(
        trace_path, manifest, expected_completions_per_group=EXPECTED_COMPLETIONS
    )
    if report.physical_weight_version != 0:
        raise StructuredGenerationAnalysisError("feasibility requires weight version zero")
    result = summarize(observations_from_events(iter_scheduler_trace(trace_path), items))
    result["plan_id"] = materializer.PLAN_ID
    result["source_artifacts"] = {
        "trace_sha256": _sha(trace_path),
        "manifest_sha256": _sha(manifest_path),
        "selection_design_sha256": _sha(design_path),
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
