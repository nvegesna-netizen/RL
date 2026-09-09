# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Analyze all three scheduler-neutral DAPO latency discovery collections."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

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
from tools import materialize_dapo_operational_latency_discovery as materializer


EXPECTED_GROUPS_PER_POOL = 16
EXPECTED_COMPLETIONS_PER_GROUP = 16


class DapoOperationalAnalysisError(ValueError):
    """A discovery input or frozen analysis requirement is inconsistent."""


@dataclass(frozen=True, slots=True)
class DesignItem:
    """Privacy-safe immutable design metadata for one unique prompt group."""

    source_pool_ordinal: int
    source_prompt_id: str
    task_name: str
    canonical_prompt_sha256: str
    source_dataset_index: int
    source_duplicate_count: int
    rendered_prompt_tokens: int


@dataclass(frozen=True, slots=True)
class Observation:
    """One completed prompt group reconstructed from the scheduler trace."""

    pool_seed: int
    item: DesignItem
    dispatch_ns: int
    ready_ns: int
    reward_mean: float
    reward_min: float
    reward_max: float
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
        raise DapoOperationalAnalysisError(f"missing numeric summary {key}")
    number = float(value)
    if not math.isfinite(number):
        raise DapoOperationalAnalysisError(f"non-finite summary {key}")
    return number


def _load_design(path: Path, expected_seed: int) -> tuple[str, tuple[DesignItem, ...]]:
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise DapoOperationalAnalysisError(
            "selection design is invalid JSON"
        ) from error
    required = {
        "schema_version",
        "analysis_status",
        "calibration_only",
        "confirmatory_eligible",
        "protocol_sha256",
        "fixed_pool_id",
        "selection_seed",
        "generation_seed",
        "items",
    }
    expected_generation = next(
        (
            generation_seed
            for selection_seed, generation_seed in materializer.POOL_SPECS
            if selection_seed == expected_seed
        ),
        None,
    )
    if (
        not isinstance(raw, dict)
        or set(raw) != required
        or raw["schema_version"] != 1
        or raw["analysis_status"] != "candidate_operational_workload_latency_discovery"
        or raw["calibration_only"] is not True
        or raw["confirmatory_eligible"] is not False
        or raw["protocol_sha256"] != materializer.PROTOCOL_SHA256
        or raw["selection_seed"] != expected_seed
        or raw["generation_seed"] != expected_generation
    ):
        raise DapoOperationalAnalysisError("selection design contract mismatch")
    fields = set(DesignItem.__dataclass_fields__)
    items = []
    for value in raw["items"]:
        if not isinstance(value, dict) or set(value) != fields:
            raise DapoOperationalAnalysisError("selection item fields mismatch")
        items.append(DesignItem(**value))
    if (
        len(items) != EXPECTED_GROUPS_PER_POOL
        or [item.source_pool_ordinal for item in items]
        != list(range(EXPECTED_GROUPS_PER_POOL))
        or len({item.canonical_prompt_sha256 for item in items})
        != EXPECTED_GROUPS_PER_POOL
        or any(
            item.task_name not in materializer.SOURCE_IDS
            or item.rendered_prompt_tokens < 1
            or item.rendered_prompt_tokens > 2048
            or item.source_duplicate_count < 1
            for item in items
        )
    ):
        raise DapoOperationalAnalysisError("selection items violate pool design")
    return str(raw["fixed_pool_id"]), tuple(items)


def _observations_from_events(
    events: Iterable[SchedulerTraceEvent],
    items: Sequence[DesignItem],
    pool_seed: int,
) -> tuple[Observation, ...]:
    events = tuple(events)
    starts = [
        event for event in events if event.event_type is SchedulerEventType.RUN_STARTED
    ]
    if len(starts) != 1:
        raise DapoOperationalAnalysisError("trace requires one run_started")
    runtime = starts[0].scalar_summaries
    if (
        runtime.get("generation_backend") != "vllm"
        or runtime.get("max_total_sequence_length") != 6144
        or runtime.get("configured_max_new_tokens") != 4096
        or runtime.get("generation_context_length") != 6144
    ):
        raise DapoOperationalAnalysisError("runtime generation bounds mismatch")
    forbidden = {
        SchedulerEventType.ATTEMPT_FAILED,
        SchedulerEventType.ATTEMPT_REMOVED,
        SchedulerEventType.GROUP_EVICTED,
        SchedulerEventType.SELECT_DECISION,
        SchedulerEventType.PROMPT_SKIPPED,
        SchedulerEventType.GROUP_REPLACED,
        SchedulerEventType.GROUP_PROMOTED,
    }
    if any(event.event_type in forbidden for event in events):
        raise DapoOperationalAnalysisError(
            "trace contains censoring, mutation, or sampler selection"
        )
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
    for event in events:
        if event.event_type not in event_maps:
            continue
        ordinal = event.source_pool_ordinal
        if ordinal not in by_ordinal or ordinal in event_maps[event.event_type]:
            raise DapoOperationalAnalysisError("unknown or duplicate source ordinal")
        event_maps[event.event_type][ordinal] = event
    expected = set(by_ordinal)
    if any(set(values) != expected for values in event_maps.values()):
        raise DapoOperationalAnalysisError("lifecycle coverage is incomplete")

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
            raise DapoOperationalAnalysisError("trace/design identity mismatch")
        if not (
            dispatch.monotonic_ns
            < completed.monotonic_ns
            <= ready.monotonic_ns
            <= archived.monotonic_ns
        ):
            raise DapoOperationalAnalysisError("non-monotonic lifecycle")
        summaries = completed.scalar_summaries
        if _number(summaries, "completion_count") != EXPECTED_COMPLETIONS_PER_GROUP:
            raise DapoOperationalAnalysisError("completion count mismatch")
        mean_tokens = _number(summaries, "mean_gen_tokens_per_sample")
        min_tokens = _number(summaries, "gen_tokens_per_sample/min")
        max_tokens = _number(summaries, "gen_tokens_per_sample/max")
        length_rate = _number(summaries, "backend_length_termination_rate")
        length_count = round(length_rate * EXPECTED_COMPLETIONS_PER_GROUP)
        if (
            not min_tokens.is_integer()
            or not max_tokens.is_integer()
            or not 0 <= min_tokens <= mean_tokens <= max_tokens <= 4096
            or not math.isclose(
                length_count,
                length_rate * EXPECTED_COMPLETIONS_PER_GROUP,
                abs_tol=1e-9,
            )
            or _number(summaries, "backend_finish_reason_availability_rate") != 1.0
        ):
            raise DapoOperationalAnalysisError("completion summaries mismatch")
        output.append(
            Observation(
                pool_seed=pool_seed,
                item=item,
                dispatch_ns=dispatch.monotonic_ns,
                ready_ns=ready.monotonic_ns,
                reward_mean=_number(summaries, "reward_mean"),
                reward_min=_number(summaries, "reward_min"),
                reward_max=_number(summaries, "reward_max"),
                mean_generated_tokens=mean_tokens,
                min_generated_tokens=int(min_tokens),
                max_generated_tokens=int(max_tokens),
                length_terminations=length_count,
            )
        )
    return tuple(output)


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average = (start + 1 + end) / 2
        for index, _ in ordered[start:end]:
            ranks[index] = average
        start = end
    return ranks


def _pooled_within_pool_spearman(observations: Sequence[Observation]) -> float:
    centered_load: list[float] = []
    centered_latency: list[float] = []
    for pool_seed, _ in materializer.POOL_SPECS:
        pool = [item for item in observations if item.pool_seed == pool_seed]
        if len(pool) != EXPECTED_GROUPS_PER_POOL:
            raise DapoOperationalAnalysisError("pool observation count mismatch")
        load_ranks = _average_ranks([item.max_generated_tokens for item in pool])
        latency_ranks = _average_ranks([item.ready_latency_ns for item in pool])
        scale = len(pool) - 1
        scaled_load = [(rank - 1) / scale for rank in load_ranks]
        scaled_latency = [(rank - 1) / scale for rank in latency_ranks]
        load_mean = statistics.fmean(scaled_load)
        latency_mean = statistics.fmean(scaled_latency)
        centered_load.extend(value - load_mean for value in scaled_load)
        centered_latency.extend(value - latency_mean for value in scaled_latency)
    load_norm = math.sqrt(sum(value * value for value in centered_load))
    latency_norm = math.sqrt(sum(value * value for value in centered_latency))
    if load_norm == 0 or latency_norm == 0:
        return 0.0
    return sum(
        load * latency
        for load, latency in zip(centered_load, centered_latency, strict=True)
    ) / (load_norm * latency_norm)


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    position = percentile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _maximum_concurrency(observations: Sequence[Observation]) -> int:
    points = [
        point
        for item in observations
        for point in ((item.dispatch_ns, 1), (item.ready_ns, -1))
    ]
    active = 0
    maximum = 0
    for _, delta in sorted(points, key=lambda value: (value[0], value[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


def _analyze_run(
    *, seed: int, run_dir: Path, materialization_root: Path
) -> tuple[tuple[Observation, ...], dict[str, object]]:
    manifest_path = materialization_root / f"fixed_pool_manifest.v1.{seed}.json"
    design_path = materialization_root / f"selection_design.v1.{seed}.json"
    trace_path = run_dir / "scheduler_trace.v1.jsonl"
    pool_id, items = _load_design(design_path, seed)
    manifest = load_fixed_pool_manifest(manifest_path)
    validate_fixed_pool_materialization(manifest)
    validate_fixed_pool_manifest_design(manifest, materializer.DESIGN_ID)
    if (
        manifest.pool_id != pool_id
        or manifest.design_protocol_sha256 != materializer.PROTOCOL_SHA256
        or manifest.order_seed != seed
    ):
        raise DapoOperationalAnalysisError("manifest/design/protocol binding mismatch")
    by_ordinal = {item.source_pool_ordinal: item for item in items}
    for manifest_item in manifest.items:
        item = by_ordinal.get(manifest_item.ordinal)
        if item is None or (
            manifest_item.source_prompt_id != item.source_prompt_id
            or manifest_item.task_name != item.task_name
            or manifest_item.source_dataset_index != item.source_dataset_index
            or manifest_item.input_token_count != item.rendered_prompt_tokens
            or manifest_item.matching_pair_id
            != f"unique-prompt-{item.canonical_prompt_sha256}"
        ):
            raise DapoOperationalAnalysisError("manifest item binding mismatch")
    trace_report = validate_fixed_pool_trace(
        trace_path,
        manifest,
        expected_completions_per_group=EXPECTED_COMPLETIONS_PER_GROUP,
    )
    if trace_report.physical_weight_version != 0:
        raise DapoOperationalAnalysisError("discovery requires weight version zero")
    observations = _observations_from_events(
        iter_scheduler_trace(trace_path), items, seed
    )
    return observations, {
        "selection_seed": seed,
        "pool_id": manifest.pool_id,
        "trace_sha256": _sha(trace_path),
        "manifest_sha256": _sha(manifest_path),
        "selection_design_sha256": _sha(design_path),
        "prompt_groups": len(observations),
        "physical_weight_version": trace_report.physical_weight_version,
        "maximum_concurrent_groups": _maximum_concurrency(observations),
    }


def analyze(
    runs: Mapping[int, Path], *, materialization_root: Path
) -> dict[str, object]:
    """Validate and aggregate the three exact discovery collections."""
    if tuple(runs) != tuple(seed for seed, _ in materializer.POOL_SPECS):
        raise DapoOperationalAnalysisError("all three runs are required in seed order")
    all_observations: list[Observation] = []
    run_records = []
    for seed, run_dir in runs.items():
        observations, record = _analyze_run(
            seed=seed,
            run_dir=run_dir,
            materialization_root=materialization_root,
        )
        all_observations.extend(observations)
        run_records.append(record)
    canonical_ids = [item.item.canonical_prompt_sha256 for item in all_observations]
    if len(canonical_ids) != len(set(canonical_ids)):
        raise DapoOperationalAnalysisError("discovery pools overlap by prompt identity")
    try:
        dataset_audit = json.loads(
            (materialization_root / "dataset_audit.v1.json").read_text()
        )
    except json.JSONDecodeError as error:
        raise DapoOperationalAnalysisError("dataset audit is invalid JSON") from error
    if (
        not isinstance(dataset_audit, dict)
        or dataset_audit.get("schema_version") != 1
        or dataset_audit.get("dataset_repo") != materializer.DATASET_REPO
        or dataset_audit.get("dataset_revision") != materializer.DATASET_REVISION
        or dataset_audit.get("dataset_file_sha256") != materializer.DATASET_FILE_SHA256
        or not isinstance(dataset_audit.get("unique_canonical_prompts"), int)
        or dataset_audit["unique_canonical_prompts"] < len(canonical_ids)
    ):
        raise DapoOperationalAnalysisError("dataset audit contract mismatch")

    latencies = [float(item.ready_latency_ns) for item in all_observations]
    p10 = _percentile(latencies, 0.1)
    p50 = _percentile(latencies, 0.5)
    p90 = _percentile(latencies, 0.9)
    latency_ratio = p90 / p10 if p10 > 0 else math.inf
    rank_association = _pooled_within_pool_spearman(all_observations)
    total_completions = len(all_observations) * EXPECTED_COMPLETIONS_PER_GROUP
    length_rate = (
        sum(item.length_terminations for item in all_observations) / total_completions
    )
    reward_variance_groups = sum(
        item.reward_min < item.reward_max for item in all_observations
    )
    reward_variance_fraction = reward_variance_groups / len(all_observations)
    checks = {
        "all_three_valid_pools": len(run_records) == 3,
        "complete_prompt_groups_48": len(all_observations) == 48,
        "conflicting_duplicate_ground_truth_count_zero": dataset_audit.get(
            "conflicting_ground_truth_count"
        )
        == 0,
        "zero_administrative_censoring": True,
        "zero_learner_steps": all(
            record["physical_weight_version"] == 0 for record in run_records
        ),
        "backend_length_termination_rate_le_0_2": length_rate <= 0.2,
        "group_ready_latency_p90_p10_ratio_ge_1_5": latency_ratio >= 1.5,
        "pooled_within_pool_spearman_ge_0_5": rank_association >= 0.5,
        "reward_variance_group_fraction_ge_0_25": reward_variance_fraction >= 0.25,
    }
    passed = all(checks.values())
    by_pool = {}
    for seed, _ in materializer.POOL_SPECS:
        pool = [item for item in all_observations if item.pool_seed == seed]
        pool_latencies = [float(item.ready_latency_ns) for item in pool]
        by_pool[str(seed)] = {
            "ready_latency_ns": {
                "p10": _percentile(pool_latencies, 0.1),
                "median": _percentile(pool_latencies, 0.5),
                "p90": _percentile(pool_latencies, 0.9),
            },
            "generated_tokens": {
                "mean_per_completion": statistics.fmean(
                    item.mean_generated_tokens for item in pool
                ),
                "median_group_max": statistics.median(
                    item.max_generated_tokens for item in pool
                ),
            },
            "reward_mean": statistics.fmean(item.reward_mean for item in pool),
            "reward_variance_groups": sum(
                item.reward_min < item.reward_max for item in pool
            ),
        }
    return {
        "schema_version": 1,
        "analysis_status": "calibration_only_operational_workload_latency_discovery",
        "calibration_only": True,
        "confirmatory_eligible": False,
        "scheduler_comparison_authorized": False,
        "replay_authorized": False,
        "training_authorized": False,
        "protocol_sha256": materializer.PROTOCOL_SHA256,
        "unit_of_analysis": "unique_dapo_prompt_group",
        "prompt_groups": len(all_observations),
        "completions_per_group": EXPECTED_COMPLETIONS_PER_GROUP,
        "source_runs": run_records,
        "dataset_audit": dataset_audit,
        "dataset_audit_sha256": _sha(materialization_root / "dataset_audit.v1.json"),
        "by_pool": by_pool,
        "pooled": {
            "ready_latency_ns": {"p10": p10, "median": p50, "p90": p90},
            "ready_latency_p90_p10_ratio": latency_ratio,
            "within_pool_spearman_group_max_tokens_ready_latency": rank_association,
            "backend_length_termination_rate": length_rate,
            "reward_mean": statistics.fmean(
                item.reward_mean for item in all_observations
            ),
            "reward_variance_groups": reward_variance_groups,
            "reward_variance_group_fraction": reward_variance_fraction,
        },
        "locked_checks": checks,
        "decision": (
            "operational_latency_discovery_supports_separate_holdout_design"
            if passed
            else "stop_no_dapo_scheduler_comparison"
        ),
        "interpretation": (
            "scheduler-neutral DAPO operational latency discovery; not a scheduler, "
            "replay, or learner-training result"
        ),
    }


def _parse_run(value: str) -> tuple[int, Path]:
    seed_text, separator, path_text = value.partition("=")
    if not separator or not seed_text.isdigit() or not path_text:
        raise argparse.ArgumentTypeError("run must be SEED=PATH")
    return int(seed_text), Path(path_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--materialization-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    parsed_runs = cast(list[tuple[int, Path]], args.run)
    runs: dict[int, Path] = dict(parsed_runs)
    if len(runs) != len(parsed_runs):
        raise DapoOperationalAnalysisError("duplicate run seed")
    result = analyze(runs, materialization_root=args.materialization_root)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
