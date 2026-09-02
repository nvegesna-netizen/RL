#!/usr/bin/env python3
"""Analyze the exposed-pool 1024-token OpenMath headroom diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

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
from tools.analyze_openmath_latency_feasibility import (
    BOOTSTRAP_REPLICATES,
    EXPECTED_COMPLETIONS,
    EXPECTED_GROUPS,
    DesignItem,
    FeasibilityAnalysisError,
    GroupObservation,
    _bootstrap,
    _count_from_rate,
    _matched_rank_biserial,
    _max_concurrency,
    _median_ratio,
    _positive_ratio,
    _required_number,
    _sha256,
    load_design,
)

CONTEXT_LENGTH = 1024


@dataclass(frozen=True, slots=True)
class TerminationDiagnostics:
    finish_available: int
    stop_finishes: int
    length_finishes: int
    abort_finishes: int
    other_finishes: int
    context_exhausted: int
    token_stop_reasons: int
    eos_matching_stop_reasons: int
    cap_hits: int
    near_cap: int


def _runtime(events: Sequence[SchedulerTraceEvent]) -> Mapping[str, object]:
    starts = [
        event for event in events if event.event_type is SchedulerEventType.RUN_STARTED
    ]
    if len(starts) != 1:
        raise FeasibilityAnalysisError("trace requires exactly one run_started event")
    runtime = starts[0].scalar_summaries
    required = {
        "generation_backend": "vllm",
        "max_total_sequence_length": CONTEXT_LENGTH,
        "configured_max_new_tokens": CONTEXT_LENGTH,
        "generation_context_length": CONTEXT_LENGTH,
        "tokenizer_eos_token_present": True,
        "effective_stop_token_count": 1,
        "effective_stop_string_count": 0,
        "vllm_skip_tokenizer_init": True,
        "generation_ignore_eos": False,
        "generation_temperature": 1.0,
        "generation_top_p": 1.0,
        "generation_top_k": -1,
        "generation_use_async_rollouts": True,
        "generation_study_seed": 52001,
        "generation_speculative_config_sha256": hashlib.sha256(b"null").hexdigest(),
        "grpo_seed": 20260901,
        "num_generations_per_prompt": 2,
        "max_rollout_turns": 1,
        "num_prompts_per_step": 4,
        "max_inflight_prompts": 4,
        "max_buffered_rollouts": 8,
        "vllm_include_stop_str_in_output": True,
        "fixed_pool_design_id": "openmath_termination_headroom_v1",
        "finish_reason_code_schema_version": 1,
    }
    for key, expected in required.items():
        if runtime.get(key) != expected:
            raise FeasibilityAnalysisError(
                f"runtime summary {key} must equal {expected!r}"
            )
    eos = runtime.get("tokenizer_eos_token_id")
    if not isinstance(eos, int) or eos < 0:
        raise FeasibilityAnalysisError("tokenizer EOS id must be a nonnegative integer")
    if runtime.get("effective_single_stop_token_id") != eos:
        raise FeasibilityAnalysisError("effective stop token must equal tokenizer EOS")
    expected_stop_digest = hashlib.sha256(f"[{eos}]".encode("utf-8")).hexdigest()
    if runtime.get("effective_stop_token_ids_sha256") != expected_stop_digest:
        raise FeasibilityAnalysisError("effective stop-token digest is inconsistent")
    if (
        runtime.get("effective_stop_strings_sha256")
        != hashlib.sha256(b"[]").hexdigest()
    ):
        raise FeasibilityAnalysisError("effective stop-string digest is inconsistent")
    tokenizer_name_digest = runtime.get("policy_tokenizer_name_sha256")
    if not isinstance(tokenizer_name_digest, str) or len(tokenizer_name_digest) != 64:
        raise FeasibilityAnalysisError("policy tokenizer name digest must be sha256")
    model_name_digest = runtime.get("policy_model_name_sha256")
    if not isinstance(model_name_digest, str) or len(model_name_digest) != 64:
        raise FeasibilityAnalysisError("policy model name digest must be sha256")
    return runtime


def _rate_count(summaries: Mapping[str, object], key: str) -> int:
    return _count_from_rate(_required_number(summaries, key), name=key)


def observations_from_events(
    events: Iterable[SchedulerTraceEvent], items: Sequence[DesignItem]
) -> tuple[
    tuple[GroupObservation, ...],
    dict[int, TerminationDiagnostics],
    Mapping[str, object],
]:
    """Join lifecycle events and require exact vLLM termination diagnostics."""
    events = tuple(events)
    runtime = _runtime(events)
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
        SchedulerEventType.ABORT_REQUESTED,
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

    observations = []
    diagnostics = {}
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
            raise FeasibilityAnalysisError("each group requires two completions")
        mean_tokens = _required_number(summaries, "mean_gen_tokens_per_sample")
        min_tokens = _required_number(summaries, "gen_tokens_per_sample/min")
        max_tokens = _required_number(summaries, "gen_tokens_per_sample/max")
        effective_cap = CONTEXT_LENGTH - item.rendered_prompt_tokens
        reported_cap_min = _required_number(summaries, "effective_max_new_tokens/min")
        reported_cap_max = _required_number(summaries, "effective_max_new_tokens/max")
        engine_seed_min = _required_number(summaries, "effective_engine_seed/min")
        engine_seed_max = _required_number(summaries, "effective_engine_seed/max")
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
            or reported_cap_min != effective_cap
            or reported_cap_max != effective_cap
            or engine_seed_min != 52001
            or engine_seed_max != 52001
            or max_tokens > effective_cap
        ):
            raise FeasibilityAnalysisError(
                "inconsistent token or effective-cap diagnostics"
            )
        finish_available = _rate_count(
            summaries, "backend_finish_reason_availability_rate"
        )
        stop_finishes = _rate_count(summaries, "backend_stop_termination_rate")
        length_finishes = _rate_count(summaries, "backend_length_termination_rate")
        abort_finishes = _rate_count(summaries, "backend_abort_termination_rate")
        other_finishes = _rate_count(summaries, "backend_other_termination_rate")
        context_exhausted = _rate_count(summaries, "backend_context_exhausted_rate")
        token_stop_reasons = _rate_count(summaries, "backend_stop_reason_token_rate")
        eos_matching_stop_reasons = _rate_count(
            summaries, "backend_stop_reason_matches_tokenizer_eos_rate"
        )
        cap_hits = _rate_count(summaries, "generated_at_effective_cap_rate")
        near_cap = _rate_count(summaries, "generated_near_effective_cap_rate")
        if (
            finish_available != EXPECTED_COMPLETIONS
            or stop_finishes
            + length_finishes
            + abort_finishes
            + other_finishes
            + context_exhausted
            != EXPECTED_COMPLETIONS
            or length_finishes != cap_hits
        ):
            raise FeasibilityAnalysisError(
                "finish reasons must be complete and length must equal cap hit"
            )
        diagnostics[ordinal] = TerminationDiagnostics(
            finish_available=finish_available,
            stop_finishes=stop_finishes,
            length_finishes=length_finishes,
            abort_finishes=abort_finishes,
            other_finishes=other_finishes,
            context_exhausted=context_exhausted,
            token_stop_reasons=token_stop_reasons,
            eos_matching_stop_reasons=eos_matching_stop_reasons,
            cap_hits=cap_hits,
            near_cap=near_cap,
        )
        observations.append(
            GroupObservation(
                item=item,
                dispatch_ns=dispatch.monotonic_ns,
                completed_ns=completed.monotonic_ns,
                ready_ns=ready.monotonic_ns,
                archived_ns=archived.monotonic_ns,
                mean_generated_tokens=mean_tokens,
                min_generated_tokens=int(min_tokens),
                max_generated_tokens=int(max_tokens),
                cap_hits=cap_hits,
                natural_terminations=_rate_count(summaries, "natural_termination_rate"),
                effective_output_cap=effective_cap,
            )
        )
    return tuple(observations), diagnostics, runtime


def analyze_observations(
    observations: Sequence[GroupObservation],
    diagnostics: Mapping[int, TerminationDiagnostics],
) -> dict[str, object]:
    """Apply the predeclared exposed-pool headroom/proxy decision contract."""
    if len(observations) != EXPECTED_GROUPS or len(diagnostics) != EXPECTED_GROUPS:
        raise FeasibilityAnalysisError("diagnostic requires 16 complete groups")
    by_pair: dict[str, dict[str, GroupObservation]] = defaultdict(dict)
    by_stratum: dict[str, list[GroupObservation]] = defaultdict(list)
    for observation in observations:
        by_pair[observation.item.pair_id][observation.item.stratum] = observation
        by_stratum[observation.item.stratum].append(observation)
    if len(by_pair) != 8 or any(
        set(pair) != {"short", "long"} for pair in by_pair.values()
    ):
        raise FeasibilityAnalysisError("diagnostic requires eight matched pairs")
    pairs = [(pair["short"], pair["long"]) for _, pair in sorted(by_pair.items())]
    short = by_stratum["short"]
    long = by_stratum["long"]

    def completion_tokens(values: Sequence[GroupObservation]) -> list[int]:
        return [
            token
            for value in values
            for token in (value.min_generated_tokens, value.max_generated_tokens)
        ]

    reference_ratio = _median_ratio(
        [value.item.reference_solution_tokens for value in long],
        [value.item.reference_solution_tokens for value in short],
    )
    output_ratio = _median_ratio(completion_tokens(long), completion_tokens(short))
    ready_ratio = _median_ratio(
        [value.ready_latency_ns for value in long],
        [value.ready_latency_ns for value in short],
    )
    ready_differences = [
        long_value.ready_latency_ns - short_value.ready_latency_ns
        for short_value, long_value in pairs
    ]
    ready_rank = _matched_rank_biserial(ready_differences)
    ready_sign_count = sum(value > 0 for value in ready_differences)
    pair_output_ratios = [
        _positive_ratio(
            long_value.mean_generated_tokens, short_value.mean_generated_tokens
        )
        for short_value, long_value in pairs
    ]
    output_sign_count = sum(value > 1 for value in pair_output_ratios)

    def count(stratum: str, field: str) -> int:
        return sum(
            getattr(diagnostics[value.item.source_pool_ordinal], field)
            for value in by_stratum[stratum]
        )

    length_counts = {name: count(name, "length_finishes") for name in by_stratum}
    stop_counts = {name: count(name, "stop_finishes") for name in by_stratum}
    near_cap_counts = {name: count(name, "near_cap") for name in by_stratum}
    invalid_counts = {
        name: sum(
            count(name, field)
            for field in ("abort_finishes", "other_finishes", "context_exhausted")
        )
        for name in by_stratum
    }
    token_stop_reason_counts = {
        name: count(name, "token_stop_reasons") for name in by_stratum
    }
    eos_matching_stop_reason_counts = {
        name: count(name, "eos_matching_stop_reasons") for name in by_stratum
    }
    denominator = len(short) * EXPECTED_COMPLETIONS
    length_rates = {name: value / denominator for name, value in length_counts.items()}
    stop_rates = {name: value / denominator for name, value in stop_counts.items()}
    near_cap_rates = {
        name: value / denominator for name, value in near_cap_counts.items()
    }
    ready_bootstrap = _bootstrap(
        pairs,
        lambda draw: _median_ratio(
            [long_value.ready_latency_ns for _, long_value in draw],
            [short_value.ready_latency_ns for short_value, _ in draw],
        ),
    )
    rank_bootstrap = _bootstrap(
        pairs,
        lambda draw: _matched_rank_biserial(
            [
                long_value.ready_latency_ns - short_value.ready_latency_ns
                for short_value, long_value in draw
            ]
        ),
    )
    maximum_concurrency = _max_concurrency(observations)
    checks = {
        "reference_median_ratio_ge_3": reference_ratio >= 3.0,
        "length_rate_lt_0_2_each": all(value < 0.20 for value in length_rates.values()),
        "length_rate_difference_le_0_1": abs(
            length_rates["long"] - length_rates["short"]
        )
        <= 0.10,
        "near_cap_rate_lt_0_2_each": all(
            value < 0.20 for value in near_cap_rates.values()
        ),
        "near_cap_rate_difference_le_0_1": abs(
            near_cap_rates["long"] - near_cap_rates["short"]
        )
        <= 0.10,
        "stop_finish_rate_ge_0_8_each": all(
            value >= 0.80 for value in stop_rates.values()
        ),
        "no_abort_other_or_context_exhausted": not any(invalid_counts.values()),
        "output_median_ratio_ge_1_5": output_ratio >= 1.5,
        "output_pair_sign_count_ge_6": output_sign_count >= 6,
        "maximum_concurrency_gt_1": maximum_concurrency > 1,
    }
    cap_checks = tuple(
        checks[key]
        for key in (
            "length_rate_lt_0_2_each",
            "length_rate_difference_le_0_1",
            "near_cap_rate_lt_0_2_each",
            "near_cap_rate_difference_le_0_1",
            "stop_finish_rate_ge_0_8_each",
            "no_abort_other_or_context_exhausted",
        )
    )
    output_checks = (
        checks["output_median_ratio_ge_1_5"],
        checks["output_pair_sign_count_ge_6"],
    )
    if any(invalid_counts.values()):
        decision = "stop_infrastructure_eos_unresolved"
    elif not all(cap_checks):
        decision = "stop_model_or_prompt_overlong"
    elif not all(output_checks):
        decision = "stop_proxy_not_transmitted"
    elif all(checks.values()):
        decision = "pass_to_fresh_disjoint_replicated_calibration_design"
    else:
        decision = "stop_technical_gate"
    return {
        "schema_version": 1,
        "analysis_status": "exploratory_post_observation_headroom",
        "calibration_only": True,
        "confirmatory_eligible": False,
        "authorizes_replay": False,
        "authorizes_training": False,
        "exposed_pool_reused": True,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "reference_solution_tokens": {"median_long_short_ratio": reference_ratio},
        "generated_output_tokens": {
            "median_long_short_ratio": output_ratio,
            "pair_long_short_ratios": pair_output_ratios,
            "pair_sign_count": output_sign_count,
        },
        "termination": {
            "length_count_by_stratum": length_counts,
            "length_rate_by_stratum": length_rates,
            "stop_count_by_stratum": stop_counts,
            "stop_rate_by_stratum": stop_rates,
            "near_cap_count_by_stratum": near_cap_counts,
            "near_cap_rate_by_stratum": near_cap_rates,
            "invalid_finish_count_by_stratum": invalid_counts,
            "token_stop_reason_count_by_stratum": token_stop_reason_counts,
            "eos_matching_stop_reason_count_by_stratum": (
                eos_matching_stop_reason_counts
            ),
            "stop_reason_token_match_is_descriptive_not_a_gate": True,
        },
        "ready_latency": {
            "median_long_short_ratio": ready_ratio,
            "matched_pairs_rank_biserial": ready_rank,
            "pair_sign_count": ready_sign_count,
            "median_ratio_bootstrap": ready_bootstrap,
            "matched_pairs_rank_biserial_bootstrap": rank_bootstrap,
            "descriptive_only_not_a_stage0_gate": True,
        },
        "maximum_concurrent_groups": maximum_concurrency,
        "locked_checks": checks,
        "decision": decision,
        "interpretation": (
            "exposed-pool capacity diagnostic only; passing permits a fresh disjoint "
            "design and cannot confirm a scheduler effect"
        ),
    }


def analyze(
    *, trace_path: Path, manifest_path: Path, design_path: Path
) -> dict[str, object]:
    pool_id, manifest_sha, items = load_design(design_path)
    manifest = load_fixed_pool_manifest(manifest_path)
    validate_fixed_pool_materialization(manifest)
    validate_fixed_pool_manifest_design(manifest, "openmath_termination_headroom_v1")
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
        trace_path, manifest, expected_completions_per_group=EXPECTED_COMPLETIONS
    )
    if trace_report.physical_weight_version != 0:
        raise FeasibilityAnalysisError("diagnostic requires physical weight version 0")
    observations, diagnostics, runtime = observations_from_events(
        iter_scheduler_trace(trace_path), items
    )
    expected_tokenizer_name = str(manifest_path.resolve().parent / "model_snapshot")
    expected_tokenizer_digest = hashlib.sha256(
        json.dumps(expected_tokenizer_name, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if (
        runtime["policy_tokenizer_name_sha256"] != expected_tokenizer_digest
        or runtime["policy_model_name_sha256"] != expected_tokenizer_digest
    ):
        raise FeasibilityAnalysisError(
            "runtime model/tokenizer paths do not match the materialized snapshot"
        )
    result = analyze_observations(observations, diagnostics)
    result["runtime_binding"] = dict(runtime)
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
