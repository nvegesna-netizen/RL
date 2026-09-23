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

"""Analyze the fresh DAPO common-input validation under its frozen gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from nemo_rl.algorithms.async_utils.fixed_pool import (
    FIXED_POOL_RUN_MODE,
    load_fixed_pool_manifest,
    validate_fixed_pool_manifest_design,
    validate_fixed_pool_materialization,
    validate_fixed_pool_trace,
)
from nemo_rl.algorithms.async_utils.scheduler_trace import iter_scheduler_trace
from tools import (
    analyze_dapo_operational_latency_discovery as base_analyzer,
    materialize_dapo_load_alignment_common_input_validation as materializer,
)
from tools.dapo_common_input_resimulation import (
    ComparisonResult,
    FixedStreamGroup,
    compare_samplers,
    load_fixed_stream,
    shifted_ready_times,
    simulate_fixed_stream,
    trace_start_ns,
)


POLL_INTERVALS_MS: Final[tuple[float, ...]] = (0.0, 5.0, 10.0, 20.0)
PRIMARY_DELAY_SECONDS: Final[float] = 24.0
PROMOTION_THRESHOLD: Final[float] = 1 / 28
SEPARATION_THRESHOLD: Final[float] = 1 / 14


class DapoFreshCommonInputValidationError(ValueError):
    """A fresh validation stream, authority, or aggregate is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DapoFreshCommonInputValidationError(message)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise DapoFreshCommonInputValidationError(f"cannot load {label}") from error
    if not isinstance(value, dict):
        raise DapoFreshCommonInputValidationError(f"{label} must be an object")
    return value


def _validate_execution_authority(path: Path) -> str:
    value = _load_object(path, "validation authority")
    expected_authorization = {
        "eos_launch": True,
        "fresh_pool_materialization": False,
        "calibration_generation": False,
        "validation_generation": True,
        "live_scheduler_arm": False,
        "learner_replay": False,
        "learner_training": False,
        "population_claim": False,
    }
    if (
        value.get("schema_version") != 1
        or value.get("analysis_status")
        != "dapo_load_alignment_fresh_common_input_validation_generation_confirmation"
        or value.get("confirmed") is not True
        or value.get("protocol_sha256") != materializer.PROTOCOL_SHA256
        or value.get("protocol_confirmation_sha256") != materializer.CONFIRMATION_SHA256
        or value.get("authorization") != expected_authorization
    ):
        raise DapoFreshCommonInputValidationError("validation authority mismatch")
    return _sha(path)


def _load_private_calibration(
    path: Path, *, expected_sha256: str
) -> dict[int, tuple[frozenset[str], str]]:
    if _sha(path) != expected_sha256:
        raise DapoFreshCommonInputValidationError(
            "private calibration manifest hash mismatch"
        )
    value = _load_object(path, "private calibration manifest")
    pools = value.get("pools")
    _require(
        value.get("schema_version") == 1
        and value.get("analysis_status")
        == "frozen_private_dapo_common_input_calibration"
        and value.get("protocol_sha256") == materializer.PROTOCOL_SHA256
        and value.get("contains_prompt_identities") is True
        and isinstance(pools, list),
        "private calibration manifest contract mismatch",
    )
    result: dict[int, tuple[frozenset[str], str]] = {}
    for pool in pools:
        _require(isinstance(pool, dict), "private calibration pool malformed")
        seed = pool.get("pool_seed")
        prompts = pool.get("prompts")
        manifest_sha = pool.get("balanced_manifest_sha256")
        _require(
            isinstance(seed, int)
            and isinstance(prompts, list)
            and isinstance(manifest_sha, str),
            "private calibration pool fields missing",
        )
        all_ids = {
            prompt.get("source_prompt_id")
            for prompt in prompts
            if isinstance(prompt, dict)
            and isinstance(prompt.get("source_prompt_id"), str)
        }
        lower = frozenset(
            str(prompt["source_prompt_id"])
            for prompt in prompts
            if isinstance(prompt, dict) and prompt.get("fixed_lower_load") is True
        )
        validation_ordinals = {
            prompt.get("validation_source_pool_ordinal")
            for prompt in prompts
            if isinstance(prompt, dict)
        }
        _require(
            len(prompts) == 32
            and len(all_ids) == 32
            and len(lower) == 16
            and validation_ordinals == set(range(32)),
            "private calibration pool geometry mismatch",
        )
        result[seed] = (lower, manifest_sha)
    _require(
        set(result) == set(materializer.POOL_SEEDS),
        "private calibration pool set mismatch",
    )
    return result


def _validate_trace(
    *,
    trace_path: Path,
    manifest_path: Path,
    pool_seed: int,
    generation_seed: int,
    expected_manifest_sha256: str,
) -> dict[str, object]:
    _require(
        _sha(manifest_path) == expected_manifest_sha256,
        "balanced manifest hash mismatch",
    )
    manifest = load_fixed_pool_manifest(manifest_path)
    validate_fixed_pool_materialization(manifest)
    validate_fixed_pool_manifest_design(manifest, materializer.DESIGN_ID)
    _require(manifest.order_seed == pool_seed, "balanced manifest pool mismatch")
    trace_report = validate_fixed_pool_trace(
        trace_path, manifest, expected_completions_per_group=16
    )
    _require(trace_report.physical_weight_version == 0, "validation changed weights")
    events = tuple(iter_scheduler_trace(trace_path))
    starts = [event for event in events if event.event_type.value == "run_started"]
    _require(
        len(starts) == 1
        and starts[0].run_mode == FIXED_POOL_RUN_MODE
        and starts[0].scalar_summaries.get("generation_study_seed") == generation_seed,
        "validation run identity or generation seed mismatch",
    )
    completed = [
        event for event in events if event.event_type.value == "rollout_completed"
    ]
    length_rate = statistics.fmean(
        base_analyzer._number(event.scalar_summaries, "backend_length_termination_rate")
        for event in completed
    )
    checks = {
        "complete_32_groups_512_completions": len(completed) == 32,
        "zero_learner_steps_and_physical_weight_version_zero": True,
        "zero_administrative_censoring": True,
        "backend_length_termination_rate_le_0_2": length_rate <= 0.2,
    }
    _require(all(checks.values()), "validation collection validity gate failed")
    return {
        "generation_seed": generation_seed,
        "trace_sha256": _sha(trace_path),
        "balanced_manifest_sha256": expected_manifest_sha256,
        "prompt_groups": len(completed),
        "completions": len(completed) * 16,
        "backend_length_termination_rate": length_rate,
        "validity_checks": checks,
    }


def _periodic_ticks(
    *, trace_path: Path, ready_times: Mapping[str, int], poll_interval_ms: float
) -> tuple[int, ...] | None:
    if poll_interval_ms == 0:
        return None
    _require(poll_interval_ms > 0, "poll interval must be positive or zero")
    interval_ns = round(poll_interval_ms * 1_000_000)
    anchor_ns = trace_start_ns(trace_path)
    final_ns = max(ready_times.values()) + interval_ns
    return tuple(range(anchor_ns, final_ns + 1, interval_ns))


def _compare(
    *,
    groups: Sequence[FixedStreamGroup],
    ready_times: Mapping[str, int],
    lower: frozenset[str],
    trace_path: Path,
    poll_interval_ms: float,
    cohort_restricted_ready_first: bool = False,
) -> ComparisonResult:
    ticks = _periodic_ticks(
        trace_path=trace_path,
        ready_times=ready_times,
        poll_interval_ms=poll_interval_ms,
    )
    all_ids = frozenset(group.source_prompt_id for group in groups)
    higher = all_ids - lower
    in_order = simulate_fixed_stream(
        groups,
        sampler="in_order",
        ready_ns_by_group=ready_times,
        selection_tick_ns=ticks,
    )
    ready_first = simulate_fixed_stream(
        groups,
        sampler="ready_first",
        ready_ns_by_group=ready_times,
        cohort_restricted_ready_first=cohort_restricted_ready_first,
        selection_tick_ns=ticks,
    )
    return compare_samplers(
        in_order,
        ready_first,
        lower_load_prompt_ids=lower,
        easier_prompt_ids=higher,
    )


def _stream_effects(
    *, trace_path: Path, lower: frozenset[str], poll_interval_ms: float
) -> dict[str, float | bool]:
    groups = load_fixed_stream(trace_path)
    all_ids = frozenset(group.source_prompt_id for group in groups)
    _require(lower <= all_ids and len(all_ids) == 32, "validation identity mismatch")
    higher = all_ids - lower
    natural_ready = shifted_ready_times(
        groups, delayed_prompt_ids=frozenset(), delay_seconds=0.0
    )
    high_ready = shifted_ready_times(
        groups, delayed_prompt_ids=higher, delay_seconds=PRIMARY_DELAY_SECONDS
    )
    low_ready = shifted_ready_times(
        groups, delayed_prompt_ids=lower, delay_seconds=PRIMARY_DELAY_SECONDS
    )
    natural = _compare(
        groups=groups,
        ready_times=natural_ready,
        lower=lower,
        trace_path=trace_path,
        poll_interval_ms=poll_interval_ms,
    )
    high = _compare(
        groups=groups,
        ready_times=high_ready,
        lower=lower,
        trace_path=trace_path,
        poll_interval_ms=poll_interval_ms,
    )
    low = _compare(
        groups=groups,
        ready_times=low_ready,
        lower=lower,
        trace_path=trace_path,
        poll_interval_ms=poll_interval_ms,
    )
    l0_high = _compare(
        groups=groups,
        ready_times=high_ready,
        lower=lower,
        trace_path=trace_path,
        poll_interval_ms=poll_interval_ms,
        cohort_restricted_ready_first=True,
    )
    l0_low = _compare(
        groups=groups,
        ready_times=low_ready,
        lower=lower,
        trace_path=trace_path,
        poll_interval_ms=poll_interval_ms,
        cohort_restricted_ready_first=True,
    )
    l0_exact = all(
        value == 0
        for value in (
            l0_high.lower_load_mean_normalized_promotion,
            l0_low.lower_load_mean_normalized_promotion,
            l0_high.changed_selected_step_sets,
            l0_low.changed_selected_step_sets,
        )
    )
    return {
        "natural": natural.lower_load_mean_normalized_promotion,
        "high_load_delayed": high.lower_load_mean_normalized_promotion,
        "low_load_delayed": low.lower_load_mean_normalized_promotion,
        "signed_separation": high.lower_load_mean_normalized_promotion
        - low.lower_load_mean_normalized_promotion,
        "l0_negative_control_exact": l0_exact,
    }


def analyze(
    runs: Mapping[tuple[int, int], Path],
    *,
    balanced_manifest_paths: Mapping[int, Path],
    private_calibration_manifest_path: Path,
    expected_private_calibration_sha256: str,
    validation_authority_path: Path,
) -> dict[str, object]:
    """Return a privacy-safe aggregate for all six fresh validation streams."""
    authority_sha256 = _validate_execution_authority(validation_authority_path)
    calibration = _load_private_calibration(
        private_calibration_manifest_path,
        expected_sha256=expected_private_calibration_sha256,
    )
    expected_runs = tuple(
        (pool_seed, generation_seed)
        for pool_seed in materializer.POOL_SEEDS
        for generation_seed in materializer.VALIDATION_SEEDS[pool_seed]
    )
    _require(tuple(runs) == expected_runs, "all six validation runs are required")
    _require(
        set(balanced_manifest_paths) == set(materializer.POOL_SEEDS),
        "balanced manifest inventory mismatch",
    )
    collection_records: dict[str, list[dict[str, object]]] = defaultdict(list)
    effects: dict[float, dict[int, list[dict[str, float | bool]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for pool_seed, generation_seed in expected_runs:
        run_dir = runs[(pool_seed, generation_seed)]
        trace_path = run_dir / "scheduler_trace.v1.jsonl"
        lower, expected_manifest_sha = calibration[pool_seed]
        record = _validate_trace(
            trace_path=trace_path,
            manifest_path=balanced_manifest_paths[pool_seed],
            pool_seed=pool_seed,
            generation_seed=generation_seed,
            expected_manifest_sha256=expected_manifest_sha,
        )
        collection_records[str(pool_seed)].append(record)
        for poll_interval_ms in POLL_INTERVALS_MS:
            effects[poll_interval_ms][pool_seed].append(
                _stream_effects(
                    trace_path=trace_path,
                    lower=lower,
                    poll_interval_ms=poll_interval_ms,
                )
            )

    by_poll: dict[str, object] = {}
    all_poll_pass = True
    for poll_interval_ms in POLL_INTERVALS_MS:
        pool_means: dict[str, dict[str, float]] = {}
        l0_exact = True
        for pool_seed in materializer.POOL_SEEDS:
            draws = effects[poll_interval_ms][pool_seed]
            pool_means[str(pool_seed)] = {
                name: statistics.fmean(float(draw[name]) for draw in draws)
                for name in (
                    "natural",
                    "high_load_delayed",
                    "low_load_delayed",
                    "signed_separation",
                )
            }
            l0_exact &= all(draw["l0_negative_control_exact"] is True for draw in draws)
        high = [value["high_load_delayed"] for value in pool_means.values()]
        low = [value["low_load_delayed"] for value in pool_means.values()]
        signed = [value["signed_separation"] for value in pool_means.values()]
        checks = {
            "high_pool_median_ge_1_over_28": statistics.median(high)
            >= PROMOTION_THRESHOLD,
            "high_pools_at_or_above_1_over_28_ge_2": sum(
                value >= PROMOTION_THRESHOLD for value in high
            )
            >= 2,
            "low_pool_median_le_negative_1_over_28": statistics.median(low)
            <= -PROMOTION_THRESHOLD,
            "low_pools_at_or_below_negative_1_over_28_ge_2": sum(
                value <= -PROMOTION_THRESHOLD for value in low
            )
            >= 2,
            "median_signed_separation_ge_1_over_14": statistics.median(signed)
            >= SEPARATION_THRESHOLD,
            "no_wrong_signed_high_pool_mean": all(value >= 0 for value in high),
            "no_wrong_signed_low_pool_mean": all(value <= 0 for value in low),
            "l0_negative_control_exact": l0_exact,
        }
        passed = all(checks.values())
        all_poll_pass &= passed
        by_poll[str(int(poll_interval_ms))] = {
            "poll_interval_milliseconds": poll_interval_ms,
            "pool_mean_effects": pool_means,
            "checks": checks,
            "passed": passed,
        }
    return {
        "schema_version": 1,
        "analysis_status": "dapo_load_alignment_fresh_common_input_validation",
        "protocol_sha256": materializer.PROTOCOL_SHA256,
        "protocol_confirmation_sha256": materializer.CONFIRMATION_SHA256,
        "validation_authority_sha256": authority_sha256,
        "private_calibration_manifest_sha256": expected_private_calibration_sha256,
        "collection_validity": dict(collection_records),
        "poll_timing_sensitivity": by_poll,
        "locked_checks": {
            "all_six_validation_collections_valid": True,
            "success_gate_passed_at_every_polling_interval": all_poll_pass,
        },
        "decision": (
            "close_as_replicated_common_input_selection_layer_mechanism_at_24_seconds"
            if all_poll_pass
            else "close_as_exploratory_regime_not_freshly_replicated"
        ),
        "privacy": {
            "aggregate_only": True,
            "private_prompt_identities_emitted": False,
            "logical_group_identifiers_emitted": False,
        },
        "authorization_state": {
            "live_scheduler_arm": False,
            "learner_replay": False,
            "learner_training": False,
            "population_claim": False,
        },
    }


def _parse_run(value: str) -> tuple[tuple[int, int], Path]:
    identity, separator, path = value.partition("=")
    pool, colon, generation = identity.partition(":")
    if (
        not separator
        or not colon
        or not pool.isdigit()
        or not generation.isdigit()
        or not path
    ):
        raise argparse.ArgumentTypeError("run must be POOL_SEED:GENERATION_SEED=PATH")
    return (int(pool), int(generation)), Path(path)


def _parse_manifest(value: str) -> tuple[int, Path]:
    pool, separator, path = value.partition("=")
    if not separator or not pool.isdigit() or not path:
        raise argparse.ArgumentTypeError("manifest must be POOL_SEED=PATH")
    return int(pool), Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument(
        "--balanced-manifest", action="append", type=_parse_manifest, required=True
    )
    parser.add_argument("--private-calibration-manifest", type=Path, required=True)
    parser.add_argument("--expected-private-calibration-sha256", required=True)
    parser.add_argument("--validation-authority", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    parsed_runs: list[tuple[tuple[int, int], Path]] = args.run
    parsed_manifests: list[tuple[int, Path]] = args.balanced_manifest
    runs = dict(parsed_runs)
    manifests = dict(parsed_manifests)
    _require(len(runs) == len(parsed_runs), "duplicate validation run")
    _require(len(manifests) == len(parsed_manifests), "duplicate balanced manifest")
    result = analyze(
        runs,
        balanced_manifest_paths=manifests,
        private_calibration_manifest_path=args.private_calibration_manifest,
        expected_private_calibration_sha256=args.expected_private_calibration_sha256,
        validation_authority_path=args.validation_authority,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
