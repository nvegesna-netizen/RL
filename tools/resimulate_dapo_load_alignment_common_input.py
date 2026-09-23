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

"""Run the frozen DAPO load-alignment common-input resimulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from dapo_common_input_resimulation import (
    ComparisonResult,
    CommonInputResimulationError,
    FixedStreamGroup,
    compare_samplers,
    comparison_to_dict,
    load_fixed_stream,
    observed_nonempty_selections,
    observed_ready_times,
    observed_selection_tick_times,
    shifted_ready_times,
    simulate_fixed_stream,
    trace_start_ns,
)


EXPECTED_CANDIDATE_STATUS = (
    "candidate_dapo_load_alignment_common_input_scheduler_resimulation"
)
EXPECTED_CONFIRMATION_STATUS = (
    "dapo_load_alignment_common_input_resimulation_confirmation"
)
EXPECTED_REPAIR_STATUS = "candidate_dapo_load_alignment_common_input_poll_timing_repair"
EXPECTED_REPAIR_CONFIRMATION_STATUS = (
    "dapo_load_alignment_common_input_poll_timing_repair_confirmation"
)
POLL_INTERVALS_MS = (0.0, 5.0, 10.0, 20.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CommonInputResimulationError(
            f"cannot load JSON {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise CommonInputResimulationError(f"JSON root must be an object: {path}")
    return value


def _validate_authority(
    *, protocol_path: Path, confirmation_path: Path
) -> tuple[dict[str, Any], str]:
    protocol = _load_json(protocol_path)
    confirmation = _load_json(confirmation_path)
    protocol_sha256 = _sha256(protocol_path)
    if protocol.get("analysis_status") != EXPECTED_CANDIDATE_STATUS:
        raise CommonInputResimulationError("unexpected protocol analysis status")
    if confirmation.get("analysis_status") != EXPECTED_CONFIRMATION_STATUS:
        raise CommonInputResimulationError("unexpected confirmation analysis status")
    if confirmation.get("candidate_sha256") != protocol_sha256:
        raise CommonInputResimulationError("confirmation does not bind protocol")
    authorized = confirmation.get("authorized")
    if not isinstance(authorized, dict) or not all(
        authorized.get(key) is True
        for key in (
            "local_implementation",
            "local_tests",
            "native_trace_parity_validation",
            "six_neutral_trace_common_input_resimulation_after_parity",
            "privacy_safe_aggregate_analysis",
        )
    ):
        raise CommonInputResimulationError("required local scope is not authorized")
    if any(
        authorized.get(key) is not False
        for key in (
            "eos_launch",
            "new_rollout",
            "learner_replay",
            "learner_training",
            "population_claim",
        )
    ):
        raise CommonInputResimulationError("confirmation scientific boundary mismatch")
    return protocol, protocol_sha256


def _validate_repair_authority(
    *,
    protocol_path: Path,
    confirmation_path: Path,
    failure_audit_path: Path,
    repair_path: Path,
    repair_confirmation_path: Path,
) -> tuple[dict[str, Any], str, str]:
    protocol, protocol_sha256 = _validate_authority(
        protocol_path=protocol_path, confirmation_path=confirmation_path
    )
    repair = _load_json(repair_path)
    repair_confirmation = _load_json(repair_confirmation_path)
    repair_sha256 = _sha256(repair_path)
    if repair.get("analysis_status") != EXPECTED_REPAIR_STATUS:
        raise CommonInputResimulationError("unexpected repair analysis status")
    if (
        repair_confirmation.get("analysis_status")
        != EXPECTED_REPAIR_CONFIRMATION_STATUS
    ):
        raise CommonInputResimulationError(
            "unexpected repair confirmation analysis status"
        )
    if repair_confirmation.get("candidate_sha256") != repair_sha256:
        raise CommonInputResimulationError("repair confirmation does not bind repair")
    bindings = repair.get("bindings")
    if not isinstance(bindings, dict) or (
        bindings.get("attempt1_protocol_sha256") != protocol_sha256
        or bindings.get("attempt1_confirmation_sha256") != _sha256(confirmation_path)
        or bindings.get("attempt1_failure_audit_sha256") != _sha256(failure_audit_path)
    ):
        raise CommonInputResimulationError("repair predecessor binding mismatch")
    failure = _load_json(failure_audit_path)
    if (
        failure.get("analysis_status")
        != "dapo_load_alignment_common_input_resimulation_attempt1_failed_before_counterfactual"
        or failure.get("decision")
        != "stop_attempt1_before_neutral_counterfactual_and_freeze_poll_timing_sensitivity_repair"
    ):
        raise CommonInputResimulationError("attempt-1 failure audit mismatch")
    authorized = repair_confirmation.get("authorized")
    if not isinstance(authorized, dict) or not all(
        authorized.get(key) is True
        for key in (
            "replacement_implementation",
            "local_tests",
            "repaired_native_parity",
            "neutral_poll_sensitivity_resimulation_after_parity",
            "privacy_safe_aggregate_analysis",
        )
    ):
        raise CommonInputResimulationError("required repair scope is not authorized")
    if any(
        authorized.get(key) is not False
        for key in (
            "eos_launch",
            "new_rollout",
            "learner_replay",
            "learner_training",
            "population_claim",
        )
    ):
        raise CommonInputResimulationError(
            "repair confirmation scientific boundary mismatch"
        )
    intervals = repair.get("repair_scope", {}).get(
        "neutral_poll_timing_sensitivity_milliseconds"
    )
    if intervals != list(POLL_INTERVALS_MS):
        raise CommonInputResimulationError("repair poll interval grid mismatch")
    return protocol, protocol_sha256, repair_sha256


def _find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if len(matches) != 1:
        raise CommonInputResimulationError(
            f"expected one {pattern} below {root}, found {len(matches)}"
        )
    return matches[0]


def _native_trace_paths(native_root: Path) -> dict[int, dict[str, Path]]:
    paths: dict[int, dict[str, Path]] = {}
    for result_path in sorted(
        native_root.rglob("dapo_load_alignment_shadow_result.v1.json")
    ):
        result = _load_json(result_path)
        pool = result.get("order_seed")
        arms = result.get("arms")
        if not isinstance(pool, int) or not isinstance(arms, dict):
            raise CommonInputResimulationError("invalid native result inventory")
        paths[pool] = {
            str(arm): result_path.parent / str(arm) / "scheduler_trace.v1.jsonl"
            for arm in arms
        }
    if set(paths) != {51001, 51002, 51003} or any(
        len(arms) != 8 for arms in paths.values()
    ):
        raise CommonInputResimulationError("native trace inventory mismatch")
    return paths


def _neutral_trace_paths(neutral_root: Path) -> dict[int, dict[int, Path]]:
    paths: dict[int, dict[int, Path]] = defaultdict(dict)
    for trace in sorted(neutral_root.rglob("scheduler_trace.v1.jsonl")):
        if not trace.parent.name.startswith("reference-"):
            continue
        seed = int(trace.parent.name.removeprefix("reference-"))
        pool = int(trace.parent.parent.name.removeprefix("pool-"))
        paths[pool][seed] = trace
    if set(paths) != {51001, 51002, 51003} or any(
        len(traces) != 2 for traces in paths.values()
    ):
        raise CommonInputResimulationError("neutral trace inventory mismatch")
    return dict(paths)


def _verify_input_hashes(
    *,
    protocol: Mapping[str, Any],
    native_paths: Mapping[int, Mapping[str, Path]],
    neutral_paths: Mapping[int, Mapping[int, Path]],
) -> None:
    expected_native = protocol.get("native_trace_sha256_by_pool_and_arm")
    expected_neutral = protocol.get("neutral_trace_sha256_by_generation_seed")
    if not isinstance(expected_native, dict) or not isinstance(expected_neutral, dict):
        raise CommonInputResimulationError("protocol trace hash inventory missing")
    for pool, arms in native_paths.items():
        pool_hashes = expected_native.get(str(pool))
        if not isinstance(pool_hashes, dict) or set(pool_hashes) != set(arms):
            raise CommonInputResimulationError("native protocol inventory mismatch")
        for arm, path in arms.items():
            if _sha256(path) != pool_hashes[arm]:
                raise CommonInputResimulationError(
                    f"native input hash mismatch: pool={pool} arm={arm}"
                )
    for traces in neutral_paths.values():
        for seed, path in traces.items():
            if _sha256(path) != expected_neutral.get(str(seed)):
                raise CommonInputResimulationError(
                    f"neutral input hash mismatch: generation_seed={seed}"
                )


def _load_reference_halves(
    *, manifest_path: Path, expected_sha256: str
) -> dict[int, tuple[frozenset[str], frozenset[str]]]:
    if _sha256(manifest_path) != expected_sha256:
        raise CommonInputResimulationError("private reference manifest hash mismatch")
    manifest = _load_json(manifest_path)
    pools = manifest.get("pools")
    if not isinstance(pools, list):
        raise CommonInputResimulationError("private reference pools missing")
    result: dict[int, tuple[frozenset[str], frozenset[str]]] = {}
    for pool in pools:
        if not isinstance(pool, dict) or not isinstance(pool.get("prompts"), list):
            raise CommonInputResimulationError("private reference pool malformed")
        seed = pool.get("pool_seed")
        if not isinstance(seed, int):
            raise CommonInputResimulationError("private reference pool seed missing")
        prompts = pool["prompts"]
        source_ids = {
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
        easier = frozenset(
            str(prompt["source_prompt_id"])
            for prompt in prompts
            if isinstance(prompt, dict) and prompt.get("fixed_harder") is False
        )
        if (
            len(prompts) != 32
            or len(source_ids) != 32
            or len(lower) != 16
            or len(easier) != 16
        ):
            raise CommonInputResimulationError("private reference halves mismatch")
        result[seed] = (lower, easier)
    if set(result) != {51001, 51002, 51003}:
        raise CommonInputResimulationError("private reference pool set mismatch")
    return result


def _sampler_from_arm(arm: str) -> str:
    if arm.endswith("_in_order"):
        return "in_order"
    if arm.endswith("_ready_first"):
        return "ready_first"
    raise CommonInputResimulationError(f"cannot infer sampler from arm {arm}")


def _run_native_parity(
    native_paths: Mapping[int, Mapping[str, Path]],
) -> dict[str, Any]:
    by_pool: dict[str, dict[str, bool]] = {}
    for pool, arms in sorted(native_paths.items()):
        by_pool[str(pool)] = {}
        for arm, path in sorted(arms.items()):
            groups = load_fixed_stream(path)
            sampler = _sampler_from_arm(arm)
            result = simulate_fixed_stream(
                groups,
                sampler=sampler,  # type: ignore[arg-type]
                ready_ns_by_group=observed_ready_times(groups),
                selection_tick_ns=observed_selection_tick_times(path),
            )
            simulated = tuple(
                selection.selected_group_ids for selection in result.selections
            )
            observed = observed_nonempty_selections(path)
            passed = simulated == observed
            by_pool[str(pool)][arm] = passed
            if not passed:
                raise CommonInputResimulationError(
                    f"native parity failed: pool={pool} arm={arm}"
                )
    return {
        "trace_count": sum(len(arms) for arms in by_pool.values()),
        "all_24_exact": all(
            passed for arms in by_pool.values() for passed in arms.values()
        ),
        "by_pool_and_arm": by_pool,
    }


def _paired_comparison(
    *,
    groups: Sequence[FixedStreamGroup],
    ready_times: Mapping[str, int],
    lower: frozenset[str],
    easier: frozenset[str],
    cohort_restricted_ready_first: bool = False,
    selection_tick_ns: Sequence[int] | None = None,
) -> ComparisonResult:
    in_order = simulate_fixed_stream(
        groups,
        sampler="in_order",
        ready_ns_by_group=ready_times,
        selection_tick_ns=selection_tick_ns,
    )
    ready_first = simulate_fixed_stream(
        groups,
        sampler="ready_first",
        ready_ns_by_group=ready_times,
        cohort_restricted_ready_first=cohort_restricted_ready_first,
        selection_tick_ns=selection_tick_ns,
    )
    return compare_samplers(
        in_order,
        ready_first,
        lower_load_prompt_ids=lower,
        easier_prompt_ids=easier,
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise CommonInputResimulationError("cannot average empty values")
    return statistics.fmean(values)


def _periodic_selection_ticks(
    *, trace_path: Path, ready_times: Mapping[str, int], poll_interval_ms: float
) -> tuple[int, ...] | None:
    if poll_interval_ms == 0:
        return None
    if poll_interval_ms < 0:
        raise CommonInputResimulationError("poll interval must be nonnegative")
    interval_ns = round(poll_interval_ms * 1_000_000)
    anchor_ns = trace_start_ns(trace_path)
    final_ns = max(ready_times.values()) + interval_ns
    return tuple(range(anchor_ns, final_ns + 1, interval_ns))


def _run_counterfactuals(
    *,
    protocol: Mapping[str, Any],
    neutral_paths: Mapping[int, Mapping[int, Path]],
    reference_halves: Mapping[int, tuple[frozenset[str], frozenset[str]]],
    poll_interval_ms: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    design = protocol.get("counterfactual_design")
    if not isinstance(design, dict) or design.get("delay_grid_seconds") != [
        0.0,
        8.0,
        16.0,
        24.0,
        30.0,
    ]:
        raise CommonInputResimulationError("counterfactual delay grid mismatch")
    delays = [float(value) for value in design["delay_grid_seconds"]]
    trace_results: dict[str, Any] = {}
    pool_values: dict[float, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    l0_exact = True

    for pool, traces in sorted(neutral_paths.items()):
        lower, easier = reference_halves[pool]
        for seed, path in sorted(traces.items()):
            groups = load_fixed_stream(path)
            observed_prompt_ids = {group.source_prompt_id for group in groups}
            if not lower <= observed_prompt_ids or not easier <= observed_prompt_ids:
                raise CommonInputResimulationError("neutral stream identity mismatch")
            trace_entry: dict[str, Any] = {"pool_seed": pool, "generation_seed": seed}
            natural_ready = shifted_ready_times(
                groups, delayed_prompt_ids=frozenset(), delay_seconds=0.0
            )
            natural = _paired_comparison(
                groups=groups,
                ready_times=natural_ready,
                lower=lower,
                easier=easier,
                selection_tick_ns=_periodic_selection_ticks(
                    trace_path=path,
                    ready_times=natural_ready,
                    poll_interval_ms=poll_interval_ms,
                ),
            )
            trace_entry["natural"] = comparison_to_dict(natural)
            signed: dict[str, Any] = {}
            for delay in delays:
                higher = frozenset(
                    group.source_prompt_id
                    for group in groups
                    if group.source_prompt_id not in lower
                )
                high_ready = shifted_ready_times(
                    groups, delayed_prompt_ids=higher, delay_seconds=delay
                )
                low_ready = shifted_ready_times(
                    groups, delayed_prompt_ids=lower, delay_seconds=delay
                )
                high = _paired_comparison(
                    groups=groups,
                    ready_times=high_ready,
                    lower=lower,
                    easier=easier,
                    selection_tick_ns=_periodic_selection_ticks(
                        trace_path=path,
                        ready_times=high_ready,
                        poll_interval_ms=poll_interval_ms,
                    ),
                )
                low = _paired_comparison(
                    groups=groups,
                    ready_times=low_ready,
                    lower=lower,
                    easier=easier,
                    selection_tick_ns=_periodic_selection_ticks(
                        trace_path=path,
                        ready_times=low_ready,
                        poll_interval_ms=poll_interval_ms,
                    ),
                )
                l0_high = _paired_comparison(
                    groups=groups,
                    ready_times=high_ready,
                    lower=lower,
                    easier=easier,
                    cohort_restricted_ready_first=True,
                    selection_tick_ns=_periodic_selection_ticks(
                        trace_path=path,
                        ready_times=high_ready,
                        poll_interval_ms=poll_interval_ms,
                    ),
                )
                l0_low = _paired_comparison(
                    groups=groups,
                    ready_times=low_ready,
                    lower=lower,
                    easier=easier,
                    cohort_restricted_ready_first=True,
                    selection_tick_ns=_periodic_selection_ticks(
                        trace_path=path,
                        ready_times=low_ready,
                        poll_interval_ms=poll_interval_ms,
                    ),
                )
                l0_exact = l0_exact and all(
                    value == 0
                    for value in (
                        l0_high.lower_load_mean_normalized_promotion,
                        l0_low.lower_load_mean_normalized_promotion,
                        l0_high.changed_selected_step_sets,
                        l0_low.changed_selected_step_sets,
                    )
                )
                signed_separation = (
                    high.lower_load_mean_normalized_promotion
                    - low.lower_load_mean_normalized_promotion
                )
                signed[str(int(delay))] = {
                    "high_load_delayed": comparison_to_dict(high),
                    "low_load_delayed": comparison_to_dict(low),
                    "signed_separation": signed_separation,
                }
                pool_values[delay][pool]["high"].append(
                    high.lower_load_mean_normalized_promotion
                )
                pool_values[delay][pool]["low"].append(
                    low.lower_load_mean_normalized_promotion
                )
                pool_values[delay][pool]["signed"].append(signed_separation)
            trace_entry["signed_by_delay_seconds"] = signed
            trace_results[str(seed)] = trace_entry

    pool_aggregates: dict[str, Any] = {}
    for delay in delays:
        per_pool: dict[str, Any] = {}
        for pool in sorted(pool_values[delay]):
            per_pool[str(pool)] = {
                "high_load_delayed_mean_promotion": _mean(
                    pool_values[delay][pool]["high"]
                ),
                "low_load_delayed_mean_promotion": _mean(
                    pool_values[delay][pool]["low"]
                ),
                "signed_separation_mean": _mean(pool_values[delay][pool]["signed"]),
            }
        pool_aggregates[str(int(delay))] = per_pool

    rule = protocol.get("prospective_progression_rule")
    if not isinstance(rule, dict):
        raise CommonInputResimulationError("progression rule missing")
    threshold = float(rule["high_load_delayed_pool_median_lower_load_promotion_min"])
    reverse_threshold = float(
        rule["low_load_delayed_pool_median_lower_load_promotion_max"]
    )
    separation_threshold = float(rule["median_pool_signed_separation_min"])
    delay_decisions: dict[str, Any] = {}
    selected_delay: float | None = None
    for delay in [8.0, 16.0, 24.0, 30.0]:
        values = list(pool_aggregates[str(int(delay))].values())
        high = [float(value["high_load_delayed_mean_promotion"]) for value in values]
        low = [float(value["low_load_delayed_mean_promotion"]) for value in values]
        signed = [float(value["signed_separation_mean"]) for value in values]
        checks = {
            "high_pool_median_ge_min": statistics.median(high) >= threshold,
            "high_pools_at_or_above_min_ge_2": sum(value >= threshold for value in high)
            >= 2,
            "low_pool_median_le_max": statistics.median(low) <= reverse_threshold,
            "low_pools_at_or_below_max_ge_2": sum(
                value <= reverse_threshold for value in low
            )
            >= 2,
            "median_signed_separation_ge_min": statistics.median(signed)
            >= separation_threshold,
            "no_wrong_signed_high_pool_mean": all(value >= 0 for value in high),
            "no_wrong_signed_low_pool_mean": all(value <= 0 for value in low),
            "l0_negative_control_exact": l0_exact,
        }
        passed = all(checks.values())
        delay_decisions[str(int(delay))] = {"checks": checks, "passed": passed}
        if passed and selected_delay is None:
            selected_delay = delay

    decision = (
        "draft_but_do_not_launch_a_separate_fresh_zero_update_controlled_release_validation_candidate"
        if selected_delay is not None
        else "stop_no_stable_signed_common_input_regime"
    )
    aggregate = {
        "poll_interval_milliseconds": poll_interval_ms,
        "l0_negative_control_exact": l0_exact,
        "pool_mean_results_by_delay_seconds": pool_aggregates,
        "progression_checks_by_delay_seconds": delay_decisions,
        "smallest_passing_delay_seconds": selected_delay,
        "decision": decision,
    }
    return trace_results, aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--attempt1-failure-audit", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--repair-confirmation", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--neutral-root", type=Path, required=True)
    parser.add_argument("--private-reference-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol, protocol_sha256, repair_sha256 = _validate_repair_authority(
        protocol_path=args.protocol,
        confirmation_path=args.confirmation,
        failure_audit_path=args.attempt1_failure_audit,
        repair_path=args.repair,
        repair_confirmation_path=args.repair_confirmation,
    )
    native_paths = _native_trace_paths(args.native_root)
    neutral_paths = _neutral_trace_paths(args.neutral_root)
    _verify_input_hashes(
        protocol=protocol,
        native_paths=native_paths,
        neutral_paths=neutral_paths,
    )
    parity = _run_native_parity(native_paths)
    if parity["all_24_exact"] is not True:
        raise CommonInputResimulationError("native parity gate failed")
    repair = _load_json(args.repair)
    expected_manifest_sha = repair.get("bindings", {}).get(
        "private_reference_manifest_sha256"
    )
    if not isinstance(expected_manifest_sha, str):
        raise CommonInputResimulationError("private reference binding missing")
    reference_halves = _load_reference_halves(
        manifest_path=args.private_reference_manifest,
        expected_sha256=expected_manifest_sha,
    )

    poll_sensitivity: dict[str, Any] = {}
    selected_delay_by_interval: dict[str, float | None] = {}
    for poll_interval_ms in POLL_INTERVALS_MS:
        trace_results, aggregate = _run_counterfactuals(
            protocol=protocol,
            neutral_paths=neutral_paths,
            reference_halves=reference_halves,
            poll_interval_ms=poll_interval_ms,
        )
        key = str(int(poll_interval_ms))
        poll_sensitivity[key] = {
            "poll_interval_milliseconds": poll_interval_ms,
            "neutral_trace_results": trace_results,
            "aggregate": aggregate,
        }
        selected_delay_by_interval[key] = aggregate["smallest_passing_delay_seconds"]
    selected_values = tuple(selected_delay_by_interval.values())
    if (
        all(value is not None for value in selected_values)
        and len(set(selected_values)) == 1
    ):
        robust_delay = selected_values[0]
        decision = "draft_but_do_not_launch_a_separate_fresh_zero_update_controlled_release_validation_candidate"
    elif any(value is None for value in selected_values):
        robust_delay = None
        decision = "stop_no_poll_robust_signed_common_input_regime"
    else:
        robust_delay = None
        decision = "stop_poll_timing_sensitive"
    robust_aggregate = {
        "smallest_passing_delay_seconds_by_poll_interval": (selected_delay_by_interval),
        "poll_robust_smallest_passing_delay_seconds": robust_delay,
        "decision": decision,
    }
    output = {
        "schema_version": 2,
        "analysis_status": (
            "exploratory_dapo_load_alignment_common_input_poll_timing_sensitivity_resimulation"
        ),
        "calibration_only": True,
        "confirmatory_eligible": False,
        "protocol_sha256": protocol_sha256,
        "confirmation_sha256": _sha256(args.confirmation),
        "attempt1_failure_audit_sha256": _sha256(args.attempt1_failure_audit),
        "repair_sha256": repair_sha256,
        "repair_confirmation_sha256": _sha256(args.repair_confirmation),
        "source_nemo_rl_commit": repair["bindings"]["nemo_rl_source_commit"],
        "input_integrity": {
            "native_trace_count": 24,
            "neutral_trace_count": 6,
            "all_30_frozen_hashes_match": True,
            "private_reference_manifest_sha256": expected_manifest_sha,
        },
        "native_parity": parity,
        "poll_timing_sensitivity": poll_sensitivity,
        "aggregate": robust_aggregate,
        "privacy": {
            "private_prompt_identities_read_for_membership_join": True,
            "private_prompt_identities_emitted": False,
            "logical_group_identifiers_emitted": False,
            "reported_outputs_are_aggregate_only": True,
        },
        "claim_boundary": protocol["claim_boundary"],
        "authorization_state": {
            "eos_launch": False,
            "new_rollout": False,
            "learner_replay": False,
            "learner_training": False,
            "population_claim": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
