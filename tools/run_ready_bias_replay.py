# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Run a hash-bound, exploratory fixed-pool scheduler replay closure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from itertools import chain
from pathlib import Path
from typing import Any, TextIO

from nemo_rl.algorithms.async_utils.fixed_pool import (
    load_fixed_pool_manifest,
    validate_fixed_pool_trace,
)
from nemo_rl.algorithms.async_utils.scheduler_replay import (
    ReleaseSchedule,
    ReplayGroup,
    SchedulerReplayError,
    decorrelated_releases,
    latency_diagnostics,
    load_groups_from_trace,
    mixture_metrics,
    natural_releases,
    replay_schedule,
)
from nemo_rl.algorithms.async_utils.scheduler_replay_plan import (
    ExploratoryClosurePlan,
    ReplayPolicyPlan,
    ReplaySourceRun,
    build_deadline_ticks,
    load_exploratory_closure_plan,
)


TRACE_NAME = "scheduler_trace.v1.jsonl"
MANIFEST_NAME = "fixed_pool_manifest.v1.json"
VALIDATION_NAME = "fixed_pool_trace_validation.json"


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_json_line(stream: TextIO, value: object) -> None:
    stream.write(_canonical_json(value))
    stream.write("\n")


def _current_commit(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        raise SchedulerReplayError("analysis worktree must be clean")
    return completed.stdout.strip()


def _find_source_run(
    run_dir: Path, plan: ExploratoryClosurePlan
) -> tuple[ReplaySourceRun, Path, Path, Path]:
    trace_path = run_dir / TRACE_NAME
    manifest_path = run_dir / MANIFEST_NAME
    validation_path = run_dir / VALIDATION_NAME
    if (
        not trace_path.is_file()
        or not manifest_path.is_file()
        or not validation_path.is_file()
    ):
        raise SchedulerReplayError(
            f"run directory lacks a required source artifact: {run_dir}"
        )
    trace_sha = _sha256_path(trace_path)
    manifest_sha = _sha256_path(manifest_path)
    validation_sha = _sha256_path(validation_path)
    matches = [
        source
        for source in plan.source_runs
        if source.trace_sha256 == trace_sha
        and source.manifest_sha256 == manifest_sha
        and source.validation_sha256 == validation_sha
    ]
    if len(matches) != 1:
        raise SchedulerReplayError(
            f"run artifacts are absent or ambiguous in the replay plan: {run_dir}"
        )
    return matches[0], trace_path, manifest_path, validation_path


def _snapshot_source_inputs(
    *,
    plan_path: Path,
    plan: ExploratoryClosurePlan,
    resolved_runs: Sequence[tuple[ReplaySourceRun, Path, Path, Path]],
    work_dir: Path,
) -> tuple[Path, list[tuple[ReplaySourceRun, Path, Path, Path]]]:
    plan_snapshot = work_dir / "source_exploratory_closure_plan.v1.json"
    shutil.copyfile(plan_path, plan_snapshot)
    if load_exploratory_closure_plan(plan_snapshot) != plan:
        raise SchedulerReplayError("replay plan changed while it was snapshotted")
    snapshots = []
    for source, trace_path, manifest_path, validation_path in resolved_runs:
        paths = []
        for label, input_path, expected_sha in (
            ("trace", trace_path, source.trace_sha256),
            ("manifest", manifest_path, source.manifest_sha256),
            ("validation", validation_path, source.validation_sha256),
        ):
            snapshot = work_dir / f"source_{source.run_id}_{label}{input_path.suffix}"
            shutil.copyfile(input_path, snapshot)
            if _sha256_path(snapshot) != expected_sha:
                raise SchedulerReplayError(
                    f"{label} changed while {source.run_id} was snapshotted"
                )
            paths.append(snapshot)
        snapshots.append((source, paths[0], paths[1], paths[2]))
    return plan_snapshot, snapshots


def _metric_or_unreached(
    groups: Sequence[ReplayGroup],
    result: Any,
    release: ReleaseSchedule,
    horizon: int,
) -> dict[str, object]:
    try:
        metrics = mixture_metrics(groups, result, horizon=horizon)
    except SchedulerReplayError as error:
        return {
            "requested_horizon": horizon,
            "horizon_reached": False,
            "selected_prompt_groups": len(result.selected_group_ids),
            "reason": str(error),
        }
    by_id = {group.logical_group_id: group for group in groups}
    selected_ids = result.selected_group_ids[:horizon]
    residence_by_task: dict[str, list[int]] = defaultdict(list)
    for group_id in selected_ids:
        group = by_id[group_id]
        residence_ns = (
            _selection_time(result, group_id)
            - release.ready_ns_by_group[group.logical_group_id]
        )
        if residence_ns < 0:
            raise SchedulerReplayError("selection precedes the active release schedule")
        residence_by_task[group.task_stratum].append(residence_ns)
    metrics["ready_to_select_mean_ns_by_task"] = {
        task: sum(values) / len(values)
        for task, values in sorted(residence_by_task.items())
    }
    horizon_tick = int(metrics["horizon_tick_id"])
    selected_at_tick = sum(
        len(decision.selected_group_ids)
        for decision in result.decisions
        if decision.tick_id <= horizon_tick
    )
    metrics["selected_total_at_horizon_tick"] = selected_at_tick
    metrics["horizon_overshoot"] = selected_at_tick - horizon
    return metrics


def _selection_time(result: Any, group_id: str) -> int:
    for decision in result.decisions:
        if group_id in decision.selected_group_ids:
            return decision.monotonic_ns
    raise SchedulerReplayError(f"selected group lacks a selection tick: {group_id}")


def _decision_record(
    *,
    run_id: str,
    deadline_ns: int,
    policy: ReplayPolicyPlan,
    release: ReleaseSchedule,
    decision: Any,
    clock_origin_ns: int,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "deadline_ns": deadline_ns,
        "policy": policy.label,
        "kernel": policy.kernel,
        "release_kind": release.kind,
        "release_seed": release.seed,
        "tick_id": decision.tick_id,
        "tick_ns": decision.monotonic_ns,
        "tick_elapsed_ns": decision.monotonic_ns - clock_origin_ns,
        "nominal_trainer_version": decision.trainer_version,
        "ready_group_ids": decision.ready_group_ids,
        "eligible_group_ids": decision.eligible_group_ids,
        "selected_group_ids": decision.selected_group_ids,
        "evicted_group_ids": decision.evicted_group_ids,
    }


def _metric_record(
    *,
    run_id: str,
    deadline_ns: int,
    policy: ReplayPolicyPlan,
    release: ReleaseSchedule,
    horizon: int,
    metrics: Mapping[str, object],
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "deadline_ns": deadline_ns,
        "policy": policy.label,
        "kernel": policy.kernel,
        "release_kind": release.kind,
        "release_seed": release.seed,
        "horizon": horizon,
        **metrics,
    }


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _interaction_summaries(
    metric_rows: Sequence[Mapping[str, object]], plan: ExploratoryClosurePlan
) -> list[dict[str, object]]:
    baseline = next(policy for policy in plan.policies if policy.kernel == "in_order")
    rows: dict[tuple[str, int, str, str, int | None, int], Mapping[str, object]] = {}
    for row in metric_rows:
        key = (
            str(row["run_id"]),
            int(row["deadline_ns"]),
            str(row["policy"]),
            str(row["release_kind"]),
            int(row["release_seed"]) if row["release_seed"] is not None else None,
            int(row["horizon"]),
        )
        rows[key] = row
    summaries: list[dict[str, object]] = []
    for source in plan.source_runs:
        for deadline_ns in plan.deadlines_ns:
            for policy in plan.policies:
                if policy.label == baseline.label:
                    continue
                for horizon in plan.horizons:
                    natural_policy = rows[
                        (
                            source.run_id,
                            deadline_ns,
                            policy.label,
                            "natural",
                            None,
                            horizon,
                        )
                    ]
                    natural_baseline = rows[
                        (
                            source.run_id,
                            deadline_ns,
                            baseline.label,
                            "natural",
                            None,
                            horizon,
                        )
                    ]
                    if (
                        not natural_policy["horizon_reached"]
                        or not natural_baseline["horizon_reached"]
                    ):
                        summaries.append(
                            {
                                "run_id": source.run_id,
                                "deadline_ns": deadline_ns,
                                "policy": policy.label,
                                "horizon": horizon,
                                "horizon_reached": False,
                            }
                        )
                        continue
                    task_names = sorted(
                        str(task) for task in dict(natural_policy["signed_share_shift"])
                    )
                    metric_specs: list[tuple[str, str | None]] = [
                        ("total_variation", None),
                        *(("signed_share_shift", task) for task in task_names),
                    ]
                    for metric_name, task_name in metric_specs:
                        output_metric_name = (
                            metric_name
                            if task_name is None
                            else f"{metric_name}:{task_name}"
                        )
                        natural_effect = _metric_value(
                            natural_policy, metric_name, task_name
                        ) - _metric_value(natural_baseline, metric_name, task_name)
                        permuted_effects = []
                        for seed in plan.release_intervention.seeds:
                            permuted_policy = rows[
                                (
                                    source.run_id,
                                    deadline_ns,
                                    policy.label,
                                    "decorrelated",
                                    seed,
                                    horizon,
                                )
                            ]
                            permuted_baseline = rows[
                                (
                                    source.run_id,
                                    deadline_ns,
                                    baseline.label,
                                    "decorrelated",
                                    seed,
                                    horizon,
                                )
                            ]
                            if (
                                not permuted_policy["horizon_reached"]
                                or not permuted_baseline["horizon_reached"]
                            ):
                                continue
                            permuted_effects.append(
                                _metric_value(permuted_policy, metric_name, task_name)
                                - _metric_value(
                                    permuted_baseline, metric_name, task_name
                                )
                            )
                        if (
                            len(permuted_effects)
                            != plan.release_intervention.seed_count
                        ):
                            summaries.append(
                                {
                                    "run_id": source.run_id,
                                    "deadline_ns": deadline_ns,
                                    "policy": policy.label,
                                    "horizon": horizon,
                                    "metric": output_metric_name,
                                    "horizon_reached": False,
                                    "reason": (
                                        "not every predeclared permutation reached "
                                        "the paired horizon"
                                    ),
                                    "paired_permutations": len(permuted_effects),
                                    "required_permutations": plan.release_intervention.seed_count,
                                }
                            )
                            continue
                        mean = statistics.fmean(permuted_effects)
                        sd = (
                            statistics.stdev(permuted_effects)
                            if len(permuted_effects) > 1
                            else 0.0
                        )
                        summaries.append(
                            {
                                "run_id": source.run_id,
                                "deadline_ns": deadline_ns,
                                "policy": policy.label,
                                "horizon": horizon,
                                "metric": output_metric_name,
                                "horizon_reached": True,
                                "natural_policy_effect": natural_effect,
                                "permuted_policy_effect_mean": mean,
                                "permuted_policy_effect_sd": sd,
                                "permuted_policy_effect_mcse": sd
                                / len(permuted_effects) ** 0.5,
                                "permuted_policy_effect_q025": _quantile(
                                    permuted_effects, 0.025
                                ),
                                "permuted_policy_effect_median": _quantile(
                                    permuted_effects, 0.5
                                ),
                                "permuted_policy_effect_q975": _quantile(
                                    permuted_effects, 0.975
                                ),
                                "interaction": natural_effect - mean,
                                "paired_permutations": len(permuted_effects),
                            }
                        )
    return summaries


def _metric_value(
    row: Mapping[str, object], metric_name: str, task_name: str | None
) -> float:
    if task_name is None:
        return float(row[metric_name])
    return float(dict(row[metric_name])[task_name])


def _progression_decision(
    summaries: Sequence[Mapping[str, object]], plan: ExploratoryClosurePlan
) -> dict[str, object]:
    primary = plan.stop_rules.primary_horizon
    eligible = [
        row
        for row in summaries
        if row.get("horizon") == primary
        and row.get("metric") == "total_variation"
        and row.get("horizon_reached") is True
    ]
    if any(
        float(row["permuted_policy_effect_mcse"])
        > plan.stop_rules.maximum_tv_interaction_mcse
        for row in eligible
    ):
        return {
            "decision": "engineering_failure",
            "reason": "TV interaction Monte Carlo error exceeded the locked bound",
        }
    by_policy_deadline: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in eligible:
        by_policy_deadline[(str(row["policy"]), int(row["deadline_ns"]))].append(
            float(row["interaction"])
        )
    nonbaseline_policies = [
        policy.label for policy in plan.policies if policy.kernel != "in_order"
    ]
    fully_reached_deadlines = {
        deadline_ns
        for deadline_ns in plan.deadlines_ns
        if all(
            any(
                row.get("run_id") == source.run_id
                and row.get("deadline_ns") == deadline_ns
                and row.get("policy") == policy
                and row.get("horizon") == primary
                and row.get("metric") == "total_variation"
                and row.get("horizon_reached") is True
                for row in summaries
            )
            for source in plan.source_runs
            for policy in nonbaseline_policies
        )
    }
    qualifying: dict[str, list[int]] = defaultdict(list)
    for (policy, deadline_ns), values in by_policy_deadline.items():
        if deadline_ns not in fully_reached_deadlines or len(values) != len(
            plan.source_runs
        ):
            continue
        same_direction = all(value > 0 for value in values) or all(
            value < 0 for value in values
        )
        median_abs = abs(statistics.median(values))
        if median_abs >= plan.stop_rules.minimum_absolute_tv_interaction and (
            same_direction or not plan.stop_rules.require_same_direction_all_runs
        ):
            qualifying[policy].append(deadline_ns)
    required = plan.stop_rules.adjacent_deadline_scenarios_required
    for policy, deadlines in qualifying.items():
        ordered = sorted(deadlines)
        for start in range(len(ordered) - required + 1):
            window = ordered[start : start + required]
            plan_positions = [plan.deadlines_ns.index(value) for value in window]
            if all(
                right == left + 1
                for left, right in zip(plan_positions, plan_positions[1:])
            ):
                return {
                    "decision": "fresh_holdout_design_eligible",
                    "policy": policy,
                    "qualifying_adjacent_deadlines_ns": window,
                    "caveat": "calibration replay is not confirmatory and cannot authorize online training",
                }
    return {
        "decision": "retire_current_workload",
        "reason": "locked robust-interaction progression rule was not met",
    }


def _across_run_interaction_summaries(
    summaries: Sequence[Mapping[str, object]], plan: ExploratoryClosurePlan
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str, int, str], list[float]] = defaultdict(list)
    for row in summaries:
        if row.get("horizon_reached") is not True or "metric" not in row:
            continue
        grouped[
            (
                int(row["deadline_ns"]),
                str(row["policy"]),
                int(row["horizon"]),
                str(row["metric"]),
            )
        ].append(float(row["interaction"]))
    output = []
    for (deadline_ns, policy, horizon, metric), values in sorted(grouped.items()):
        if len(values) != len(plan.source_runs):
            continue
        output.append(
            {
                "deadline_ns": deadline_ns,
                "policy": policy,
                "horizon": horizon,
                "metric": metric,
                "run_count": len(values),
                "interaction_median": statistics.median(values),
                "interaction_min": min(values),
                "interaction_max": max(values),
            }
        )
    return output


def _group_record(
    run_id: str, group: ReplayGroup, *, clock_origin_ns: int
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "logical_group_id": group.logical_group_id,
        "prompt_uid": group.prompt_uid,
        "source_prompt_id": group.source_prompt_id,
        "task_stratum": group.task_stratum,
        "repeated_prompt_cluster_uid": group.repeated_prompt_cluster_uid,
        "dispatch_cohort": group.dispatch_cohort,
        "decorrelation_block": group.decorrelation_block,
        "slot_order": group.slot_order,
        "dispatch_ns": group.dispatch_ns,
        "dispatch_elapsed_ns": group.dispatch_ns - clock_origin_ns,
        "ready_ns": group.ready_ns,
        "ready_elapsed_ns": group.ready_ns - clock_origin_ns,
        "archived_ns": group.archived_ns,
        "natural_latency_ns": group.natural_latency_ns,
        "ready_to_archive_ns": (
            group.archived_ns - group.ready_ns
            if group.archived_ns is not None
            else None
        ),
        "nominal_start_version": group.nominal_start_version,
        "target_step": group.target_step,
        "physical_start_version": group.physical_start_version,
        "physical_end_version": group.physical_end_version,
        "scalar_summaries": dict(group.scalar_summaries),
    }


def _collection_diagnostics(groups: Sequence[ReplayGroup]) -> dict[str, object]:
    by_block: dict[str, list[ReplayGroup]] = defaultdict(list)
    by_task_archive: dict[str, list[int]] = defaultdict(list)
    dispatch_anchors: dict[int, int] = {}
    for group in groups:
        by_block[group.decorrelation_block].append(group)
        dispatch_anchors[group.nominal_start_version] = max(
            dispatch_anchors.get(group.nominal_start_version, 0), group.dispatch_ns
        )
        if group.archived_ns is not None:
            by_task_archive[group.task_stratum].append(
                group.archived_ns - group.ready_ns
            )
    ordered_anchors = [dispatch_anchors[key] for key in sorted(dispatch_anchors)]
    return {
        "blocks": {
            block_id: {
                "size": len(members),
                "task_counts": {
                    task: sum(member.task_stratum == task for member in members)
                    for task in sorted({member.task_stratum for member in members})
                },
                "distinct_latency_values": len(
                    {member.natural_latency_ns for member in members}
                ),
            }
            for block_id, members in sorted(by_block.items())
        },
        "dispatch_anchor_gaps_ns": [
            right - left for left, right in zip(ordered_anchors, ordered_anchors[1:])
        ],
        "ready_to_archive_median_ns_by_task": {
            task: statistics.median(values)
            for task, values in sorted(by_task_archive.items())
        },
    }


def run_analysis(
    *, plan_path: Path, run_dirs: Sequence[Path], output_dir: Path, repo_root: Path
) -> None:
    """Execute the complete replay plan and write hash-addressed artifacts."""
    plan = load_exploratory_closure_plan(plan_path)
    commit = _current_commit(repo_root)
    if commit != plan.analysis_code_commit:
        raise SchedulerReplayError(
            f"analysis code commit mismatch: plan={plan.analysis_code_commit}, actual={commit}"
        )
    if len(run_dirs) != len(plan.source_runs):
        raise SchedulerReplayError("run directory count does not match the replay plan")
    if output_dir.exists():
        raise FileExistsError(f"replay output already exists: {output_dir}")
    output_dir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    work_dir = output_dir.with_name(f".{output_dir.name}.partial")
    work_dir.mkdir(mode=0o700, exist_ok=False)
    os.chmod(work_dir, 0o700)

    resolved_runs = [_find_source_run(path, plan) for path in run_dirs]
    if {source.run_id for source, _, _, _ in resolved_runs} != {
        source.run_id for source in plan.source_runs
    }:
        raise SchedulerReplayError("run directories do not cover the replay plan")
    plan_snapshot, resolved_runs = _snapshot_source_inputs(
        plan_path=plan_path,
        plan=plan,
        resolved_runs=resolved_runs,
        work_dir=work_dir,
    )

    metric_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    groups_path = work_dir / "groups.v1.jsonl"
    assignments_path = work_dir / "release_assignments.v1.jsonl"
    decisions_path = work_dir / "decisions.v1.jsonl"
    release_diagnostics_path = work_dir / "release_diagnostics.v1.jsonl"
    metrics_long_path = work_dir / "metrics_long.v1.jsonl"
    with (
        groups_path.open("x", encoding="utf-8") as groups_stream,
        assignments_path.open("x", encoding="utf-8") as assignments_stream,
        decisions_path.open("x", encoding="utf-8") as decisions_stream,
        release_diagnostics_path.open(
            "x", encoding="utf-8"
        ) as release_diagnostics_stream,
        metrics_long_path.open("x", encoding="utf-8") as metrics_long_stream,
    ):
        for source, trace_path, manifest_path, validation_path in sorted(
            resolved_runs, key=lambda item: item[0].run_id
        ):
            manifest = load_fixed_pool_manifest(manifest_path)
            if manifest.pool_id != source.pool_id:
                raise SchedulerReplayError(f"pool ID mismatch for {source.run_id}")
            if manifest.order_seed != source.order_seed:
                raise SchedulerReplayError(f"order seed mismatch for {source.run_id}")
            try:
                uploaded_validation = json.loads(validation_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise SchedulerReplayError(
                    f"cannot read source validation for {source.run_id}: {error}"
                ) from error
            if (
                uploaded_validation.get("status") != "passed"
                or uploaded_validation.get("generation_study_seed")
                != source.generation_seed
                or uploaded_validation.get("order_seed") != source.order_seed
                or uploaded_validation.get("pool_id") != source.pool_id
                or uploaded_validation.get("manifest_sha256") != source.manifest_sha256
            ):
                raise SchedulerReplayError(
                    f"uploaded source validation mismatch for {source.run_id}"
                )
            strict_report = validate_fixed_pool_trace(
                trace_path,
                manifest,
                expected_completions_per_group=plan.expected_completions_per_group,
            )
            groups = load_groups_from_trace(
                trace_path,
                manifest_path,
                expected_completions_per_group=plan.expected_completions_per_group,
            )
            clock_origin_ns = min(group.dispatch_ns for group in groups)
            for group in groups:
                _write_json_line(
                    groups_stream,
                    _group_record(
                        source.run_id, group, clock_origin_ns=clock_origin_ns
                    ),
                )
            validation_rows.append(
                {
                    "run_id": source.run_id,
                    "trace_sha256": source.trace_sha256,
                    "manifest_sha256": source.manifest_sha256,
                    "pool_id": source.pool_id,
                    "clock_origin": "first_attempt_dispatched",
                    "clock_origin_ns": clock_origin_ns,
                    "strict_fixed_pool_validation": asdict(strict_report),
                    "natural_latency_diagnostics": latency_diagnostics(
                        groups, natural_releases(groups)
                    ),
                    "collection_diagnostics": _collection_diagnostics(groups),
                }
            )
            schedules: Iterable[ReleaseSchedule] = chain(
                (natural_releases(groups),),
                (
                    decorrelated_releases(groups, seed=seed, namespace=source.pool_id)
                    for seed in plan.release_intervention.seeds
                ),
            )
            for release in schedules:
                _write_json_line(
                    release_diagnostics_stream,
                    {
                        "run_id": source.run_id,
                        "release_kind": release.kind,
                        "release_seed": release.seed,
                        **latency_diagnostics(groups, release),
                    },
                )
                for assignment in release.assignments:
                    _write_json_line(
                        assignments_stream,
                        {
                            "run_id": source.run_id,
                            "release_seed": release.seed,
                            **asdict(assignment),
                        },
                    )
                for deadline_ns in plan.deadlines_ns:
                    ticks = build_deadline_ticks(
                        groups, deadline_ns=deadline_ns, tick_plan=plan.tick_plan
                    )
                    for policy in plan.policies:
                        result = replay_schedule(
                            groups, ticks, policy.to_policy(), releases=release
                        )
                        for decision in result.decisions:
                            _write_json_line(
                                decisions_stream,
                                _decision_record(
                                    run_id=source.run_id,
                                    deadline_ns=deadline_ns,
                                    policy=policy,
                                    release=release,
                                    decision=decision,
                                    clock_origin_ns=clock_origin_ns,
                                ),
                            )
                        for horizon in plan.horizons:
                            metrics = _metric_or_unreached(
                                groups, result, release, horizon
                            )
                            metric_row = _metric_record(
                                run_id=source.run_id,
                                deadline_ns=deadline_ns,
                                policy=policy,
                                release=release,
                                horizon=horizon,
                                metrics=metrics,
                            )
                            metric_rows.append(metric_row)
                            _write_json_line(metrics_long_stream, metric_row)

    interactions = _interaction_summaries(metric_rows, plan)
    summary = {
        "schema_version": 1,
        "analysis_status": plan.analysis_status,
        "calibration_only": plan.calibration_only,
        "confirmatory_eligible": plan.confirmatory_eligible,
        "fidelity_status": "unavailable_for_fixed_pool_source_trace",
        "plan_id": plan.plan_id,
        "plan_file_sha256": _sha256_path(plan_snapshot),
        "analysis_code_commit": commit,
        "validations": validation_rows,
        "interactions": interactions,
        "across_run_interactions": _across_run_interaction_summaries(
            interactions, plan
        ),
        "progression": _progression_decision(interactions, plan),
    }
    (work_dir / "metrics.v1.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (work_dir / "metrics.v1.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        fieldnames = (
            "run_id",
            "deadline_ns",
            "policy",
            "kernel",
            "release_kind",
            "release_seed",
            "horizon",
            "horizon_reached",
            "observed_prefix_groups",
            "total_variation",
            "worst_representation_ratio",
            "inclusion_disparity",
            "eviction_fraction",
            "eviction_disparity",
            "selected_share",
            "signed_share_shift",
            "reason",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in metric_rows:
            serialized = dict(row)
            for key in ("selected_share", "signed_share_shift"):
                if key in serialized:
                    serialized[key] = _canonical_json(serialized[key])
            writer.writerow(serialized)

    provenance = {
        "schema_version": 1,
        "plan_id": plan.plan_id,
        "analysis_code_commit": commit,
        "source_runs": [source.model_dump(mode="json") for source in plan.source_runs],
        "semantics": plan.semantics,
        "fidelity_status": "unavailable_for_fixed_pool_source_trace",
        "interpretation": (
            "post-observation descriptive closure; not native throughput, "
            "not a trained-policy counterfactual, and not confirmatory evidence"
        ),
    }
    (work_dir / "provenance.v1.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copyfile(plan_snapshot, work_dir / "exploratory_closure_plan.v1.json")

    artifact_rows = []
    for path in sorted(work_dir.iterdir()):
        if path.name == "artifact_manifest.v1.json":
            continue
        os.chmod(path, 0o600)
        artifact_rows.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256_path(path),
            }
        )
    (work_dir / "artifact_manifest.v1.json").write_text(
        json.dumps(
            {"schema_version": 1, "artifacts": artifact_rows}, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    os.chmod(work_dir / "artifact_manifest.v1.json", 0o600)
    work_dir.rename(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    run_analysis(
        plan_path=args.plan,
        run_dirs=args.run_dir,
        output_dir=args.output_dir,
        repo_root=repo_root,
    )


if __name__ == "__main__":
    main()
