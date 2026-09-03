# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Run the frozen, pure-CPU scheduler-latency positive control."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from nemo_rl.algorithms.async_utils.scheduler_replay import (
    ReplayGroup,
    ReplayPolicy,
    ReplayTick,
    decorrelated_releases,
    mixture_metrics,
    natural_releases,
    replay_schedule,
)


class PositiveControlError(ValueError):
    """The frozen control contract or its result is invalid."""


EXPECTED_PLAN_KEYS = {
    "analysis_code_commit",
    "analysis_status",
    "calibration_only",
    "confirmatory_eligible",
    "fast_latency_ns",
    "groups_per_stratum",
    "horizon",
    "min_max_groups_per_tick",
    "permutation_seed_count",
    "permutation_seed_start",
    "plan_id",
    "pool_groups",
    "replay_authorized",
    "schema_version",
    "slow_latency_ns",
    "thresholds",
    "tick_times_ns",
    "training_authorized",
}
EXPECTED_THRESHOLDS = {
    "all_permutation_horizons_reached": True,
    "fast_share_interaction_min": 0.4,
    "natural_fast_share_min": 0.99,
    "permuted_mean_fast_share_max": 0.55,
    "permuted_mean_fast_share_min": 0.45,
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plan_id(payload: Mapping[str, object]) -> str:
    without_id = {key: value for key, value in payload.items() if key != "plan_id"}
    return _sha256_bytes(_canonical_json(without_id).encode())


def load_plan(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PositiveControlError("positive-control plan is unreadable") from error
    if not isinstance(value, dict) or set(value) != EXPECTED_PLAN_KEYS:
        raise PositiveControlError("positive-control plan keys do not match schema v1")
    if value.get("plan_id") != _plan_id(value):
        raise PositiveControlError("positive-control plan_id mismatch")
    expected = {
        "schema_version": 1,
        "analysis_status": "engineering_scheduler_latency_positive_control",
        "calibration_only": True,
        "confirmatory_eligible": False,
        "replay_authorized": False,
        "training_authorized": False,
        "pool_groups": 32,
        "groups_per_stratum": 16,
        "fast_latency_ns": 100_000_000,
        "slow_latency_ns": 900_000_000,
        "tick_times_ns": [200_000_000, 400_000_000],
        "min_max_groups_per_tick": 4,
        "horizon": 8,
        "permutation_seed_start": 0,
        "permutation_seed_count": 1000,
        "thresholds": EXPECTED_THRESHOLDS,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise PositiveControlError(f"positive-control plan has invalid {key}")
    commit = value.get("analysis_code_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise PositiveControlError("analysis_code_commit must be 40 lowercase hex")
    return value


def _groups(plan: Mapping[str, Any]) -> tuple[ReplayGroup, ...]:
    groups = []
    for slot in range(plan["pool_groups"]):
        task = "fast" if slot % 2 == 0 else "slow"
        latency = (
            plan["fast_latency_ns"] if task == "fast" else plan["slow_latency_ns"]
        )
        prompt_uid = hashlib.sha256(f"synthetic-slot-{slot}".encode()).hexdigest()
        groups.append(
            ReplayGroup(
                logical_group_id=f"group-{slot:02d}",
                prompt_uid=prompt_uid,
                source_prompt_id=prompt_uid,
                task_stratum=task,
                repeated_prompt_cluster_uid=prompt_uid,
                dispatch_cohort="all-at-zero",
                decorrelation_block="all-groups",
                slot_order=slot,
                dispatch_ns=0,
                ready_ns=latency,
                nominal_start_version=0,
                target_step=0,
            )
        )
    return tuple(groups)


def _ticks(plan: Mapping[str, Any]) -> tuple[ReplayTick, ...]:
    size = plan["min_max_groups_per_tick"]
    return tuple(
        ReplayTick(index, clock, 0, size, size)
        for index, clock in enumerate(plan["tick_times_ns"])
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * probability)
    return ordered[index]


def analyze_plan(
    plan: Mapping[str, Any],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    groups = _groups(plan)
    ticks = _ticks(plan)
    policy = ReplayPolicy("ready_first")
    horizon = plan["horizon"]
    natural_release = natural_releases(groups)
    natural_result = replay_schedule(groups, ticks, policy, natural_release)
    natural_metrics = mixture_metrics(groups, natural_result, horizon=horizon)
    natural_fast_share = natural_metrics["selected_share"]["fast"]

    permutation_rows = []
    assignment_rows = []
    permuted_fast_shares = []
    all_horizons_reached = True
    first_seed = plan["permutation_seed_start"]
    for seed in range(first_seed, first_seed + plan["permutation_seed_count"]):
        release = decorrelated_releases(
            groups, seed=seed, namespace="scheduler-positive-control-v1"
        )
        result = replay_schedule(groups, ticks, policy, release)
        reached = len(result.selected_group_ids) >= horizon
        all_horizons_reached = all_horizons_reached and reached
        if not reached:
            permutation_rows.append({"seed": seed, "horizon_reached": False})
            continue
        metrics = mixture_metrics(groups, result, horizon=horizon)
        fast_share = metrics["selected_share"]["fast"]
        permuted_fast_shares.append(fast_share)
        permutation_rows.append(
            {
                "seed": seed,
                "horizon_reached": True,
                "fast_share": fast_share,
                "total_variation": metrics["total_variation"],
                "assignment_fixed_points": release.assignment_fixed_points,
            }
        )
        assignment_rows.extend(
            {"row_type": "release_assignment", "seed": seed, **asdict(assignment)}
            for assignment in release.assignments
        )
    if not permuted_fast_shares:
        raise PositiveControlError("no permutation reached the selected-group horizon")
    permuted_mean = statistics.fmean(permuted_fast_shares)
    permuted_sd = statistics.stdev(permuted_fast_shares)
    interaction = (natural_fast_share - 0.5) - (permuted_mean - 0.5)
    checks = {
        "all_permutation_horizons_reached": all_horizons_reached,
        "natural_fast_share_min": natural_fast_share
        >= EXPECTED_THRESHOLDS["natural_fast_share_min"],
        "permuted_mean_fast_share_min": permuted_mean
        >= EXPECTED_THRESHOLDS["permuted_mean_fast_share_min"],
        "permuted_mean_fast_share_max": permuted_mean
        <= EXPECTED_THRESHOLDS["permuted_mean_fast_share_max"],
        "fast_share_interaction_min": interaction
        >= EXPECTED_THRESHOLDS["fast_share_interaction_min"],
    }
    result = {
        "schema_version": 1,
        "analysis_status": plan["analysis_status"],
        "calibration_only": True,
        "confirmatory_eligible": False,
        "replay_authorized": False,
        "training_authorized": False,
        "plan_id": plan["plan_id"],
        "pool_groups": len(groups),
        "horizon": horizon,
        "natural": {
            "selected_group_ids": list(natural_result.selected_group_ids),
            "fast_share": natural_fast_share,
            "total_variation": natural_metrics["total_variation"],
        },
        "permutations": {
            "planned": plan["permutation_seed_count"],
            "horizon_reached": len(permuted_fast_shares),
            "fast_share_mean": permuted_mean,
            "fast_share_sd": permuted_sd,
            "fast_share_mcse": permuted_sd / len(permuted_fast_shares) ** 0.5,
            "fast_share_quantiles": {
                "q025": _quantile(permuted_fast_shares, 0.025),
                "q50": _quantile(permuted_fast_shares, 0.5),
                "q975": _quantile(permuted_fast_shares, 0.975),
            },
        },
        "fast_share_interaction": interaction,
        "locked_checks": checks,
        "decision": "positive_control_passed"
        if all(checks.values())
        else "positive_control_failed",
        "interpretation": (
            "engineering control with synthetic latencies; not evidence about "
            "natural workloads or trained-policy outcomes"
        ),
    }
    natural_decisions = [
        {"row_type": "natural_decision", **asdict(decision)}
        for decision in natural_result.decisions
    ]
    return result, permutation_rows, assignment_rows + natural_decisions


def _git_commit(repo_root: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        raise PositiveControlError("analysis worktree must be clean")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json(path: Path, value: object) -> None:
    path.write_text(_canonical_json(value) + "\n")


def run(*, plan_path: Path, output_dir: Path) -> dict[str, object]:
    plan = load_plan(plan_path)
    repo_root = Path(__file__).resolve().parents[1]
    commit = _git_commit(repo_root)
    if commit != plan["analysis_code_commit"]:
        raise PositiveControlError("plan analysis_code_commit does not match HEAD")
    if output_dir.exists():
        raise PositiveControlError("output directory already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        plan_snapshot = temporary / "positive_control_plan.v1.json"
        shutil.copyfile(plan_path, plan_snapshot)
        if load_plan(plan_snapshot) != plan:
            raise PositiveControlError("plan changed while being snapshotted")
        result, permutation_rows, combined_rows = analyze_plan(plan)
        _write_json(temporary / "result.v1.json", result)
        with (temporary / "permutations.v1.jsonl").open("w") as stream:
            for row in permutation_rows:
                stream.write(_canonical_json(row) + "\n")
        with (temporary / "release_assignments_and_natural_decisions.v1.jsonl").open(
            "w"
        ) as stream:
            for row in combined_rows:
                stream.write(_canonical_json(row) + "\n")
        with (temporary / "groups.v1.jsonl").open("w") as stream:
            for group in _groups(plan):
                stream.write(_canonical_json(asdict(group)) + "\n")
        _write_json(
            temporary / "provenance.v1.json",
            {
                "schema_version": 1,
                "analysis_code_commit": commit,
                "plan_sha256": _sha256_path(plan_snapshot),
                "plan_id": plan["plan_id"],
            },
        )
        artifacts = sorted(
            path for path in temporary.iterdir() if path.name != "SHA256SUMS"
        )
        (temporary / "SHA256SUMS").write_text(
            "".join(f"{_sha256_path(path)}  {path.name}\n" for path in artifacts)
        )
        temporary.rename(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(plan_path=arguments.plan, output_dir=arguments.output_dir)
    print(_canonical_json(result))


if __name__ == "__main__":
    main()
