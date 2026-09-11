#!/usr/bin/env python3
"""Summarize dose context from authenticated M4 lifecycle ledgers."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


class DoseContextError(ValueError):
    """Raised when a lifecycle ledger cannot support the audit."""


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(finite),
        "mean": math.fsum(finite) / len(finite) if finite else None,
        "p50": _quantile(finite, 0.50),
        "p90": _quantile(finite, 0.90),
        "p99": _quantile(finite, 0.99),
        "maximum": max(finite) if finite else None,
    }


def _load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise DoseContextError(f"{path}:{line_number}: invalid JSON") from error
        if row.get("schema_version") != 4:
            raise DoseContextError(f"{path}:{line_number}: lifecycle schema is not v4")
        rows.append(row)
    if not rows:
        raise DoseContextError(f"{path}: empty lifecycle ledger")
    if len({row["run_id"] for row in rows}) != 1:
        raise DoseContextError(f"{path}: multiple run identifiers")
    if len({row["clock_domain_id"] for row in rows}) != 1:
        raise DoseContextError(f"{path}: multiple monotonic clock domains")
    return rows


def _analyze(label: str, path: Path) -> dict[str, Any]:
    rows = _load(path)
    by_group: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    stages = Counter()
    learner_times: list[int] = []
    generation_seconds: list[float] = []
    environment_seconds: list[float] = []
    for row in rows:
        stage = str(row["stage"])
        stages[stage] += 1
        if stage == "learner_version_advanced":
            learner_times.append(int(row["timestamp_ns"]))
        elif row["group_id"] != "__learner__":
            by_group[str(row["group_id"])][stage].append(row)
        if stage == "sibling_done":
            if row["generation_duration_ns"] is not None:
                generation_seconds.append(float(row["generation_duration_ns"]) / 1e9)
            if row["environment_duration_ns"] is not None:
                environment_seconds.append(float(row["environment_duration_ns"]) / 1e9)

    post_sibling_to_hold_start: dict[str, list[float]] = defaultdict(list)
    observed_hold: dict[str, list[float]] = defaultdict(list)
    hold_overshoot: dict[str, list[float]] = defaultdict(list)
    ready_to_terminal: list[float] = []
    group_generation_span: list[float] = []
    assigned_counts = Counter()
    for group_stages in by_group.values():
        assignments = group_stages.get("release_delay_assigned", [])
        starts = group_stages.get("release_delay_started", [])
        completes = group_stages.get("release_delay_completed", [])
        siblings = group_stages.get("sibling_done", [])
        if assignments:
            if len(assignments) != 1:
                raise DoseContextError(f"{path}: duplicate group assignment")
            assigned_counts[str(assignments[0]["release_arm"])] += 1
        if siblings:
            sibling_times = [int(row["timestamp_ns"]) for row in siblings]
            group_generation_span.append(
                (max(sibling_times) - min(sibling_times)) / 1e9
            )
        if starts and completes and siblings:
            if len(starts) != 1 or len(completes) != 1:
                raise DoseContextError(f"{path}: duplicate hold lifecycle")
            arm = str(starts[0]["release_arm"])
            requested = float(starts[0]["release_delay_seconds"])
            start_ns = int(starts[0]["timestamp_ns"])
            complete_ns = int(completes[0]["timestamp_ns"])
            post_sibling_to_hold_start[arm].append(
                (start_ns - max(int(row["timestamp_ns"]) for row in siblings)) / 1e9
            )
            actual = (complete_ns - start_ns) / 1e9
            observed_hold[arm].append(actual)
            hold_overshoot[arm].append(actual - requested)
        ready = group_stages.get("group_ready", [])
        removed = group_stages.get("removed", [])
        if ready and removed:
            if len(ready) != 1 or len(removed) != 1:
                raise DoseContextError(f"{path}: duplicate ready/terminal lifecycle")
            ready_to_terminal.append(
                (int(removed[0]["timestamp_ns"]) - int(ready[0]["timestamp_ns"])) / 1e9
            )

    learner_times.sort()
    update_intervals = [
        (right - left) / 1e9 for left, right in zip(learner_times, learner_times[1:])
    ]
    median_update = _quantile(update_intervals, 0.50)
    return {
        "label": label,
        "path": str(path),
        "run_id": rows[0]["run_id"],
        "row_count": len(rows),
        "stage_counts": dict(sorted(stages.items())),
        "assignment_counts": dict(sorted(assigned_counts.items())),
        "generation_duration_seconds": _summary(generation_seconds),
        "environment_duration_seconds": _summary(environment_seconds),
        "within_group_sibling_completion_span_seconds": _summary(group_generation_span),
        "last_sibling_to_hold_start_seconds": {
            arm: _summary(values)
            for arm, values in sorted(post_sibling_to_hold_start.items())
        },
        "observed_hold_seconds": {
            arm: _summary(values) for arm, values in sorted(observed_hold.items())
        },
        "hold_overshoot_seconds": {
            arm: _summary(values) for arm, values in sorted(hold_overshoot.items())
        },
        "ready_to_terminal_seconds": _summary(ready_to_terminal),
        "learner_update_interval_seconds": _summary(update_intervals),
        "five_seconds_over_median_update": (
            5.0 / median_update
            if median_update is not None and median_update > 0
            else None
        ),
    }


def _parse_run(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("run must have LABEL=PATH form")
    return label, Path(path)


def main() -> None:
    """Run the lifecycle dose-context audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=_parse_run)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    labels = [label for label, _ in args.run]
    if len(set(labels)) != len(labels):
        raise DoseContextError("run labels must be unique")
    result = {
        "schema": "m4-downstream-quality-dose-context-v1",
        "status": "RETROSPECTIVE_DESCRIPTIVE_NOT_A_CAUSAL_REESTIMATION",
        "interpretation": {
            "five_second_ratio": "fixed treatment divided by median learner-update interval",
            "last_sibling_to_hold_start": "controller path timing, not production-delay prevalence",
            "prohibited": [
                "reclassify_registered_m4_endpoints",
                "claim_natural_delay_prevalence",
                "claim_final_quality_effect",
            ],
        },
        "runs": [_analyze(label, path) for label, path in args.run],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
