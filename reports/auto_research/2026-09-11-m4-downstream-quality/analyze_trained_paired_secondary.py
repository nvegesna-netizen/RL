#!/usr/bin/env python3
"""Reconstruct the frozen downstream-quality secondary analyses offline."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.opportunity_ledger_join import (  # noqa: E402
    LedgerJoinProtocol,
    ReleaseArm,
    join_opportunity_ledgers,
)
from tools.opportunity_loss_analysis import bound_opportunity_loss  # noqa: E402


HERE = Path(__file__).resolve().parent
SESSION = ROOT / "session/20260909_m4_llama_lifecycle_derived_transport"
ARCHIVES = SESSION / "downstream-quality-terminal-artifacts"
AUTH = HERE / "trained_paired_terminal_authentication.json"
PRIMARY = HERE / "trained_paired_analysis_result.json"
RUN_MANIFEST = HERE / "trained_paired_run_manifest.json"
COMMON_START = 8
COMMON_END = 407
T_975 = {15: 2.131449545559323, 14: 2.1447866879169273}


def _member(archive: zipfile.ZipFile, suffix: str) -> bytes:
    names = [name for name in archive.namelist() if name.endswith("/" + suffix)]
    if len(names) != 1:
        raise RuntimeError(f"expected one archive member ending in {suffix}: {names}")
    return archive.read(names[0])


def _jsonl(raw: bytes) -> list[dict[str, Any]]:
    if not raw.endswith(b"\n"):
        raise RuntimeError("authenticated JSONL lacks terminal newline")
    return [json.loads(line) for line in raw.splitlines()]


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _paired_summary(values: list[float]) -> dict[str, Any]:
    if len(values) != 16:
        raise RuntimeError("paired secondary estimator requires all 16 blocks")
    mean = _mean(values)
    sd = statistics.stdev(values)
    se = sd / math.sqrt(len(values))
    half = T_975[15] * se
    return {
        "block_count": len(values),
        "estimate_mixed_d5_minus_immediate": mean,
        "paired_sample_standard_deviation": sd,
        "standard_error": se,
        "student_interval_95": [mean - half, mean + half],
    }


def _correlation(left: list[float], right: list[float]) -> float:
    lx, ly = _mean(left), _mean(right)
    numerator = math.fsum((x - lx) * (y - ly) for x, y in zip(left, right, strict=True))
    denominator = math.sqrt(
        math.fsum((x - lx) ** 2 for x in left)
        * math.fsum((y - ly) ** 2 for y in right)
    )
    return numerator / denominator


def _regression(x: list[float], y: list[float]) -> dict[str, Any]:
    if len(x) != 16 or len(y) != 16:
        raise RuntimeError("mediation diagnostic requires all 16 blocks")
    xbar, ybar = _mean(x), _mean(y)
    sxx = math.fsum((value - xbar) ** 2 for value in x)
    slope = math.fsum((a - xbar) * (b - ybar) for a, b in zip(x, y, strict=True)) / sxx
    intercept = ybar - slope * xbar
    residuals = [b - intercept - slope * a for a, b in zip(x, y, strict=True)]
    residual_sd = math.sqrt(math.fsum(value * value for value in residuals) / 14)
    slope_se = residual_sd / math.sqrt(sxx)
    half = T_975[14] * slope_se
    return {
        "block_count": 16,
        "pearson_correlation": _correlation(x, y),
        "intercept": intercept,
        "slope": slope,
        "slope_standard_error": slope_se,
        "slope_interval_95": [slope - half, slope + half],
        "role": "descriptive_mediation_diagnostic_not_causal_mediator_estimate",
    }


def _run_metrics(
    run: dict[str, Any], auth: dict[str, Any], primary_run: dict[str, Any]
) -> dict[str, Any]:
    identity = f"{run['block_id']}_{run['regime']}"
    archive_path = next((ARCHIVES / identity).glob("workload-job-*.zip"))
    with zipfile.ZipFile(archive_path) as archive:
        lifecycle_raw = _member(archive, "lifecycle.jsonl")
        opportunity_raw = _member(archive, "opportunity.jsonl")
        duty_raw = _member(archive, "observer-duty.json")
    expected = auth["files"]
    for name, raw in (
        ("lifecycle.jsonl", lifecycle_raw),
        ("opportunity.jsonl", opportunity_raw),
        ("observer-duty.json", duty_raw),
    ):
        if hashlib.sha256(raw).hexdigest() != expected[name]["sha256"]:
            raise RuntimeError(f"{identity}: {name} hash differs from terminal authentication")
    lifecycle = _jsonl(lifecycle_raw)
    opportunity = _jsonl(opportunity_raw)
    duty = json.loads(duty_raw)
    arms = (
        (ReleaseArm("control", 0.0, 1),)
        if run["regime"] == "immediate"
        else (ReleaseArm("control", 0.0, 1), ReleaseArm("d5", 5.0, 1))
    )
    joined = join_opportunity_ledgers(
        protocol=LedgerJoinProtocol(
            assignment_domain=run["assignment_domain"],
            assignment_seed=run["assignment_seed"],
            arms=arms,
            primary_start_version=COMMON_START,
            primary_end_version=COMMON_END,
            siblings_per_group=8,
            train_batch_size=32,
        ),
        lifecycle_rows=lifecycle,
        opportunity_rows=opportunity,
    )
    by_group: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    unique_group_stages = {
        "release_delay_started", "release_delay_completed", "group_completed",
        "group_ready", "removed",
    }
    for row in lifecycle:
        if row["group_id"] != "__learner__" and row["stage"] in unique_group_stages:
            if row["stage"] in by_group[row["group_id"]]:
                raise RuntimeError(f"{identity}: duplicate {row['stage']}")
            by_group[row["group_id"]][row["stage"]] = row

    version_advances: list[float] = []
    direct_chains: list[float] = []
    selected_version_age: list[float] = []
    ready_latency: list[float] = []
    for assignment in joined:
        stages = by_group[assignment.assignment_id]
        started, completed = stages.get("release_delay_started"), stages.get("release_delay_completed")
        if started is None or completed is None:
            raise RuntimeError(f"{identity}: common-window assignment lacks completed hold")
        advance = completed["learner_weight_version"] - started["learner_weight_version"]
        version_advances.append(float(advance))
        chain = False
        if all(name in stages for name in ("group_completed", "group_ready", "removed")):
            sequence = [
                completed["controller_sequence"],
                stages["group_completed"]["controller_sequence"],
                stages["group_ready"]["controller_sequence"],
                stages["removed"]["controller_sequence"],
            ]
            chain = bool(
                started["learner_weight_version"] <= assignment.start_version + 1
                < completed["learner_weight_version"]
                and sequence == sorted(sequence)
                and len(set(sequence)) == 4
                and stages["removed"]["removal_reason"] == "stale_evicted"
            )
            ready_latency.append(
                (stages["removed"]["timestamp_ns"] - stages["group_ready"]["timestamp_ns"])
                / 1e9
            )
            if stages["removed"]["removal_reason"] == "selected":
                selected_version_age.append(
                    float(stages["removed"]["end_weight_version"] - assignment.start_version)
                )
        direct_chains.append(float(chain))

    total_q = math.fsum(row.opportunity for row in joined)
    lost_q = math.fsum(row.opportunity for row in joined if row.delivered is False)
    positive_q_lost = sum(row.opportunity > 0 and row.delivered is False for row in joined)
    positive_q = sum(row.opportunity > 0 for row in joined)
    group_rows = [row for row in opportunity if row.get("event_type") == "group"]
    all_tokens = [int(sibling["valid_actor_tokens"]) for row in group_rows for sibling in row["siblings"]]
    wall_seconds = int(primary_run["systems"]["wall_seconds"])
    within_mixed = None
    if run["regime"] == "mixed_d5":
        within_mixed = bound_opportunity_loss(
            (row.analysis_assignment() for row in joined),
            material_threshold=0.2,
            max_missing_fraction=0.01,
        ).to_dict()
    return {
        "identity": identity,
        "block_id": run["block_id"],
        "regime": run["regime"],
        "common_window": [COMMON_START, COMMON_END],
        "assignment_count": len(joined),
        "terminal_missing_count": sum(row.delivered is None for row in joined),
        "delivered_count": sum(row.delivered is True for row in joined),
        "lost_count": sum(row.delivered is False for row in joined),
        "positive_opportunity_assignment_count": positive_q,
        "positive_opportunity_lost_count": positive_q_lost,
        "normalized_realized_opportunity_loss": lost_q / total_q,
        "mean_version_advance_during_release": _mean(version_advances),
        "direct_chain_rate": _mean(direct_chains),
        "mean_selected_version_age": _mean(selected_version_age),
        "mean_ready_to_terminal_seconds": _mean(ready_latency),
        "generated_valid_actor_tokens": sum(all_tokens),
        "mean_response_tokens": _mean([float(value) for value in all_tokens]),
        "learner_updates_per_second": 448 / wall_seconds,
        "generated_tokens_per_second": sum(all_tokens) / wall_seconds,
        "wall_seconds": wall_seconds,
        "terminal_accuracy": primary_run["accuracy"],
        "zero_accuracy_indicator": int(primary_run["accuracy"] == 0.0),
        "observer_duty": duty["corrected_observer_duty"],
        "within_mixed_d5_minus_control_opportunity_loss": within_mixed,
        "workload_job_id": auth["canonical_job_id"],
        "authenticated_member_sha256": {
            name: expected[name]["sha256"]
            for name in ("lifecycle.jsonl", "opportunity.jsonl", "observer-duty.json")
        },
    }


def _scatter_svg(points: list[dict[str, float]], path: Path) -> None:
    width, height, left, right, top, bottom = 760, 500, 90, 735, 35, 430
    xs = [row["opportunity_loss_difference"] for row in points]
    ys = [row["accuracy_difference"] for row in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    padx, pady = (xmax - xmin) * 0.08, (ymax - ymin) * 0.08
    xmin, xmax, ymin, ymax = xmin - padx, xmax + padx, ymin - pady, ymax + pady
    sx = lambda x: left + (x - xmin) / (xmax - xmin) * (right - left)
    sy = lambda y: bottom - (y - ymin) / (ymax - ymin) * (bottom - top)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Helvetica,Arial,sans-serif;fill:#202124}.axis{font-size:13px}.point{font-size:11px}</style>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#202124"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#202124"/>',
        f'<line x1="{sx(0):.1f}" y1="{top}" x2="{sx(0):.1f}" y2="{bottom}" stroke="#bbb" stroke-dasharray="5 4"/>',
        f'<line x1="{left}" y1="{sy(0):.1f}" x2="{right}" y2="{sy(0):.1f}" stroke="#bbb" stroke-dasharray="5 4"/>',
    ]
    for row in points:
        x, y = sx(row["opportunity_loss_difference"]), sy(row["accuracy_difference"])
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#1769aa"/>')
        parts.append(f'<text x="{x + 7:.1f}" y="{y - 5:.1f}" class="point">{row["block_id"]}</text>')
    parts.extend([
        f'<text x="{(left + right) / 2:.1f}" y="480" text-anchor="middle" class="axis">mixed-d5 minus immediate normalized realized opportunity loss</text>',
        f'<text transform="translate(20 {(top + bottom) / 2:.1f}) rotate(-90)" text-anchor="middle" class="axis">mixed-d5 minus immediate terminal accuracy</text>',
        '</svg>',
    ])
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    authentication = json.loads(AUTH.read_bytes())
    primary = json.loads(PRIMARY.read_bytes())
    manifest = json.loads(RUN_MANIFEST.read_bytes())
    if authentication["status"] != "PASS" or not authentication["all_artifacts_authenticated"]:
        raise RuntimeError("terminal authentication is not complete")
    auth_by_id = {row["identity"]: row for row in authentication["runs"]}
    primary_by_id = {
        f"{row['block_id']}_{row['regime']}": row for row in primary["runs"]
    }
    runs = [
        _run_metrics(run, auth_by_id[f"{run['block_id']}_{run['regime']}"], primary_by_id[f"{run['block_id']}_{run['regime']}"])
        for run in manifest["runs"]
    ]
    by_block = defaultdict(dict)
    for row in runs:
        by_block[row["block_id"]][row["regime"]] = row
    metric_names = [
        "normalized_realized_opportunity_loss",
        "direct_chain_rate",
        "mean_version_advance_during_release",
        "learner_updates_per_second",
        "generated_tokens_per_second",
        "wall_seconds",
        "generated_valid_actor_tokens",
        "mean_response_tokens",
        "zero_accuracy_indicator",
        "mean_selected_version_age",
        "mean_ready_to_terminal_seconds",
    ]
    contrasts = {}
    differences = {}
    for name in metric_names:
        values = [
            float(by_block[block]["mixed_d5"][name])
            - float(by_block[block]["immediate"][name])
            for block in sorted(by_block)
        ]
        differences[name] = values
        contrasts[name] = _paired_summary(values)
    accuracy = primary["block_differences_mixed_minus_immediate"]
    diagnostic = _regression(differences["normalized_realized_opportunity_loss"], accuracy)
    points = [
        {
            "block_id": block,
            "opportunity_loss_difference": differences["normalized_realized_opportunity_loss"][index],
            "accuracy_difference": accuracy[index],
        }
        for index, block in enumerate(sorted(by_block))
    ]
    selected_age = [row["mean_selected_version_age"] for row in runs]
    policy_loss = [row["normalized_realized_opportunity_loss"] for row in runs]
    ready_latency = [row["mean_ready_to_terminal_seconds"] for row in runs]
    result = {
        "schema": "m4-downstream-quality-trained-paired-secondary-analysis-v1",
        "status": "COMPLETE_OFFLINE_SECONDARY",
        "analysis_role": "prespecified_secondary_cannot_override_primary_accuracy_result",
        "common_window": [COMMON_START, COMMON_END],
        "definitions": {
            "normalized_realized_opportunity_loss": "sum(Q for undelivered assignments) / sum(Q for all assignments) within each run and common window",
            "paired_release_policy_contrast": "equal_weight mean over the 16 matched blocks of mixed_d5 minus immediate",
            "direct_chain": "delay crosses the maximum-staleness boundary and is followed in controller order by completion, ready, and stale eviction",
            "generated_token_metrics": "all opportunity-ledger group siblings over the complete 448-update run",
            "collapse_incidence": "indicator that the registered 1024-prompt terminal accuracy equals zero",
        },
        "runs": runs,
        "paired_release_policy_contrasts": contrasts,
        "opportunity_loss_accuracy_diagnostic": diagnostic,
        "scatter_points": points,
        "metric_discriminant": {
            "run_count": 32,
            "correlation_policy_loss_with_consumed_rollout_version_age": _correlation(policy_loss, selected_age),
            "correlation_policy_loss_with_ready_to_terminal_latency": _correlation(policy_loss, ready_latency),
            "positive_opportunity_lost_assignments": sum(row["positive_opportunity_lost_count"] for row in runs),
            "positive_opportunity_assignments": sum(row["positive_opportunity_assignment_count"] for row in runs),
            "consumed_rollout_staleness_scope": "selected_groups_only",
            "m4_scope": "all_common_window_assignments_with_pre_release_opportunity",
            "interpretation": "descriptive_discriminant_not_an_additional_causal_estimand",
        },
        "unsupported_secondary_endpoints": {
            "equal_wall_clock_quality": "not_identified_only_terminal_equal_update_evaluation_was_frozen_and_preserved",
            "time_to_fixed_quality": "not_identified_no_in_loop_validation_or_preselected_intermediate_evaluations",
        },
        "primary_result_modified": False,
        "training_rerun": False,
    }
    output = HERE / "trained_paired_secondary_analysis.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _scatter_svg(points, HERE / "trained_paired_opportunity_accuracy_scatter.svg")
    print(json.dumps({
        "status": result["status"],
        "runs": len(runs),
        "opportunity_loss_contrast": contrasts["normalized_realized_opportunity_loss"],
        "diagnostic": diagnostic,
        "metric_discriminant": result["metric_discriminant"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
