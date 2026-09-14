#!/usr/bin/env python3
"""Show empirically why consumed-rollout staleness does not recover M4 loss."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from tools.m4_llama3b_terminal_analysis import THREE_B_CELLS, rows as load_rows_file  # noqa: E402
from tools.m4_llama_v5_terminal_analysis import CELLS as ONE_B_CELLS  # noqa: E402
from tools.opportunity_ledger_join import LedgerJoinProtocol, ReleaseArm, join_opportunity_ledgers  # noqa: E402
from tools.opportunity_loss_analysis import bound_opportunity_loss  # noqa: E402
from tools.opportunity_loss_pipeline import _parse_protocol  # noqa: E402

from run_publication_synthesis import numina_join_protocol  # noqa: E402


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _corr(left: list[float], right: list[float]) -> float:
    a, b = _mean(left), _mean(right)
    return math.fsum((x-a)*(y-b) for x,y in zip(left,right,strict=True)) / math.sqrt(
        math.fsum((x-a)**2 for x in left) * math.fsum((y-b)**2 for y in right)
    )


def _summarize(label: str, joined, lifecycle: list[dict[str, Any]]) -> dict[str, Any]:
    stages: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    keep = {"group_ready", "removed"}
    for row in lifecycle:
        if row.get("stage") in keep:
            stages[row["group_id"]][row["stage"]] = row
    age: dict[str, list[float]] = defaultdict(list)
    latency: dict[str, list[float]] = defaultdict(list)
    for assignment in joined:
        group = stages.get(assignment.assignment_id, {})
        removed = group.get("removed")
        ready = group.get("group_ready")
        if removed is not None and removed.get("removal_reason") == "selected":
            age[assignment.arm].append(float(removed["end_weight_version"] - assignment.start_version))
            if ready is not None:
                latency[assignment.arm].append((removed["timestamp_ns"] - ready["timestamp_ns"]) / 1e9)
    raw = bound_opportunity_loss(row.analysis_assignment() for row in joined)
    if raw.complete_data_delta_l is None:
        raise RuntimeError(f"{label}: terminally incomplete common window")
    return {
        "label": label,
        "assignment_count": len(joined),
        "m4_unadjusted_estimate": raw.complete_data_delta_l,
        "selected_count": {arm: len(age[arm]) for arm in ("control", "d5")},
        "selected_mean_version_age": {arm: _mean(age[arm]) for arm in ("control", "d5")},
        "selected_version_age_contrast_d5_minus_control": _mean(age["d5"]) - _mean(age["control"]),
        "selected_mean_ready_latency_seconds": {arm: _mean(latency[arm]) for arm in ("control", "d5")},
        "selected_ready_latency_contrast_d5_minus_control": _mean(latency["d5"]) - _mean(latency["control"]),
        "positive_opportunity_lost_count": sum(row.opportunity > 0 and row.delivered is False for row in joined),
        "positive_opportunity_count": sum(row.opportunity > 0 for row in joined),
    }


def _llama_cells(prefix: str, cells, artifact_root: Path) -> list[dict[str, Any]]:
    result = []
    for cell, (_, _, domain, seed, _) in cells.items():
        lifecycle = load_rows_file(artifact_root / cell / "lifecycle.jsonl")
        opportunity = load_rows_file(artifact_root / cell / "opportunity.jsonl")
        joined = join_opportunity_ledgers(
            protocol=LedgerJoinProtocol(
                assignment_domain=domain,
                assignment_seed=seed,
                arms=(ReleaseArm("control", 0.0, 1), ReleaseArm("d5", 5.0, 1)),
                primary_start_version=8,
                primary_end_version=407,
                siblings_per_group=8,
                train_batch_size=32,
            ),
            lifecycle_rows=lifecycle,
            opportunity_rows=opportunity,
        )
        result.append(_summarize(f"{prefix}-{cell}", joined, lifecycle))
    return result


def _qwen_cells(*, common_window: bool) -> list[dict[str, Any]]:
    synthesis = json.loads((HERE / "six_cell_synthesis.json").read_bytes())
    robustness = json.loads((HERE / "robustness_results.json").read_bytes())
    numina_protocol_path = (
        ROOT
        / "reports/auto_research/2026-09-07-m4-opportunity-loss-numinamath-generalization/"
        "protocol_config.json"
    )
    numina_protocol = json.loads(numina_protocol_path.read_bytes())
    result = []
    for cell in sorted(synthesis["cell_results"]):
        evidence = synthesis["evidence"][cell]
        lifecycle_path = ROOT / evidence["lifecycle_path"]
        opportunity_path = ROOT / evidence["opportunity_path"]
        protocol_path = ROOT / evidence["protocol_path"]
        for path, expected in (
            (lifecycle_path, evidence["lifecycle_sha256"]),
            (opportunity_path, evidence["opportunity_sha256"]),
            (protocol_path, evidence["protocol_sha256"]),
        ):
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"{cell}: evidence hash mismatch for {path}")
        if cell.endswith("numinamath"):
            protocol = numina_join_protocol(numina_protocol, cell)
        else:
            _, protocol, _ = _parse_protocol(protocol_path.read_bytes())
            if common_window:
                protocol = dataclasses.replace(
                    protocol, primary_start_version=8, primary_end_version=407
                )
        lifecycle = load_rows_file(lifecycle_path)
        joined = join_opportunity_ledgers(
            protocol=protocol,
            lifecycle_rows=lifecycle,
            opportunity_rows=load_rows_file(opportunity_path),
        )
        count_key = (
            "common_window_assignment_count"
            if common_window
            else "full_window_assignment_count"
        )
        expected_count = robustness["cells"][cell][count_key]
        if len(joined) != expected_count:
            raise RuntimeError(f"{cell}: assignment count differs from frozen synthesis")
        result.append(_summarize(f"qwen-{cell}", joined, lifecycle))
    return result


def _correlation_summary(cells: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "correlation_m4_with_consumed_version_age_contrast": _corr(
            [row["m4_unadjusted_estimate"] for row in cells],
            [row["selected_version_age_contrast_d5_minus_control"] for row in cells],
        ),
        "correlation_m4_with_consumed_ready_latency_contrast": _corr(
            [row["m4_unadjusted_estimate"] for row in cells],
            [row["selected_ready_latency_contrast_d5_minus_control"] for row in cells],
        ),
    }


def main() -> None:
    session = ROOT / "session/20260909_m4_llama_lifecycle_derived_transport"
    qwen_cells = _qwen_cells(common_window=False)
    qwen_common_cells = _qwen_cells(common_window=True)
    llama_cells = _llama_cells(
        "llama1b", ONE_B_CELLS, session / "v5-terminal-artifacts/authenticated"
    )
    llama_cells.extend(
        _llama_cells(
            "llama3b", THREE_B_CELLS, session / "llama3b-terminal-artifacts/authenticated"
        )
    )
    cells = qwen_cells + llama_cells
    output = {
        "schema": "m4-metric-discriminant-v2",
        "status": "COMPLETE_RETROSPECTIVE_DESCRIPTIVE",
        "analysis_role": "metric_discriminant_not_new_causal_estimator",
        "raw_data_scope": "all_six_qwen_cells_and_eight_authenticated_llama_acquisitions",
        "cell_count": len(cells),
        "assignment_count": sum(row["assignment_count"] for row in cells),
        "cells": cells,
        "summary": {
            **_correlation_summary(cells),
            "positive_opportunity_lost_count": sum(row["positive_opportunity_lost_count"] for row in cells),
            "positive_opportunity_count": sum(row["positive_opportunity_count"] for row in cells),
            "consumed_staleness_conditions_on_selection": True,
            "m4_retains_pre_release_opportunity_for_unselected_groups": True,
        },
        "subgroup_summaries": {
            "qwen_six_cells": {
                "cell_count": len(qwen_cells),
                "assignment_count": sum(row["assignment_count"] for row in qwen_cells),
                **_correlation_summary(qwen_cells),
            },
            "qwen_six_cells_common_window": {
                "cell_count": len(qwen_common_cells),
                "assignment_count": sum(
                    row["assignment_count"] for row in qwen_common_cells
                ),
                **_correlation_summary(qwen_common_cells),
            },
            "llama_eight_acquisitions": {
                "cell_count": len(llama_cells),
                "assignment_count": sum(row["assignment_count"] for row in llama_cells),
                **_correlation_summary(llama_cells),
            },
        },
        "claim_boundary": "Correlations are descriptive. The identification distinction follows from observability: consumed-rollout staleness excludes candidates that never train.",
    }
    if output["assignment_count"] != 106653:
        raise RuntimeError(f"authenticated assignment total differs: {output['assignment_count']}")
    (HERE / "metric_discriminant.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: output[key] for key in ("status", "cell_count", "assignment_count")}, indent=2))
    print(json.dumps(output["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
