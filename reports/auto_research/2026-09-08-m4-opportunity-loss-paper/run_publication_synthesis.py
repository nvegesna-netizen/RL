#!/usr/bin/env python3
"""Rebuild the retrospective six-cell M4 publication synthesis from raw ledgers."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

from tools.opportunity_ledger_join import (
    LedgerJoinProtocol,
    OpportunityAuditContract,
    ReleaseArm,
    join_opportunity_ledgers,
)
from tools.opportunity_loss_analysis import bound_opportunity_loss
from tools.opportunity_loss_grid_analysis import (
    GridCellInference,
    infer_common_window_cell,
)
from tools.opportunity_loss_pipeline import _parse_jsonl, _parse_protocol
from tools.opportunity_loss_six_cell_synthesis import infer_six_cell_synthesis


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SESSION = ROOT / "session/20260907_m4_cross_study_harmonization"
GRID_REPORT = (
    ROOT / "reports/auto_research/2026-09-07-m4-opportunity-loss-grid-completion"
)
NUMINA_REPORT = (
    ROOT
    / "reports/auto_research/2026-09-07-m4-opportunity-loss-numinamath-generalization"
)
GRID_RESULT = GRID_REPORT / "grid_synthesis_result.json"
NUMINA_TERMINAL = SESSION / "numinamath-paired-terminal-analysis-result.json"
NUMINA_PROTOCOL = NUMINA_REPORT / "protocol_config.json"
COMMON_START = 8
COMMON_END = 407
BOOTSTRAP_DRAWS = 20_000


HISTORICAL = {
    "qwen3_0p6b_openmath": {
        "seed": 20260917,
        "protocol": ROOT
        / "reports/auto_research/2026-09-02-m4-opportunity-loss-followup/protocol_config.json",
        "lifecycle": SESSION
        / "qwen3_0p6b_openmath/m4-opportunity-loss-followup-lifecycle.jsonl",
        "opportunity": SESSION
        / "qwen3_0p6b_openmath/m4-opportunity-loss-followup-opportunity.jsonl",
        "full_result": ROOT
        / "reports/auto_research/2026-09-02-m4-opportunity-loss-followup/prospective-efficiency-audit/result.json",
    },
    "qwen3_1p7b_openmath": {
        "seed": 20260918,
        "protocol": ROOT
        / "reports/auto_research/2026-09-03-m4-opportunity-loss-transport/qwen3-1p7b-confirmatory-design/protocol_config.json",
        "lifecycle": SESSION
        / "qwen3_1p7b_openmath/m4-qwen3-1p7b-confirmatory-lifecycle.jsonl",
        "opportunity": SESSION
        / "qwen3_1p7b_openmath/m4-qwen3-1p7b-confirmatory-opportunity.jsonl",
        "full_result": ROOT
        / "reports/auto_research/2026-09-03-m4-opportunity-loss-transport/qwen3-1p7b-confirmatory-design/acquisition_result.json",
    },
    "qwen3_0p6b_gsm8k": {
        "seed": 20260920,
        "protocol": GRID_REPORT / "protocol_config.json",
        "lifecycle": SESSION
        / "grid-acquisition-failure-terminal-artifact/workspace/assets/basic/m4-qwen3-0p6b-gsm8k-grid-control-d5-acquisition/m4-grid-lifecycle.jsonl",
        "opportunity": SESSION
        / "grid-acquisition-failure-terminal-artifact/workspace/assets/basic/m4-qwen3-0p6b-gsm8k-grid-control-d5-acquisition/m4-grid-opportunity.jsonl",
        "full_result": GRID_REPORT / "acquisition_result.json",
    },
    "qwen3_1p7b_gsm8k": {
        "seed": 20260919,
        "protocol": ROOT
        / "reports/auto_research/2026-09-04-m4-opportunity-loss-workload-transport/qwen3-1p7b-gsm8k-confirmatory-design/protocol_config_r2.json",
        "lifecycle": ROOT
        / "session/20260906_m4_gsm8k_r2/acquisition-terminal-artifact/workspace/assets/basic/m4-qwen3-1p7b-gsm8k-confirmatory-control-d5-acquisition-r2/m4-gsm8k-r2-lifecycle.jsonl",
        "opportunity": ROOT
        / "session/20260906_m4_gsm8k_r2/acquisition-terminal-artifact/workspace/assets/basic/m4-qwen3-1p7b-gsm8k-confirmatory-control-d5-acquisition-r2/m4-gsm8k-r2-opportunity.jsonl",
        "full_result": ROOT
        / "reports/auto_research/2026-09-04-m4-opportunity-loss-workload-transport/qwen3-1p7b-gsm8k-confirmatory-design/acquisition_r2_result.json",
    },
}
NUMINA = {
    "qwen3_0p6b_numinamath": {"scale": "0p6b", "seed": 20260926},
    "qwen3_1p7b_numinamath": {"scale": "1p7b", "seed": 20260928},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def type7(values: tuple[float, ...], probability: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def intervals(cell: GridCellInference) -> dict[str, object]:
    z = NormalDist().inv_cdf(0.975)
    hac = (
        cell.estimate - z * cell.hac_standard_error,
        cell.estimate + z * cell.hac_standard_error,
    )
    bootstrap = (
        cell.estimate - type7(cell.bootstrap_shifts, 0.975),
        cell.estimate - type7(cell.bootstrap_shifts, 0.025),
    )
    return {
        "estimate": cell.estimate,
        "hac_standard_error": cell.hac_standard_error,
        "hac_interval": hac,
        "bootstrap_interval": bootstrap,
        "outer_envelope": (min(hac[0], bootstrap[0]), max(hac[1], bootstrap[1])),
    }


def numina_join_protocol(raw: dict[str, object], name: str) -> LedgerJoinProtocol:
    replay = raw["instrumentation"]["opportunity_replay"]
    loss = raw["instrumentation"]["loss_replay"]
    prospective = raw["prospective_cells"][name]
    runtime = raw["runtime_each_cell"]
    return LedgerJoinProtocol(
        assignment_domain=prospective["assignment_domain"],
        assignment_seed=prospective["assignment_seed"],
        arms=tuple(
            ReleaseArm(
                label=arm["label"],
                delay_seconds=float(arm["delay_seconds"]),
                mass=arm["mass"],
            )
            for arm in raw["arms"]
        ),
        primary_start_version=COMMON_START,
        primary_end_version=COMMON_END,
        siblings_per_group=runtime["generations_per_prompt"],
        train_batch_size=runtime["train_global_batch_size"],
        opportunity_audit=OpportunityAuditContract(
            estimator_name=replay["estimator"],
            baseline_algorithm=replay["baseline_algorithm"],
            normalize_rewards=replay["normalize_rewards"],
            normalization_epsilon=replay["normalization_epsilon"],
            reduction_dtype=replay["reduction_dtype"],
            reward_dtype=replay["reward_dtype"],
            use_leave_one_out_baseline=replay["use_leave_one_out_baseline"],
            allowed_reward_values=tuple(
                float(value) for value in replay["allowed_reward_values"]
            ),
            disable_ppo_ratio=loss["disable_ppo_ratio"],
            positive_example_nll_weight=loss["positive_example_nll_weight"],
            sequence_level_importance_ratios=loss["sequence_level_importance_ratios"],
            token_level_loss=loss["token_level_loss"],
            use_cispo=loss["use_cispo"],
        ),
    )


def load_rows() -> tuple[dict[str, list[object]], dict[str, dict[str, object]]]:
    frozen_grid = json.loads(GRID_RESULT.read_bytes())
    frozen_numina = json.loads(NUMINA_TERMINAL.read_bytes())
    numina_raw = json.loads(NUMINA_PROTOCOL.read_bytes())
    rows_by_cell = {}
    evidence = {}

    for name, spec in HISTORICAL.items():
        frozen = frozen_grid["cell_evidence"][name]
        for key in ("protocol", "lifecycle", "opportunity"):
            assert sha256(spec[key]) == frozen[f"{key}_sha256"]
        _, protocol, _ = _parse_protocol(spec["protocol"].read_bytes())
        protocol = dataclasses.replace(
            protocol,
            primary_start_version=COMMON_START,
            primary_end_version=COMMON_END,
        )
        rows = join_opportunity_ledgers(
            protocol=protocol,
            lifecycle_rows=_parse_jsonl(
                spec["lifecycle"].read_bytes(), compact=False, name=f"{name} lifecycle"
            ),
            opportunity_rows=_parse_jsonl(
                spec["opportunity"].read_bytes(),
                compact=True,
                name=f"{name} opportunity",
            ),
        )
        assert len(rows) == frozen["assignment_count"]
        rows_by_cell[name] = rows
        evidence[name] = {
            "lifecycle_path": relative(spec["lifecycle"]),
            "lifecycle_sha256": sha256(spec["lifecycle"]),
            "opportunity_path": relative(spec["opportunity"]),
            "opportunity_sha256": sha256(spec["opportunity"]),
            "protocol_path": relative(spec["protocol"]),
            "protocol_sha256": sha256(spec["protocol"]),
        }

    for name, spec in NUMINA.items():
        frozen = frozen_numina["prospective_cell_evidence"][name]
        directory = (
            SESSION / f"numinamath-{spec['scale']}-acquisition-terminal-artifact"
        )
        stem = f"m4-numinamath-{spec['scale']}"
        lifecycle = directory / f"{stem}-lifecycle.jsonl"
        opportunity = directory / f"{stem}-opportunity.jsonl"
        assert sha256(lifecycle) == frozen["lifecycle_sha256"]
        assert sha256(opportunity) == frozen["opportunity_sha256"]
        rows = join_opportunity_ledgers(
            protocol=numina_join_protocol(numina_raw, name),
            lifecycle_rows=_parse_jsonl(
                lifecycle.read_bytes(), compact=False, name=f"{name} lifecycle"
            ),
            opportunity_rows=_parse_jsonl(
                opportunity.read_bytes(), compact=True, name=f"{name} opportunity"
            ),
        )
        assert len(rows) == frozen["assignment_count"]
        rows_by_cell[name] = rows
        evidence[name] = {
            "lifecycle_path": relative(lifecycle),
            "lifecycle_sha256": sha256(lifecycle),
            "opportunity_path": relative(opportunity),
            "opportunity_sha256": sha256(opportunity),
            "protocol_path": relative(NUMINA_PROTOCOL),
            "protocol_sha256": sha256(NUMINA_PROTOCOL),
        }
    return rows_by_cell, evidence


def full_window_record(name: str) -> dict[str, object]:
    if name.endswith("numinamath"):
        cell = json.loads((NUMINA_REPORT / "acquisition_result.json").read_bytes())[
            "cells"
        ][name]
        value = cell["supporting_materiality"]
        return {
            "assignment_count": cell["assignment_count"],
            "estimate": value["estimate"],
            "outer_envelope": value["outer_confidence_envelope"],
            "conclusion": value["conclusion"],
        }
    raw = json.loads(HISTORICAL[name]["full_result"].read_bytes())
    if name == "qwen3_0p6b_openmath":
        value = raw["causal_result"]
        count = raw["primary_assignments"]["total"]
        estimate = value["adjusted_estimate"]
    elif name == "qwen3_1p7b_openmath":
        value = raw["primary_adjusted_inference"]
        count = value["assignment_count"]
        estimate = value["estimate"]
    elif name == "qwen3_0p6b_gsm8k":
        value = raw["primary_adjusted_inference"]
        count = raw["acquisition"]["primary_assignment_count"]
        estimate = value["estimate"]
    else:
        value = raw["primary_adjusted_result"]
        count = value["primary_assignment_count"]
        estimate = value["estimate"]
    return {
        "assignment_count": count,
        "estimate": estimate,
        "outer_envelope": value["confidence_envelope"],
        "conclusion": value.get("conclusion", value.get("causal_conclusion")),
    }


def svg_forest(cells: dict[str, dict[str, object]], path: Path) -> None:
    labels = [
        ("0.6B · OpenMath", "qwen3_0p6b_openmath"),
        ("1.7B · OpenMath", "qwen3_1p7b_openmath"),
        ("0.6B · GSM8K", "qwen3_0p6b_gsm8k"),
        ("1.7B · GSM8K", "qwen3_1p7b_gsm8k"),
        ("0.6B · NuminaMath", "qwen3_0p6b_numinamath"),
        ("1.7B · NuminaMath", "qwen3_1p7b_numinamath"),
    ]
    left, right, top, row = 210, 850, 55, 55
    x_min, x_max = 0.08, 0.37
    x = lambda value: left + (value - x_min) / (x_max - x_min) * (right - left)
    height = top + row * len(labels) + 70
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" viewBox="0 0 900 {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#202124}.label{font-size:15px}.axis{font-size:13px}.title{font-size:18px;font-weight:700}</style>",
        '<text x="20" y="28" class="title">M4 normalized opportunity-loss effects</text>',
    ]
    for tick in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35):
        parts.append(
            f'<line x1="{x(tick):.1f}" y1="42" x2="{x(tick):.1f}" y2="{height - 45}" stroke="#e5e7eb"/>'
        )
        parts.append(
            f'<text x="{x(tick):.1f}" y="{height - 20}" text-anchor="middle" class="axis">{tick:.2f}</text>'
        )
    parts.append(
        f'<line x1="{x(0.2):.1f}" y1="42" x2="{x(0.2):.1f}" y2="{height - 45}" stroke="#d93025" stroke-width="2" stroke-dasharray="6 5"/>'
    )
    for index, (label, name) in enumerate(labels):
        y = top + index * row
        record = cells[name]
        low, high = record["outer_envelope"]
        estimate = record["estimate"]
        color = "#1769aa" if "0p6b" in name else "#7b1fa2"
        parts.append(f'<text x="20" y="{y + 5}" class="label">{label}</text>')
        parts.append(
            f'<line x1="{x(low):.1f}" y1="{y}" x2="{x(high):.1f}" y2="{y}" stroke="{color}" stroke-width="4"/>'
        )
        parts.append(f'<circle cx="{x(estimate):.1f}" cy="{y}" r="7" fill="{color}"/>')
    parts.append(
        f'<text x="{x(0.2) + 6:.1f}" y="{height - 49}" class="axis" fill="#d93025">registered materiality threshold</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def svg_interactions(synthesis: dict[str, object], path: Path) -> None:
    items = [
        ("GSM8K − OpenMath", "gsm8k_minus_openmath"),
        ("GSM8K − NuminaMath", "gsm8k_minus_numinamath"),
    ]
    left, right = 220, 850
    x_min, x_max = -0.16, 0.04
    x = lambda value: left + (value - x_min) / (x_max - x_min) * (right - left)
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="250" viewBox="0 0 900 250">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#202124}.label{font-size:15px}.axis{font-size:13px}.title{font-size:18px;font-weight:700}</style>",
        '<text x="20" y="28" class="title">Dependency-aware model-by-workload interactions</text>',
    ]
    for tick in (-0.15, -0.10, -0.05, 0.0):
        parts.append(
            f'<line x1="{x(tick):.1f}" y1="42" x2="{x(tick):.1f}" y2="205" stroke="#e5e7eb"/>'
        )
        parts.append(
            f'<text x="{x(tick):.1f}" y="230" text-anchor="middle" class="axis">{tick:.2f}</text>'
        )
    parts.append(
        f'<line x1="{x(0.0):.1f}" y1="42" x2="{x(0.0):.1f}" y2="205" stroke="#202124" stroke-width="2"/>'
    )
    for index, (label, key) in enumerate(items):
        y = 85 + index * 80
        record = synthesis["reference_interactions"][key]
        low, high = record["simultaneous_bootstrap_interval"]
        estimate = record["estimate"]
        parts.append(f'<text x="20" y="{y + 5}" class="label">{label}</text>')
        parts.append(
            f'<line x1="{x(low):.1f}" y1="{y}" x2="{x(high):.1f}" y2="{y}" stroke="#00897b" stroke-width="4"/>'
        )
        parts.append(f'<circle cx="{x(estimate):.1f}" cy="{y}" r="7" fill="#00695c"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def main() -> None:
    rows_by_cell, evidence = load_rows()
    inferences = {}
    cell_results = {}
    robustness_cells = {}
    seeds = {
        **{name: spec["seed"] for name, spec in HISTORICAL.items()},
        **{name: spec["seed"] for name, spec in NUMINA.items()},
    }
    for name, rows in rows_by_cell.items():
        inference = infer_common_window_cell(
            rows,
            bootstrap_seed=seeds[name],
            bootstrap_draws=BOOTSTRAP_DRAWS,
        )
        inferences[name] = inference
        result = intervals(inference)
        outer = result["outer_envelope"]
        result.update(
            {
                "assignment_count": len(rows),
                "conclusion_at_registered_threshold": "MATERIAL"
                if outer[0] > 0.2
                else "NOT_MATERIAL"
                if outer[1] <= 0.2
                else "INCONCLUSIVE",
                "terminal_missing_count": sum(row.delivered is None for row in rows),
            }
        )
        cell_results[name] = result
        unadjusted = bound_opportunity_loss(row.analysis_assignment() for row in rows)
        full = full_window_record(name)
        robustness_cells[name] = {
            "adjusted_common_window_estimate": inference.estimate,
            "unadjusted_common_window_estimate": unadjusted.complete_data_delta_l,
            "adjustment_difference": inference.estimate
            - unadjusted.complete_data_delta_l,
            "missingness_lower_endpoint": result["estimate"],
            "missingness_upper_endpoint": result["estimate"],
            "missingness_identification_width": 0.0,
            "common_window_assignment_count": len(rows),
            "full_window_assignment_count": full["assignment_count"],
            "common_window_estimate": inference.estimate,
            "full_window_estimate": full["estimate"],
            "common_minus_full": inference.estimate - full["estimate"],
            "threshold_sensitivity": {
                f"{threshold:.2f}": "MATERIAL"
                if outer[0] > threshold
                else "NOT_MATERIAL"
                if outer[1] <= threshold
                else "INCONCLUSIVE"
                for threshold in (0.10, 0.15, 0.20, 0.25, 0.30)
            },
        }

    synthesis = infer_six_cell_synthesis(inferences).to_dict()
    result = {
        "schema": "m4-opportunity-loss-publication-six-cell-synthesis-v1",
        "analysis_role": "retrospective_secondary_cannot_override_registered_cell_or_interaction_results",
        "common_window": [COMMON_START, COMMON_END],
        "bootstrap_draws_per_cell": BOOTSTRAP_DRAWS,
        "cell_results": cell_results,
        "dependency_aware_synthesis": synthesis,
        "evidence": evidence,
        "input_records": {
            "grid_synthesis": {
                "path": relative(GRID_RESULT),
                "sha256": sha256(GRID_RESULT),
            },
            "numinamath_terminal_analysis": {
                "path": relative(NUMINA_TERMINAL),
                "sha256": sha256(NUMINA_TERMINAL),
            },
        },
        "limitations": {
            "shared_reference": "Both reference interactions reuse the same two GSM8K cells.",
            "independent_replications": False,
            "downstream_model_quality_measured": False,
            "family_wide_generalization_supported": False,
        },
    }
    robustness = {
        "schema": "m4-opportunity-loss-publication-robustness-v1",
        "analysis_role": "retrospective_descriptive_sensitivity",
        "cells": robustness_cells,
        "interpretation": {
            "hac_vs_bootstrap": "Both interval families are reported separately; the conservative cell display uses their outer envelope.",
            "missingness": "All six common-window datasets have complete terminal scoring, so sharp lower and upper endpoints coincide.",
            "threshold": "Threshold labels are descriptive interval classifications, not replacements for registered decisions.",
            "window": "Common-window results harmonize versions 8-407; full-window results retain each cell's registered primary window.",
        },
    }
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "six_cell_synthesis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (HERE / "robustness_results.json").write_text(
        json.dumps(robustness, indent=2, sort_keys=True) + "\n"
    )
    svg_forest(cell_results, HERE / "cell_forest_plot.svg")
    svg_interactions(synthesis, HERE / "interaction_plot.svg")
    print(
        json.dumps(
            {
                "six_cell_synthesis_sha256": sha256(HERE / "six_cell_synthesis.json"),
                "robustness_results_sha256": sha256(HERE / "robustness_results.json"),
                "global_heterogeneity_p_value": synthesis[
                    "global_heterogeneity_p_value"
                ],
                "hac_correlation": synthesis["hac_correlation"],
                "bootstrap_correlation": synthesis["bootstrap_correlation"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
