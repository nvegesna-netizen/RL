#!/usr/bin/env python3
"""Evaluate the frozen baseline-budgeted OARS shadow-policy repair."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


G_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(G_HERE))

from run_offline_gate import (  # noqa: E402
    CellData,
    GroupRecord,
    OfflineGateError,
    load_json,
    load_llama_cells,
    load_qwen_cells,
    policy_order,
    selected_group_ids,
    sha256,
    type7,
)


def combination_score(
    selected: Sequence[GroupRecord],
    *,
    current_version: int,
) -> tuple[float, float, int]:
    """Return the frozen opportunity-first lexicographic objective."""
    imminent_l1 = math.fsum(
        group.metrics.registered_l1
        for group in selected
        if group.start_version + 1 <= current_version
    )
    total_l1 = math.fsum(group.metrics.registered_l1 for group in selected)
    total_tokens = sum(group.metrics.valid_actor_tokens for group in selected)
    return imminent_l1, total_l1, -total_tokens


def baseline_budgeted_oars(
    candidates: Sequence[GroupRecord],
    *,
    current_version: int,
    groups_per_step: int = 4,
    budget_multiplier: float = 1.02,
) -> tuple[tuple[GroupRecord, ...], Mapping[str, Any]]:
    """Choose the exact best opportunity batch within the baseline token budget."""
    if len(candidates) < groups_per_step:
        raise OfflineGateError("candidate set is smaller than one learner batch")
    if not math.isfinite(budget_multiplier) or budget_multiplier < 1.0:
        raise OfflineGateError("budget multiplier must be finite and at least one")
    baseline = tuple(
        policy_order(
            candidates,
            policy="weight_fifo",
            current_version=current_version,
        )[:groups_per_step]
    )
    baseline_tokens = sum(group.metrics.valid_actor_tokens for group in baseline)
    token_budget = math.floor(baseline_tokens * budget_multiplier + 1e-12)
    ordered_candidates = sorted(candidates, key=lambda group: group.group_id)
    best: tuple[GroupRecord, ...] | None = None
    best_score: tuple[float, float, int] | None = None
    feasible_count = 0
    combination_count = 0
    for combination in itertools.combinations(ordered_candidates, groups_per_step):
        combination_count += 1
        tokens = sum(group.metrics.valid_actor_tokens for group in combination)
        if tokens > token_budget:
            continue
        feasible_count += 1
        score = combination_score(combination, current_version=current_version)
        # Candidate iteration is lexicographic by group identity. Strict
        # improvement preserves the first identity tuple on an exact tie.
        if best_score is None or score > best_score:
            best = combination
            best_score = score
    if feasible_count == 0 or best is None:
        raise OfflineGateError("baseline batch was unexpectedly infeasible")
    selected_tokens = sum(group.metrics.valid_actor_tokens for group in best)
    if len(best) != groups_per_step or selected_tokens > token_budget:
        raise OfflineGateError("budgeted OARS violated its feasibility contract")
    return tuple(best), {
        "baseline_group_ids": [group.group_id for group in baseline],
        "baseline_tokens": baseline_tokens,
        "token_budget": token_budget,
        "selected_tokens": selected_tokens,
        "combination_count": combination_count,
        "feasible_combination_count": feasible_count,
    }


def sample_owner(cell: CellData) -> Mapping[str, str]:
    """Return the validated sample-to-group identity map."""
    result: dict[str, str] = {}
    for row in cell.opportunity_rows:
        if row.get("event_type") != "group":
            continue
        group_id = str(row["group_id"])
        for raw_sample_id in row["sample_ids"]:
            sample_id = str(raw_sample_id)
            if sample_id in result:
                raise OfflineGateError(f"{cell.label}: duplicate sample identity")
            result[sample_id] = group_id
    return result


def repaired_shadow_summary(
    cell: CellData,
    *,
    max_staleness: int = 1,
) -> Mapping[str, Any]:
    """Evaluate repaired choices on the parent's reconstructed decision sets."""
    owner = sample_owner(cell)
    steps = sorted(
        (
            row
            for row in cell.opportunity_rows
            if row.get("event_type") == "train_step_completed"
        ),
        key=lambda row: int(row["learner_version"]),
    )
    decision_count = 0
    baseline_l1 = 0.0
    selected_l1 = 0.0
    baseline_l2 = 0.0
    selected_l2 = 0.0
    baseline_tokens = 0
    selected_tokens = 0
    overlap_sum = 0.0
    combinations_evaluated = 0
    feasible_combinations = 0
    candidate_counts = []
    maximum_decision_token_ratio = 0.0
    for step in steps:
        actual_ids = selected_group_ids(step, owner)
        if len(actual_ids) != 4 or any(group_id not in cell.groups for group_id in actual_ids):
            continue
        actual = [cell.groups[group_id] for group_id in actual_ids]
        decision_timestamp = min(group.removed_timestamp_ns for group in actual)
        current_version = int(step["previous_learner_version"])
        minimum_version = max(0, current_version - max_staleness)
        candidates = [
            group
            for group in cell.groups.values()
            if group.ready_timestamp_ns <= decision_timestamp
            and group.removed_timestamp_ns >= decision_timestamp
            and minimum_version <= group.start_version <= current_version
        ]
        if len(candidates) <= 4:
            continue
        baseline = tuple(
            policy_order(
                candidates,
                policy="weight_fifo",
                current_version=current_version,
            )[:4]
        )
        if tuple(group.group_id for group in baseline) != actual_ids:
            raise OfflineGateError(
                f"{cell.label}: reconstructed weight-FIFO differs from actual selection"
            )
        selected, audit = baseline_budgeted_oars(
            candidates,
            current_version=current_version,
        )
        decision_count += 1
        candidate_counts.append(len(candidates))
        combinations_evaluated += audit["combination_count"]
        feasible_combinations += audit["feasible_combination_count"]
        baseline_l1 += math.fsum(group.metrics.registered_l1 for group in baseline)
        selected_l1 += math.fsum(group.metrics.registered_l1 for group in selected)
        baseline_l2 += math.fsum(group.metrics.l2 for group in baseline)
        selected_l2 += math.fsum(group.metrics.l2 for group in selected)
        decision_baseline_tokens = int(audit["baseline_tokens"])
        decision_selected_tokens = int(audit["selected_tokens"])
        baseline_tokens += decision_baseline_tokens
        selected_tokens += decision_selected_tokens
        maximum_decision_token_ratio = max(
            maximum_decision_token_ratio,
            decision_selected_tokens / decision_baseline_tokens,
        )
        actual_set = set(actual_ids)
        overlap_sum += sum(group.group_id in actual_set for group in selected) / 4
    if decision_count == 0 or baseline_l1 <= 0.0 or baseline_tokens <= 0:
        raise OfflineGateError(f"{cell.label}: no informative contended decisions")
    return {
        "decision_count": decision_count,
        "candidate_group_count": {
            "minimum": min(candidate_counts),
            "median": type7(candidate_counts, 0.5),
            "maximum": max(candidate_counts),
        },
        "combinations_evaluated": combinations_evaluated,
        "feasible_combinations": feasible_combinations,
        "baseline_selected_l1_mass": baseline_l1,
        "repaired_selected_l1_mass": selected_l1,
        "l1_gain_over_baseline_fraction": selected_l1 / baseline_l1 - 1.0,
        "baseline_selected_l2_mass": baseline_l2,
        "repaired_selected_l2_mass": selected_l2,
        "l2_gain_over_baseline_fraction": selected_l2 / baseline_l2 - 1.0,
        "baseline_selected_valid_actor_tokens": baseline_tokens,
        "repaired_selected_valid_actor_tokens": selected_tokens,
        "aggregate_selected_token_ratio_to_baseline": selected_tokens
        / baseline_tokens,
        "maximum_decision_token_ratio_to_baseline": maximum_decision_token_ratio,
        "mean_overlap_with_actual": overlap_sum / decision_count,
        "baseline_feasible_every_decision": True,
        "four_groups_selected_every_decision": True,
        "priority_uses_only_predecision_fields": True,
    }


def evaluate_repaired_gate(
    *,
    parent: Mapping[str, Any],
    repaired_cells: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Apply the unchanged parent gate to the repaired policy."""
    parent_conditions = parent["gate"]["conditions"]
    inherited = {
        name: bool(parent_conditions[name]["passed"])
        for name in (
            "authentication_and_coverage",
            "metric_robustness",
            "natural_relevance",
        )
    }
    shadow_count = sum(
        cell["l1_gain_over_baseline_fraction"] >= 0.10
        for cell in repaired_cells.values()
    )
    service_pass = all(
        cell["aggregate_selected_token_ratio_to_baseline"] <= 1.02
        and cell["maximum_decision_token_ratio_to_baseline"] <= 1.02
        for cell in repaired_cells.values()
    )
    additional_pass = all(
        cell["baseline_feasible_every_decision"]
        and cell["four_groups_selected_every_decision"]
        and cell["priority_uses_only_predecision_fields"]
        for cell in repaired_cells.values()
    )
    conditions = {
        **{name: {"passed": passed, "inherited": True} for name, passed in inherited.items()},
        "shadow_headroom": {
            "passed": shadow_count >= 8,
            "acquisitions_at_or_above_0p10": shadow_count,
            "required": 8,
        },
        "service_cost": {
            "passed": service_pass,
            "maximum_allowed_token_ratio": 1.02,
            "scope": "each_decision_and_acquisition_aggregate",
        },
        "repair_contract": {
            "passed": additional_pass,
            "baseline_feasible_every_decision": additional_pass,
        },
    }
    return {
        "all_required": True,
        "passed": all(condition["passed"] for condition in conditions.values()),
        "conditions": conditions,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--parent-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run the immutable service-budget repair analysis."""
    args = parse_args()
    protocol_path = G_HERE / "service_budget_repair_protocol.json"
    protocol = load_json(protocol_path)
    if protocol["schema"] != "m4-opportunity-control-service-budget-repair-v1":
        raise OfflineGateError("service-budget repair protocol schema differs")
    require_parent = str(protocol["parent_result_sha256"])
    if sha256(args.parent_result) != require_parent:
        raise OfflineGateError("parent gate result digest differs")
    parent = load_json(args.parent_result)
    cells = load_qwen_cells(args.evidence_root)
    cells.extend(load_llama_cells(args.evidence_root))
    repaired = {
        cell.label: repaired_shadow_summary(cell)
        for cell in sorted(cells, key=lambda value: value.label)
    }
    gate = evaluate_repaired_gate(parent=parent, repaired_cells=repaired)
    result = {
        "schema": "m4-opportunity-control-service-budget-repair-result-v1",
        "status": "COMPLETE_RETROSPECTIVE_POLICY_REPAIR",
        "analysis_role": protocol["claim_boundary"],
        "parent_result_sha256": require_parent,
        "protocol_sha256": sha256(protocol_path),
        "analysis_script_sha256": sha256(Path(__file__)),
        "cells": repaired,
        "gate": gate,
        "next_action": (
            "implement_baseline_budgeted_oars_in_shadow_mode_without_training"
            if gate["passed"]
            else "do_not_implement_or_launch_policy"
        ),
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "gate": gate}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
