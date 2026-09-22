# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Outcome-excluded systems gate for one enacted OARS-v2 scorer."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.m4_oars_v2_live_shadow_qualification import (
    OARSV2_SCORERS,
    _read_json,
    _read_jsonl,
)

EXPECTED_STEPS = 64
EXPECTED_GROUPS = 4
EXPECTED_CANDIDATES = 8
EXPECTED_COMBINATIONS = 70
ACTUATION_SCORERS = ("reward_variance_risk", "absolute_m4_risk")


def _finite_nonnegative(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _nondecreasing(values: Sequence[int]) -> bool:
    return all(left <= right for left, right in zip(values, values[1:]))


def assess_v2_actuation(
    *,
    scorer: str,
    oars_rows: Sequence[Mapping[str, Any]],
    lifecycle_rows: Sequence[Mapping[str, Any]],
    observer_duty: Mapping[str, Any],
    source_commit: str,
    run_start_ns: int,
    run_end_ns: int,
) -> dict[str, Any]:
    """Evaluate scorer identity, exact choice, liveness, and duty gates."""
    if scorer not in ACTUATION_SCORERS:
        raise ValueError("unsupported OARS-v2 actuation scorer")
    if len(oars_rows) < 2 or run_start_ns < 0 or run_end_ns <= run_start_ns:
        raise ValueError("invalid OARS-v2 actuation inputs")
    header, *decisions = oars_rows
    advances = sorted(
        int(row["learner_weight_version"])
        for row in lifecycle_rows
        if row.get("stage") == "learner_version_advanced"
    )
    removals = Counter(
        str(row.get("removal_reason"))
        for row in lifecycle_rows
        if row.get("stage") == "removed"
    )
    accounting = [row.get("liveness_accounting") for row in decisions]
    accounting_complete = all(isinstance(row, dict) for row in accounting)
    accounting_rows = [row for row in accounting if isinstance(row, dict)]
    accounting_keys = (
        "candidate_excess_removed_groups_total",
        "stale_evicted_groups_total",
        "replenishment_batches_earned_total",
        "replenishment_batches_consumed_total",
    )
    accounting_values = {
        key: [int(row.get(key, -1)) for row in accounting_rows]
        for key in accounting_keys
    }
    final_accounting = accounting_rows[-1] if accounting_rows else {}
    decision_latencies = [int(row.get("decision_latency_ns", -1)) for row in decisions]
    active_window_ns = int(observer_duty.get("active_window_ns", 0))
    decision_duty = (
        sum(decision_latencies) / active_window_ns
        if active_window_ns > 0 and all(value >= 0 for value in decision_latencies)
        else math.inf
    )
    gradient_duty = observer_duty.get("corrected_observer_duty")
    combined_duty = (
        float(gradient_duty) + decision_duty
        if _finite_nonnegative(gradient_duty) and math.isfinite(decision_duty)
        else math.inf
    )
    proposal_rows = [
        proposal
        for row in decisions
        for proposal in (row.get("proposals", {}).get(name) for name in OARSV2_SCORERS)
        if isinstance(proposal, dict)
    ]
    checks = {
        "header": (
            header.get("event_type") == "header"
            and header.get("schema_version") == 2
            and header.get("mode") == "act"
            and header.get("policy") == "multi_scorer_oars_v2_actuator"
            and header.get("actuation_scorer") == scorer
            and header.get("candidate_window_policy") == "controlled_frontier"
            and header.get("selection_candidate_watermark") == EXPECTED_CANDIDATES
            and header.get("minimum_service_multiplier") == 0.98
            and header.get("maximum_service_multiplier") == 1.02
            and header.get("max_candidate_groups") == 64
            and header.get("exact_search_max_candidates") == 16
            and header.get("candidate_mutation") == "configured_scorer_exact_removal"
            and header.get("stale_replenishment_policy")
            == "one_batch_drop_newest_excess_v1"
        ),
        "complete_learner_steps": advances == list(range(1, EXPECTED_STEPS + 1)),
        "one_decision_per_step": (
            len(decisions) == EXPECTED_STEPS
            and all(row.get("event_type") == "decision" for row in decisions)
        ),
        "exact_candidate_set": all(
            row.get("candidate_group_count") == EXPECTED_CANDIDATES
            and len(row.get("candidates") or ()) == EXPECTED_CANDIDATES
            for row in decisions
        ),
        "all_scorers_exact_no_fallback": (
            len(proposal_rows) == EXPECTED_STEPS * len(OARSV2_SCORERS)
            and all(
                proposal.get("status") == "proposed"
                and proposal.get("combination_count") == EXPECTED_COMBINATIONS
                and proposal.get("search_candidate_count") == EXPECTED_CANDIDATES
                and proposal.get("search_strategy") == "exact"
                for proposal in proposal_rows
            )
        ),
        "complete_metadata_no_skip": all(
            row.get("skip_reason") is None for row in decisions
        ),
        "configured_proposal_identity": all(
            row.get("mode") == "act"
            and row.get("actuation_scorer") == scorer
            and row.get("proposal_matches_actual") is True
            and row.get("actual_selected_group_count") == EXPECTED_GROUPS
            and len(row.get("actual_selected_group_ids") or ()) == EXPECTED_GROUPS
            and len(set(row.get("actual_selected_group_ids") or ())) == EXPECTED_GROUPS
            and row.get("actual_selected_group_ids")
            == row.get("proposals", {}).get(scorer, {}).get("proposed_group_ids")
            for row in decisions
        ),
        "two_sided_service_band": all(
            _finite_nonnegative(proposal.get("proposed_valid_actor_tokens"))
            and _finite_nonnegative(proposal.get("minimum_tokens"))
            and _finite_nonnegative(proposal.get("maximum_tokens"))
            and int(proposal["minimum_tokens"])
            <= int(proposal["proposed_valid_actor_tokens"])
            <= int(proposal["maximum_tokens"])
            for proposal in proposal_rows
        ),
        "bounded_candidate_excess": all(
            isinstance(row.get("eligible_candidate_count"), int)
            and EXPECTED_CANDIDATES
            <= int(row["eligible_candidate_count"])
            <= EXPECTED_CANDIDATES + EXPECTED_GROUPS - 1
            and row.get("candidate_excess_count")
            == int(row["eligible_candidate_count"]) - EXPECTED_CANDIDATES
            for row in decisions
        ),
        "complete_liveness_accounting": (
            accounting_complete
            and len(accounting_rows) == EXPECTED_STEPS
            and all(_nondecreasing(values) for values in accounting_values.values())
            and all(
                row.get("candidate_excess_pending_groups")
                == decision.get("candidate_excess_count")
                and row.get("replenishment_credits_outstanding") == 0
                and row.get("replenishment_batches_earned_total")
                == row.get("replenishment_batches_consumed_total")
                for row, decision in zip(accounting_rows, decisions, strict=True)
            )
            and int(final_accounting.get("candidate_excess_removed_groups_total", 0))
            == removals["oars_candidate_excess"]
            and int(final_accounting.get("stale_evicted_groups_total", 0))
            == removals["stale_evicted"]
            and int(final_accounting.get("candidate_excess_removed_groups_total", 0))
            > 0
            and int(final_accounting.get("stale_evicted_groups_total", 0)) > 0
            and int(final_accounting.get("replenishment_batches_earned_total", 0)) > 0
        ),
        "combined_observer_controller_duty": combined_duty <= 0.01,
        "runtime": (run_end_ns - run_start_ns) / 1e9 <= 14_400.0,
    }
    status = "PASS_OARS_V2_ACTUATION_QUALIFIED" if all(checks.values()) else "FAIL"
    return {
        "schema": "m4-oars-v2-actuation-qualification-result-v1",
        "status": status,
        "source_commit": source_commit,
        "actuation_scorer": scorer,
        "checks": checks,
        "decision_count": len(decisions),
        "removals": dict(sorted(removals.items())),
        "liveness_accounting": final_accounting,
        "gradient_observer_duty": gradient_duty,
        "oars_v2_decision_duty": decision_duty,
        "combined_observer_controller_duty": combined_duty,
        "runtime_seconds": (run_end_ns - run_start_ns) / 1e9,
        "oars_v2_actuated": True,
        "training_quality_analyzed": False,
        "scientific_outcome_acquisition": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer", choices=ACTUATION_SCORERS, required=True)
    parser.add_argument("--oars", type=Path, required=True)
    parser.add_argument("--lifecycle", type=Path, required=True)
    parser.add_argument("--observer-duty", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-start-ns", type=int, required=True)
    parser.add_argument("--run-end-ns", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess_v2_actuation(
        scorer=args.scorer,
        oars_rows=_read_jsonl(args.oars),
        lifecycle_rows=_read_jsonl(args.lifecycle),
        observer_duty=_read_json(args.observer_duty),
        source_commit=args.source_commit,
        run_start_ns=args.run_start_ns,
        run_end_ns=args.run_end_ns,
    )
    args.output.write_text(
        json.dumps(result, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    if result["status"] != "PASS_OARS_V2_ACTUATION_QUALIFIED":
        raise SystemExit(f"OARS-v2 actuation qualification failed: {result['checks']}")
    print(result["status"])


if __name__ == "__main__":
    main()
