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

"""Frozen systems gate for the common-frontier OARS-v2 qualification."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.m4_oars_v2_live_shadow_qualification import (
    OARSV2_SCORERS,
    _read_json,
    _read_jsonl,
    assess_live_shadow,
)

EXPECTED_STEPS = 64
EXPECTED_CANDIDATES = 8
EXPECTED_COMBINATIONS = 70


def assess_controlled_frontier(
    *,
    oars_rows: Sequence[Mapping[str, Any]],
    lifecycle_rows: Sequence[Mapping[str, Any]],
    observer_duty: Mapping[str, Any],
    source_commit: str,
    run_start_ns: int,
    run_end_ns: int,
) -> dict[str, Any]:
    """Require complete choice-bearing decisions under controlled FIFO."""
    base = assess_live_shadow(
        oars_rows=oars_rows,
        lifecycle_rows=lifecycle_rows,
        observer_duty=observer_duty,
        source_commit=source_commit,
        run_start_ns=run_start_ns,
        run_end_ns=run_end_ns,
        expected_candidate_window_policy="controlled_frontier",
        expected_selection_candidate_watermark=EXPECTED_CANDIDATES,
        minimum_contended_decisions=EXPECTED_STEPS,
    )
    decisions = tuple(oars_rows[1:])
    proposal_rows = [
        proposal
        for row in decisions
        for scorer in OARSV2_SCORERS
        if isinstance((proposal := row.get("proposals", {}).get(scorer)), dict)
    ]
    controlled_checks = {
        "eight_candidates_every_decision": all(
            row.get("candidate_group_count") == EXPECTED_CANDIDATES for row in decisions
        ),
        "exact_choice_space_every_proposal": (
            len(proposal_rows) == EXPECTED_STEPS * len(OARSV2_SCORERS)
            and all(
                proposal.get("status") == "proposed"
                and proposal.get("combination_count") == EXPECTED_COMBINATIONS
                and proposal.get("search_candidate_count") == EXPECTED_CANDIDATES
                and proposal.get("search_strategy") == "exact"
                for proposal in proposal_rows
            )
        ),
        "zero_fallbacks": base["fallback_fraction"] == 0.0,
    }
    checks = {**base["checks"], **controlled_checks}
    status = (
        "PASS_CONTROLLED_FRONTIER_SYSTEMS_READY"
        if all(checks.values())
        else "FAIL_CONTROLLED_FRONTIER_SYSTEMS_GATE"
    )
    return {
        **base,
        "schema": "m4-oars-v2-controlled-frontier-qualification-result-v1",
        "status": status,
        "checks": checks,
        "controlled_frontier_systems_ready": status
        == "PASS_CONTROLLED_FRONTIER_SYSTEMS_READY",
        "acting_policy": "weight_fifo_under_common_eight_candidate_frontier",
        "natural_eager_fifo_representativeness_claim": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oars", type=Path, required=True)
    parser.add_argument("--lifecycle", type=Path, required=True)
    parser.add_argument("--observer-duty", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-start-ns", type=int, required=True)
    parser.add_argument("--run-end-ns", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess_controlled_frontier(
        oars_rows=_read_jsonl(args.oars),
        lifecycle_rows=_read_jsonl(args.lifecycle),
        observer_duty=_read_json(args.observer_duty),
        source_commit=args.source_commit,
        run_start_ns=args.run_start_ns,
        run_end_ns=args.run_end_ns,
    )
    args.output.write_text(json.dumps(result, allow_nan=False, sort_keys=True) + "\n")
    if not result["controlled_frontier_systems_ready"]:
        raise SystemExit(f"controlled-frontier gate failed: {result['checks']}")
    print(result["status"])


if __name__ == "__main__":
    main()
