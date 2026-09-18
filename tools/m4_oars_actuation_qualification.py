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

"""Outcome-neutral gates for one FIFO or enacted-OARS qualification arm."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

EXPECTED_STEPS = 64
EXPECTED_GROUPS = 4
EXPECTED_CANDIDATES = 8


class OARSActuationQualificationError(ValueError):
    """Raised when an actuation qualification artifact is malformed."""


def _finite_nonnegative(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and value >= 0
    )


def _reject_constant(value: str) -> None:
    raise OARSActuationQualificationError(f"non-finite JSON constant {value}")


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=_reject_constant
    )
    if not isinstance(value, dict):
        raise OARSActuationQualificationError(f"{path.name} must contain one object")
    return value


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise OARSActuationQualificationError(
            f"{path.name} must be nonempty newline JSONL"
        )
    rows = [
        json.loads(line, parse_constant=_reject_constant) for line in raw.splitlines()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise OARSActuationQualificationError(f"{path.name} rows must be objects")
    return rows


def assess_oars_actuation_qualification(
    *,
    mode: str,
    oars_rows: Sequence[Mapping[str, Any]],
    lifecycle_rows: Sequence[Mapping[str, Any]],
    observer_duty: Mapping[str, Any],
    source_commit: str,
    run_start_ns: int,
    run_end_ns: int,
) -> dict[str, Any]:
    """Evaluate frozen systems gates without reading training outcomes."""
    if mode not in {"observe", "act"}:
        raise OARSActuationQualificationError("mode must be observe or act")
    if len(oars_rows) < 2 or run_start_ns < 0 or run_end_ns <= run_start_ns:
        raise OARSActuationQualificationError("invalid qualification inputs")
    header, *decisions = oars_rows
    if any(row.get("event_type") != "decision" for row in decisions):
        raise OARSActuationQualificationError(
            "OARS ledger contains unknown event types"
        )

    advances = sorted(
        int(row["learner_weight_version"])
        for row in lifecycle_rows
        if row.get("stage") == "learner_version_advanced"
    )
    identity_key = (
        "baseline_matches_actual" if mode == "observe" else "proposal_matches_actual"
    )
    decision_latencies = [int(row.get("decision_latency_ns", -1)) for row in decisions]
    active_window_ns = int(observer_duty.get("active_window_ns", 0))
    oars_duty = (
        sum(decision_latencies) / active_window_ns
        if active_window_ns > 0 and all(value >= 0 for value in decision_latencies)
        else math.inf
    )
    gradient_duty = observer_duty.get("corrected_observer_duty")
    checks = {
        "header": (
            header.get("event_type") == "header"
            and header.get("schema_version") == 3
            and header.get("policy") == "baseline_budgeted_oars_v1"
            and header.get("candidate_window_policy")
            == "oldest_ready_exact_watermark_v1"
            and header.get("mode") == mode
            and header.get("service_budget_multiplier") == 1.02
            and header.get("max_candidate_groups") == 25
            and header.get("selection_candidate_watermark") == EXPECTED_CANDIDATES
            and header.get("stale_replenishment_policy")
            == ("one_batch_drop_newest_excess_v1" if mode == "act" else "none")
        ),
        "complete_learner_steps": advances == list(range(1, EXPECTED_STEPS + 1)),
        "one_decision_per_step": len(decisions) == EXPECTED_STEPS,
        "exact_candidate_set": all(
            row.get("candidate_group_count") == EXPECTED_CANDIDATES
            and row.get("combination_count") == 70
            for row in decisions
        ),
        "bounded_replenishment": all(
            isinstance(row.get("eligible_candidate_count"), int)
            and EXPECTED_CANDIDATES
            <= int(row["eligible_candidate_count"])
            <= EXPECTED_CANDIDATES + EXPECTED_GROUPS - 1
            and row.get("candidate_excess_count")
            == int(row["eligible_candidate_count"]) - EXPECTED_CANDIDATES
            for row in decisions
        ),
        "complete_metadata_no_skip": all(
            row.get("skip_reason") is None for row in decisions
        ),
        "exact_policy_identity": all(
            row.get(identity_key) is True for row in decisions
        ),
        "selection_cardinality_and_uniqueness": all(
            row.get("actual_selected_group_count") == EXPECTED_GROUPS
            and len(row.get("actual_selected_group_ids") or ()) == EXPECTED_GROUPS
            and len(set(row.get("actual_selected_group_ids") or ())) == EXPECTED_GROUPS
            for row in decisions
        ),
        "service_budget": all(
            _finite_nonnegative(row.get("baseline_valid_actor_tokens"))
            and _finite_nonnegative(row.get("proposed_valid_actor_tokens"))
            and _finite_nonnegative(row.get("token_budget"))
            and int(row["proposed_valid_actor_tokens"]) <= int(row["token_budget"])
            and int(row["token_budget"])
            == math.floor(int(row["baseline_valid_actor_tokens"]) * 1.02 + 1e-12)
            for row in decisions
        ),
        "gradient_observer_duty": (
            _finite_nonnegative(gradient_duty) and float(gradient_duty) <= 0.01
        ),
        "oars_decision_duty": math.isfinite(oars_duty) and oars_duty <= 0.01,
        "runtime": (run_end_ns - run_start_ns) / 1e9 <= 14_400.0,
    }
    return {
        "schema": "m4-oars-actuation-qualification-arm-result-v1",
        "source_commit": source_commit,
        "mode": mode,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "decision_count": len(decisions),
        "candidate_excess_count": sum(
            int(row.get("candidate_excess_count", 0)) for row in decisions
        ),
        "gradient_observer_duty": gradient_duty,
        "oars_decision_duty": oars_duty,
        "runtime_seconds": (run_end_ns - run_start_ns) / 1e9,
        "oars_actuated": mode == "act",
        "training_quality_analyzed": False,
        "scientific_outcome_acquisition": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("observe", "act"), required=True)
    parser.add_argument("--oars", type=Path, required=True)
    parser.add_argument("--lifecycle", type=Path, required=True)
    parser.add_argument("--observer-duty", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-start-ns", type=int, required=True)
    parser.add_argument("--run-end-ns", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess_oars_actuation_qualification(
        mode=args.mode,
        oars_rows=_read_jsonl(args.oars),
        lifecycle_rows=_read_jsonl(args.lifecycle),
        observer_duty=_read_json(args.observer_duty),
        source_commit=args.source_commit,
        run_start_ns=args.run_start_ns,
        run_end_ns=args.run_end_ns,
    )
    args.output.write_text(
        json.dumps(result, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    if result["status"] != "PASS":
        raise SystemExit(f"OARS actuation qualification failed: {result['checks']}")
    print(f"M4_OARS_QUALIFICATION_{args.mode.upper()}_PASS")


if __name__ == "__main__":
    main()
