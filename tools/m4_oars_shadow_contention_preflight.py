# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Outcome-neutral gates for the repaired OARS contention preflight."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.m4_oars_shadow_preflight import (
    _read_json,
    _read_jsonl,
    assess_oars_shadow_preflight,
)

EXPECTED_STEPS = 64
EXPECTED_BATCH_GROUPS = 4
EXPECTED_CANDIDATE_WATERMARK = 8


def assess_oars_shadow_contention_preflight(
    *,
    oars_rows: Sequence[Mapping[str, Any]],
    lifecycle_rows: Sequence[Mapping[str, Any]],
    observer_duty: Mapping[str, Any],
    source_commit: str,
    run_start_ns: int,
    run_end_ns: int,
) -> dict[str, Any]:
    """Require a complete, FIFO-controlled eight-candidate systems run."""
    base = assess_oars_shadow_preflight(
        oars_rows=oars_rows,
        lifecycle_rows=lifecycle_rows,
        observer_duty=observer_duty,
        source_commit=source_commit,
        run_start_ns=run_start_ns,
        run_end_ns=run_end_ns,
        expected_steps=EXPECTED_STEPS,
        expected_batch_groups=EXPECTED_BATCH_GROUPS,
        minimum_contended_decisions=EXPECTED_STEPS,
    )
    header = oars_rows[0]
    decisions = tuple(
        row for row in oars_rows[1:] if row.get("event_type") == "decision"
    )
    repair_checks = {
        "declared_selection_candidate_watermark": (
            header.get("selection_candidate_watermark") == EXPECTED_CANDIDATE_WATERMARK
        ),
        "candidate_watermark_reached_every_decision": all(
            row.get("candidate_group_count") == EXPECTED_CANDIDATE_WATERMARK
            for row in decisions
        ),
        "nontrivial_choice_set_every_decision": all(
            row.get("combination_count") == 70 for row in decisions
        ),
    }
    checks = {**base["checks"], **repair_checks}
    return {
        **base,
        "schema": "m4-oars-shadow-contention-systems-preflight-result-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "selection_candidate_watermark": EXPECTED_CANDIDATE_WATERMARK,
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
    result = assess_oars_shadow_contention_preflight(
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
        raise SystemExit(f"OARS contention preflight failed: {result['checks']}")
    print("M4_OARS_FIFO_CONTROLLED_SHADOW_CONTENTION_PREFLIGHT_PASS")


if __name__ == "__main__":
    main()
