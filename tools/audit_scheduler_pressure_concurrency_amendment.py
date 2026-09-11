# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Audit amended concurrency accounting without reclassifying frozen replication 49001."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_ARMS = {
    f"natural_{level}_{sampler}"
    for level in ("l0", "l1", "l3")
    for sampler in ("ready_first", "in_order")
} | {
    f"controlled_positive_control_l3_{mapping}_{sampler}"
    for mapping in ("short_delayed", "long_delayed")
    for sampler in ("ready_first", "in_order")
}


class SchedulerPressureConcurrencyAuditError(ValueError):
    """Archived calibration artifacts do not match the amendment audit contract."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _events(path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(json.loads(line) for line in path.read_text().splitlines())


def _maximum_concurrency(
    events: tuple[dict[str, Any], ...], *, terminal_event_types: frozenset[str]
) -> int:
    changes = []
    for event in events:
        if event.get("event_type") == "attempt_dispatched":
            changes.append((int(event["monotonic_ns"]), 1))
        elif event.get("event_type") in terminal_event_types:
            changes.append((int(event["monotonic_ns"]), -1))
    active = maximum = 0
    for _, change in sorted(changes, key=lambda item: (item[0], item[1])):
        active += change
        maximum = max(maximum, active)
    return maximum


def audit(
    *,
    runs_root: Path,
    frozen_result_path: Path,
    expected_frozen_result_sha256: str,
) -> dict[str, Any]:
    if _sha(frozen_result_path) != expected_frozen_result_sha256:
        raise SchedulerPressureConcurrencyAuditError("frozen result SHA mismatch")
    frozen = json.loads(frozen_result_path.read_text())
    if (
        frozen.get("order_seed") != 49001
        or frozen.get("decision") != "stop_no_replay_or_training"
        or frozen.get("replication_checks", {}).get("all_ten_arms_valid") is not False
    ):
        raise SchedulerPressureConcurrencyAuditError(
            "frozen 49001 failure identity mismatch"
        )
    run_dirs = {
        path.name: path
        for path in runs_root.iterdir()
        if path.is_dir() and path.name in EXPECTED_ARMS
    }
    if set(run_dirs) != EXPECTED_ARMS:
        raise SchedulerPressureConcurrencyAuditError(
            "expected exactly ten arm directories"
        )
    arms = {}
    for arm_id, run_dir in sorted(run_dirs.items()):
        trace_path = run_dir / "scheduler_trace.v1.jsonl"
        if not trace_path.is_file():
            raise SchedulerPressureConcurrencyAuditError(f"missing trace for {arm_id}")
        events = _events(trace_path)
        arms[arm_id] = {
            "trace_sha256": _sha(trace_path),
            "maximum_active_generation_groups": _maximum_concurrency(
                events,
                terminal_event_types=frozenset({"rollout_completed", "attempt_failed"}),
            ),
            "maximum_unreleased_groups": _maximum_concurrency(
                events,
                terminal_event_types=frozenset({"group_ready", "attempt_failed"}),
            ),
        }
    natural = [value for key, value in arms.items() if key.startswith("natural_")]
    return {
        "schema_version": 1,
        "analysis_status": "calibration_only_concurrency_amendment_event_accounting_audit",
        "source_order_seed": 49001,
        "source_frozen_result_sha256": expected_frozen_result_sha256,
        "source_frozen_decision_unchanged": "stop_no_replay_or_training",
        "reclassifies_source_replication": False,
        "eligible_for_amended_acceptance": False,
        "arms": arms,
        "diagnostic_checks": {
            "all_ten_maximum_active_generation_groups_equal_4": all(
                value["maximum_active_generation_groups"] == 4
                for value in arms.values()
            ),
            "all_six_natural_maximum_unreleased_groups_equal_4": all(
                value["maximum_unreleased_groups"] == 4 for value in natural
            ),
        },
        "counterfactual_replay_authorized": False,
        "learner_training_authorized": False,
        "population_claim_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--frozen-result", type=Path, required=True)
    parser.add_argument("--expected-frozen-result-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        runs_root=args.runs_root,
        frozen_result_path=args.frozen_result,
        expected_frozen_result_sha256=args.expected_frozen_result_sha256,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
