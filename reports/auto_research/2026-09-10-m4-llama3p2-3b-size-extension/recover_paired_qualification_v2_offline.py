#!/usr/bin/env python3
"""Recover the frozen Llama 3B qualification gates from completed-run artifacts."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import zipfile
from pathlib import Path

from tools.m4_llama3b_qualification import assess_llama3b_qualification
from tools.neutral_qualification_lifecycle import assess_neutral_qualification_lifecycle
from tools.opportunity_ledger_join import (
    LedgerJoinProtocol,
    ReleaseArm,
    join_opportunity_ledgers,
)


SOURCE_COMMIT = "88424bdb7bfb526e541b4a7b07830bc31028e64f"
CELLS = {
    "openmath": {
        "job_id": 434414383,
        "zip_sha256": "c1b93302f8f561269eec96feb905cde18a476bda61120b4ee8d39a15f4abdf81",
        "trace_sha256": "1fc22d292e98f82f33b0964d43cef161c5662da8cec2a9c0a2319be2fc6c2552",
        "stem": "m4-llama-openmath-qualification",
        "domain": "m4-llama3p2-3b-openmath-neutral-qualification-v1",
        "assignment_seed": 20261101,
        "assignment_bootstrap_seed": 20261107,
        "timing_bootstrap_seed": 20261109,
    },
    "gsm8k": {
        "job_id": 434414503,
        "zip_sha256": "691e410ac464d5156cf002573b5cc0993982683f7445be78f3fc270df1eeb113",
        "trace_sha256": "72868253b8c215c06dd542cfd9a4f151f523171de961f251513a4ef8ea6af6a2",
        "stem": "m4-llama-gsm8k-qualification",
        "domain": "m4-llama3p2-3b-gsm8k-neutral-qualification-v1",
        "assignment_seed": 20261102,
        "assignment_bootstrap_seed": 20261108,
        "timing_bootstrap_seed": 20261110,
    },
}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def exactly_one(archive: zipfile.ZipFile, suffix: str) -> bytes:
    names = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(names) != 1:
        raise RuntimeError(f"expected one archive member ending {suffix!r}, found {names}")
    return archive.read(names[0])


def json_rows(raw: bytes) -> list[dict[str, object]]:
    if not raw.endswith(b"\n"):
        raise RuntimeError("JSONL artifact lacks terminal newline")
    return [json.loads(line) for line in raw.splitlines()]


def recover(cell: str, archive_path: Path, trace_path: Path) -> dict[str, object]:
    registered = CELLS[cell]
    archive_raw = archive_path.read_bytes()
    trace_raw = trace_path.read_bytes()
    if sha256(archive_raw) != registered["zip_sha256"]:
        raise RuntimeError(f"{cell} workload archive hash differs")
    if sha256(trace_raw) != registered["trace_sha256"]:
        raise RuntimeError(f"{cell} workload trace hash differs")
    trace = trace_raw.decode("utf-8", errors="replace")
    if f"M4_LLAMA3B_{cell.upper()}_V2_AUTHORIZED_EVIDENCE_PASS" not in trace:
        raise RuntimeError(f"{cell} authorization marker absent")
    if "SC run complete: {'train_steps': 64, 'trainer_version': 64}" not in trace:
        raise RuntimeError(f"{cell} completed-training marker absent")
    starts = re.findall(r"readonly START_NS=(\d+)", trace)
    ends = re.findall(r"readonly END_NS=(\d+)", trace)
    if len(starts) != 1 or len(ends) != 1:
        raise RuntimeError(f"{cell} timing boundary count differs")
    start_ns, end_ns = int(starts[0]), int(ends[0])
    if end_ns <= start_ns:
        raise RuntimeError(f"{cell} nonpositive wall time")
    stem = registered["stem"]
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"{cell} corrupt archive member: {bad}")
        evidence = {
            "run.log": exactly_one(archive, f"/{stem}-run.log"),
            "lifecycle.jsonl": exactly_one(archive, f"/{stem}-lifecycle.jsonl"),
            "opportunity.jsonl": exactly_one(archive, f"/{stem}-opportunity.jsonl"),
            "lifecycle-duty.json": exactly_one(
                archive, f"/{stem}-lifecycle-duty.json"
            ),
            "derivation.json": exactly_one(archive, f"/{stem}-derivation.json"),
        }
    run_log = evidence["run.log"].decode("utf-8", errors="strict")
    if "SC run complete: {'train_steps': 64, 'trainer_version': 64}" not in run_log:
        raise RuntimeError(f"{cell} archive run log lacks completion marker")
    lifecycle = json_rows(evidence["lifecycle.jsonl"])
    opportunity = json_rows(evidence["opportunity.jsonl"])
    duty = json.loads(evidence["lifecycle-duty.json"])
    derivation = json.loads(evidence["derivation.json"])
    groups = [row for row in opportunity if row.get("event_type") == "group"]
    steps = [
        row for row in opportunity if row.get("event_type") == "train_step_completed"
    ]
    group_ids = {row["group_id"] for row in groups}
    rewards = {sibling["reward"] for row in groups for sibling in row["siblings"]}
    lifecycle_assessment = assess_neutral_qualification_lifecycle(
        lifecycle, opportunity_group_ids=group_ids
    )
    protocol = LedgerJoinProtocol(
        assignment_domain=str(registered["domain"]),
        assignment_seed=int(registered["assignment_seed"]),
        arms=(ReleaseArm("neutral", 0.0, 1),),
        primary_start_version=8,
        primary_end_version=55,
        siblings_per_group=8,
        train_batch_size=32,
    )
    joined = join_opportunity_ledgers(
        protocol=protocol,
        lifecycle_rows=lifecycle,
        opportunity_rows=opportunity,
    )
    # The submitted block used row["start_weight_version"]. The join returns a
    # JoinedOpportunityAssignment dataclass whose frozen field is start_version.
    versions = collections.Counter(row.start_version for row in joined)
    advances = sorted(
        (row for row in lifecycle if row.get("stage") == "learner_version_advanced"),
        key=lambda row: row["learner_weight_version"],
    )
    advance_time = {
        row["learner_weight_version"]: row["timestamp_ns"] for row in advances
    }
    intervals = [
        (advance_time[version + 1] - advance_time[version]) / 1e9
        for version in range(8, 56)
    ]
    duration = (end_ns - start_ns) / 1e9
    qualification = assess_llama3b_qualification(
        versions,
        intervals,
        total_qualification_runtime_seconds=duration,
        assignment_bootstrap_seed=int(registered["assignment_bootstrap_seed"]),
        timing_bootstrap_seed=int(registered["timing_bootstrap_seed"]),
    ).to_dict()
    expected_corrected = max(
        0,
        duty["raw_observer_ns"]
        - duty["observation_count"] * duty["paired_clock_overhead_ns"],
    )
    checks = {
        "complete_steps": len(steps) == 64
        and [row["learner_version"] for row in steps] == list(range(1, 65))
        and [row["learner_weight_version"] for row in advances] == list(range(1, 65)),
        "neutral_lifecycle": lifecycle_assessment["supported"],
        "group_ids_unique": len(groups) == len(group_ids),
        "binary_reward_support": rewards == {0.0, 1.0},
        "exact_evaluable_window": set(versions) == set(range(8, 56)),
        "strict_join": len(joined)
        == sum(8 <= row["start_weight_version"] <= 55 for row in groups),
        "assignment_support": qualification["assignment_support_passed"],
        "timing_support": qualification["timing_support_passed"],
        "joint_qualification_gate": qualification["qualified"],
        "lifecycle_duty": duty["corrected_observer_ns"] == expected_corrected
        and duty["corrected_observer_duty"] <= 0.01,
        "derivation": derivation["completed_step_count"] == 64
        and derivation["group_count"] == len(groups)
        and derivation["derivation_method"]
        == "lifecycle_derived_grpo_opportunity_v1",
    }
    return {
        "schema": "m4-llama3p2-3b-neutral-qualification-offline-recovery-v2",
        "cell": cell,
        "source_commit": SOURCE_COMMIT,
        "workload_job_id": registered["job_id"],
        "workload_archive_sha256": registered["zip_sha256"],
        "workload_trace_sha256": registered["trace_sha256"],
        "artifact_sha256": {name: sha256(raw) for name, raw in evidence.items()},
        "trainer_steps": 64,
        "gpus": 2,
        "wall_time_seconds": duration,
        "opportunity_group_count": len(groups),
        "strict_join_assignment_count": len(joined),
        "lifecycle_event_count": len(lifecycle),
        "corrected_lifecycle_duty": duty["corrected_observer_duty"],
        "reward_values": sorted(rewards),
        "qualification": qualification,
        "checks": checks,
        "qualification_complete": all(checks.values()),
        "causal_estimate_produced": False,
        "acquisition_started": False,
        "automatic_retry": False,
        "automatic_extension": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for cell in CELLS:
        parser.add_argument(f"--{cell}-archive", type=Path, required=True)
        parser.add_argument(f"--{cell}-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cells = {
        cell: recover(
            cell,
            getattr(args, f"{cell}_archive"),
            getattr(args, f"{cell}_trace"),
        )
        for cell in CELLS
    }
    result = {
        "schema": "m4-llama3p2-3b-paired-neutral-qualification-offline-recovery-v2",
        "status": "BOTH_QUALIFIED" if all(row["qualification_complete"] for row in cells.values()) else "QUALIFICATION_GATE_FAILED",
        "recovery_delta": "JoinedOpportunityAssignment.start_version_field_access_only",
        "training_rerun": False,
        "scientific_configuration_changed": False,
        "cells": cells,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
