"""Outcome-free completion evidence; no-event admission also requires zero events."""

from __future__ import annotations

import ast
import argparse
import hashlib
import json
import re
from pathlib import Path

from research.m4_eligible_replay.contracts import canonical_json

NO_CAPTURE_ERROR = "eligible live decision capture ended without a capsule"
SCHEMA = "m4-capture-full-window-proof-v1"


def prove_full_window(
    *,
    run_exit_code: int,
    run_log: bytes,
    lifecycle: bytes,
    oars: bytes,
    capsule_written: bool,
    watchdog_record_present: bool,
) -> dict:
    """Classify a bounded native exit without rewriting its observed exit code.

    The native unsuppressed controller raises NO_CAPTURE_ERROR after completing
    all 64 updates without an eligible decision. That specific boundary is valid
    only with independently inventoried full-window evidence; arbitrary runtime
    errors, truncated windows, and watchdog exits remain failures.
    """
    lifecycle_rows = [json.loads(line) for line in lifecycle.splitlines() if line]
    advances = [
        r for r in lifecycle_rows if r.get("stage") == "learner_version_advanced"
    ]
    versions = [r["learner_weight_version"] for r in advances]
    rows = [json.loads(line) for line in oars.splitlines() if line]
    decisions = [r for r in rows if r.get("event_type") == "decision"]
    headers = [r for r in rows if r.get("event_type") == "header"]
    text = re.sub(r"\x1b\[[0-9;]*m", "", run_log.decode())
    terminal_error = None
    if run_exit_code == 1:
        errors = [
            line.strip()
            for line in text.splitlines()
            if re.search(r"(?:Error|Exception):", line)
        ]
        if not errors or not errors[-1].endswith("RuntimeError: " + NO_CAPTURE_ERROR):
            raise ValueError("unexpected native capture failure")
        terminal_error = NO_CAPTURE_ERROR
    elif run_exit_code == 0:
        match = re.search(r"(?m)^SC run complete: (\{.*\})$", text)
        if not match:
            raise ValueError("native successful completion record missing")
        controller = ast.literal_eval(match.group(1))
        if controller["train_steps"] != 64 or controller["trainer_version"] != 64:
            raise ValueError("native successful completion window incomplete")
    else:
        raise ValueError("unexpected native capture exit code")
    record = {
        "schema": SCHEMA,
        "train_steps": len(advances),
        "learner_version": versions[-1] if versions else None,
        "learner_versions": versions,
        "run_exit_code": run_exit_code,
        "expected_no_capture_exception": terminal_error,
        "capsule_written": capsule_written,
        "watchdog_record_present": watchdog_record_present,
        "oars_header_count": len(headers),
        "oars_header_scorer": headers[0].get("actuation_scorer")
        if len(headers) == 1
        else None,
        "oars_decisions": len(decisions),
        "eligible_divergence_count": sum(
            r.get("skip_reason") is None
            and r.get("proposals", {}).get("m4_shield", {}).get("proposed_group_ids")
            != r.get("proposals", {})
            .get("reward_variance_risk", {})
            .get("proposed_group_ids")
            for r in decisions
        ),
        "oars_actions_matching_shield": sum(
            r.get("proposal_matches_actual") is True
            and isinstance(r.get("actual_selected_group_ids"), list)
            and len(r["actual_selected_group_ids"]) == 4
            and len(set(r["actual_selected_group_ids"])) == 4
            and r.get("actual_selected_group_ids")
            == r.get("proposals", {}).get("m4_shield", {}).get("proposed_group_ids")
            for r in decisions
        ),
        "source_sha256": {
            "run.log": hashlib.sha256(run_log).hexdigest(),
            "lifecycle.jsonl": hashlib.sha256(lifecycle).hexdigest(),
            "oars-v2.jsonl": hashlib.sha256(oars).hexdigest(),
        },
    }
    validate_full_window(record)
    return record


def validate_full_window(record: dict) -> None:
    """Validate completion, not absence of eligibility (checked at admission)."""
    if (
        record.get("schema") != SCHEMA
        or type(record.get("train_steps")) is not int
        or record["train_steps"] != 64
        or record.get("learner_version") != 64
        or record.get("learner_versions") != list(range(1, 65))
        or any(type(v) is not int for v in record["learner_versions"])
        or record.get("capsule_written") is not False
        or record.get("watchdog_record_present") is not False
        or record.get("oars_header_count") != 1
        or record.get("oars_header_scorer") != "m4_shield"
        or record.get("oars_decisions") != 64
        or record.get("oars_actions_matching_shield") != 64
        or type(record.get("eligible_divergence_count")) is not int
        or not 0 <= record["eligible_divergence_count"] <= 64
        or type(record.get("run_exit_code")) is not int
        or (record["run_exit_code"], record.get("expected_no_capture_exception"))
        not in ((0, None), (1, NO_CAPTURE_ERROR))
    ):
        raise ValueError("no-eligibility requires complete authenticated window")
    sources = record.get("source_sha256", {})
    if set(sources) != {"run.log", "lifecycle.jsonl", "oars-v2.jsonl"} or any(
        not isinstance(h, str) or re.fullmatch(r"[a-f0-9]{64}", h) is None
        for h in sources.values()
    ):
        raise ValueError("full-window source hashes incomplete")


def main() -> None:
    """Emit an immutable proof from closed workload files, never training outcomes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-exit-code", type=int, required=True)
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--lifecycle", type=Path, required=True)
    parser.add_argument("--oars", type=Path, required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--watchdog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = prove_full_window(
        run_exit_code=args.run_exit_code,
        run_log=args.run_log.read_bytes(),
        lifecycle=args.lifecycle.read_bytes(),
        oars=args.oars.read_bytes(),
        capsule_written=args.capsule.exists(),
        watchdog_record_present=args.watchdog.exists(),
    )
    with args.output.open("xb") as stream:
        stream.write(canonical_json(record))


if __name__ == "__main__":
    main()
