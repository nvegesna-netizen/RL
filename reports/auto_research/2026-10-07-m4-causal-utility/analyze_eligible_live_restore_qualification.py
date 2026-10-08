#!/usr/bin/env python3
"""Apply the frozen gate to two independent eligible-live restore results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def analyze(
    protocol_path: Path, restore_a_path: Path, restore_b_path: Path
) -> dict[str, Any]:
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    rows = [json.loads(path.read_bytes()) for path in (restore_a_path, restore_b_path)]
    expected_parameter = protocol["restore"]["expected_parameter_sha256"]
    absolute_tolerance = protocol["evaluation"][
        "absolute_loss_reproducibility_tolerance"
    ]
    relative_tolerance = protocol["evaluation"][
        "relative_loss_reproducibility_tolerance"
    ]
    losses = [float(row.get("heldout_loss", math.nan)) for row in rows]
    checks = {
        "schemas": all(
            row.get("schema") == "m4-shield-eligible-live-single-restore-result-v1"
            for row in rows
        ),
        "labels": [row.get("restore_label") for row in rows] == ["a", "b"],
        "single_restore_gates": all(
            row.get("status") == "PASS_SINGLE_RESTORE" for row in rows
        ),
        "protocol_binding": all(
            row.get("protocol_sha256") == protocol_sha for row in rows
        ),
        "manifest_binding": all(
            row.get("capsule_manifest_sha256")
            == protocol["source_capsule"]["manifest_sha256"]
            for row in rows
        ),
        "checkpoint_binding": all(
            row.get("checkpoint_authentication_sha256")
            == protocol["source_capsule"]["checkpoint_authentication_sha256"]
            for row in rows
        ),
        "heldout_only": all(
            row.get("heldout_groups") == protocol["evaluation"]["groups"]
            and row.get("heldout_samples") == protocol["evaluation"]["samples"]
            and row.get("arm_group_files_loaded") is False
            for row in rows
        ),
        "same_heldout_groups": rows[0].get("heldout_group_ids")
        == rows[1].get("heldout_group_ids"),
        "parameter_hashes": all(
            row.get("parameter_sha256_before_evaluation") == expected_parameter
            and row.get("parameter_sha256_after_evaluation") == expected_parameter
            and row.get("parameter_hash_unchanged") is True
            for row in rows
        ),
        "optimizer_restore_requested": all(
            row.get("optimizer_restore_requested") is True
            and row.get("optimizer_path_equals_weights_path") is True
            for row in rows
        ),
        "forward_only": all(
            row.get("eval_mode") is True
            and row.get("torch_no_grad") is True
            and row.get("optimizer_steps") == 0
            and row.get("scheduler_steps") == 0
            for row in rows
        ),
        "losses_finite": all(math.isfinite(value) for value in losses),
        "losses_reproducible": all(math.isfinite(value) for value in losses)
        and math.isclose(
            losses[0],
            losses[1],
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ),
        "outcome_exclusion": all(
            row.get("causal_arm_outcomes_opened") is False
            and row.get("causal_effect_estimated") is False
            and row.get("paired_acquisition_started") is False
            for row in rows
        ),
    }
    passed = all(checks.values())
    difference = losses[1] - losses[0] if checks["losses_finite"] else None
    return {
        "schema": "m4-shield-eligible-live-restore-qualification-result-v1",
        "status": "PASS_RESTORE_AND_EVALUATION_GATE"
        if passed
        else "FAIL_RESTORE_AND_EVALUATION_GATE",
        "protocol_sha256": protocol_sha,
        "source_capsule_manifest_sha256": protocol["source_capsule"]["manifest_sha256"],
        "checks": checks,
        "restore_a_heldout_loss": losses[0] if math.isfinite(losses[0]) else None,
        "restore_b_heldout_loss": losses[1] if math.isfinite(losses[1]) else None,
        "restore_loss_difference_b_minus_a": difference,
        "absolute_loss_reproducibility_tolerance": absolute_tolerance,
        "relative_loss_reproducibility_tolerance": relative_tolerance,
        "causal_arm_outcomes_opened": False,
        "causal_effect_estimated": False,
        "paired_acquisition_started": False,
        "decision": protocol["decision_ladder"][
            "pass" if passed else "fail_scientific"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--restore-a", type=Path, required=True)
    parser.add_argument("--restore-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.protocol, args.restore_a, args.restore_b)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(result))
    print(json.dumps({"output": str(args.output), "status": result["status"]}))
    if result["status"] != "PASS_RESTORE_AND_EVALUATION_GATE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
