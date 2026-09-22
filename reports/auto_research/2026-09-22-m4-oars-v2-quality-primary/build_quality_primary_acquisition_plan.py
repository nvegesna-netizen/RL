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

"""Build the sealed OARS-v2 18-block acquisition and analysis plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PROTOCOL_SHA256 = "e0e0b26cc1a592bf14c4b0f77d990662f6c3ff3e09bfb6a772a3efbf149b9d2d"
GATE_SHA256 = "7d2660bc1d1e651f83b40e037ea1890c5a3389681536f20fc8f5e389e4afc470"
ARM_CONFIG = {
    "fifo": {"mode": "observe", "actuation_scorer": None},
    "reward_variance": {
        "mode": "act",
        "actuation_scorer": "reward_variance_risk",
    },
    "absolute_m4": {"mode": "act", "actuation_scorer": "absolute_m4_risk"},
}


def sha256(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Validate the frozen gate and emit deterministic acquisition records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--qualification-gate", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--analysis-plan", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.protocol) != PROTOCOL_SHA256:
        raise RuntimeError("frozen quality-primary protocol moved")
    if GATE_SHA256 == "PLACEHOLDER":
        raise RuntimeError("qualification gate hash has not been locked")
    if sha256(args.qualification_gate) != GATE_SHA256:
        raise RuntimeError("joint actuation qualification gate moved")
    protocol = json.loads(args.protocol.read_bytes())
    gate = json.loads(args.qualification_gate.read_bytes())
    if gate["status"] != "PASS_ACQUISITION_GATE_READY_NOT_LAUNCHED":
        raise RuntimeError("joint actuation qualification gate did not pass")
    runs = []
    predecessor = None
    sequence = 0
    for block in protocol["design"]["blocks"]:
        for arm in block["order"]:
            sequence += 1
            identity = f"b{block['block']:02d}-{arm.replace('_', '-')}"
            runs.append(
                {
                    "global_sequence": sequence,
                    "identity": identity,
                    "predecessor": predecessor,
                    "block": block["block"],
                    "arm": arm,
                    "training_seed": block["training_seed"],
                    "assignment_seed": block["assignment_seed"],
                    "assignment_domain": block["assignment_domain"],
                    **ARM_CONFIG[arm],
                }
            )
            predecessor = identity
    if len(runs) != 54 or sequence != 54:
        raise RuntimeError("frozen run count moved")
    manifest = {
        "schema": "m4-oars-v2-quality-primary-run-manifest-v1",
        "status": "SEALED_NOT_AUTHORIZED_FOR_SUBMISSION",
        "protocol_sha256": PROTOCOL_SHA256,
        "qualification_gate_sha256": GATE_SHA256,
        "run_count": 54,
        "block_count": 18,
        "sequence_policy": "strict_global_sequence_one_run_at_a_time",
        "gpus_per_run": 2,
        "scheduler_time_limit_seconds_per_run": 14400,
        "queue_deadline_override": None,
        "maximum_gpu_hours": 432,
        "complete_outcome_embargo": True,
        "automatic_retry": False,
        "automatic_replacement": False,
        "automatic_extension": False,
        "runs": runs,
    }
    analysis = {
        "schema": "m4-oars-v2-quality-primary-analysis-plan-v1",
        "status": "SEALED_BEFORE_ACQUISITION_AUTHORIZATION",
        "protocol_sha256": PROTOCOL_SHA256,
        "qualification_gate_sha256": GATE_SHA256,
        "run_manifest_sha256": None,
        "analysis_unit": "matched three-arm training-seed block",
        "primary": {
            "estimand": (
                "mean across 18 blocks of absolute-M4 minus FIFO terminal "
                "GSM8K accuracy"
            ),
            "interval": {
                "method": "two_sided_matched_block_t",
                "confidence": 0.95,
                "df": 17,
                "t_critical": 2.10981557783318,
            },
            "robustness": {
                "method": "inclusive_two_sided_exact_sign_flip",
                "enumeration_count": 262144,
            },
            "evidence_criterion": "95% interval lower bound above zero",
        },
        "secondary": [
            "reward-variance minus FIFO terminal GSM8K accuracy",
            "absolute-M4 minus reward-variance terminal GSM8K accuracy",
            "matched wall-time log ratios",
            "matched valid-actor-token-per-update log ratios",
            "registered opportunity and scheduler mechanism statistics",
        ],
        "multiplicity": (
            "One sole primary contrast. Secondary contrasts are descriptive "
            "and do not replace or modify the primary."
        ),
        "outcome_access_gate": (
            "Authenticate all 54 frozen identities and both terminal artifact "
            "copies per identity before opening any arm outcome."
        ),
        "incomplete_block_rule": (
            "Any missing or invalid arm makes its block incomplete; no automatic "
            "retry, replacement, or extension is allowed."
        ),
        "reporting": (
            "Report all numeric estimates, intervals, block outcomes, policy "
            "compliance, dose, and wall time without suppressing mixed results."
        ),
    }
    args.run_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    analysis["run_manifest_sha256"] = sha256(args.run_manifest)
    args.analysis_plan.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print(args.run_manifest, sha256(args.run_manifest))
    print(args.analysis_plan, sha256(args.analysis_plan))


if __name__ == "__main__":
    main()
