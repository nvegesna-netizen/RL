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

"""Freeze the OARS-v2 three-arm quality-primary study design."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path

DESIGN_SEED = 20260922
SIMULATION_SEED = 20260923
SIMULATION_DRAWS = 100_000
BLOCK_COUNT = 18
ARMS = ("fifo", "reward_variance", "absolute_m4")
T_CRITICAL = {
    12: 2.200985160082949,
    15: 2.1447866879169273,
    18: 2.1098155778331806,
}


def simulated_power(*, effect: float, sample_sd: float, count: int, seed: int) -> float:
    """Estimate paired-t superiority power with a frozen Gaussian simulation."""
    generator = random.Random(seed)
    critical = T_CRITICAL[count]
    passed = 0
    for _ in range(SIMULATION_DRAWS):
        values = [generator.gauss(effect, sample_sd) for _ in range(count)]
        mean = math.fsum(values) / count
        variance = math.fsum((value - mean) ** 2 for value in values) / (count - 1)
        lower = mean - critical * math.sqrt(variance / count)
        passed += lower > 0.0
    return passed / SIMULATION_DRAWS


def main() -> None:
    """Write the deterministic protocol from preserved predecessor evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor-analysis", type=Path, required=True)
    parser.add_argument("--systems-finding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predecessor = json.loads(args.predecessor_analysis.read_text())
    systems = json.loads(args.systems_finding.read_text())
    sample_sd = predecessor["secondary_terminal_gsm8k_accuracy"]["sample_sd"]
    if sample_sd != 0.1009742867702437:
        raise RuntimeError("predecessor paired-quality variance moved")
    if systems["status"] != "PASS_CONTROLLED_FRONTIER_SYSTEMS_READY":
        raise RuntimeError("controlled-frontier systems gate did not pass")

    permutations = list(itertools.permutations(ARMS)) * 3
    random.Random(DESIGN_SEED).shuffle(permutations)
    blocks = [
        {
            "assignment_domain": f"m4-oars-v2-quality-block-{index:02d}-v1",
            "assignment_seed": 20262400 + index,
            "block": index,
            "order": list(order),
            "training_seed": 20262300 + index,
        }
        for index, order in enumerate(permutations, start=1)
    ]
    order_counts = {
        "-".join(order): permutations.count(order)
        for order in itertools.permutations(ARMS)
    }
    power = {
        str(count): {
            str(effect): simulated_power(
                effect=effect,
                sample_sd=sample_sd,
                count=count,
                seed=SIMULATION_SEED + count * 1000 + round(effect * 1000),
            )
            for effect in (0.05, 0.075, 0.10)
        }
        for count in T_CRITICAL
    }
    protocol = {
        "schema": "m4-oars-v2-quality-primary-protocol-v1",
        "status": "FROZEN_BEFORE_ACTUATION_IMPLEMENTATION_COMMIT",
        "user_direction": "do it",
        "objective": (
            "Estimate whether absolute-M4 scheduling changes terminal GSM8K "
            "accuracy relative to controlled FIFO under a shared eight-candidate "
            "frontier, with reward variance as an active mechanistic comparator."
        ),
        "prerequisites": {
            "controlled_frontier_systems_gate": systems["status"],
            "reward_variance_actuation_qualification": "REQUIRED_AND_EXCLUDED",
            "absolute_m4_actuation_qualification": "REQUIRED_AND_EXCLUDED",
            "acquisition_launch_requires_both_qualifications": True,
        },
        "arms": {
            "fifo": "controlled WeightFIFO at a common eight-ready-group frontier",
            "reward_variance": (
                "OARS-v2 exact four-of-eight selection maximizing reward-variance "
                "risk within the 0.98-1.02 FIFO token band"
            ),
            "absolute_m4": (
                "OARS-v2 exact four-of-eight selection maximizing absolute M4 "
                "risk within the 0.98-1.02 FIFO token band"
            ),
        },
        "design": {
            "analysis_unit": "matched three-arm training-seed block",
            "block_count": BLOCK_COUNT,
            "total_run_count": BLOCK_COUNT * len(ARMS),
            "model": "meta-llama/Llama-3.2-1B-Instruct",
            "workload": "gsm8k",
            "updates": 64,
            "nodes_per_run": 1,
            "gpus_per_run": 2,
            "selection_candidate_watermark": 8,
            "selection_cardinality": 4,
            "maximum_staleness_versions": 1,
            "execution": "all 54 arms serial in the frozen global order",
            "order_randomization_seed": DESIGN_SEED,
            "balanced_order_counts": order_counts,
            "blocks": blocks,
        },
        "estimands": {
            "primary": (
                "mean matched-block difference in terminal GSM8K accuracy, "
                "absolute_m4 minus fifo"
            ),
            "secondary": [
                "reward_variance minus fifo terminal GSM8K accuracy",
                "absolute_m4 minus reward_variance terminal GSM8K accuracy",
                "matched log wall-time ratios",
                "matched log valid-actor-token-per-update ratios",
                "policy disagreement, overlap, and retained opportunity summaries",
            ],
        },
        "analysis": {
            "primary_interval": (
                "two-sided 95% matched-block t interval over 18 absolute_m4-minus-"
                "fifo differences"
            ),
            "primary_evidence_criterion": "the primary 95% interval lower bound exceeds zero",
            "primary_robustness": (
                "inclusive exact two-sided sign-flip randomization p-value over "
                "all 2^18 matched-block sign assignments"
            ),
            "secondary_intervals": (
                "two-sided 95% matched-block t intervals, reported descriptively "
                "without replacing or modifying the primary"
            ),
            "reporting": (
                "Always report estimates, intervals, all block outcomes, policy "
                "compliance, dose, and wall time; do not suppress mixed results."
            ),
        },
        "power": {
            "planning_source": (
                "The authenticated predecessor v1 OARS/FIFO study's terminal-"
                "quality paired SD; its mean is not assumed as the new effect."
            ),
            "planning_paired_sd": sample_sd,
            "simulation_seed": SIMULATION_SEED,
            "simulation_draws_per_cell": SIMULATION_DRAWS,
            "two_sided_alpha": 0.05,
            "simulated_power": power,
            "chosen_design": (
                "18 blocks provide about 85% sensitivity for a +0.075 accuracy "
                "effect under the planning variance."
            ),
        },
        "qualification": {
            "order": ["reward_variance", "absolute_m4"],
            "training_seeds": {
                "reward_variance": 20262231,
                "absolute_m4": 20262232,
            },
            "updates": 64,
            "excluded_from_outcome_analysis": True,
            "required_gates": [
                "exactly 64 complete learner steps and decisions",
                "exactly eight candidates and 70 combinations per decision",
                "actual selections exactly match the configured scorer proposal",
                "four unique ready groups selected per decision",
                "zero skips and zero fallbacks",
                "complete candidate-excess and stale-replenishment accounting",
                "combined observer/controller duty at most one percent",
                "runtime at most 14400 seconds",
            ],
        },
        "outcome_embargo": {
            "interim_outcome_access": "PROHIBITED",
            "completion_gate": (
                "All 54 frozen identities and duplicate terminal artifact copies "
                "must authenticate before any arm outcome is opened."
            ),
            "incomplete_block_rule": (
                "A missing or invalid arm makes its block incomplete. No automatic "
                "retry, replacement, or extension is allowed."
            ),
        },
        "claim_boundary": {
            "allowed": (
                "A causal scheduler effect on terminal GSM8K accuracy in the tested "
                "Llama-3.2-1B, 64-update, two-GPU controlled-frontier setting."
            ),
            "forbidden": [
                "generalization beyond the tested model, workload, horizon, or hardware",
                "calling proposal disagreement a quality effect",
                "pooling either actuation qualification with acquisition outcomes",
                "changing the primary endpoint after any outcome access",
                "treating pipeline status as a scientific endpoint",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps(protocol["power"], indent=2, sort_keys=True))
    print(json.dumps(protocol["design"]["balanced_order_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
