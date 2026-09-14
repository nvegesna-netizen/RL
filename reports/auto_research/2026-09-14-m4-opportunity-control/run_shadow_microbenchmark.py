#!/usr/bin/env python3
"""Benchmark the frozen maximum-cardinality OARS exact selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import nemo_rl.algorithms.async_utils.opportunity_at_risk as oars  # noqa: E402


def sha256(path: Path) -> str:
    """Return the SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[int], probability: float) -> int:
    """Return the nearest-rank percentile for positive observations."""
    ordered = sorted(values)
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[rank - 1]


def parse_args() -> argparse.Namespace:
    """Parse benchmark controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=250)
    parser.add_argument("--warmup", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    """Run a deterministic 25-choose-4 benchmark and emit canonical JSON."""
    args = parse_args()
    if args.trials < 1 or args.warmup < 0:
        raise ValueError("trials must be positive and warmup must be nonnegative")
    candidates = tuple(
        oars.OpportunityCandidate(
            group_id=f"g{index:02d}",
            start_weight_version=1,
            l1=float(index + 1),
            valid_actor_tokens=100,
        )
        for index in range(25)
    )
    kwargs = {
        "baseline_group_ids": tuple(group.group_id for group in candidates[:4]),
        "current_train_weight": 2,
        "service_budget_multiplier": 1.02,
    }
    for _ in range(args.warmup):
        oars.select_baseline_budgeted_opportunity(candidates, **kwargs)

    elapsed_ns: list[int] = []
    selections = []
    for _ in range(args.trials):
        started_ns = time.perf_counter_ns()
        selection = oars.select_baseline_budgeted_opportunity(candidates, **kwargs)
        elapsed_ns.append(time.perf_counter_ns() - started_ns)
        selections.append(selection)

    first = selections[0]
    if any(selection != first for selection in selections):
        raise RuntimeError("selector was nondeterministic across benchmark trials")
    expected_combinations = math.comb(25, 4)
    if first.combination_count != expected_combinations:
        raise RuntimeError("selector did not enumerate the frozen maximum choice set")
    if first.proposed_valid_actor_tokens > first.token_budget:
        raise RuntimeError("selector violated its frozen service budget")

    source_path = Path(oars.__file__).resolve()
    result = {
        "schema": "m4-opportunity-control-shadow-microbenchmark-v1",
        "status": "COMPLETE_CPU_MICROBENCHMARK",
        "candidate_groups": 25,
        "batch_groups": 4,
        "combinations_per_trial": expected_combinations,
        "all_combinations_feasible": (
            first.feasible_combination_count == expected_combinations
        ),
        "deterministic_across_trials": True,
        "service_budget_satisfied": True,
        "proposed_group_ids": list(first.proposed_group_ids),
        "trials": args.trials,
        "warmup_trials": args.warmup,
        "elapsed_ns": {
            "minimum": min(elapsed_ns),
            "median": int(statistics.median(elapsed_ns)),
            "p95_nearest_rank": percentile(elapsed_ns, 0.95),
            "maximum": max(elapsed_ns),
        },
        "environment": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "implementation_path": str(source_path.relative_to(REPO_ROOT)),
        "implementation_sha256": sha256(source_path),
        "benchmark_script_sha256": sha256(Path(__file__).resolve()),
        "claim_boundary": (
            "local_cpu_selector_latency_not_controller_end_to_end_latency_or_"
            "training_throughput"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
