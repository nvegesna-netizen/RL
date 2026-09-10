# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Deterministic capacity and conservative power planning for Llama M4 V5."""

from __future__ import annotations

import json
import math
import random
from statistics import NormalDist


COUNTS = {
    "openmath": (22,14,18,14,15,17,11,16,15,16,12,19,9,17,15,22,8,24,21,5,25,17,10,20,16,16,18,15,16,16,16,18,12,23,8,25,16,17,16,16,16,16,16,17,17,16,16,16),
    "gsm8k": (43,25,43,25,43,25,43,25,43,25,43,25,43,25,43,25,43,25,43,25,43,25,43,25,43,25,43,25,43,25,43,25,43,25,43,25,43,25,35,21,12,10,22,11,21,16,16,17),
}
SEEDS = {"openmath": 20261019, "gsm8k": 20261020}


def lower(values: list[float], alpha: float = 0.05) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(alpha * len(ordered)) - 1)]


def capacity(counts: tuple[int, ...], seed: int, draws: int = 100_000) -> dict[str, float]:
    rng = random.Random(seed)
    combined = []
    for _ in range(draws):
        total = 0
        for _replicate in range(2):
            for _block in range(6):
                start = rng.randrange(48)
                total += sum(counts[(start + offset) % 48] for offset in range(8))
        combined.append(total * 400 / 48)
    point = 2 * sum(counts) * 400 / 48
    bound = lower(combined)
    return {
        "combined_point_assignments": point,
        "combined_lower_95_assignments": bound,
        "lower_margin_over_6900": bound - 6900,
        "degradation_to_6900": 1 - 6900 / bound,
    }


def one_sided_power(effect: float, null: float, standard_error: float) -> float:
    return NormalDist().cdf((effect - null) / standard_error - NormalDist().inv_cdf(0.95))


def main() -> None:
    result = {
        "schema": "m4-llama-v5-replicated-acquisition-planning-v1",
        "capacity": {
            name: capacity(COUNTS[name], SEEDS[name]) for name in sorted(COUNTS)
        },
        "power": {},
    }
    assumptions = {
        "openmath_materiality": (0.301, 0.2, 0.01880691796),
        "gsm8k_positive_direction": (0.185, 0.0, 0.01565807921),
        "openmath_minus_gsm8k": (0.116, 0.0, 0.02447193511),
    }
    for name, (effect, null, historical_se) in assumptions.items():
        base = historical_se / math.sqrt(2)
        conservative = base * 1.25
        result["power"][name] = {
            "planning_effect": effect,
            "null": null,
            "historical_single_run_se_already_inflated_25pct": historical_se,
            "two_run_se": base,
            "two_run_se_with_additional_25pct_run_heterogeneity_inflation": conservative,
            "one_sided_power": one_sided_power(effect, null, conservative),
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
