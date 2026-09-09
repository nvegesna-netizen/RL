#!/usr/bin/env python3
"""Describe the registered five-second dose relative to learner update cadence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COMMON_START = 8
COMMON_END = 407
DELAY_SECONDS = 5.0
LABELS = {
    "qwen3_0p6b_openmath": ("Qwen3-0.6B", "OpenMath"),
    "qwen3_1p7b_openmath": ("Qwen3-1.7B", "OpenMath"),
    "qwen3_0p6b_gsm8k": ("Qwen3-0.6B", "GSM8K"),
    "qwen3_1p7b_gsm8k": ("Qwen3-1.7B", "GSM8K"),
    "qwen3_0p6b_numinamath": ("Qwen3-0.6B", "NuminaMath"),
    "qwen3_1p7b_numinamath": ("Qwen3-1.7B", "NuminaMath"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def cadence(path: Path) -> dict[str, object]:
    times: dict[int, int] = {}
    run_ids: set[str] = set()
    clock_ids: set[str] = set()
    with path.open() as source:
        for line in source:
            if '"stage": "learner_version_advanced"' not in line:
                continue
            row = json.loads(line)
            version = row["learner_weight_version"]
            assert isinstance(version, int) and version not in times
            times[version] = row["timestamp_ns"]
            run_ids.add(row["run_id"])
            clock_ids.add(row["clock_domain_id"])
    assert len(run_ids) == 1 and len(clock_ids) == 1
    assert all(version in times for version in range(COMMON_START, COMMON_END + 2))
    seconds = [
        (times[version + 1] - times[version]) / 1_000_000_000
        for version in range(COMMON_START, COMMON_END + 1)
    ]
    assert len(seconds) == 400 and min(seconds) > 0
    median = quantile(seconds, 0.5)
    return {
        "interval_count": len(seconds),
        "mean_seconds": sum(seconds) / len(seconds),
        "median_seconds": median,
        "p25_seconds": quantile(seconds, 0.25),
        "p75_seconds": quantile(seconds, 0.75),
        "p05_seconds": quantile(seconds, 0.05),
        "p95_seconds": quantile(seconds, 0.95),
        "five_seconds_over_median_cadence": DELAY_SECONDS / median,
    }


def main() -> None:
    synthesis = json.loads((HERE / "six_cell_synthesis.json").read_bytes())
    cells = {}
    for name, labels in LABELS.items():
        evidence = synthesis["evidence"][name]
        lifecycle = ROOT / evidence["lifecycle_path"]
        assert sha256(lifecycle) == evidence["lifecycle_sha256"]
        cells[name] = {
            "model": labels[0],
            "workload": labels[1],
            "lifecycle_path": evidence["lifecycle_path"],
            "lifecycle_sha256": evidence["lifecycle_sha256"],
            **cadence(lifecycle),
        }
    result = {
        "schema": "m4-publication-update-cadence-sensitivity-v1",
        "analysis_role": "retrospective_descriptive_not_a_registered_causal_estimand",
        "delay_seconds": DELAY_SECONDS,
        "preceding_version_window": [COMMON_START, COMMON_END],
        "interval_definition": "timestamp(version+1)-timestamp(version) for preceding learner versions 8-407",
        "cells": cells,
        "limitations": [
            "The delay-to-median ratio is descriptive and was not a registered endpoint.",
            "Cadence is an observed post-assignment system property and is not used to renormalize causal effects.",
            "The fixed five-second intervention remains the treatment in every registered analysis.",
        ],
    }
    output = HERE / "cadence_results.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Update-cadence sensitivity",
        "",
        "This retrospective descriptive analysis places the fixed five-second",
        "treatment on the observed learner-update time scale. It does not redefine",
        "the treatment or any registered causal estimand.",
        "",
        "| Model | Workload | Median update (s) | IQR (s) | 5 s / median update |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for value in cells.values():
        lines.append(
            f"| {value['model']} | {value['workload']} | "
            f"{value['median_seconds']:.3f} | "
            f"[{value['p25_seconds']:.3f}, {value['p75_seconds']:.3f}] | "
            f"{value['five_seconds_over_median_cadence']:.3f} |"
        )
    lines += [
        "",
        "Each cell contributes exactly 400 authenticated inter-update intervals:",
        "the time from learner version `v` to `v+1` for preceding versions 8--407.",
        "The ratios are context descriptors only. Using them to rescale the effect",
        "after observing outcomes would change the estimand and is not done.",
        "",
        f"Result SHA-256: `{sha256(output)}`",
    ]
    (HERE / "cadence_results.md").write_text("\n".join(lines) + "\n")
    print("UPDATE_CADENCE_ANALYSIS_PASS")
    print(f"cadence_results_sha256={sha256(output)}")


if __name__ == "__main__":
    main()
