#!/usr/bin/env python3
"""Credential-free verification and synthetic M4 replay (stdlib only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    return json.loads(path.read_bytes())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest() -> None:
    manifest = load(ROOT / "MANIFEST.json")
    expected = {entry["path"] for entry in manifest["files"]}
    actual = {
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file() and path.name != "MANIFEST.json" and "__pycache__" not in path.parts
    }
    assert actual == expected, (sorted(actual - expected), sorted(expected - actual))
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        assert path.stat().st_size == entry["bytes"]
        assert sha256(path) == entry["sha256"]


def published_checks() -> dict[str, object]:
    result = load(ROOT / "data/published_results.json")
    cells = result["cell_results"]
    assert len(cells) == 6
    assert sum(cell["assignment_count"] for cell in cells.values()) == 43_756
    assert all(cell["terminal_missing_count"] == 0 for cell in cells.values())
    synthesis = result["dependency_aware_synthesis"]
    assert math.isclose(synthesis["hac_correlation"], 0.3793770869832136)
    assert math.isclose(synthesis["bootstrap_correlation"], 0.40530226396440766)
    assert math.isclose(synthesis["global_heterogeneity_p_value"], 0.0031390066345424925)
    extension = result["llama_v5_extension"]["combined_workloads"]
    assert math.isclose(extension["openmath"]["identification_interval"][0], 0.39907889029747823)
    assert math.isclose(extension["gsm8k"]["identification_interval"][0], 0.34607413746273974)
    assert sum(value["assignment_count"] for value in extension.values()) == 28_712
    extension_3b = result["llama_3b_extension"]
    combined_3b = extension_3b["combined_workloads"]
    assert math.isclose(combined_3b["openmath"]["identification_interval"][0], 0.2620625084474564)
    assert math.isclose(combined_3b["gsm8k"]["identification_interval"][0], 0.21352290408701716)
    assert combined_3b["openmath"]["conclusion"] == "MATERIAL"
    assert combined_3b["gsm8k"]["conclusion"] == "INCONCLUSIVE"
    assert all(value["simultaneous_two_contrast_interval"][1] < 0 for value in extension_3b["secondary_3b_minus_1b"].values())
    downstream = result["downstream_quality"]
    assert downstream["block_count"] == 16
    assert downstream["run_count"] == 32
    assert downstream["conclusion"] == "INCONCLUSIVE"
    assert math.isclose(downstream["estimate"], -0.044921875)
    assert downstream["student_interval_95"] == [-0.13394335582884473, 0.04409960582884473]
    secondary = downstream["secondary"]
    assert math.isclose(secondary["normalized_realized_opportunity_loss"]["estimate_mixed_d5_minus_immediate"], 0.007385549503757312)
    assert math.isclose(secondary["opportunity_loss_accuracy_diagnostic"]["pearson_correlation"], -0.09593746695743842)
    oars = result["oars_confirmatory"]
    assert oars["classification"] == "INCONCLUSIVE"
    assert oars["primary"]["n"] == 10
    assert math.isclose(oars["primary"]["mean"], 302.028300505341)
    assert math.isclose(oars["secondary_terminal_gsm8k_accuracy"]["mean"], 0.09977255496588325)
    assert math.isclose(oars["wall_time_ratio"]["geometric_mean"], 0.7667760632976284)
    assert oars["gates"]["policy_compliance"]
    assert not oars["gates"]["primary_superiority"]
    discriminant = result["metric_discriminant"]
    assert discriminant["assignment_count"] == 106_653
    assert discriminant["summary"]["positive_opportunity_lost_count"] == 23_254
    provenance = load(ROOT / "data/provenance.json")
    assert set(provenance["acquisitions"]) == {f"A{i}" for i in range(1, 15)}
    assert sum(item["full_window_assignments"] for item in provenance["acquisitions"].values()) == 106_653
    assert all(len(item["terminal_artifact_sha256"]) == 64 for item in provenance["acquisitions"].values())
    return {
        "common_window_assignments": 43_756,
        "full_window_assignments": 106_653,
        "llama_1b_extension_assignments": 28_712,
        "llama_3b_extension_assignments": 28_788,
        "downstream_quality_blocks": downstream["block_count"],
        "downstream_quality_runs": downstream["run_count"],
        "oars_pairs": oars["primary"]["n"],
        "oars_runs": 20,
        "metric_discriminant_assignments": discriminant["assignment_count"],
        "hac_correlation": synthesis["hac_correlation"],
        "bootstrap_correlation": synthesis["bootstrap_correlation"],
    }


def cell_estimates(rows: list[dict[str, object]]) -> tuple[dict[str, float], dict[str, list[float]]]:
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        grouped[row["cell"]][int(row["start_version"])][row["arm"]].append(row)
    estimates = {}
    version_effects = {}
    for cell, versions in grouped.items():
        effects = []
        for version in sorted(versions):
            arms = versions[version]
            pooled_q = sum(float(r["Q"]) for arm in arms.values() for r in arm) / sum(len(arm) for arm in arms.values())
            d5 = sum(float(r["Q"]) * int(r["D"]) for r in arms["d5"]) / len(arms["d5"])
            control = sum(float(r["Q"]) * int(r["D"]) for r in arms["control"]) / len(arms["control"])
            effects.append((d5 - control) / pooled_q)
        estimates[cell] = sum(effects) / len(effects)
        version_effects[cell] = effects
    return estimates, version_effects


def hac_se(values: list[float], lag: int = 1) -> float:
    n = len(values)
    mean = sum(values) / n
    centered = [value - mean for value in values]
    long_run = sum(value * value for value in centered) / n
    for offset in range(1, min(lag, n - 1) + 1):
        covariance = sum(centered[i] * centered[i - offset] for i in range(offset, n)) / n
        long_run += 2 * (1 - offset / (lag + 1)) * covariance
    return math.sqrt(max(long_run, 0.0) / n)


def circular_bootstrap(values: list[float], draws: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    observed = sum(values) / n
    shifts = []
    for _ in range(draws):
        sampled = []
        while len(sampled) < n:
            start = rng.randrange(n)
            sampled.extend((values[start], values[(start + 1) % n]))
        shifts.append(sum(sampled[:n]) / n - observed)
    return shifts


def assert_summary_close(actual, expected) -> None:
    if isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_summary_close(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            assert_summary_close(actual_item, expected_item)
    else:
        assert actual == expected


def synthetic_replay() -> dict[str, object]:
    rows = [json.loads(line) for line in (ROOT / "synthetic/miniature_ledger.jsonl").read_text().splitlines()]
    estimates, effects = cell_estimates(rows)
    shifts = {name: circular_bootstrap(values, 1_000, 7100 + index) for index, (name, values) in enumerate(sorted(effects.items()))}
    scale = {
        workload: estimates[f"qwen3_1p7b_{workload}"] - estimates[f"qwen3_0p6b_{workload}"]
        for workload in ("openmath", "gsm8k", "numinamath")
    }
    interactions = {
        "gsm8k_minus_openmath": scale["gsm8k"] - scale["openmath"],
        "gsm8k_minus_numinamath": scale["gsm8k"] - scale["numinamath"],
    }
    paired = []
    for index in range(1_000):
        gsm = shifts["qwen3_1p7b_gsm8k"][index] - shifts["qwen3_0p6b_gsm8k"][index]
        openmath = shifts["qwen3_1p7b_openmath"][index] - shifts["qwen3_0p6b_openmath"][index]
        numina = shifts["qwen3_1p7b_numinamath"][index] - shifts["qwen3_0p6b_numinamath"][index]
        paired.append((gsm - openmath, gsm - numina))
    mean_x = sum(x for x, _ in paired) / len(paired)
    mean_y = sum(y for _, y in paired) / len(paired)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in paired) / (len(paired) - 1)
    sd_x = math.sqrt(sum((x - mean_x) ** 2 for x, _ in paired) / (len(paired) - 1))
    sd_y = math.sqrt(sum((y - mean_y) ** 2 for _, y in paired) / (len(paired) - 1))
    summary = {
        "row_count": len(rows),
        "cell_estimates": estimates,
        "cell_hac_se": {name: hac_se(values) for name, values in effects.items()},
        "model_scale_effects": scale,
        "reference_interactions": interactions,
        "shared_reference_bootstrap_correlation": covariance / (sd_x * sd_y),
    }
    expected = load(ROOT / "synthetic/expected_summary.json")
    assert_summary_close(summary, expected)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", required=True)
    parser.parse_args()
    verify_manifest()
    published = published_checks()
    synthetic = synthetic_replay()
    output = {"status": "PASS", "published": published, "synthetic": synthetic}
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
