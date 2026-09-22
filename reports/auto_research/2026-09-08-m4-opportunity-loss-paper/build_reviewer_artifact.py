#!/usr/bin/env python3
"""Build a deterministic, anonymous M4 reviewer artifact using only stdlib."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path


HERE = Path(__file__).resolve().parent

CELL_IDS = {
    "qwen3_0p6b_openmath": "A1",
    "qwen3_1p7b_openmath": "A2",
    "qwen3_0p6b_gsm8k": "A3",
    "qwen3_1p7b_gsm8k": "A4",
    "qwen3_0p6b_numinamath": "A5",
    "qwen3_1p7b_numinamath": "A6",
    "llama3p2_1b_openmath_r1": "A7",
    "llama3p2_1b_openmath_r2": "A8",
    "llama3p2_1b_gsm8k_r1": "A9",
    "llama3p2_1b_gsm8k_r2": "A10",
    "llama3p2_3b_openmath_r1": "A11",
    "llama3p2_3b_openmath_r2": "A12",
    "llama3p2_3b_gsm8k_r1": "A13",
    "llama3p2_3b_gsm8k_r2": "A14",
}

EXTERNAL_ARTIFACTS = {
    "A1": ("07cddc489ea20055161f9c435d44d0a52b91113fda9eac16472bdb531dc6ffb6", 54_950_561),
    "A2": ("022e4d18f2eedd2339c3c7eb7decf6e21f929761de46bcd0f85a73f0a2700ac7", 131_003_667),
    "A3": ("6c2ebeb2b2bf67df16be815f372ea87cde8eeff4d23e63291d8a98ac94659075", 137_287_850),
    "A4": ("966dfbf60548e5fd791d59d3baa2bc3b3a1b8d6b0b1a15de1ef510fc618720ca", 134_908_947),
    "A5": ("ad7c556f435b0f9e30a65c201680bf6dab74e077af16e38f299bfc04af969c35", 129_888_875),
    "A6": ("6aa61968730833387a15b45a6d0f2be541c6f84b6d4f16a29e7dbf583981fc9b", 128_543_493),
    "A7": ("7fc1d62dba8bf82777e9d86abc3e205afa5f5012ce067082fd3a2babe88139b6", 66_178_095),
    "A8": ("37da83c526bef6cf13b36090d21dd49a9d73a768422e3fea9381521eb8e71a33", 65_947_753),
    "A9": ("c321dcf94a138d2a289142b0991c452be848d28dedf5f5c468fba78cd85ffa59", 66_467_861),
    "A10": ("fcb236655ed1554215e370cca6d21fe5873b385e4f4b1576cdc27ff2dbe4735b", 67_119_843),
    "A11": ("b79b7d1bcd861bdacf3d161bbdc5fe236edc35da41f427d0adf20a1895ddcd38", 66_561_497),
    "A12": ("034d6e57440593a48a28f705534474b1c36c8aeb0170c22e54d869f4e33a9aaa", 66_110_870),
    "A13": ("6714f77466a5d3e3c854339f69d19ac4d8312276cc3627c25fe3d8ef79b26163", 67_019_231),
    "A14": ("e535f0ce6c1fdb4f1863eca4142596f4543190d53dfd70cee7ddaf35d25a5920", 67_286_792),
}

PROTOCOL_SOURCES = {
    "A1": HERE.parent / "2026-09-02-m4-opportunity-loss-followup/protocol_config.json",
    "A2": HERE.parent / "2026-09-03-m4-opportunity-loss-transport/qwen3-1p7b-confirmatory-design/protocol_config.json",
    "A3": HERE.parent / "2026-09-07-m4-opportunity-loss-grid-completion/protocol_config.json",
    "A4": HERE.parent / "2026-09-04-m4-opportunity-loss-workload-transport/qwen3-1p7b-gsm8k-confirmatory-design/protocol_config_r2.json",
    "A5": HERE.parent / "2026-09-07-m4-opportunity-loss-numinamath-generalization/protocol_config.json",
    "A6": HERE.parent / "2026-09-07-m4-opportunity-loss-numinamath-generalization/protocol_config.json",
    "A7": HERE.parent / "2026-09-09-m4-llama-lifecycle-derived-transport/v5_protocol_config.json",
    "A8": HERE.parent / "2026-09-09-m4-llama-lifecycle-derived-transport/v5_protocol_config.json",
    "A9": HERE.parent / "2026-09-09-m4-llama-lifecycle-derived-transport/v5_protocol_config.json",
    "A10": HERE.parent / "2026-09-09-m4-llama-lifecycle-derived-transport/v5_protocol_config.json",
    "A11": HERE.parent / "2026-09-10-m4-llama3p2-3b-size-extension/prospective_protocol.json",
    "A12": HERE.parent / "2026-09-10-m4-llama3p2-3b-size-extension/prospective_protocol.json",
    "A13": HERE.parent / "2026-09-10-m4-llama3p2-3b-size-extension/prospective_protocol.json",
    "A14": HERE.parent / "2026-09-10-m4-llama3p2-3b-size-extension/prospective_protocol.json",
}

PROTOCOL_HASHES = {
    "A1": "6631a4ca8cfc6010150d1c135a1f5821c717f0dea189a3309c5f0f58f9b0709c",
    "A2": "eef87a5e26f2cc1a39428628b0c28288b17a297f49d449a80ed9699f9ee17cb7",
    "A3": "b6682795071d12a5d6b3eddcdf73536e045949f08ce881885dd330fba9885db6",
    "A4": "3413bde3718563157cee5d504f174710f65406dd7f1effd8ad2755a03d73b69e",
    "A5": "bbe2c07211952e3e46d4511524f68e864606a7c63f271ae8e064263044dad0c8",
    "A6": "bbe2c07211952e3e46d4511524f68e864606a7c63f271ae8e064263044dad0c8",
    "A7": "56a3311f78a0ac12cb9dc7ecdeee1a6fc0c5ace889799ebfaba1faaa0e610521",
    "A8": "56a3311f78a0ac12cb9dc7ecdeee1a6fc0c5ace889799ebfaba1faaa0e610521",
    "A9": "56a3311f78a0ac12cb9dc7ecdeee1a6fc0c5ace889799ebfaba1faaa0e610521",
    "A10": "56a3311f78a0ac12cb9dc7ecdeee1a6fc0c5ace889799ebfaba1faaa0e610521",
    "A11": "4f2a5e496030e836b3449697928f4f363d7d36e3cec668105b2cdcc701b9814c",
    "A12": "4f2a5e496030e836b3449697928f4f363d7d36e3cec668105b2cdcc701b9814c",
    "A13": "4f2a5e496030e836b3449697928f4f363d7d36e3cec668105b2cdcc701b9814c",
    "A14": "4f2a5e496030e836b3449697928f4f363d7d36e3cec668105b2cdcc701b9814c",
}


README = """# M4 anonymous reviewer artifact

This credential-free bundle verifies the compact published M4 results and
exercises the estimand and dependency-aware interaction path on a synthetic
miniature ledger. It requires only Python 3.10+ and the standard library.

Run from the extracted directory:

```sh
python3 analysis/replay.py --verify
python3 analysis/render_figures.py --verify
python3 -m unittest discover -s tests -v
```

`data/published_results.json` contains the six-cell Qwen common-window results,
the four replicated Llama size/workload endpoints, the separate 16-pair
downstream-quality result, and the 10-pair OARS/FIFO policy result after removal
of private filesystem paths. It also includes the compact 106,653-assignment
Qwen-plus-Llama metric-discriminant summary; raw empirical ledgers remain external.
`data/provenance.json` binds opaque
acquisition IDs A1--A14 to the frozen protocols, compact records, and external
terminal archives by SHA-256. The large empirical ledgers are not included;
therefore this bundle verifies compact results but does not independently
re-estimate the empirical cells. The synthetic ledger contains no experimental
observations and is used only to test the analysis structure.
"""

REPLAY = r'''#!/usr/bin/env python3
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
    assert "classification" not in oars
    assert oars["pair_count"] == 10
    assert oars["run_count"] == 20
    assert oars["primary"]["n"] == 10
    assert math.isclose(oars["primary"]["mean"], 302.028300505341)
    assert math.isclose(oars["secondary_terminal_gsm8k_accuracy"]["mean"], 0.09977255496588325)
    assert math.isclose(oars["wall_time_ratio"]["geometric_mean"], 0.7667760632976284)
    assert oars["policy_compliance"]
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
'''

FIGURES = r'''#!/usr/bin/env python3
"""Render and verify publication SVGs from compact publication results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def forest(cells: dict[str, dict[str, object]]) -> str:
    labels = [
        ("0.6B - OpenMath", "qwen3_0p6b_openmath"),
        ("1.7B - OpenMath", "qwen3_1p7b_openmath"),
        ("0.6B - GSM8K", "qwen3_0p6b_gsm8k"),
        ("1.7B - GSM8K", "qwen3_1p7b_gsm8k"),
        ("0.6B - NuminaMath", "qwen3_0p6b_numinamath"),
        ("1.7B - NuminaMath", "qwen3_1p7b_numinamath"),
        ("Llama 1B - OpenMath (2 reps)", "llama3p2_1b_openmath"),
        ("Llama 1B - GSM8K (2 reps)", "llama3p2_1b_gsm8k"),
        ("Llama 3B - OpenMath (2 reps)", "llama3p2_3b_openmath"),
        ("Llama 3B - GSM8K (2 reps)", "llama3p2_3b_gsm8k"),
    ]
    left, right, top, row = 255, 850, 30, 55
    x_min, x_max = 0.08, 0.48
    x = lambda value: left + (value - x_min) / (x_max - x_min) * (right - left)
    height = top + row * len(labels) + 70
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" viewBox="0 0 900 {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Helvetica,Arial,sans-serif;fill:#202124}.label{font-size:15px}.axis{font-size:13px}</style>',
    ]
    for tick in (0.1, 0.2, 0.3, 0.4):
        parts.append(f'<line x1="{x(tick):.1f}" y1="42" x2="{x(tick):.1f}" y2="{height - 45}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x(tick):.1f}" y="{height - 20}" text-anchor="middle" class="axis">{tick:.2f}</text>')
    parts.append(f'<line x1="{x(0.2):.1f}" y1="42" x2="{x(0.2):.1f}" y2="{height - 45}" stroke="#d93025" stroke-width="2" stroke-dasharray="6 5"/>')
    for index, (label, name) in enumerate(labels):
        y = top + index * row
        record = cells[name]
        low, high = record["outer_envelope"]
        estimate = record["estimate"]
        color = "#e67e22" if "llama3p2_3b" in name else "#00897b" if "llama" in name else "#1769aa" if "0p6b" in name else "#7b1fa2"
        parts.append(f'<text x="20" y="{y + 5}" class="label">{label}</text>')
        parts.append(f'<line x1="{x(low):.1f}" y1="{y}" x2="{x(high):.1f}" y2="{y}" stroke="{color}" stroke-width="4"/>')
        for endpoint in (low, high):
            parts.append(f'<line x1="{x(endpoint):.1f}" y1="{y - 7}" x2="{x(endpoint):.1f}" y2="{y + 7}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<circle cx="{x(estimate):.1f}" cy="{y}" r="7" fill="{color}"/>')
    parts.append(f'<text x="{x(0.2) + 6:.1f}" y="{height - 49}" class="axis" fill="#d93025">registered materiality threshold</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def interactions(synthesis: dict[str, object]) -> str:
    items = [("GSM8K - OpenMath", "gsm8k_minus_openmath"), ("GSM8K - NuminaMath", "gsm8k_minus_numinamath")]
    left, right = 220, 850
    x_min, x_max = -0.16, 0.04
    x = lambda value: left + (value - x_min) / (x_max - x_min) * (right - left)
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="220" viewBox="0 0 900 220">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Helvetica,Arial,sans-serif;fill:#202124}.label{font-size:15px}.axis{font-size:13px}</style>',
    ]
    for tick in (-0.15, -0.10, -0.05, 0.0):
        parts.append(f'<line x1="{x(tick):.1f}" y1="15" x2="{x(tick):.1f}" y2="180" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x(tick):.1f}" y="205" text-anchor="middle" class="axis">{tick:.2f}</text>')
    parts.append(f'<line x1="{x(0.0):.1f}" y1="15" x2="{x(0.0):.1f}" y2="180" stroke="#202124" stroke-width="2"/>')
    for index, (label, key) in enumerate(items):
        y = 60 + index * 80
        record = synthesis["reference_interactions"][key]
        low, high = record["simultaneous_bootstrap_interval"]
        estimate = record["estimate"]
        parts.append(f'<text x="20" y="{y + 5}" class="label">{label}</text>')
        parts.append(f'<line x1="{x(low):.1f}" y1="{y}" x2="{x(high):.1f}" y2="{y}" stroke="#00897b" stroke-width="4"/>')
        for endpoint in (low, high):
            parts.append(f'<line x1="{x(endpoint):.1f}" y1="{y - 7}" x2="{x(endpoint):.1f}" y2="{y + 7}" stroke="#00897b" stroke-width="2"/>')
        parts.append(f'<circle cx="{x(estimate):.1f}" cy="{y}" r="7" fill="#00695c"/>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = json.loads((ROOT / "data/published_results.json").read_bytes())
    cells = dict(result["cell_results"])
    for workload in ("openmath", "gsm8k"):
        record = result["llama_v5_extension"]["combined_workloads"][workload]
        cells[f"llama3p2_1b_{workload}"] = {
            "estimate": record["identification_interval"][0],
            "outer_envelope": record["confidence_envelope"],
        }
        record_3b = result["llama_3b_extension"]["combined_workloads"][workload]
        cells[f"llama3p2_3b_{workload}"] = {
            "estimate": record_3b["identification_interval"][0],
            "outer_envelope": record_3b["confidence_envelope"],
        }
    outputs = {
        ROOT / "figures/cell_forest_plot.svg": forest(cells),
        ROOT / "figures/interaction_plot.svg": interactions(result["dependency_aware_synthesis"]),
    }
    for path, content in outputs.items():
        if args.verify:
            assert path.read_text() == content
        else:
            path.parent.mkdir(exist_ok=True)
            path.write_text(content)
    print("FIGURE_RENDER_VERIFY_PASS" if args.verify else "FIGURE_RENDER_PASS")


if __name__ == "__main__":
    main()
'''

TEST = r'''import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReviewerArtifactTest(unittest.TestCase):
    def test_offline_replay(self) -> None:
        result = subprocess.run(
            [sys.executable, "analysis/replay.py", "--verify"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        output = json.loads(result.stdout)
        self.assertEqual(output["status"], "PASS")
        self.assertEqual(output["published"]["full_window_assignments"], 106_653)
        self.assertEqual(output["published"]["llama_1b_extension_assignments"], 28_712)
        self.assertEqual(output["published"]["llama_3b_extension_assignments"], 28_788)
        self.assertEqual(output["published"]["downstream_quality_blocks"], 16)
        self.assertEqual(output["published"]["downstream_quality_runs"], 32)
        self.assertEqual(output["published"]["oars_pairs"], 10)
        self.assertEqual(output["published"]["oars_runs"], 20)
        self.assertEqual(output["synthetic"]["row_count"], 192)

    def test_figure_reproduction(self) -> None:
        result = subprocess.run(
            [sys.executable, "analysis/render_figures.py", "--verify"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("FIGURE_RENDER_VERIFY_PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def public_protocol(raw: dict[str, object], artifact_id: str) -> dict[str, object]:
    if artifact_id in {"A11", "A12", "A13", "A14"}:
        names = {"A11": "openmath_r1", "A12": "openmath_r2", "A13": "gsm8k_r1", "A14": "gsm8k_r2"}
        name = names[artifact_id]
        cell = raw["cells"][name]
        return {
            "artifact_id": artifact_id,
            "analysis": raw["analysis"],
            "assignment": {"domain": cell["domain"], "seed": cell["seed"]},
            "instrument": raw["instrument"],
            "model": raw["model"],
            "protocol": raw["schema"],
            "runtime": raw["acquisition"],
            "windows": {
                "primary": raw["acquisition"]["primary_start_versions"],
                "guard": raw["acquisition"]["terminal_guard_start_versions"],
            },
            "workload": "OpenMath" if name.startswith("openmath") else "GSM8K",
        }
    if artifact_id in {"A7", "A8", "A9", "A10"}:
        names = {
            "A7": "openmath_r1",
            "A8": "openmath_r2",
            "A9": "gsm8k_r1",
            "A10": "gsm8k_r2",
        }
        name = names[artifact_id]
        cell = raw["prospective_replicates"][name]
        return {
            "artifact_id": artifact_id,
            "analysis": raw["analysis"],
            "assignment": {"domain": cell["assignment_domain"], "seed": cell["assignment_seed"]},
            "instrument": raw["instrument"],
            "model": raw["model"],
            "protocol": raw["protocol"],
            "runtime": raw["runtime_each_replicate"],
            "windows": raw["windows_each_replicate"],
            "workload": cell["workload"],
        }
    result = {
        "artifact_id": artifact_id,
        "analysis": raw["analysis"],
        "arms": raw["arms"],
        "instrumentation": raw["instrumentation"],
        "protocol": raw["protocol"],
        "windows": raw["windows"],
    }
    if artifact_id in {"A5", "A6"}:
        name = "qwen3_0p6b_numinamath" if artifact_id == "A5" else "qwen3_1p7b_numinamath"
        result["assignment"] = {
            "domain": raw["prospective_cells"][name]["assignment_domain"],
            "seed": raw["prospective_cells"][name]["assignment_seed"],
        }
        result["runtime"] = raw["runtime_each_cell"]
        result["model"] = raw["prospective_cells"][name]["model"]
        result["workload"] = {
            key: raw["workload"][key]
            for key in ("huggingface_path", "repository_revision", "expected_source_rows", "expected_filtered_rows", "expected_unique_filtered_problems", "filter")
        }
    else:
        result["assignment"] = raw["assignment"]
        result["runtime"] = raw["runtime"]
        if "workload" in raw:
            result["workload"] = raw["workload"]
        else:
            result["workload"] = {"name": "OpenMath"}
    return result


def synthetic_rows() -> list[dict[str, object]]:
    counts = {
        "qwen3_0p6b_openmath": ([0, 1, 0, 1], [3, 4, 4, 3]),
        "qwen3_1p7b_openmath": ([0, 1, 1, 0], [2, 3, 3, 2]),
        "qwen3_0p6b_gsm8k": ([0, 0, 1, 1], [2, 4, 4, 4]),
        "qwen3_1p7b_gsm8k": ([0, 1, 0, 1], [0, 3, 1, 2]),
        "qwen3_0p6b_numinamath": ([1, 0, 1, 0], [4, 2, 4, 4]),
        "qwen3_1p7b_numinamath": ([1, 0, 0, 1], [3, 1, 3, 3]),
    }
    rows = []
    for cell, (control, d5) in counts.items():
        for version in range(4):
            for arm, lost_count in (("control", control[version]), ("d5", d5[version])):
                for sibling in range(4):
                    rows.append({
                        "D": int(sibling < lost_count),
                        "Q": 1.0,
                        "arm": arm,
                        "cell": cell,
                        "start_version": version,
                        "synthetic": True,
                    })
    return rows


def create_archive(root: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as tar:
                for path in sorted(root.rglob("*")):
                    if not path.is_file():
                        continue
                    info = tar.gettarinfo(str(path), arcname=f"m4-reviewer-artifact/{path.relative_to(root)}")
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")

    (output / "analysis").mkdir(parents=True)
    (output / "data").mkdir()
    (output / "docs").mkdir()
    (output / "figures").mkdir()
    (output / "protocols").mkdir()
    (output / "synthetic").mkdir()
    (output / "tests").mkdir()
    (output / "README.md").write_text(README)
    (output / "analysis/replay.py").write_text(REPLAY)
    (output / "analysis/render_figures.py").write_text(FIGURES)
    (output / "tests/test_replay.py").write_text(TEST)

    published = json.loads((HERE / "six_cell_synthesis.json").read_bytes())
    evidence = published.pop("evidence")
    published.pop("input_records")
    extension = json.loads((HERE / "llama_v5_publication_extension.json").read_bytes())
    for cell in extension["cells"].values():
        cell.pop("selected_artifact_sha256", None)
    published["llama_v5_extension"] = extension
    extension_3b = json.loads((HERE / "llama_3b_publication_extension.json").read_bytes())
    for cell in extension_3b["cells"].values():
        cell.pop("selected_artifact_sha256", None)
    published["llama_3b_extension"] = extension_3b
    published["downstream_quality"] = {
        "schema": "m4-downstream-quality-public-summary-v1",
        "causal_unit": "matched_training_seed_block",
        "block_count": 16,
        "run_count": 32,
        "prompt_count_per_run": 1024,
        "immediate_mean_accuracy": 0.259765625,
        "mixed_d5_mean_accuracy": 0.21484375,
        "estimate": -0.044921875,
        "student_interval_95": [-0.13394335582884473, 0.04409960582884473],
        "exact_sign_flip_p_value": 0.29815673828125,
        "practical_absolute_accuracy_margin": 0.02,
        "conclusion": "INCONCLUSIVE",
    }
    secondary = json.loads((HERE.parent / "2026-09-11-m4-downstream-quality/trained_paired_secondary_analysis.json").read_bytes())
    published["downstream_quality"]["secondary"] = {
        "analysis_role": secondary["analysis_role"],
        "normalized_realized_opportunity_loss": secondary["paired_release_policy_contrasts"]["normalized_realized_opportunity_loss"],
        "direct_chain_rate": secondary["paired_release_policy_contrasts"]["direct_chain_rate"],
        "mean_version_advance_during_release": secondary["paired_release_policy_contrasts"]["mean_version_advance_during_release"],
        "wall_seconds": secondary["paired_release_policy_contrasts"]["wall_seconds"],
        "opportunity_loss_accuracy_diagnostic": secondary["opportunity_loss_accuracy_diagnostic"],
        "unsupported_secondary_endpoints": secondary["unsupported_secondary_endpoints"],
    }
    oars = json.loads(
        (HERE.parent / "2026-09-15-m4-oars-randomized/confirmatory_terminal_analysis_result.json").read_bytes()
    )
    published["oars_confirmatory"] = {
        "schema": "m4-oars-public-summary-v1",
        "pair_count": oars["primary"]["n"],
        "run_count": 20,
        "policy_compliance": oars["gates"]["policy_compliance"],
        "primary": oars["primary"],
        "secondary_terminal_gsm8k_accuracy": oars["secondary_terminal_gsm8k_accuracy"],
        "training_dose_ratio": oars["training_dose_ratio"],
        "wall_time_ratio": oars["wall_time_ratio"],
    }
    discriminant = json.loads((HERE / "metric_discriminant.json").read_bytes())
    published["metric_discriminant"] = {
        key: discriminant[key]
        for key in ("schema", "status", "analysis_role", "raw_data_scope", "cell_count", "assignment_count", "summary", "subgroup_summaries", "claim_boundary")
    }
    published["evidence_commitments"] = {
        CELL_IDS[name]: {
            "cell": name,
            "lifecycle_sha256": item["lifecycle_sha256"],
            "opportunity_sha256": item["opportunity_sha256"],
            "protocol_sha256": item["protocol_sha256"],
        }
        for name, item in evidence.items()
    }
    write_json(output / "data/published_results.json", published)
    write_json(output / "data/robustness.json", json.loads((HERE / "robustness_results.json").read_bytes()))
    cadence = json.loads((HERE / "cadence_results.json").read_bytes())
    for item in cadence["cells"].values():
        item.pop("lifecycle_path")
    write_json(output / "data/cadence.json", cadence)
    subprocess.run(
        [sys.executable, str(output / "analysis/render_figures.py")],
        cwd=output,
        check=True,
        capture_output=True,
        text=True,
    )

    full_counts = {"A1": 7199, "A2": 8673, "A3": 9573, "A4": 9429, "A5": 7229, "A6": 7050, "A7": 6908, "A8": 6809, "A9": 7314, "A10": 7681, "A11": 7073, "A12": 6612, "A13": 7176, "A14": 7927}
    provenance = {
        "schema": "m4-anonymous-provenance-v1",
        "access_scope": "compact_records_only_external_raw_archives_not_included",
        "acquisitions": {
            artifact_id: {
                "cell": next(name for name, value in CELL_IDS.items() if value == artifact_id),
                "full_window_assignments": full_counts[artifact_id],
                "protocol_sha256": PROTOCOL_HASHES[artifact_id],
                "terminal_artifact_sha256": EXTERNAL_ARTIFACTS[artifact_id][0],
                "terminal_artifact_bytes": EXTERNAL_ARTIFACTS[artifact_id][1],
            }
            for artifact_id in sorted(EXTERNAL_ARTIFACTS)
        },
        "downstream_quality": {
            "protocol_sha256": "dbe7a4f7d1938ef43d24e536ccf9dd57110a01bab5e6cb511980218e472ef2c0",
            "run_manifest_sha256": "a461e83c7dbccdefa1c5c779062e709f34bfc502639a3e7de663a717fefa93bf",
            "terminal_topology_sha256": "b644466687e59f89d4ddb33d75f3b1c096db9e10f7090b122d263a4b2068d72c",
            "terminal_authentication_sha256": "f1c00b24877ba54fc68ff267af86824aa2233f8bdf674416bfaa18c9a801a441",
            "completion_gate_sha256": "54167b9517dc37b3ae05122e035c95da5bcd50f75d2326c1d6a841c7ddc43887",
            "analysis_sha256": "b412c5bb61ae637bf8e52442df09b8fec8e21800123ed2d900b987feca6da306",
            "secondary_analysis_sha256": sha256(HERE.parent / "2026-09-11-m4-downstream-quality/trained_paired_secondary_analysis.json"),
        },
        "oars_confirmatory": {
            "runtime_source_commit": "90dbb632026591f24c966a43edd05b1e4933358b",
            "runtime_source_archive_sha256": "1d9e17a430fc1a184d767413de829ad9ee1375091485b5e7e0b24572f61f7690",
            "protocol_sha256": "3bc49412a212419635e55b616e080708e17cf265c1926c1d93faebf072a87bfd",
            "run_manifest_sha256": "1f59d8cf3339d70b58b084af48bd2f192a880d7a43fca8190fb1d54649db5b39",
            "completion_gate_sha256": "4f844e49aaaeb99e19edd2b24bd3e7eb0664a9cb466321ddb20d2d3bab3fcb11",
            "extraction_receipt_sha256": "2911b5fad4cae38ec75aa24a5cf87f989830a4678796acb933f61cdd2bedd144",
            "analysis_sha256": "480180bee539b88c46f36274c9627ef99f8dae1185995f5ddaffbbbf248a25a2",
            "analysis_execution_receipt_sha256": "86165211b84972c8b24266b1655706b15579265e9c133a74552dd9900058d3db",
            "extractor_sha256": "b8577148b22c93392b0954c680848226e36125a673e4e49e9d7eb9fe8a64125e",
            "analyzer_sha256": "37d9553fec2bdd491d742341099f1a2916598b82b4d16eefb3022177385105b8",
            "pair_count": 10,
            "run_count": 20,
        },
        "metric_discriminant": {
            "analysis_sha256": sha256(HERE / "metric_discriminant.json"),
            "qwen_recovery_receipt_sha256": sha256(HERE / "qwen_raw_recovery_receipt.json"),
            "registered_window_assignments": discriminant["assignment_count"],
        },
    }
    write_json(output / "data/provenance.json", provenance)

    protocols = {}
    for artifact_id, source in PROTOCOL_SOURCES.items():
        assert sha256(source) == PROTOCOL_HASHES[artifact_id]
        protocols[artifact_id] = public_protocol(json.loads(source.read_bytes()), artifact_id)
    write_json(output / "protocols/public_protocols.json", {"schema": "m4-public-protocol-projections-v1", "protocols": protocols})

    claim_ledger = (HERE / "claim_ledger.md").read_text()
    claim_ledger = claim_ledger.replace(
        "`evidence_map.json`; implementation commits `c0d12e61f`, `5940059c8`, `766351123`, `4e6f6a993`, `9cc2e9c6e`",
        "`protocols/public_protocols.json`; `data/provenance.json`",
    ).replace(
        "all runs retain the same algorithm and accelerator environment",
        "all runs retain the same algorithm and one accelerator environment",
    )
    (output / "docs/claim_ledger.md").write_text(claim_ledger)
    (output / "docs/primary_table.md").write_bytes((HERE / "primary_table.md").read_bytes())
    robustness_text = (HERE / "robustness.md").read_text()
    robustness_text = robustness_text.replace(
        "`robustness_results.json`; its SHA-256 is\n`4f12bafe8cca627a48b9ee21f9fb0d05a3964d2004994a5d28568f8d7c3b168a`",
        f"`data/robustness.json`; its SHA-256 is\n`{sha256(output / 'data/robustness.json')}`",
    )
    (output / "docs/robustness.md").write_text(robustness_text)
    rows = synthetic_rows()
    (output / "synthetic/miniature_ledger.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))

    # Compute expected synthetic output by importing the just-written replay module.
    namespace = {"__file__": str(output / "analysis/replay.py"), "__name__": "artifact_replay_build"}
    exec(compile(REPLAY, str(output / "analysis/replay.py"), "exec"), namespace)
    original_load = namespace["load"]
    namespace["load"] = lambda path: {} if path.name == "expected_summary.json" else original_load(path)
    # Temporarily duplicate the replay calculation without its final equality check.
    estimates, effects = namespace["cell_estimates"](rows)
    shifts = {name: namespace["circular_bootstrap"](values, 1_000, 7100 + index) for index, (name, values) in enumerate(sorted(effects.items()))}
    scale = {workload: estimates[f"qwen3_1p7b_{workload}"] - estimates[f"qwen3_0p6b_{workload}"] for workload in ("openmath", "gsm8k", "numinamath")}
    paired = []
    for index in range(1_000):
        gsm = shifts["qwen3_1p7b_gsm8k"][index] - shifts["qwen3_0p6b_gsm8k"][index]
        openmath = shifts["qwen3_1p7b_openmath"][index] - shifts["qwen3_0p6b_openmath"][index]
        numina = shifts["qwen3_1p7b_numinamath"][index] - shifts["qwen3_0p6b_numinamath"][index]
        paired.append((gsm - openmath, gsm - numina))
    mean_x = sum(x for x, _ in paired) / len(paired)
    mean_y = sum(y for _, y in paired) / len(paired)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in paired) / (len(paired) - 1)
    sd_x = (sum((x - mean_x) ** 2 for x, _ in paired) / (len(paired) - 1)) ** 0.5
    sd_y = (sum((y - mean_y) ** 2 for _, y in paired) / (len(paired) - 1)) ** 0.5
    expected = {
        "row_count": len(rows),
        "cell_estimates": estimates,
        "cell_hac_se": {name: namespace["hac_se"](values) for name, values in effects.items()},
        "model_scale_effects": scale,
        "reference_interactions": {"gsm8k_minus_openmath": scale["gsm8k"] - scale["openmath"], "gsm8k_minus_numinamath": scale["gsm8k"] - scale["numinamath"]},
        "shared_reference_bootstrap_correlation": covariance / (sd_x * sd_y),
    }
    write_json(output / "synthetic/expected_summary.json", expected)

    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            files.append({"path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_json(output / "MANIFEST.json", {"schema": "m4-reviewer-artifact-manifest-v1", "files": files, "external_artifacts": provenance["acquisitions"]})
    create_archive(output, args.archive.resolve())
    receipt = {
        "archive": args.archive.name,
        "archive_bytes": args.archive.stat().st_size,
        "archive_sha256": sha256(args.archive),
        "manifest_sha256": sha256(output / "MANIFEST.json"),
    }
    write_json(args.archive.with_suffix(args.archive.suffix + ".json"), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
