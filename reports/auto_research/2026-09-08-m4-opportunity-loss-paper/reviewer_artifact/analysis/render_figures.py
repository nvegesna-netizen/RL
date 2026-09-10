#!/usr/bin/env python3
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
        color = "#00897b" if "llama" in name else "#1769aa" if "0p6b" in name else "#7b1fa2"
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
