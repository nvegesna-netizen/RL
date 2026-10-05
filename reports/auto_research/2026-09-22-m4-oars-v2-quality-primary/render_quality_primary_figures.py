#!/usr/bin/env python3
"""Render deterministic SVG figures from the frozen quality-primary result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

WIDTH = 900
HEIGHT = 520
LEFT = 92
RIGHT = 32
TOP = 62
BOTTOM = 72
COLORS = {
    "fifo": "#4C78A8",
    "reward_variance": "#F58518",
    "absolute_m4": "#54A24B",
}
LABELS = {
    "fifo": "FIFO",
    "reward_variance": "Reward variance",
    "absolute_m4": "Absolute M4",
}


def svg_text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 14,
    anchor: str = "start",
    weight: str = "normal",
) -> str:
    """Return one SVG text element."""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="#222">{value}</text>'
    )


def write_svg(path: Path, body: list[str], *, title: str, description: str) -> None:
    """Write a complete accessible SVG document."""
    payload = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img">',
        f"<title>{title}</title>",
        f"<desc>{description}</desc>",
        '<rect width="100%" height="100%" fill="white"/>',
        *body,
        "</svg>",
    ]
    path.write_text("\n".join(payload) + "\n")


def render_block_accuracy(result: dict[str, Any], output: Path) -> None:
    """Render all three terminal accuracies for each matched block."""
    plot_width = WIDTH - LEFT - RIGHT
    plot_height = HEIGHT - TOP - BOTTOM
    x_scale = lambda block: LEFT + (block - 1) * plot_width / 17
    y_scale = lambda value: TOP + plot_height * (1 - value / 0.45)
    body = [
        svg_text(
            WIDTH / 2,
            30,
            "Terminal GSM8K accuracy by matched block",
            size=20,
            anchor="middle",
            weight="bold",
        )
    ]
    for tick in (0.0, 0.1, 0.2, 0.3, 0.4):
        y = y_scale(tick)
        body.append(
            f'<line x1="{LEFT}" y1="{y:.2f}" x2="{WIDTH - RIGHT}" y2="{y:.2f}" '
            'stroke="#dddddd" stroke-width="1"/>'
        )
        body.append(svg_text(LEFT - 12, y + 5, f"{tick:.1f}", anchor="end"))
    for block in range(1, 19):
        x = x_scale(block)
        body.append(svg_text(x, HEIGHT - 43, str(block), size=12, anchor="middle"))
    body.extend(
        [
            f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{HEIGHT - BOTTOM}" stroke="#333"/>',
            f'<line x1="{LEFT}" y1="{HEIGHT - BOTTOM}" x2="{WIDTH - RIGHT}" y2="{HEIGHT - BOTTOM}" stroke="#333"/>',
            svg_text(
                WIDTH / 2, HEIGHT - 12, "Matched training-seed block", anchor="middle"
            ),
            f'<text x="22" y="{TOP + plot_height / 2:.2f}" text-anchor="middle" '
            'font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#222" '
            'transform="rotate(-90 22 '
            f'{TOP + plot_height / 2:.2f})">Terminal accuracy</text>',
        ]
    )
    rows = result["block_rows"]
    for arm in ("fifo", "reward_variance", "absolute_m4"):
        points = " ".join(
            f"{x_scale(row['block']):.2f},{y_scale(row['terminal_gsm8k_accuracy'][arm]):.2f}"
            for row in rows
        )
        body.append(
            f'<polyline points="{points}" fill="none" stroke="{COLORS[arm]}" '
            'stroke-width="2" stroke-opacity="0.75"/>'
        )
        for row in rows:
            body.append(
                f'<circle cx="{x_scale(row["block"]):.2f}" '
                f'cy="{y_scale(row["terminal_gsm8k_accuracy"][arm]):.2f}" r="4" '
                f'fill="{COLORS[arm]}" stroke="white" stroke-width="1"/>'
            )
    legend_x = LEFT + 8
    for index, arm in enumerate(("fifo", "reward_variance", "absolute_m4")):
        x = legend_x + index * 210
        body.append(
            f'<line x1="{x}" y1="48" x2="{x + 24}" y2="48" '
            f'stroke="{COLORS[arm]}" stroke-width="3"/>'
        )
        body.append(svg_text(x + 31, 53, LABELS[arm], size=13))
    write_svg(
        output,
        body,
        title="Terminal GSM8K accuracy by matched block",
        description=(
            "FIFO, reward-variance, and absolute-M4 terminal accuracy across all "
            "18 matched training-seed blocks."
        ),
    )


def render_accuracy_forest(result: dict[str, Any], output: Path) -> None:
    """Render the three frozen paired terminal-accuracy contrasts."""
    x_min = -0.15
    x_max = 0.12
    plot_left = 330
    plot_right = WIDTH - RIGHT
    plot_width = plot_right - plot_left
    x_scale = lambda value: plot_left + (value - x_min) * plot_width / (x_max - x_min)
    contrasts = [
        (
            "Absolute M4 minus FIFO (primary)",
            result["primary_absolute_m4_minus_fifo_terminal_gsm8k_accuracy"],
            COLORS["absolute_m4"],
        ),
        (
            "Reward variance minus FIFO",
            result["secondary_terminal_gsm8k_accuracy"]["reward_variance_vs_fifo"],
            COLORS["reward_variance"],
        ),
        (
            "Absolute M4 minus reward variance",
            result["secondary_terminal_gsm8k_accuracy"][
                "absolute_m4_vs_reward_variance"
            ],
            "#8C6BB1",
        ),
    ]
    body = [
        svg_text(
            WIDTH / 2,
            34,
            "Matched terminal-accuracy contrasts",
            size=20,
            anchor="middle",
            weight="bold",
        )
    ]
    for tick in (-0.15, -0.10, -0.05, 0.0, 0.05, 0.10):
        x = x_scale(tick)
        body.append(
            f'<line x1="{x:.2f}" y1="72" x2="{x:.2f}" y2="{HEIGHT - BOTTOM}" '
            f'stroke="{"#555" if tick == 0 else "#dddddd"}" '
            f'stroke-width="{"2" if tick == 0 else "1"}"/>'
        )
        body.append(svg_text(x, HEIGHT - 42, f"{tick:+.2f}", size=12, anchor="middle"))
    for index, (label, summary, color) in enumerate(contrasts):
        y = 145 + index * 105
        mean = float(summary["mean"])
        lower = float(summary["ci95_lower"])
        upper = float(summary["ci95_upper"])
        body.append(svg_text(plot_left - 18, y + 5, label, size=14, anchor="end"))
        body.append(
            f'<line x1="{x_scale(lower):.2f}" y1="{y}" x2="{x_scale(upper):.2f}" '
            f'y2="{y}" stroke="{color}" stroke-width="4"/>'
        )
        body.append(
            f'<line x1="{x_scale(lower):.2f}" y1="{y - 9}" '
            f'x2="{x_scale(lower):.2f}" y2="{y + 9}" stroke="{color}" stroke-width="2"/>'
        )
        body.append(
            f'<line x1="{x_scale(upper):.2f}" y1="{y - 9}" '
            f'x2="{x_scale(upper):.2f}" y2="{y + 9}" stroke="{color}" stroke-width="2"/>'
        )
        body.append(
            f'<circle cx="{x_scale(mean):.2f}" cy="{y}" r="7" fill="{color}" '
            'stroke="white" stroke-width="1.5"/>'
        )
        body.append(
            svg_text(
                plot_left,
                y + 31,
                f"{mean:+.4f} [{lower:+.4f}, {upper:+.4f}]",
                size=12,
            )
        )
    body.append(
        svg_text(
            (plot_left + plot_right) / 2,
            HEIGHT - 12,
            "Accuracy difference (left minus right)",
            anchor="middle",
        )
    )
    write_svg(
        output,
        body,
        title="Matched terminal-accuracy contrasts",
        description=(
            "Point estimates and two-sided 95 percent matched-block t intervals "
            "for the three frozen scheduler contrasts."
        ),
    )


def main() -> None:
    """Parse arguments and write both frozen-result figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--block-output", type=Path, required=True)
    parser.add_argument("--forest-output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_bytes())
    if result.get("status") != "PASS_FROZEN_ANALYSIS_COMPLETE":
        raise ValueError("frozen analysis result did not pass")
    render_block_accuracy(result, args.block_output)
    render_accuracy_forest(result, args.forest_output)


if __name__ == "__main__":
    main()
