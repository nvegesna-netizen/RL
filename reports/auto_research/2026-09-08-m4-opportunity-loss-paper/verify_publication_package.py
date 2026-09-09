#!/usr/bin/env python3
"""Offline integrity checks for the MLSys M4 publication package."""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_bytes())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_text(path: Path, values: list[str]) -> None:
    text = path.read_text()
    for value in values:
        assert value in text, f"{path}: missing {value}"


def main() -> None:
    synthesis_path = HERE / "six_cell_synthesis.json"
    synthesis = load(synthesis_path)
    cells = synthesis["cell_results"]
    assert len(cells) == 6
    assert sum(cell["assignment_count"] for cell in cells.values()) == 43_756
    assert all(cell["terminal_missing_count"] == 0 for cell in cells.values())
    assert abs(
        synthesis["dependency_aware_synthesis"]["hac_correlation"] - 0.3793770869832136
    ) < 1e-9
    assert abs(
        synthesis["dependency_aware_synthesis"]["bootstrap_correlation"]
        - 0.40530226396440766
    ) < 1e-9
    assert sha256(synthesis_path) == (
        "fd4c74c245b5294be175e2117c136e5cc7a3dcd5ce302713ba8041768c6902be"
    )

    evidence = load(
        ROOT
        / "reports/auto_research/2026-09-02-m4-opportunity-loss-followup/evidence_map.json"
    )
    numina = load(
        ROOT
        / "reports/auto_research/2026-09-07-m4-opportunity-loss-numinamath-generalization/acquisition_result.json"
    )
    grid = load(
        ROOT
        / "reports/auto_research/2026-09-07-m4-opportunity-loss-grid-completion/acquisition_result.json"
    )
    definitive_total = (
        evidence["prospective_followup_result"]["completion"]["primary_assignments"]
        + evidence["model_scale_transport_qwen3_1p7b_openmath"]["completion"][
            "primary_assignments"
        ]
        + evidence["workload_transport_qwen3_1p7b_gsm8k"]["completion"][
            "primary_assignments"
        ]
        + grid["acquisition"]["primary_assignment_count"]
        + sum(cell["assignment_count"] for cell in numina["cells"].values())
    )
    assert definitive_total == 49_153

    cadence_path = HERE / "cadence_results.json"
    cadence = load(cadence_path)
    assert len(cadence["cells"]) == 6
    assert all(cell["interval_count"] == 400 for cell in cadence["cells"].values())
    ratios = [
        cell["five_seconds_over_median_cadence"]
        for cell in cadence["cells"].values()
    ]
    assert 0.378 < min(ratios) < 0.380
    assert 0.643 < max(ratios) < 0.645

    require_text(
        HERE / "manuscript.md",
        [
            "49,153",
            "χ²(2)=11.53",
            "p=0.00314",
            "0.00329",
            "Qwen3-0.6B/GSM8K",
            sha256(synthesis_path),
        ],
    )
    require_text(
        HERE / "reproducibility_appendix.md",
        [
            "66675506",
            "failed after training, at analysis invocation",
            grid["raw_evidence"]["terminal_artifact_sha256"],
            evidence["prospective_followup_result"]["artifact"]["sha256"],
            evidence["model_scale_transport_qwen3_1p7b_openmath"]["artifact"][
                "sha256"
            ],
            evidence["workload_transport_qwen3_1p7b_gsm8k"]["artifact"]["sha256"],
            numina["cells"]["qwen3_0p6b_numinamath"]["artifact_sha256"],
            numina["cells"]["qwen3_1p7b_numinamath"]["artifact_sha256"],
        ],
    )

    for name in (
        "cell_forest_plot.svg",
        "interaction_plot.svg",
        "causal_diagram.svg",
        "provenance_diagram.svg",
    ):
        ET.parse(HERE / name)

    bib = (HERE / "references.bib").read_text()
    assert "and others" not in bib
    bib_keys = set(re.findall(r"^@\w+\{([^,]+),", bib, flags=re.MULTILINE))
    tex = (HERE / "mlsys2027/paper.tex").read_text()
    cited = {
        key.strip()
        for group in re.findall(r"\\cite\{([^}]+)\}", tex)
        for key in group.split(",")
    }
    assert cited <= bib_keys, f"missing BibTeX keys: {sorted(cited - bib_keys)}"
    print("PUBLICATION_PACKAGE_VERIFY_PASS")
    print(f"definitive_full_window_assignments={definitive_total}")
    print(f"common_window_assignments={sum(c['assignment_count'] for c in cells.values())}")
    print(f"six_cell_synthesis_sha256={sha256(synthesis_path)}")


if __name__ == "__main__":
    main()
