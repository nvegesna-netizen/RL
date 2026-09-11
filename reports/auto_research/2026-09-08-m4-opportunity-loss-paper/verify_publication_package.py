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
    extension = load(HERE / "llama_v5_publication_extension.json")
    llama = extension["combined_workloads"]
    assert sum(cell["assignment_count"] for cell in llama.values()) == 28_712
    assert all(cell["conclusion"] == "MATERIAL" for cell in llama.values())
    assert llama["openmath"]["confidence_envelope"][0] > 0.20
    assert llama["gsm8k"]["confidence_envelope"][0] > 0.20
    assert extension["secondary_openmath_minus_gsm8k"]["conclusion"] == "INCONCLUSIVE"
    three_b_extension = load(HERE / "llama_3b_publication_extension.json")
    three_b = three_b_extension["combined_workloads"]
    assert sum(cell["assignment_count"] for cell in three_b.values()) == 28_788
    assert three_b["openmath"]["conclusion"] == "MATERIAL"
    assert three_b["gsm8k"]["conclusion"] == "INCONCLUSIVE"
    assert three_b["openmath"]["confidence_envelope"][0] > 0.20
    assert three_b["gsm8k"]["confidence_envelope"][0] < 0.20
    assert all(
        contrast["conclusion"] == "NEGATIVE"
        and contrast["simultaneous_two_contrast_interval"][1] < 0
        for contrast in three_b_extension["secondary_3b_minus_1b"].values()
    )
    campaign_total = definitive_total + sum(
        cell["assignment_count"] for cell in llama.values()
    ) + sum(cell["assignment_count"] for cell in three_b.values())
    assert campaign_total == 106_653

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
            "106,653",
            "28,712",
            "28,788",
            "0.3991",
            "0.3461",
            "0.2621",
            "0.2135",
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
            "7fc1d62dba8bf82777e9d86abc3e205afa5f5012ce067082fd3a2babe88139b6",
            "37da83c526bef6cf13b36090d21dd49a9d73a768422e3fea9381521eb8e71a33",
            "c321dcf94a138d2a289142b0991c452be848d28dedf5f5c468fba78cd85ffa59",
            "fcb236655ed1554215e370cca6d21fe5873b385e4f4b1576cdc27ff2dbe4735b",
            "b79b7d1bcd861bdacf3d161bbdc5fe236edc35da41f427d0adf20a1895ddcd38",
            "034d6e57440593a48a28f705534474b1c36c8aeb0170c22e54d869f4e33a9aaa",
            "6714f77466a5d3e3c854339f69d19ac4d8312276cc3627c25fe3d8ef79b26163",
            "e535f0ce6c1fdb4f1863eca4142596f4543190d53dfd70cee7ddaf35d25a5920",
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
    print(f"qwen_full_window_assignments={definitive_total}")
    print(f"llama_1b_extension_assignments={sum(cell['assignment_count'] for cell in llama.values())}")
    print(f"llama_3b_extension_assignments={sum(cell['assignment_count'] for cell in three_b.values())}")
    print(f"campaign_full_window_assignments={campaign_total}")
    print(f"common_window_assignments={sum(c['assignment_count'] for c in cells.values())}")
    print(f"six_cell_synthesis_sha256={sha256(synthesis_path)}")


if __name__ == "__main__":
    main()
