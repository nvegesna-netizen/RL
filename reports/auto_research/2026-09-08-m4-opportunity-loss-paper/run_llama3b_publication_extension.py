#!/usr/bin/env python3
"""Recompute publication-facing Llama 3.2 3B robustness from frozen ledgers."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from statistics import NormalDist

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from tools.m4_llama3b_terminal_analysis import THREE_B_CELLS, rows, type7
from tools.m4_llama_v5_analysis import combine_replicates, infer_replicate
from tools.opportunity_ledger_join import LedgerJoinProtocol, ReleaseArm, join_opportunity_ledgers
from tools.opportunity_loss_analysis import bound_opportunity_loss


ARTIFACTS = ROOT / "session/20260909_m4_llama_lifecycle_derived_transport/llama3b-terminal-artifacts/authenticated"
AUTH = ROOT / "reports/auto_research/2026-09-10-m4-llama3p2-3b-size-extension/terminal_authentication.json"
TERMINAL = ROOT / "reports/auto_research/2026-09-10-m4-llama3p2-3b-size-extension/terminal_result.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def threshold_class(envelope: list[float], threshold: float) -> str:
    if envelope[0] > threshold:
        return "ABOVE_THRESHOLD"
    if envelope[1] <= threshold:
        return "BELOW_OR_EQUAL_THRESHOLD"
    return "CROSSES_THRESHOLD"


def interval(inference) -> dict[str, object]:
    z = NormalDist().inv_cdf(0.975)
    estimate = inference.lower.estimate
    hac = [estimate - z * inference.lower.hac_standard_error, estimate + z * inference.lower.hac_standard_error]
    shifts = inference.lower.bootstrap_shifts
    bootstrap = [estimate - type7(list(shifts), 0.975), estimate - type7(list(shifts), 0.025)]
    return {
        "estimate": estimate,
        "hac_interval": hac,
        "bootstrap_interval": bootstrap,
        "outer_envelope": [min(hac[0], bootstrap[0]), max(hac[1], bootstrap[1])],
    }


def main() -> None:
    terminal = json.loads(TERMINAL.read_bytes())
    auth = json.loads(AUTH.read_bytes())
    inferences = {}
    unadjusted = {}
    cells = {}
    for cell, (workload, replicate, domain, seed, bootstrap_seed) in THREE_B_CELLS.items():
        source = ARTIFACTS / cell
        protocol = LedgerJoinProtocol(
            assignment_domain=domain,
            assignment_seed=seed,
            arms=(ReleaseArm("control", 0.0, 1), ReleaseArm("d5", 5.0, 1)),
            primary_start_version=8,
            primary_end_version=407,
            siblings_per_group=8,
            train_batch_size=32,
        )
        joined = join_opportunity_ledgers(
            protocol=protocol,
            lifecycle_rows=rows(source / "lifecycle.jsonl"),
            opportunity_rows=rows(source / "opportunity.jsonl"),
        )
        inference = infer_replicate(joined, replicate=replicate, bootstrap_seed=bootstrap_seed)
        raw = bound_opportunity_loss(row.analysis_assignment() for row in joined)
        assert raw.complete_data_delta_l is not None
        inferences[cell] = inference
        unadjusted[cell] = raw.complete_data_delta_l
        endpoint = interval(inference)
        cells[cell] = {
            "workload": workload,
            "replicate": replicate,
            "assignment_count": len(joined),
            "adjusted": endpoint,
            "unadjusted_estimate": raw.complete_data_delta_l,
            "adjustment_shift": endpoint["estimate"] - raw.complete_data_delta_l,
            "terminal_missing_count": raw.control.missing_terminals + raw.treatment.missing_terminals,
            "selected_artifact_sha256": auth["cells"][cell.replace("-", "_")]["workload_artifact_sha256"],
        }

    combined = {}
    for workload in ("openmath", "gsm8k"):
        reps = {replicate: inferences[f"{workload}-{replicate}"] for replicate in ("r1", "r2")}
        result = combine_replicates(reps).to_dict()
        envelope = list(result["confidence_envelope"])
        result.update({
            "assignment_count": sum(cells[f"{workload}-{replicate}"]["assignment_count"] for replicate in ("r1", "r2")),
            "unadjusted_estimate": sum(unadjusted[f"{workload}-{replicate}"] for replicate in ("r1", "r2")) / 2,
            "threshold_sensitivity": {
                f"{threshold:.2f}": threshold_class(envelope, threshold)
                for threshold in (0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4)
            },
            "leave_one_replicate_out": {
                replicate: cells[f"{workload}-{replicate}"]["adjusted"]
                for replicate in ("r1", "r2")
            },
            "common_vs_full_window": {
                "status": "IDENTICAL_BY_DESIGN",
                "registered_primary_window": [8, 407],
                "common_window": [8, 407],
                "guard_window_excluded": [408, 447],
            },
        })
        combined[workload] = result

    for workload in ("openmath", "gsm8k"):
        expected = terminal["co_primary_3b_workloads"][workload]
        assert abs(combined[workload]["identification_interval"][0] - expected["estimate"]) < 1e-12
        assert list(combined[workload]["confidence_envelope"]) == expected["confidence_envelope"]

    output = {
        "schema": "m4-llama3p2-3b-publication-extension-v1",
        "analysis_role": "prospective_within_family_size_extension",
        "source_terminal_result_sha256": sha256(TERMINAL),
        "authentication_sha256": sha256(AUTH),
        "cells": cells,
        "combined_workloads": combined,
        "secondary_3b_minus_1b": terminal["secondary_3b_minus_1b"],
        "secondary_cross_workload_size_interaction_gsm8k_minus_openmath": terminal[
            "secondary_cross_workload_size_interaction_gsm8k_minus_openmath"
        ],
        "robustness_notes": {
            "missingness": "Sharp endpoints coincide in all four acquisitions because every primary assignment was terminally scored.",
            "common_window": "Registered full and common windows are both versions 8--407; versions 408--447 are the frozen guard window.",
            "scope": "The 3B-minus-1B contrasts are prespecified fixed-configuration secondary estimates, not a randomized model-size effect or general scaling law.",
        },
    }
    target = HERE / "llama_3b_publication_extension.json"
    target.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(target)


if __name__ == "__main__":
    main()
