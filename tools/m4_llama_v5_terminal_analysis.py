# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Execute the frozen all-cell terminal analysis for the Llama M4 V5 study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

from tools.m4_llama_v5_analysis import combine_replicates, infer_replicate
from tools.opportunity_ledger_join import (
    LedgerJoinProtocol,
    ReleaseArm,
    join_opportunity_ledgers,
)


CELLS = {
    "openmath-r1": (
        "openmath",
        "r1",
        "m4-llama3p2-1b-openmath-lifecycle-derived-transport-v5-r1",
        20261021,
        20261025,
    ),
    "openmath-r2": (
        "openmath",
        "r2",
        "m4-llama3p2-1b-openmath-lifecycle-derived-transport-v5-r2",
        20261022,
        20261026,
    ),
    "gsm8k-r1": (
        "gsm8k",
        "r1",
        "m4-llama3p2-1b-gsm8k-lifecycle-derived-transport-v5-r1",
        20261023,
        20261027,
    ),
    "gsm8k-r2": (
        "gsm8k",
        "r2",
        "m4-llama3p2-1b-gsm8k-lifecycle-derived-transport-v5-r2",
        20261024,
        20261028,
    ),
}


def rows(path: Path) -> list[dict[str, object]]:
    raw = path.read_bytes()
    assert raw and raw.endswith(b"\n")
    return [json.loads(line) for line in raw.splitlines()]


def type7(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    return (
        ordered[lower]
        if lower == upper
        else ordered[lower] * (upper - location) + ordered[upper] * (location - lower)
    )


def contrast(left, right) -> dict[str, object]:
    estimate = left.lower.estimate - right.lower.estimate
    se = math.hypot(left.lower.hac_standard_error, right.lower.hac_standard_error)
    shifts = [
        a - b
        for a, b in zip(
            left.lower.bootstrap_shifts, right.lower.bootstrap_shifts, strict=True
        )
    ]
    z = NormalDist().inv_cdf(0.975)
    hac = (estimate - z * se, estimate + z * se)
    bootstrap = (estimate - type7(shifts, 0.975), estimate - type7(shifts, 0.025))
    envelope = (min(hac[0], bootstrap[0]), max(hac[1], bootstrap[1]))
    return {
        "estimate": estimate,
        "hac_standard_error": se,
        "hac_interval": hac,
        "bootstrap_interval": bootstrap,
        "confidence_envelope": envelope,
        "conclusion": "POSITIVE"
        if envelope[0] > 0
        else "NEGATIVE"
        if envelope[1] < 0
        else "INCONCLUSIVE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--authentication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    authentication = json.loads(args.authentication.read_bytes())
    inferences = {}
    cells = {}
    for cell, (workload, replicate, domain, seed, bootstrap_seed) in CELLS.items():
        source = args.artifacts / cell
        lifecycle = rows(source / "lifecycle.jsonl")
        opportunity = rows(source / "opportunity.jsonl")
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
            protocol=protocol, lifecycle_rows=lifecycle, opportunity_rows=opportunity
        )
        expected = sum(
            row.get("event_type") == "group"
            and 8 <= int(row["start_weight_version"]) <= 407
            for row in opportunity
        )
        assert len(joined) == expected and len(joined) > 0
        inference = infer_replicate(
            joined, replicate=replicate, bootstrap_seed=bootstrap_seed
        )
        inferences[cell] = inference
        cells[cell] = {
            "workload": workload,
            "replicate": replicate,
            "joined_assignment_count": len(joined),
            "lower_estimate": inference.lower.estimate,
            "upper_estimate": inference.upper.estimate,
            "lower_hac_standard_error": inference.lower.hac_standard_error,
            "upper_hac_standard_error": inference.upper.hac_standard_error,
            "control_missing_fraction": inference.control_missing_fraction,
            "treatment_missing_fraction": inference.treatment_missing_fraction,
            "artifact_sha256": authentication[cell]["selected_sha256"],
        }
    combined = {}
    for workload in ("openmath", "gsm8k"):
        reps = {
            replicate: inferences[f"{workload}-{replicate}"]
            for replicate in ("r1", "r2")
        }
        combined[workload] = combine_replicates(reps).to_dict()
        combined[workload]["replicate_heterogeneity_lower_endpoint_r1_minus_r2"] = (
            contrast(reps["r1"], reps["r2"])
        )
    open_meta = type(
        "Meta",
        (),
        {
            "lower": type(
                "Endpoint",
                (),
                {
                    "estimate": sum(
                        inferences[f"openmath-{r}"].lower.estimate for r in ("r1", "r2")
                    )
                    / 2,
                    "hac_standard_error": math.sqrt(
                        sum(
                            inferences[f"openmath-{r}"].lower.hac_standard_error ** 2
                            for r in ("r1", "r2")
                        )
                    )
                    / 2,
                    "bootstrap_shifts": tuple(
                        (
                            inferences["openmath-r1"].lower.bootstrap_shifts[i]
                            + inferences["openmath-r2"].lower.bootstrap_shifts[i]
                        )
                        / 2
                        for i in range(20000)
                    ),
                },
            )()
        },
    )()
    gsm_meta = type(
        "Meta",
        (),
        {
            "lower": type(
                "Endpoint",
                (),
                {
                    "estimate": sum(
                        inferences[f"gsm8k-{r}"].lower.estimate for r in ("r1", "r2")
                    )
                    / 2,
                    "hac_standard_error": math.sqrt(
                        sum(
                            inferences[f"gsm8k-{r}"].lower.hac_standard_error ** 2
                            for r in ("r1", "r2")
                        )
                    )
                    / 2,
                    "bootstrap_shifts": tuple(
                        (
                            inferences["gsm8k-r1"].lower.bootstrap_shifts[i]
                            + inferences["gsm8k-r2"].lower.bootstrap_shifts[i]
                        )
                        / 2
                        for i in range(20000)
                    ),
                },
            )()
        },
    )()
    result = {
        "schema": "m4-llama-v5-four-acquisition-terminal-analysis-v1",
        "analysis_role": "prospective_replicated_family_transport",
        "bootstrap_draws_each_replicate": 20000,
        "cells": cells,
        "combined_workloads": combined,
        "secondary_openmath_minus_gsm8k_lower_endpoint": contrast(open_meta, gsm_meta),
        "authentication_sha256": hashlib.sha256(
            args.authentication.read_bytes()
        ).hexdigest(),
        "qualification_data_entered_estimator": False,
        "automatic_retry": False,
        "automatic_extension": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
