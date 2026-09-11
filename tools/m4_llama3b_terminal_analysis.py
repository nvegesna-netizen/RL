# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Execute the frozen Llama 3.2 3B acquisition and size-extension analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import NormalDist

from tools.m4_llama_size_extension_analysis import infer_size_contrast
from tools.m4_llama_v5_analysis import (
    ReplicateInference,
    combine_replicates,
    infer_replicate,
)
from tools.opportunity_ledger_join import (
    LedgerJoinProtocol,
    ReleaseArm,
    join_opportunity_ledgers,
)


THREE_B_CELLS = {
    "openmath-r1": (
        "openmath",
        "r1",
        "m4-llama3p2-3b-openmath-size-extension-v1-r1",
        20261103,
        20261111,
    ),
    "openmath-r2": (
        "openmath",
        "r2",
        "m4-llama3p2-3b-openmath-size-extension-v1-r2",
        20261104,
        20261112,
    ),
    "gsm8k-r1": (
        "gsm8k",
        "r1",
        "m4-llama3p2-3b-gsm8k-size-extension-v1-r1",
        20261105,
        20261113,
    ),
    "gsm8k-r2": (
        "gsm8k",
        "r2",
        "m4-llama3p2-3b-gsm8k-size-extension-v1-r2",
        20261106,
        20261114,
    ),
}
ONE_B_CELLS = {
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, object]]:
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise RuntimeError(f"malformed ledger {path}")
    return [json.loads(line) for line in raw.splitlines()]


def type7(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_inferences(
    root: Path,
    cells: Mapping[str, tuple[str, str, str, int, int]],
    authentication: Mapping[str, object],
) -> tuple[dict[str, ReplicateInference], dict[str, object]]:
    inferences = {}
    summaries = {}
    auth_cells = authentication["cells"]
    for cell, (workload, replicate, domain, seed, bootstrap_seed) in cells.items():
        source = root / cell
        auth = auth_cells[cell.replace("-", "_")]
        if (
            sha256(source / "lifecycle.jsonl") != auth["lifecycle_sha256"]
            or sha256(source / "opportunity.jsonl") != auth["opportunity_sha256"]
        ):
            raise RuntimeError(f"{cell} authenticated ledger digest differs")
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
            protocol=protocol,
            lifecycle_rows=lifecycle,
            opportunity_rows=opportunity,
        )
        expected = sum(
            row.get("event_type") == "group"
            and 8 <= int(row["start_weight_version"]) <= 407
            for row in opportunity
        )
        if len(joined) != expected or not joined:
            raise RuntimeError(f"{cell} strict join coverage differs")
        inference = infer_replicate(
            joined, replicate=replicate, bootstrap_seed=bootstrap_seed
        )
        inferences[cell] = inference
        summaries[cell] = {
            "workload": workload,
            "replicate": replicate,
            "joined_assignment_count": len(joined),
            "lower_estimate": inference.lower.estimate,
            "upper_estimate": inference.upper.estimate,
            "lower_hac_standard_error": inference.lower.hac_standard_error,
            "upper_hac_standard_error": inference.upper.hac_standard_error,
            "control_missing_fraction": inference.control_missing_fraction,
            "treatment_missing_fraction": inference.treatment_missing_fraction,
            "lifecycle_sha256": auth["lifecycle_sha256"],
            "opportunity_sha256": auth["opportunity_sha256"],
        }
    return inferences, summaries


def by_workload(
    inferences: Mapping[str, ReplicateInference], workload: str
) -> dict[str, ReplicateInference]:
    return {
        replicate: inferences[f"{workload}-{replicate}"] for replicate in ("r1", "r2")
    }


def endpoint_contrast(
    estimate: float, standard_error: float, shifts: Sequence[float]
) -> dict[str, object]:
    z = NormalDist().inv_cdf(0.975)
    hac = (estimate - z * standard_error, estimate + z * standard_error)
    bootstrap = (estimate - type7(shifts, 0.975), estimate - type7(shifts, 0.025))
    envelope = (min(hac[0], bootstrap[0]), max(hac[1], bootstrap[1]))
    conclusion = (
        "POSITIVE"
        if envelope[0] > 0.0
        else "NEGATIVE"
        if envelope[1] < 0.0
        else "INCONCLUSIVE"
    )
    return {
        "estimate": estimate,
        "hac_standard_error": standard_error,
        "hac_interval": hac,
        "bootstrap_interval": bootstrap,
        "confidence_envelope": envelope,
        "conclusion": conclusion,
    }


def contrast_components(
    one_b: Mapping[str, ReplicateInference],
    three_b: Mapping[str, ReplicateInference],
) -> tuple[float, float, list[float]]:
    runs = [one_b["r1"], one_b["r2"], three_b["r1"], three_b["r2"]]
    estimate = (
        three_b["r1"].lower.estimate
        + three_b["r2"].lower.estimate
        - one_b["r1"].lower.estimate
        - one_b["r2"].lower.estimate
    ) / 2.0
    standard_error = (
        math.sqrt(sum(run.lower.hac_standard_error**2 for run in runs)) / 2.0
    )
    shifts = [
        (
            three_b["r1"].lower.bootstrap_shifts[index]
            + three_b["r2"].lower.bootstrap_shifts[index]
            - one_b["r1"].lower.bootstrap_shifts[index]
            - one_b["r2"].lower.bootstrap_shifts[index]
        )
        / 2.0
        for index in range(20_000)
    ]
    return estimate, standard_error, shifts


def holm_adjusted(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[name] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--three-b-artifacts", type=Path, required=True)
    parser.add_argument("--three-b-authentication", type=Path, required=True)
    parser.add_argument("--one-b-artifacts", type=Path, required=True)
    parser.add_argument("--one-b-authentication", type=Path, required=True)
    parser.add_argument("--one-b-terminal-result", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    three_b_auth = json.loads(args.three_b_authentication.read_bytes())
    one_b_auth = json.loads(args.one_b_authentication.read_bytes())
    protocol = json.loads(args.protocol.read_bytes())
    if three_b_auth["status"] != "AUTHENTICATED_READY_FOR_FROZEN_JOINT_ANALYSIS":
        raise RuntimeError("3B authentication gate is not open")
    if one_b_auth["status"] != "AUTHENTICATED_READY_FOR_FROZEN_JOINT_ANALYSIS":
        raise RuntimeError("1B authentication gate is not open")
    if protocol["schema"] != "m4-llama3p2-3b-within-family-size-extension-v1":
        raise RuntimeError("prospective size-extension protocol differs")

    three_b, cells = load_inferences(
        args.three_b_artifacts, THREE_B_CELLS, three_b_auth
    )
    one_b, _ = load_inferences(args.one_b_artifacts, ONE_B_CELLS, one_b_auth)
    primary = {
        workload: combine_replicates(by_workload(three_b, workload)).to_dict()
        for workload in ("openmath", "gsm8k")
    }
    primary_p = {name: value["material_p_value"] for name, value in primary.items()}
    adjusted = holm_adjusted(primary_p)
    for workload in primary:
        primary[workload]["holm_adjusted_material_p_value"] = adjusted[workload]

    one_b_terminal = json.loads(args.one_b_terminal_result.read_bytes())
    for workload in ("openmath", "gsm8k"):
        reproduced = combine_replicates(by_workload(one_b, workload))
        expected = one_b_terminal["analysis"]["combined_workloads"][workload]
        if (
            abs(reproduced.identification_interval[0] - expected["estimate"]) > 1e-12
            or reproduced.conclusion != expected["conclusion"]
        ):
            raise RuntimeError(f"1B {workload} reference did not reproduce")

    components = {
        workload: contrast_components(
            by_workload(one_b, workload), by_workload(three_b, workload)
        )
        for workload in ("openmath", "gsm8k")
    }
    size_contrasts = {
        workload: infer_size_contrast(
            by_workload(one_b, workload), by_workload(three_b, workload)
        ).to_dict()
        for workload in ("openmath", "gsm8k")
    }
    maxima = [
        max(
            abs(components[workload][2][index] / components[workload][1])
            for workload in ("openmath", "gsm8k")
        )
        for index in range(20_000)
    ]
    simultaneous_critical = type7(maxima, 0.95)
    for workload in ("openmath", "gsm8k"):
        estimate, standard_error, _ = components[workload]
        size_contrasts[workload]["simultaneous_two_contrast_bootstrap_interval"] = (
            estimate - simultaneous_critical * standard_error,
            estimate + simultaneous_critical * standard_error,
        )
    interaction_estimate = components["gsm8k"][0] - components["openmath"][0]
    interaction_se = math.hypot(components["gsm8k"][1], components["openmath"][1])
    interaction_shifts = [
        gsm - openmath
        for gsm, openmath in zip(
            components["gsm8k"][2], components["openmath"][2], strict=True
        )
    ]
    interaction = endpoint_contrast(
        interaction_estimate, interaction_se, interaction_shifts
    )
    joint_success = all(
        primary[workload]["conclusion"] == "MATERIAL"
        for workload in ("openmath", "gsm8k")
    )
    result = {
        "schema": "m4-llama3p2-3b-four-acquisition-terminal-analysis-v1",
        "analysis_role": "prospective_within_family_size_extension",
        "status": (
            "TERMINAL_MATERIAL_BOTH_WORKLOADS_REPLICATED"
            if joint_success
            else "TERMINAL_JOINT_MATERIALITY_NOT_CONFIRMED"
        ),
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "materiality_threshold": 0.2,
        "bootstrap_draws_each_replicate": 20_000,
        "cells": cells,
        "co_primary_3b_workloads": primary,
        "joint_success": joint_success,
        "secondary_3b_minus_1b_lower_endpoint": size_contrasts,
        "secondary_cross_workload_size_interaction_gsm8k_minus_openmath": interaction,
        "simultaneous_two_contrast_method": "95th_percentile_max_absolute_studentized_independent_block_bootstrap_shift",
        "simultaneous_two_contrast_critical_value": simultaneous_critical,
        "size_interpretation": "descriptive_fixed_configuration_contrasts_not_a_population_scaling_law",
        "three_b_authentication_sha256": sha256(args.three_b_authentication),
        "one_b_authentication_sha256": sha256(args.one_b_authentication),
        "one_b_terminal_result_sha256": sha256(args.one_b_terminal_result),
        "protocol_sha256": sha256(args.protocol),
        "one_b_reference_reproduced": True,
        "qualification_data_entered_estimator": False,
        "automatic_retry": False,
        "automatic_extension": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
