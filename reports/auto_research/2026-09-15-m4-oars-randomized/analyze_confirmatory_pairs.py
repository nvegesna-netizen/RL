#!/usr/bin/env python3
"""Apply the frozen paired OARS confirmatory analysis after authentication."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

T_CRITICAL_DF9 = 2.2621571628540993
EXPECTED_PAIRS = 10
EXPECTED_RUNS = 20
RUN_MANIFEST_SHA256 = (
    "1f59d8cf3339d70b58b084af48bd2f192a880d7a43fca8190fb1d54649db5b39"
)


class ConfirmatoryAnalysisError(ValueError):
    """Raised when the complete authenticated input contract is not met."""


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfirmatoryAnalysisError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ConfirmatoryAnalysisError(f"{name} must be finite")
    return result


def paired_t_summary(values: list[float]) -> dict[str, float | int]:
    """Return the preregistered paired-t summary for exactly ten values."""
    if len(values) != EXPECTED_PAIRS:
        raise ConfirmatoryAnalysisError("paired analysis requires exactly 10 pairs")
    mean = statistics.fmean(values)
    sample_sd = statistics.stdev(values)
    half_width = T_CRITICAL_DF9 * sample_sd / math.sqrt(EXPECTED_PAIRS)
    return {
        "n": EXPECTED_PAIRS,
        "mean": mean,
        "sample_sd": sample_sd,
        "ci95_lower": mean - half_width,
        "ci95_upper": mean + half_width,
    }


def exact_two_sided_sign_flip_p(values: list[float]) -> float:
    """Return the inclusive exact two-sided paired sign-flip p-value."""
    if len(values) != EXPECTED_PAIRS:
        raise ConfirmatoryAnalysisError("sign-flip test requires exactly 10 pairs")
    observed = abs(statistics.fmean(values))
    tolerance = 1e-15 * max(1.0, observed)
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=EXPECTED_PAIRS):
        permuted = abs(
            math.fsum(sign * value for sign, value in zip(signs, values, strict=True))
            / EXPECTED_PAIRS
        )
        extreme += permuted + tolerance >= observed
    return extreme / (2**EXPECTED_PAIRS)


def _validate_rows(
    rows: list[dict[str, Any]], manifest: dict[str, Any]
) -> dict[tuple[int, str], dict[str, Any]]:
    expected = {run["identity"]: run for run in manifest["runs"]}
    if manifest.get("run_count") != EXPECTED_RUNS or len(expected) != EXPECTED_RUNS:
        raise ConfirmatoryAnalysisError("run manifest is not the frozen 20-run design")
    if len(rows) != EXPECTED_RUNS:
        raise ConfirmatoryAnalysisError("all 20 authenticated results are required")
    observed: dict[tuple[int, str], dict[str, Any]] = {}
    prompt_hashes = set()
    for row in rows:
        identity = row.get("identity")
        if identity not in expected:
            raise ConfirmatoryAnalysisError(f"unknown run identity: {identity}")
        run = expected[identity]
        for key in (
            "arm",
            "assignment_domain",
            "assignment_seed",
            "mode",
            "pair",
            "training_seed",
        ):
            if row.get(key) != run[key]:
                raise ConfirmatoryAnalysisError(f"{identity}: {key} differs")
        if (
            row.get("schema") != "m4-oars-randomized-confirmatory-arm-result-v1"
            or row.get("status") != "PASS"
            or row.get("protocol_sha256") != manifest["protocol_sha256"]
            or row.get("run_manifest_sha256") != RUN_MANIFEST_SHA256
            or row.get("source_commit") != manifest["runtime_source"]["commit"]
            or row.get("training_quality_analyzed") is not True
            or row.get("scientific_outcome_acquisition") is not True
        ):
            raise ConfirmatoryAnalysisError(f"{identity}: result contract differs")
        systems = row.get("systems")
        outcomes = row.get("outcomes")
        if not isinstance(systems, dict) or not isinstance(outcomes, dict):
            raise ConfirmatoryAnalysisError(f"{identity}: result sections missing")
        checks = systems.get("checks")
        if (
            systems.get("qualification_status") != "PASS"
            or not isinstance(checks, dict)
            or not checks
            or not all(value is True for value in checks.values())
        ):
            raise ConfirmatoryAnalysisError(f"{identity}: systems gate failed")
        if outcomes.get("terminal_gsm8k_prompt_count") != 1319:
            raise ConfirmatoryAnalysisError(f"{identity}: evaluation count differs")
        for name in (
            "retained_l1_per_selected_group",
            "valid_actor_tokens_per_update",
            "terminal_gsm8k_accuracy",
        ):
            value = _finite(outcomes.get(name), name=f"{identity}.{name}")
            if value < 0:
                raise ConfirmatoryAnalysisError(f"{identity}.{name} is negative")
        if float(outcomes["valid_actor_tokens_per_update"]) <= 0:
            raise ConfirmatoryAnalysisError(f"{identity}: training dose is not positive")
        if float(outcomes["terminal_gsm8k_accuracy"]) > 1:
            raise ConfirmatoryAnalysisError(f"{identity}: accuracy exceeds one")
        correct = outcomes.get("terminal_gsm8k_correct")
        if (
            isinstance(correct, bool)
            or not isinstance(correct, int)
            or not 0 <= correct <= 1319
            or not math.isclose(
                float(outcomes["terminal_gsm8k_accuracy"]),
                correct / 1319,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise ConfirmatoryAnalysisError(f"{identity}: accuracy count differs")
        time_to_64 = _finite(
            systems.get("time_to_update_64_seconds"),
            name=f"{identity}.time_to_update_64_seconds",
        )
        if time_to_64 <= 0:
            raise ConfirmatoryAnalysisError(f"{identity}: runtime is not positive")
        prompt_hashes.add(outcomes.get("terminal_gsm8k_prompt_manifest_sha256"))
        key = (int(run["pair"]), str(run["arm"]))
        if key in observed:
            raise ConfirmatoryAnalysisError(f"duplicate pair arm: {key}")
        observed[key] = row
    if len(prompt_hashes) != 1 or None in prompt_hashes:
        raise ConfirmatoryAnalysisError("terminal evaluation prompt manifests differ")
    if set(observed) != {
        (pair, arm)
        for pair in range(1, EXPECTED_PAIRS + 1)
        for arm in ("fifo", "oars")
    }:
        raise ConfirmatoryAnalysisError("paired arm coverage differs")
    return observed


def analyze(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    """Compute all frozen paired estimands and the registered classification."""
    indexed = _validate_rows(rows, manifest)
    primary_differences = []
    wall_log_ratios = []
    token_log_ratios = []
    quality_differences = []
    pair_rows = []
    for pair in range(1, EXPECTED_PAIRS + 1):
        fifo = indexed[(pair, "fifo")]
        oars = indexed[(pair, "oars")]
        fifo_outcomes = fifo["outcomes"]
        oars_outcomes = oars["outcomes"]
        primary = float(oars_outcomes["retained_l1_per_selected_group"]) - float(
            fifo_outcomes["retained_l1_per_selected_group"]
        )
        wall = math.log(
            float(oars["systems"]["time_to_update_64_seconds"])
            / float(fifo["systems"]["time_to_update_64_seconds"])
        )
        token = math.log(
            float(oars_outcomes["valid_actor_tokens_per_update"])
            / float(fifo_outcomes["valid_actor_tokens_per_update"])
        )
        quality = float(oars_outcomes["terminal_gsm8k_accuracy"]) - float(
            fifo_outcomes["terminal_gsm8k_accuracy"]
        )
        primary_differences.append(primary)
        wall_log_ratios.append(wall)
        token_log_ratios.append(token)
        quality_differences.append(quality)
        pair_rows.append(
            {
                "pair": pair,
                "primary_oars_minus_fifo": primary,
                "quality_oars_minus_fifo": quality,
                "token_log_oars_over_fifo": token,
                "wall_log_oars_over_fifo": wall,
            }
        )

    primary = paired_t_summary(primary_differences)
    wall_log = paired_t_summary(wall_log_ratios)
    token_log = paired_t_summary(token_log_ratios)
    quality = paired_t_summary(quality_differences)
    primary_superiority = float(primary["ci95_lower"]) > 0.0
    policy_compliance = all(
        all(row["systems"]["checks"].values()) for row in indexed.values()
    )
    wall_utility = float(wall_log["ci95_upper"]) < math.log(1.10)
    if primary_superiority and policy_compliance and wall_utility:
        classification = "MATERIAL"
    elif float(primary["ci95_upper"]) <= 0.0:
        classification = "NON_MATERIAL"
    else:
        classification = "INCONCLUSIVE"
    return {
        "classification": classification,
        "gates": {
            "all_ten_pairs_valid": True,
            "policy_compliance": policy_compliance,
            "primary_superiority": primary_superiority,
            "wall_time_utility": wall_utility,
        },
        "pair_rows": pair_rows,
        "primary": {
            **primary,
            "exact_two_sided_sign_flip_p": exact_two_sided_sign_flip_p(
                primary_differences
            ),
        },
        "protocol_sha256": manifest["protocol_sha256"],
        "schema": "m4-oars-randomized-confirmatory-analysis-result-v1",
        "secondary_terminal_gsm8k_accuracy": quality,
        "training_dose_ratio": {
            "geometric_mean": math.exp(float(token_log["mean"])),
            "ci95_lower": math.exp(float(token_log["ci95_lower"])),
            "ci95_upper": math.exp(float(token_log["ci95_upper"])),
            "log_scale": token_log,
        },
        "wall_time_ratio": {
            "geometric_mean": math.exp(float(wall_log["mean"])),
            "ci95_lower": math.exp(float(wall_log["ci95_lower"])),
            "ci95_upper": math.exp(float(wall_log["ci95_upper"])),
            "log_scale": wall_log,
        },
    }


def main() -> None:
    """Authenticate compact inputs against the completion gate, then analyze."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--authentication-gate", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.run_manifest.read_bytes())
    gate = json.loads(args.authentication_gate.read_bytes())
    expected_identities = {run["identity"] for run in manifest["runs"]}
    if (
        gate.get("schema")
        != "m4-oars-randomized-confirmatory-terminal-authentication-v1"
        or gate.get("status") != "PASS_ALL_20_AUTHENTICATED_RESULTS_UNOPENED"
        or set(gate.get("results", {})) != expected_identities
    ):
        raise ConfirmatoryAnalysisError("terminal authentication gate differs")
    rows = []
    for identity in sorted(expected_identities):
        path = args.results_dir / f"{identity}.json"
        if sha256(path) != gate["results"][identity]["sha256"]:
            raise ConfirmatoryAnalysisError(f"{identity}: compact result hash differs")
        rows.append(json.loads(path.read_bytes()))
    result = analyze(rows, manifest)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
