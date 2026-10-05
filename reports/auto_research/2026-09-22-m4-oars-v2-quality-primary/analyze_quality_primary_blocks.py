#!/usr/bin/env python3
"""Apply the frozen 18-block OARS-v2 quality-primary analysis."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

T_CRITICAL_DF17 = 2.10981557783318
EXPECTED_BLOCKS = 18
EXPECTED_RUNS = 54
ARMS = ("fifo", "reward_variance", "absolute_m4")
RUN_MANIFEST_SHA256 = "44fe686d444545965afd6964387ae662c2a7f640512b1262da19b90ddfabd20f"
ANALYSIS_PLAN_SHA256 = (
    "af09cbcc891ebf1f6eb1287defd04e2708aec8ec8c8d0ba7592975c7564e0500"
)
PROTOCOL_SHA256 = "e0e0b26cc1a592bf14c4b0f77d990662f6c3ff3e09bfb6a772a3efbf149b9d2d"
SOURCE_COMMIT = "50443383825adeddca9cd1417c9dd261d8413670"
EXTRACTION_SCHEMA = "m4-oars-v2-quality-primary-result-extraction-v1"
EXTRACTION_STATUS = "PASS_54_RESULTS_EXTRACTED_AND_HASH_AUTHENTICATED"


class QualityPrimaryAnalysisError(ValueError):
    """Raised when the frozen analysis input contract is not met."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: object, *, name: str) -> float:
    """Return a finite numeric value or fail closed."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualityPrimaryAnalysisError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise QualityPrimaryAnalysisError(f"{name} must be finite")
    return result


def paired_t_summary(values: list[float]) -> dict[str, float | int]:
    """Return the frozen two-sided matched-block t summary."""
    if len(values) != EXPECTED_BLOCKS:
        raise QualityPrimaryAnalysisError("paired analysis requires exactly 18 blocks")
    mean = statistics.fmean(values)
    sample_sd = statistics.stdev(values)
    standard_error = sample_sd / math.sqrt(EXPECTED_BLOCKS)
    half_width = T_CRITICAL_DF17 * standard_error
    return {
        "n": EXPECTED_BLOCKS,
        "mean": mean,
        "sample_sd": sample_sd,
        "standard_error": standard_error,
        "ci95_lower": mean - half_width,
        "ci95_upper": mean + half_width,
    }


def exact_two_sided_sign_flip_p(values: list[float]) -> float:
    """Return the inclusive exact two-sided paired sign-flip p-value."""
    if len(values) != EXPECTED_BLOCKS:
        raise QualityPrimaryAnalysisError("sign-flip test requires exactly 18 blocks")
    observed = abs(statistics.fmean(values))
    tolerance = 1e-15 * max(1.0, observed)
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=EXPECTED_BLOCKS):
        permuted = abs(
            math.fsum(sign * value for sign, value in zip(signs, values, strict=True))
            / EXPECTED_BLOCKS
        )
        extreme += permuted + tolerance >= observed
    return extreme / (2**EXPECTED_BLOCKS)


def arm_summary(values: list[float]) -> dict[str, float | int]:
    """Return a descriptive summary for one arm across all blocks."""
    if len(values) != EXPECTED_BLOCKS:
        raise QualityPrimaryAnalysisError("arm summary requires exactly 18 blocks")
    return {
        "n": EXPECTED_BLOCKS,
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def ratio_summary(log_ratios: list[float]) -> dict[str, Any]:
    """Return a matched log-ratio summary and exponentiated interval."""
    log_scale = paired_t_summary(log_ratios)
    return {
        "geometric_mean_ratio": math.exp(float(log_scale["mean"])),
        "ci95_lower": math.exp(float(log_scale["ci95_lower"])),
        "ci95_upper": math.exp(float(log_scale["ci95_upper"])),
        "log_scale": log_scale,
    }


def validate_manifest(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate the sealed 54-run manifest and return runs by identity."""
    runs = manifest.get("runs")
    if (
        manifest.get("schema") != "m4-oars-v2-quality-primary-run-manifest-v1"
        or manifest.get("run_count") != EXPECTED_RUNS
        or manifest.get("block_count") != EXPECTED_BLOCKS
        or manifest.get("protocol_sha256") != PROTOCOL_SHA256
        or not isinstance(runs, list)
        or len(runs) != EXPECTED_RUNS
    ):
        raise QualityPrimaryAnalysisError("run manifest differs from frozen design")
    expected = {run.get("identity"): run for run in runs}
    if None in expected or len(expected) != EXPECTED_RUNS:
        raise QualityPrimaryAnalysisError("run-manifest identities differ")
    if [run.get("global_sequence") for run in runs] != list(
        range(1, EXPECTED_RUNS + 1)
    ):
        raise QualityPrimaryAnalysisError("global sequence differs")
    block_arms = {
        block: {run.get("arm") for run in runs if run.get("block") == block}
        for block in range(1, EXPECTED_BLOCKS + 1)
    }
    if any(arms != set(ARMS) for arms in block_arms.values()):
        raise QualityPrimaryAnalysisError("three-arm block coverage differs")
    return expected


def validate_rows(
    rows: list[dict[str, Any]], manifest: dict[str, Any]
) -> dict[tuple[int, str], dict[str, Any]]:
    """Validate all compact outcomes against the frozen three-arm design."""
    expected = validate_manifest(manifest)
    if len(rows) != EXPECTED_RUNS:
        raise QualityPrimaryAnalysisError("all 54 authenticated results are required")
    observed: dict[tuple[int, str], dict[str, Any]] = {}
    prompt_hashes: set[object] = set()
    for row in rows:
        identity = row.get("identity")
        if identity not in expected:
            raise QualityPrimaryAnalysisError(f"unknown run identity: {identity}")
        run = expected[identity]
        for key in (
            "actuation_scorer",
            "arm",
            "assignment_domain",
            "assignment_seed",
            "block",
            "global_sequence",
            "mode",
            "training_seed",
        ):
            if row.get(key) != run[key]:
                raise QualityPrimaryAnalysisError(f"{identity}: {key} differs")
        if (
            row.get("schema") != "m4-oars-v2-quality-primary-arm-result-v1"
            or row.get("status") != "PASS"
            or row.get("protocol_sha256") != PROTOCOL_SHA256
            or row.get("run_manifest_sha256") != RUN_MANIFEST_SHA256
            or row.get("analysis_plan_sha256") != ANALYSIS_PLAN_SHA256
            or row.get("source_commit") != SOURCE_COMMIT
            or row.get("scientific_outcome_acquisition") is not True
            or row.get("training_quality_analyzed") is not True
        ):
            raise QualityPrimaryAnalysisError(f"{identity}: result contract differs")
        systems = row.get("systems")
        mechanism = row.get("mechanism")
        outcomes = row.get("outcomes")
        if not all(isinstance(value, dict) for value in (systems, mechanism, outcomes)):
            raise QualityPrimaryAnalysisError(f"{identity}: result sections missing")
        expected_system_status = (
            "PASS_CONTROLLED_FRONTIER_SYSTEMS_READY"
            if run["arm"] == "fifo"
            else "PASS_OARS_V2_ACTUATION_QUALIFIED"
        )
        checks = systems.get("checks")
        if (
            systems.get("qualification_status") != expected_system_status
            or not isinstance(checks, dict)
            or not checks
            or not all(value is True for value in checks.values())
        ):
            raise QualityPrimaryAnalysisError(f"{identity}: systems gate failed")
        for name in (
            "time_to_update_64_seconds",
            "gradient_observer_duty",
            "scheduler_decision_duty",
        ):
            value = finite(systems.get(name), name=f"{identity}.systems.{name}")
            if value < 0 or (name == "time_to_update_64_seconds" and value == 0):
                raise QualityPrimaryAnalysisError(f"{identity}.{name} is invalid")
        retained_l1 = finite(
            mechanism.get("retained_l1_per_selected_group"),
            name=f"{identity}.mechanism.retained_l1_per_selected_group",
        )
        valid_tokens = finite(
            mechanism.get("valid_actor_tokens_per_update"),
            name=f"{identity}.mechanism.valid_actor_tokens_per_update",
        )
        fifo_overlap = finite(
            mechanism.get("mean_fifo_overlap"),
            name=f"{identity}.mechanism.mean_fifo_overlap",
        )
        if retained_l1 < 0 or valid_tokens <= 0 or not 0 <= fifo_overlap <= 4:
            raise QualityPrimaryAnalysisError(f"{identity}: mechanism values invalid")
        if run["arm"] == "fifo" and not math.isclose(fifo_overlap, 4.0):
            raise QualityPrimaryAnalysisError(f"{identity}: FIFO overlap differs")
        if outcomes.get("terminal_gsm8k_prompt_count") != 1319:
            raise QualityPrimaryAnalysisError(f"{identity}: evaluation count differs")
        accuracy = finite(
            outcomes.get("terminal_gsm8k_accuracy"),
            name=f"{identity}.outcomes.terminal_gsm8k_accuracy",
        )
        correct = outcomes.get("terminal_gsm8k_correct")
        if (
            not 0 <= accuracy <= 1
            or isinstance(correct, bool)
            or not isinstance(correct, int)
            or not 0 <= correct <= 1319
            or not math.isclose(accuracy, correct / 1319, rel_tol=0.0, abs_tol=1e-15)
        ):
            raise QualityPrimaryAnalysisError(f"{identity}: accuracy count differs")
        prompt_hashes.add(outcomes.get("terminal_gsm8k_prompt_manifest_sha256"))
        key = (int(run["block"]), str(run["arm"]))
        if key in observed:
            raise QualityPrimaryAnalysisError(f"duplicate block arm: {key}")
        observed[key] = row
    if len(prompt_hashes) != 1 or None in prompt_hashes:
        raise QualityPrimaryAnalysisError("terminal evaluation prompt manifests differ")
    expected_keys = {
        (block, arm) for block in range(1, EXPECTED_BLOCKS + 1) for arm in ARMS
    }
    if set(observed) != expected_keys:
        raise QualityPrimaryAnalysisError("three-arm block coverage differs")
    return observed


def metric_values(
    indexed: dict[tuple[int, str], dict[str, Any]],
    getter: Callable[[dict[str, Any]], float],
) -> dict[str, list[float]]:
    """Collect one numeric metric by arm in block order."""
    return {
        arm: [getter(indexed[(block, arm)]) for block in range(1, EXPECTED_BLOCKS + 1)]
        for arm in ARMS
    }


def difference(left: list[float], right: list[float]) -> list[float]:
    """Return matched left-minus-right differences."""
    return [a - b for a, b in zip(left, right, strict=True)]


def log_ratio(left: list[float], right: list[float]) -> list[float]:
    """Return matched log(left/right) values."""
    return [math.log(a / b) for a, b in zip(left, right, strict=True)]


def three_contrasts(
    values: dict[str, list[float]], *, ratio: bool = False
) -> dict[str, Any]:
    """Summarize the three registered pairwise arm contrasts."""
    summarize = ratio_summary if ratio else paired_t_summary
    transform = log_ratio if ratio else difference
    return {
        "absolute_m4_vs_fifo": summarize(
            transform(values["absolute_m4"], values["fifo"])
        ),
        "reward_variance_vs_fifo": summarize(
            transform(values["reward_variance"], values["fifo"])
        ),
        "absolute_m4_vs_reward_variance": summarize(
            transform(values["absolute_m4"], values["reward_variance"])
        ),
    }


def analyze(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    """Compute the frozen primary and descriptive secondary analyses."""
    indexed = validate_rows(rows, manifest)
    accuracy = metric_values(
        indexed, lambda row: float(row["outcomes"]["terminal_gsm8k_accuracy"])
    )
    correct = metric_values(
        indexed, lambda row: float(row["outcomes"]["terminal_gsm8k_correct"])
    )
    wall_time = metric_values(
        indexed, lambda row: float(row["systems"]["time_to_update_64_seconds"])
    )
    valid_tokens = metric_values(
        indexed,
        lambda row: float(row["mechanism"]["valid_actor_tokens_per_update"]),
    )
    retained_l1 = metric_values(
        indexed,
        lambda row: float(row["mechanism"]["retained_l1_per_selected_group"]),
    )
    fifo_overlap = metric_values(
        indexed, lambda row: float(row["mechanism"]["mean_fifo_overlap"])
    )
    gradient_duty = metric_values(
        indexed, lambda row: float(row["systems"]["gradient_observer_duty"])
    )
    scheduler_duty = metric_values(
        indexed, lambda row: float(row["systems"]["scheduler_decision_duty"])
    )
    primary_differences = difference(accuracy["absolute_m4"], accuracy["fifo"])
    primary = paired_t_summary(primary_differences)
    primary["exact_two_sided_sign_flip_p"] = exact_two_sided_sign_flip_p(
        primary_differences
    )
    primary["evidence_criterion_lower_bound_above_zero"] = (
        float(primary["ci95_lower"]) > 0.0
    )

    block_rows = []
    for block in range(1, EXPECTED_BLOCKS + 1):
        index = block - 1
        block_rows.append(
            {
                "block": block,
                "terminal_gsm8k_accuracy": {arm: accuracy[arm][index] for arm in ARMS},
                "terminal_gsm8k_correct": {
                    arm: int(correct[arm][index]) for arm in ARMS
                },
                "absolute_m4_minus_fifo_accuracy": primary_differences[index],
                "reward_variance_minus_fifo_accuracy": (
                    accuracy["reward_variance"][index] - accuracy["fifo"][index]
                ),
                "absolute_m4_minus_reward_variance_accuracy": (
                    accuracy["absolute_m4"][index] - accuracy["reward_variance"][index]
                ),
                "time_to_update_64_seconds": {
                    arm: wall_time[arm][index] for arm in ARMS
                },
                "valid_actor_tokens_per_update": {
                    arm: valid_tokens[arm][index] for arm in ARMS
                },
                "retained_l1_per_selected_group": {
                    arm: retained_l1[arm][index] for arm in ARMS
                },
                "mean_fifo_overlap": {arm: fifo_overlap[arm][index] for arm in ARMS},
            }
        )

    arm_metrics = {
        "terminal_gsm8k_accuracy": {arm: arm_summary(accuracy[arm]) for arm in ARMS},
        "terminal_gsm8k_correct": {arm: arm_summary(correct[arm]) for arm in ARMS},
        "time_to_update_64_seconds": {arm: arm_summary(wall_time[arm]) for arm in ARMS},
        "valid_actor_tokens_per_update": {
            arm: arm_summary(valid_tokens[arm]) for arm in ARMS
        },
        "retained_l1_per_selected_group": {
            arm: arm_summary(retained_l1[arm]) for arm in ARMS
        },
        "mean_fifo_overlap": {arm: arm_summary(fifo_overlap[arm]) for arm in ARMS},
        "gradient_observer_duty": {
            arm: arm_summary(gradient_duty[arm]) for arm in ARMS
        },
        "scheduler_decision_duty": {
            arm: arm_summary(scheduler_duty[arm]) for arm in ARMS
        },
    }
    return {
        "schema": "m4-oars-v2-quality-primary-analysis-result-v1",
        "status": "PASS_FROZEN_ANALYSIS_COMPLETE",
        "protocol_sha256": PROTOCOL_SHA256,
        "run_manifest_sha256": RUN_MANIFEST_SHA256,
        "analysis_plan_sha256": ANALYSIS_PLAN_SHA256,
        "source_commit": SOURCE_COMMIT,
        "block_count": EXPECTED_BLOCKS,
        "run_count": EXPECTED_RUNS,
        "policy_compliance": True,
        "primary_absolute_m4_minus_fifo_terminal_gsm8k_accuracy": primary,
        "secondary_terminal_gsm8k_accuracy": three_contrasts(accuracy),
        "matched_wall_time_ratios": three_contrasts(wall_time, ratio=True),
        "matched_valid_actor_token_ratios": three_contrasts(valid_tokens, ratio=True),
        "mechanism_retained_l1_contrasts": three_contrasts(retained_l1),
        "mechanism_fifo_overlap_contrasts": three_contrasts(fifo_overlap),
        "arm_metrics": arm_metrics,
        "block_rows": block_rows,
    }


def main() -> None:
    """Authenticate compact inputs against the extraction receipt and analyze."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--analysis-plan", type=Path, required=True)
    parser.add_argument("--extraction-receipt", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.run_manifest) != RUN_MANIFEST_SHA256:
        raise QualityPrimaryAnalysisError("frozen run-manifest hash differs")
    if sha256_file(args.analysis_plan) != ANALYSIS_PLAN_SHA256:
        raise QualityPrimaryAnalysisError("frozen analysis-plan hash differs")
    manifest = json.loads(args.run_manifest.read_bytes())
    extraction = json.loads(args.extraction_receipt.read_bytes())
    expected_identities = {run["identity"] for run in manifest["runs"]}
    if (
        extraction.get("schema") != EXTRACTION_SCHEMA
        or extraction.get("status") != EXTRACTION_STATUS
        or extraction.get("result_count") != EXPECTED_RUNS
        or extraction.get("evaluation_data_accessed") is not False
        or set(extraction.get("results", {})) != expected_identities
    ):
        raise QualityPrimaryAnalysisError("result-extraction receipt differs")
    rows = []
    for identity in sorted(expected_identities):
        path = args.results_dir / f"{identity}.json"
        if sha256_file(path) != extraction["results"][identity]["sha256"]:
            raise QualityPrimaryAnalysisError(
                f"{identity}: compact result hash differs"
            )
        rows.append(json.loads(path.read_bytes()))
    result = analyze(rows, manifest)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
