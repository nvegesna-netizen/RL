# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fail-closed run-level analysis for the M4 downstream-quality study."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from statistics import fmean


PROTOCOL_SHA256 = "dbe7a4f7d1938ef43d24e536ccf9dd57110a01bab5e6cb511980218e472ef2c0"
EMBARGO_SHA256 = "3d292020ff6d9a42a9209de3c65010603a24a5c9ffabbeaf77521be99ac84997"
T_CRITICAL_95_DF15 = 2.131449545559323
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20261401


class DownstreamQualityAnalysisError(ValueError):
    """Raised before or during a fail-closed downstream-quality analysis."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path, *, name: str) -> tuple[bytes, dict[str, object]]:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise DownstreamQualityAnalysisError(f"cannot read {name}: {path}") from error
    if not data.endswith(b"\n"):
        raise DownstreamQualityAnalysisError(f"{name} is not newline-terminated")
    try:
        value = json.loads(data)
    except json.JSONDecodeError as error:
        raise DownstreamQualityAnalysisError(f"malformed {name}") from error
    if not isinstance(value, dict):
        raise DownstreamQualityAnalysisError(f"{name} must be an object")
    return data, value


def _safe_relative_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str):
        raise DownstreamQualityAnalysisError(f"{name} must be a string")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise DownstreamQualityAnalysisError(f"unsafe {name}")
    return Path(*pure.parts)


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DownstreamQualityAnalysisError(f"{name} must be a lowercase SHA-256")
    return value


def _sample_sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise DownstreamQualityAnalysisError(
            "sample standard deviation needs two values"
        )
    center = fmean(values)
    return math.sqrt(
        math.fsum((value - center) ** 2 for value in values) / (len(values) - 1)
    )


def _type7(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise DownstreamQualityAnalysisError("invalid quantile input")
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _classification(interval: tuple[float, float], margin: float) -> str:
    lower, upper = interval
    if upper < -margin:
        return "MATERIALLY_WORSE"
    if lower > margin:
        return "MATERIALLY_BETTER"
    if lower >= -margin and upper <= margin:
        return "PRACTICALLY_EQUIVALENT"
    if upper < 0.0 or lower > 0.0:
        return "DETECTABLE_BUT_NOT_MATERIAL"
    return "INCONCLUSIVE"


def _paired_inference(
    differences: Sequence[float], *, margin: float
) -> dict[str, object]:
    if len(differences) != 16:
        raise DownstreamQualityAnalysisError("primary inference requires 16 blocks")
    estimate = fmean(differences)
    sd = _sample_sd(differences)
    standard_error = sd / math.sqrt(len(differences))
    interval = (
        estimate - T_CRITICAL_95_DF15 * standard_error,
        estimate + T_CRITICAL_95_DF15 * standard_error,
    )

    observed = abs(estimate)
    exceedances = 0
    sign_flip_estimates = []
    for signs in itertools.product((-1.0, 1.0), repeat=16):
        candidate = fmean(sign * value for sign, value in zip(signs, differences))
        sign_flip_estimates.append(candidate)
        exceedances += abs(candidate) >= observed - 1e-15

    generator = random.Random(BOOTSTRAP_SEED)
    bootstrap_means = []
    bootstrap_t = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [differences[generator.randrange(16)] for _ in range(16)]
        sample_mean = fmean(sample)
        sample_se = _sample_sd(sample) / 4.0
        bootstrap_means.append(sample_mean)
        if sample_se > 0.0:
            bootstrap_t.append((sample_mean - estimate) / sample_se)
    percentile = (_type7(bootstrap_means, 0.025), _type7(bootstrap_means, 0.975))
    if standard_error == 0.0 or not bootstrap_t:
        studentized = (estimate, estimate)
    else:
        studentized = (
            estimate - _type7(bootstrap_t, 0.975) * standard_error,
            estimate - _type7(bootstrap_t, 0.025) * standard_error,
        )
    return {
        "block_count": 16,
        "degrees_of_freedom": 15,
        "estimate": estimate,
        "paired_sample_standard_deviation": sd,
        "standard_error": standard_error,
        "student_interval_95": interval,
        "practical_absolute_accuracy_margin": margin,
        "conclusion": _classification(interval, margin),
        "exact_sign_flip": {
            "enumerated_assignments": len(sign_flip_estimates),
            "two_sided_zero_effect_p_value": exceedances / len(sign_flip_estimates),
            "estimate_minimum": min(sign_flip_estimates),
            "estimate_maximum": max(sign_flip_estimates),
        },
        "paired_bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "percentile_interval_95": percentile,
            "studentized_interval_95": studentized,
        },
    }


def _prompt_cluster_sensitivity(
    immediate: Sequence[Sequence[int]], mixed: Sequence[Sequence[int]]
) -> dict[str, object]:
    blocks = len(immediate)
    prompts = len(immediate[0])
    differences = [
        [mixed[b][j] - immediate[b][j] for j in range(prompts)] for b in range(blocks)
    ]
    estimate = fmean(value for block in differences for value in block)
    block_means = [fmean((*immediate[b], *mixed[b])) for b in range(blocks)]
    prompt_means = [
        fmean(
            [immediate[b][j] for b in range(blocks)]
            + [mixed[b][j] for b in range(blocks)]
        )
        for j in range(prompts)
    ]
    grand = fmean(block_means)
    cluster_scores = []
    x_squared = blocks * prompts * 0.5
    for b in range(blocks):
        score = 0.0
        for treatment, outcomes in ((0.0, immediate[b]), (1.0, mixed[b])):
            x = treatment - 0.5
            for j, outcome in enumerate(outcomes):
                residual = (
                    outcome - block_means[b] - prompt_means[j] + grand - estimate * x
                )
                score += x * residual
        cluster_scores.append(score)
    observations = blocks * prompts * 2
    parameters = blocks + prompts
    correction = (blocks / (blocks - 1)) * (
        (observations - 1) / (observations - parameters)
    )
    variance = (
        correction * math.fsum(score**2 for score in cluster_scores) / (x_squared**2)
    )
    return {
        "method": "balanced_block_and_prompt_fixed_effects_clustered_by_training_block",
        "estimate": estimate,
        "cluster_count": blocks,
        "prompt_count": prompts,
        "cluster_robust_standard_error": math.sqrt(max(0.0, variance)),
        "cannot_override_primary": True,
    }


def _base_covariate_sensitivity(
    *, prompt_differences: Sequence[float], base_correctness: Sequence[int]
) -> dict[str, object]:
    if len(prompt_differences) != len(base_correctness):
        raise DownstreamQualityAnalysisError("base covariate prompt count differs")
    mean_x = fmean(base_correctness)
    mean_y = fmean(prompt_differences)
    denominator = math.fsum((value - mean_x) ** 2 for value in base_correctness)
    slope = (
        math.fsum(
            (x - mean_x) * (y - mean_y)
            for x, y in zip(base_correctness, prompt_differences)
        )
        / denominator
        if denominator > 0.0
        else 0.0
    )
    return {
        "method": "prompt_level_effect_heterogeneity_on_v8_base_correctness",
        "base_correctness_mean": mean_x,
        "heterogeneity_slope": slope,
        "standardized_mean_effect": mean_y,
        "unadjusted_mean_effect": mean_y,
        "point_estimate_is_identical_by_balanced_fixed_prompt_design": True,
        "cannot_override_primary": True,
    }


def _validate_gate_before_outcomes(
    *,
    protocol_path: Path,
    run_manifest_path: Path,
    embargo_path: Path,
    completion_gate_path: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], str]:
    protocol_data, protocol = _read_json(protocol_path, name="protocol")
    if _sha256(protocol_data) != PROTOCOL_SHA256:
        raise DownstreamQualityAnalysisError("frozen protocol moved")
    manifest_data, manifest = _read_json(run_manifest_path, name="run manifest")
    embargo_data, embargo = _read_json(embargo_path, name="outcome embargo")
    if _sha256(embargo_data) != EMBARGO_SHA256:
        raise DownstreamQualityAnalysisError("frozen outcome embargo moved")
    gate_data, gate = _read_json(completion_gate_path, name="completion gate")

    if (
        protocol["schema"]
        != "m4-downstream-quality-trained-paired-acquisition-protocol-v1"
        or manifest["schema"] != "m4-downstream-quality-trained-paired-run-manifest-v1"
        or manifest["status"] != "PASS_LOCAL_IDENTITIES_UNLAUNCHABLE"
        or manifest["protocol_sha256"] != PROTOCOL_SHA256
        or manifest["run_count"] != 32
        or manifest["block_count"] != 16
    ):
        raise DownstreamQualityAnalysisError("protocol or run manifest differs")
    if (
        embargo["status"] != "LOCKED_NO_TRAINED_OUTCOMES"
        or embargo["required_complete_pairs"] != 16
        or embargo["required_complete_runs"] != 32
        or embargo["partial_result_analysis_allowed"] is not False
    ):
        raise DownstreamQualityAnalysisError("outcome embargo differs")
    if (
        gate.get("schema") != embargo["required_completion_gate_schema"]
        or gate.get("status") != embargo["required_completion_status"]
        or gate.get("protocol_sha256") != PROTOCOL_SHA256
        or gate.get("run_manifest_sha256") != _sha256(manifest_data)
        or gate.get("all_runs_terminal") is not True
        or gate.get("all_artifacts_authenticated") is not True
        or gate.get("resource_caps_passed") is not True
        or gate.get("outcome_release_authorized") is not True
        or gate.get("complete_pair_count") != 16
        or gate.get("complete_run_count") != 32
        or not isinstance(gate.get("summed_wall_hours"), (int, float))
        or not 0.0 <= gate["summed_wall_hours"] <= 128.0
        or not isinstance(gate.get("aggregate_h100_gpu_hours"), (int, float))
        or not 0.0 <= gate["aggregate_h100_gpu_hours"] <= 256.0
        or not isinstance(gate.get("runs"), list)
        or len(gate["runs"]) != 32
    ):
        raise DownstreamQualityAnalysisError(
            "outcome embargo remains locked: completion gate is not complete"
        )
    return protocol, manifest, gate, _sha256(gate_data)


def analyze(
    *,
    protocol_path: Path,
    run_manifest_path: Path,
    embargo_path: Path,
    completion_gate_path: Path,
    results_root: Path,
) -> dict[str, object]:
    protocol, manifest, gate, gate_sha = _validate_gate_before_outcomes(
        protocol_path=protocol_path,
        run_manifest_path=run_manifest_path,
        embargo_path=embargo_path,
        completion_gate_path=completion_gate_path,
    )
    registered = {(run["block_id"], run["regime"]): run for run in manifest["runs"]}
    gate_runs = {(run["block_id"], run["regime"]): run for run in gate["runs"]}
    if set(registered) != set(gate_runs) or len(registered) != 32:
        raise DownstreamQualityAnalysisError("completion identities differ")

    scores: dict[tuple[str, str], list[int]] = {}
    prompt_hashes: list[str] | None = None
    run_summaries = []
    for identity in sorted(registered):
        expected = registered[identity]
        authenticated = gate_runs[identity]
        if (
            authenticated.get("config_sha256") != expected["config_sha256"]
            or authenticated.get("training_seed") != expected["training_seed"]
            or authenticated.get("assignment_seed") != expected["assignment_seed"]
            or authenticated.get("assignment_domain") != expected["assignment_domain"]
            or authenticated.get("train_steps") != 448
            or authenticated.get("trainer_version") != 448
            or authenticated.get("prompt_manifest_sha256")
            != protocol["evaluation"]["prompt_manifest_sha256"]
            or not isinstance(authenticated.get("pipeline_id"), int)
            or not isinstance(authenticated.get("job_id"), int)
            or not isinstance(authenticated.get("wall_hours"), (int, float))
            or not 0.0 <= authenticated["wall_hours"] <= 4.0
            or not isinstance(authenticated.get("h100_gpu_hours"), (int, float))
            or not 0.0 <= authenticated["h100_gpu_hours"] <= 8.0
        ):
            raise DownstreamQualityAnalysisError(
                f"authenticated identity differs: {identity}"
            )
        if authenticated.get("result_path") != expected["expected_result_path"]:
            raise DownstreamQualityAnalysisError(f"result path differs: {identity}")
        _safe_relative_path(
            authenticated.get("terminal_export_path"), name="terminal export path"
        )
        for key in (
            "terminal_export_sha256",
            "converted_weights_sha256",
            "evaluation_data_sha256",
        ):
            _require_sha256(authenticated.get(key), name=key)
        relative = _safe_relative_path(
            authenticated.get("result_path"), name="result path"
        )
        result_data, result = _read_json(
            results_root / relative, name=f"run result {identity}"
        )
        if _sha256(result_data) != authenticated.get("result_sha256"):
            raise DownstreamQualityAnalysisError(f"result digest differs: {identity}")
        block_id, regime = identity
        if (
            result.get("schema") != "m4-downstream-quality-trained-run-result-v1"
            or result.get("protocol_sha256") != PROTOCOL_SHA256
            or result.get("block_id") != block_id
            or result.get("regime") != regime
            or result.get("training_seed") != expected["training_seed"]
            or result.get("assignment_seed") != expected["assignment_seed"]
            or result.get("assignment_domain") != expected["assignment_domain"]
            or result.get("config_sha256") != expected["config_sha256"]
            or result.get("train_steps") != 448
            or result.get("trainer_version") != 448
            or result.get("prompt_manifest_sha256")
            != protocol["evaluation"]["prompt_manifest_sha256"]
            or result.get("optimizer_exported") is not False
            or result.get("resumable_checkpoint") is not False
        ):
            raise DownstreamQualityAnalysisError(
                f"run result contract differs: {identity}"
            )
        rows = result.get("scores")
        if not isinstance(rows, list) or len(rows) != 1024:
            raise DownstreamQualityAnalysisError(f"score count differs: {identity}")
        current_hashes = []
        current_scores = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(
                row.get("prompt_row_sha256"), str
            ):
                raise DownstreamQualityAnalysisError(f"malformed score row: {identity}")
            reward = row.get("reward")
            if isinstance(reward, bool) or reward not in (0, 1, 0.0, 1.0):
                raise DownstreamQualityAnalysisError(f"nonbinary score: {identity}")
            current_hashes.append(row["prompt_row_sha256"])
            current_scores.append(int(reward))
        if len(set(current_hashes)) != 1024:
            raise DownstreamQualityAnalysisError(f"duplicate prompt hash: {identity}")
        if prompt_hashes is None:
            prompt_hashes = current_hashes
        elif current_hashes != prompt_hashes:
            raise DownstreamQualityAnalysisError(f"prompt order differs: {identity}")
        scores[identity] = current_scores
        run_summaries.append(
            {
                "block_id": block_id,
                "regime": regime,
                "accuracy": fmean(current_scores),
                "correct": sum(current_scores),
                "result_sha256": _sha256(result_data),
                "pipeline_id": authenticated["pipeline_id"],
                "job_id": authenticated["job_id"],
                "systems": result.get("systems"),
                "mechanism": result.get("mechanism"),
            }
        )

    block_ids = [block["id"] for block in protocol["blocks"]]
    immediate = [scores[(block_id, "immediate")] for block_id in block_ids]
    mixed = [scores[(block_id, "mixed_d5")] for block_id in block_ids]
    block_differences = [
        fmean(mixed[index]) - fmean(immediate[index]) for index in range(16)
    ]
    primary = _paired_inference(
        block_differences,
        margin=protocol["inference"]["practical_absolute_accuracy_margin"],
    )
    prompt_sensitivity = _prompt_cluster_sensitivity(immediate, mixed)

    base_relative = _safe_relative_path(
        gate.get("base_covariate_path"), name="base covariate path"
    )
    base_data, base = _read_json(
        results_root / base_relative, name="base prompt covariate"
    )
    if _sha256(base_data) != gate.get("base_covariate_sha256"):
        raise DownstreamQualityAnalysisError("base prompt covariate digest differs")
    base_rows = base.get("scores")
    if (
        base.get("schema") != "m4-downstream-quality-v8-base-prompt-covariate-v1"
        or base.get("prompt_manifest_sha256")
        != protocol["evaluation"]["prompt_manifest_sha256"]
        or not isinstance(base_rows, list)
        or len(base_rows) != 1024
        or [row.get("prompt_row_sha256") for row in base_rows] != prompt_hashes
    ):
        raise DownstreamQualityAnalysisError("base prompt covariate contract differs")
    base_correctness = []
    for row in base_rows:
        reward = row.get("reward")
        if isinstance(reward, bool) or reward not in (0, 1, 0.0, 1.0):
            raise DownstreamQualityAnalysisError("base prompt covariate is nonbinary")
        base_correctness.append(int(reward))
    prompt_differences = [
        fmean(mixed[b][j] - immediate[b][j] for b in range(16)) for j in range(1024)
    ]

    margin_sensitivity = {
        str(margin): _classification(tuple(primary["student_interval_95"]), margin)
        for margin in (0.01, 0.015, 0.025, 0.03)
    }
    return {
        "schema": "m4-downstream-quality-trained-paired-analysis-result-v1",
        "status": "COMPLETE",
        "protocol_sha256": PROTOCOL_SHA256,
        "run_manifest_sha256": gate["run_manifest_sha256"],
        "completion_gate_sha256": gate_sha,
        "causal_unit": "matched_training_seed_block",
        "primary": primary,
        "block_differences_mixed_minus_immediate": block_differences,
        "runs": run_summaries,
        "sensitivities": {
            "prompt_fixed_effect_clustered_by_training_block": prompt_sensitivity,
            "v8_base_correctness": _base_covariate_sensitivity(
                prompt_differences=prompt_differences,
                base_correctness=base_correctness,
            ),
            "materiality_margin": margin_sensitivity,
        },
        "prompt_count_is_degrees_of_freedom": False,
        "historical_v8_enters_primary_estimator": False,
        "as_treated_primary_analysis": False,
        "automatic_retry_replacement_or_extension": False,
        "claim_boundary": "total effect of randomized release policy in the tested setting; exclusive M4 mediation is not identified",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--embargo", type=Path, required=True)
    parser.add_argument("--completion-gate", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise DownstreamQualityAnalysisError(f"output already exists: {args.output}")
    result = analyze(
        protocol_path=args.protocol,
        run_manifest_path=args.run_manifest,
        embargo_path=args.embargo,
        completion_gate_path=args.completion_gate,
        results_root=args.results_root,
    )
    args.output.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
