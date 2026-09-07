# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Build the no-training lock for Qwen3-0.6B/GSM8K grid completion."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import NormalDist

from omegaconf import OmegaConf

from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    validate_single_controller_config,
)
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from tools.opportunity_loss_pipeline import (
    OpportunityLossPipelineError,
    _parse_protocol,
)
from tools.opportunity_loss_workload_transport_pipeline import (
    GRID_SECONDARY_CONTRACT,
    validate_workload_transport_contract,
)
from tools.opportunity_loss_workload_transport_preflight import (
    IMAGE_COMMIT,
    IMAGE_PATH,
    IMAGE_SHA256,
    WorkloadTransportPreflightError,
    _lower_hex,
    _sha,
    _write_canonical,
)
from tools.opportunity_loss_workload_transport_r2_preflight import (
    PINNED_FILES as R2_PINNED_FILES,
)
from tools.opportunity_loss_workload_transport_r2_preflight import (
    validate_r2_workload_transport_config,
)


CONFIG_PATH = (
    "examples/configs/"
    "grpo_math_1B_megatron_single_controller_m4_qwen3_0p6b_gsm8k_grid.yaml"
)
PROTOCOL_PATH = (
    "reports/auto_research/2026-09-07-m4-opportunity-loss-grid-completion/"
    "protocol_config.json"
)
POWER_PATH = (
    "reports/auto_research/2026-09-07-m4-opportunity-loss-grid-completion/"
    "power_capacity_plan.json"
)
AUDIT_PATH = (
    "reports/auto_research/2026-09-07-m4-opportunity-loss-grid-completion/"
    "harmonization_audit.json"
)
COMMON_WINDOW_AUDIT_PATH = (
    "reports/auto_research/2026-09-07-m4-opportunity-loss-grid-completion/"
    "historical_common_window_audit.json"
)
LOCAL_VALIDATION_PATH = (
    "reports/auto_research/2026-09-07-m4-opportunity-loss-grid-completion/"
    "local_validation.json"
)
QWEN_0P6B_OPENMATH_RESULT_PATH = (
    "reports/auto_research/2026-09-02-m4-opportunity-loss-followup/"
    "prospective-efficiency-audit/result.json"
)
QWEN_1P7B_OPENMATH_RESULT_PATH = (
    "reports/auto_research/2026-09-03-m4-opportunity-loss-transport/"
    "qwen3-1p7b-confirmatory-design/acquisition_result.json"
)
QWEN_1P7B_GSM8K_RESULT_PATH = (
    "reports/auto_research/2026-09-04-m4-opportunity-loss-workload-transport/"
    "qwen3-1p7b-gsm8k-confirmatory-design/acquisition_r2_result.json"
)
CONFIG_SHA256 = "daa70f06c2faba8162a4e5b2123cc16755df6f1ec8ab5603ead45d9df435522e"
PROTOCOL_SHA256 = "b6682795071d12a5d6b3eddcdf73536e045949f08ce881885dd330fba9885db6"
POWER_SHA256 = "a73ebfc96e00f40c28a9a55ece3eb4afeda7a48afcf2158d0f5d7755d55d95fc"
AUDIT_SHA256 = "55b84a5c34de8aa07b101b46fdf400c84036d7dd3088b2f923507d623fbbacfa"
COMMON_WINDOW_AUDIT_SHA256 = (
    "f7cb20cdc5fea17ea9edd5b66f4ecc21e27fd1c4342af2392e6d2d6480e5c27d"
)
QWEN_0P6B_OPENMATH_RESULT_SHA256 = (
    "971ee49d367628a24dfa4e5fac4b8c24a12ef2b3e0835181b26f3e9fbfa3067e"
)
QWEN_1P7B_OPENMATH_RESULT_SHA256 = (
    "7eec5871da27004eb665c7662abe813ff990c1a8cdad4a45773af24d0b8f7ade"
)
QWEN_1P7B_GSM8K_RESULT_SHA256 = (
    "3204cd86c95723ea312396e98d7b2ee41d07ce2392e59ad76bdf5425cf2cc9d6"
)
PINNED_FILES = tuple(
    dict.fromkeys(
        (
            *R2_PINNED_FILES,
            CONFIG_PATH,
            PROTOCOL_PATH,
            POWER_PATH,
            AUDIT_PATH,
            COMMON_WINDOW_AUDIT_PATH,
            LOCAL_VALIDATION_PATH,
            QWEN_0P6B_OPENMATH_RESULT_PATH,
            QWEN_1P7B_OPENMATH_RESULT_PATH,
            QWEN_1P7B_GSM8K_RESULT_PATH,
            "tools/opportunity_loss_grid_analysis.py",
            "tools/opportunity_loss_grid_completion_preflight.py",
            "tests/unit/tools/test_opportunity_loss_grid_analysis.py",
            "tests/unit/tools/test_opportunity_loss_grid_completion_protocol.py",
        )
    )
)


def validate_grid_completion_config(config: Mapping[str, object]) -> None:
    """Require a model-only scientific delta from the accepted GSM8K R2 config."""
    candidate = copy.deepcopy(dict(config))
    try:
        policy = candidate["policy"]
        async_rl = candidate["async_rl"]
        logger = candidate["logger"]
        if not isinstance(policy, dict) or not isinstance(async_rl, dict):
            raise KeyError
        tokenizer = policy["tokenizer"]
        release = async_rl["controlled_release_delay"]
        audit = async_rl["gradient_opportunity_audit"]
        if (
            not isinstance(tokenizer, dict)
            or not isinstance(release, dict)
            or not isinstance(audit, dict)
            or not isinstance(logger, dict)
        ):
            raise KeyError
        if policy.get("model_name") != "Qwen/Qwen3-0.6B":
            raise WorkloadTransportPreflightError("grid model identity disagrees")
        if tokenizer.get("name") != "Qwen/Qwen3-0.6B":
            raise WorkloadTransportPreflightError("grid tokenizer identity disagrees")
        if release.get("seed") != 20260915 or release.get("assignment_domain") != (
            "m4-opportunity-loss-qwen3-0p6b-gsm8k-grid-v1"
        ):
            raise WorkloadTransportPreflightError(
                "grid randomization identity disagrees"
            )
        if (
            async_rl.get("lifecycle_audit_path"),
            audit.get("output_path"),
            audit.get("observer_duty_path"),
            logger.get("log_dir"),
        ) != (
            "results/m4-qwen3-0p6b-gsm8k-grid/lifecycle.jsonl",
            "results/m4-qwen3-0p6b-gsm8k-grid/opportunity.jsonl",
            "results/m4-qwen3-0p6b-gsm8k-grid/observer-duty.json",
            "results/m4-qwen3-0p6b-gsm8k-grid/metrics",
        ):
            raise WorkloadTransportPreflightError("grid output paths disagree")

        policy["model_name"] = "Qwen/Qwen3-1.7B"
        tokenizer["name"] = "Qwen/Qwen3-1.7B"
        release["seed"] = 20260913
        release["assignment_domain"] = "m4-opportunity-loss-qwen3-1p7b-gsm8k-r2-v1"
        async_rl["lifecycle_audit_path"] = (
            "results/m4-qwen3-1p7b-gsm8k-transport-r2/lifecycle.jsonl"
        )
        audit["output_path"] = (
            "results/m4-qwen3-1p7b-gsm8k-transport-r2/opportunity.jsonl"
        )
        audit["observer_duty_path"] = (
            "results/m4-qwen3-1p7b-gsm8k-transport-r2/observer-duty.json"
        )
        logger["log_dir"] = "results/m4-qwen3-1p7b-gsm8k-transport-r2/metrics"
    except KeyError as error:
        raise WorkloadTransportPreflightError(
            "grid config structure disagrees"
        ) from error
    validate_r2_workload_transport_config(candidate)
    validate_single_controller_config(MasterConfig(**config))


def validate_power_capacity_plan(raw: Mapping[str, object]) -> None:
    """Recompute the exact prospective precision and classification-power gate."""
    try:
        capacity = raw["capacity"]
        precision = raw["planning_precision"]
        power = raw["power"]
        if not all(
            isinstance(value, Mapping) for value in (capacity, precision, power)
        ):
            raise KeyError
        standard_error = (
            float(precision["base_hac_standard_error"])
            * math.sqrt(
                int(precision["base_assignment_count"])
                / int(capacity["minimum_primary_assignments"])
            )
            * float(precision["inflation_factor_for_model_transport"])
        )
        critical_value = NormalDist().inv_cdf(0.975)
        wide_power = NormalDist().cdf(0.05 / standard_error - critical_value)
        near_power = NormalDist().cdf(0.025 / standard_error - critical_value)
        if (
            not math.isclose(
                standard_error,
                float(precision["planned_hac_standard_error"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or not math.isclose(
                wide_power,
                float(power["classification_power_at_0p25_material"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or not math.isclose(
                wide_power,
                float(power["classification_power_at_0p15_not_material"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or not math.isclose(
                near_power,
                float(power["classification_power_at_0p225_material"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or wide_power < float(power["minimum_required_at_0p15_and_0p25"])
            or near_power >= float(power["minimum_required_at_0p15_and_0p25"])
        ):
            raise WorkloadTransportPreflightError("grid power gate disagrees")
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, WorkloadTransportPreflightError):
            raise
        raise WorkloadTransportPreflightError("grid power plan is malformed") from error


def build_grid_completion_preflight_lock(
    *, repo: Path, source_commit: str, source_archive_sha256: str
) -> dict[str, object]:
    """Resolve and bind the fourth-cell package without permitting execution."""
    source_commit = _lower_hex(source_commit, length=40, name="source commit")
    source_archive_sha256 = _lower_hex(
        source_archive_sha256, length=64, name="source archive SHA256"
    )
    register_omegaconf_resolvers()
    resolved = OmegaConf.to_container(load_config(repo / CONFIG_PATH), resolve=True)
    if not isinstance(resolved, Mapping):
        raise WorkloadTransportPreflightError("resolved grid config must be an object")
    validate_grid_completion_config(resolved)

    expected_hashes = {
        CONFIG_PATH: CONFIG_SHA256,
        PROTOCOL_PATH: PROTOCOL_SHA256,
        POWER_PATH: POWER_SHA256,
        AUDIT_PATH: AUDIT_SHA256,
        COMMON_WINDOW_AUDIT_PATH: COMMON_WINDOW_AUDIT_SHA256,
        QWEN_0P6B_OPENMATH_RESULT_PATH: QWEN_0P6B_OPENMATH_RESULT_SHA256,
        QWEN_1P7B_OPENMATH_RESULT_PATH: QWEN_1P7B_OPENMATH_RESULT_SHA256,
        QWEN_1P7B_GSM8K_RESULT_PATH: QWEN_1P7B_GSM8K_RESULT_SHA256,
    }
    for name, expected in expected_hashes.items():
        if _sha((repo / name).read_bytes()) != expected:
            raise WorkloadTransportPreflightError(f"frozen file moved: {name}")

    protocol_raw = (repo / PROTOCOL_PATH).read_bytes()
    try:
        protocol_value, protocol, options = _parse_protocol(protocol_raw)
        validate_workload_transport_contract(protocol_value, protocol, options)
    except OpportunityLossPipelineError as error:
        raise WorkloadTransportPreflightError("grid protocol is invalid") from error
    if protocol_value.get("status") != (
        "FROZEN_LOCAL_PROTOCOL_PENDING_NO_TRAINING_PREFLIGHT"
    ):
        raise WorkloadTransportPreflightError("grid protocol status disagrees")
    if protocol_value.get("grid_secondary_analysis") != GRID_SECONDARY_CONTRACT:
        raise WorkloadTransportPreflightError("grid secondary contract disagrees")
    evidence = protocol_value.get("design_evidence")
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("prior_observations_may_enter_estimator") is not False
        or evidence.get("harmonization_audit_sha256") != AUDIT_SHA256
        or evidence.get("historical_common_window_audit_sha256")
        != COMMON_WINDOW_AUDIT_SHA256
        or evidence.get("power_capacity_plan_sha256") != POWER_SHA256
        or evidence.get("qwen3_0p6b_openmath_result_record_sha256")
        != QWEN_0P6B_OPENMATH_RESULT_SHA256
        or evidence.get("qwen3_1p7b_openmath_result_record_sha256")
        != QWEN_1P7B_OPENMATH_RESULT_SHA256
        or evidence.get("qwen3_1p7b_gsm8k_result_record_sha256")
        != QWEN_1P7B_GSM8K_RESULT_SHA256
    ):
        raise WorkloadTransportPreflightError("prior-study exclusion disagrees")

    power_raw = json.loads((repo / POWER_PATH).read_bytes())
    if not isinstance(power_raw, Mapping):
        raise WorkloadTransportPreflightError("grid power plan must be an object")
    validate_power_capacity_plan(power_raw)
    result_statuses = (
        json.loads((repo / QWEN_0P6B_OPENMATH_RESULT_PATH).read_bytes()).get("status"),
        json.loads((repo / QWEN_1P7B_OPENMATH_RESULT_PATH).read_bytes()).get("status"),
        json.loads((repo / QWEN_1P7B_GSM8K_RESULT_PATH).read_bytes()).get("status"),
    )
    if result_statuses != ("SUCCESS", "MATERIAL", "TERMINAL_COMPLETE_NOT_MATERIAL"):
        raise WorkloadTransportPreflightError("grid evidence statuses disagree")

    files: dict[str, dict[str, object]] = {}
    for name in PINNED_FILES:
        raw = (repo / name).read_bytes()
        files[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
    return {
        "schema": "m4-qwen3-0p6b-gsm8k-grid-no-training-preflight-lock-v1",
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "files": files,
        "protocol": {"sha256": _sha(protocol_raw), "size": len(protocol_raw)},
        "image": {
            "path": IMAGE_PATH,
            "sha256": IMAGE_SHA256,
            "embedded_commit": IMAGE_COMMIT,
        },
        "design": {
            "model": "Qwen3-0.6B",
            "dataset": "openai/gsm8k:main:train",
            "arms": ["control", "d5"],
            "trainer_steps": 558,
            "primary_versions": 500,
            "common_grid_versions": 400,
            "minimum_primary_assignments": 7395,
            "max_num_epochs": 2,
            "gpus": 2,
            "wall_clock_cap_hours": 4.0,
            "gpu_hour_cap": 8.0,
            "prior_observations_enter_estimator": False,
        },
        "training_allowed": False,
        "qualification_submission_allowed": False,
        "scientific_acquisition_allowed": False,
        "eos_submission_allowed": False,
        "automatic_retry": False,
        "automatic_extension": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_grid_completion_preflight_lock(
        repo=args.repo.resolve(),
        source_commit=args.source_commit,
        source_archive_sha256=args.source_archive_sha256,
    )
    _write_canonical(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
