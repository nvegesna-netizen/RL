# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Build the fail-closed no-training lock for M4 Llama family transport."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import NormalDist

from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


REPORT_ROOT = "reports/auto_research/2026-09-08-m4-llama-family-transport"
IMAGE_PATH = (
    "/lustre/fsw/coreai_dlalgo_ci/nvegesna/nemo_rl_images/nemo-rl-nightly-5802754.sqsh"
)
IMAGE_SHA256 = "3df8114a0b3e60ef95ce13f8b982c7cc45d164aca71c63a87388b1e8434ee470"
IMAGE_COMMIT = "ae07eafe8035b5b2e84efa7234e70e7fd7e493c1"
PROTOCOL_PATH = f"{REPORT_ROOT}/protocol_config.json"
DESIGN_PATH = f"{REPORT_ROOT}/design.md"
POWER_PATH = f"{REPORT_ROOT}/power_capacity_plan.json"
HISTORICAL_SYNTHESIS_PATH = (
    "reports/auto_research/2026-09-08-m4-opportunity-loss-paper/six_cell_synthesis.json"
)
CONFIG_PATHS = {
    "llama3p2_1b_openmath": (
        "examples/configs/"
        "grpo_math_1B_megatron_single_controller_m4_"
        "llama3p2_1b_openmath_family_transport.yaml"
    ),
    "llama3p2_1b_gsm8k": (
        "examples/configs/"
        "grpo_math_1B_megatron_single_controller_m4_"
        "llama3p2_1b_gsm8k_family_transport.yaml"
    ),
}
QUALIFICATION_CONFIG_PATHS = {
    "llama3p2_1b_openmath": (
        "examples/configs/"
        "grpo_math_1B_megatron_single_controller_m4_"
        "llama3p2_1b_openmath_neutral_qualification.yaml"
    ),
    "llama3p2_1b_gsm8k": (
        "examples/configs/"
        "grpo_math_1B_megatron_single_controller_m4_"
        "llama3p2_1b_gsm8k_neutral_qualification.yaml"
    ),
}
PINNED_FILES = (
    DESIGN_PATH,
    POWER_PATH,
    PROTOCOL_PATH,
    HISTORICAL_SYNTHESIS_PATH,
    *CONFIG_PATHS.values(),
    *QUALIFICATION_CONFIG_PATHS.values(),
    "examples/configs/grpo_math_1B.yaml",
    "examples/configs/grpo_math_1B_megatron_single_controller.yaml",
    "examples/configs/grpo_math_1B_megatron_single_controller_m4_opportunity_loss_followup.yaml",
    "examples/configs/recipes/llm/grpo-llama3.2-1b-instruct-1n8g-megatron.yaml",
    "nemo_rl/algorithms/async_utils/controlled_release.py",
    "nemo_rl/algorithms/async_utils/gradient_opportunity.py",
    "nemo_rl/algorithms/async_utils/observer_duty.py",
    "nemo_rl/algorithms/async_utils/rollout_lifecycle.py",
    "nemo_rl/algorithms/single_controller.py",
    "nemo_rl/algorithms/single_controller_utils/config.py",
    "nemo_rl/data/datasets/response_datasets/gsm8k.py",
    "nemo_rl/data/datasets/response_datasets/openmathinstruct2.py",
    "nemo_rl/data/processors.py",
    "nemo_rl/environments/math_environment.py",
    "nemo_rl/experience/rollout_manager.py",
    "tools/m4_llama_family_transport_preflight.py",
    "tools/neutral_qualification_lifecycle.py",
    "tools/opportunity_loss_adjusted_inference.py",
    "tools/opportunity_loss_family_transport_analysis.py",
    "tools/opportunity_loss_inference.py",
    "tools/opportunity_loss_mechanism.py",
    "tools/opportunity_loss_pipeline.py",
    "tests/unit/single_controller/test_gradient_opportunity.py",
    "tests/unit/tools/test_m4_llama_qualification_repair_amendment.py",
    "tests/unit/tools/test_m4_llama_family_transport_preflight.py",
    "tests/unit/tools/test_m4_llama_family_transport_protocol.py",
    "tests/unit/tools/test_neutral_qualification_lifecycle.py",
    "tests/unit/tools/test_opportunity_loss_family_transport_analysis.py",
    f"{REPORT_ROOT}/qualification_repair.md",
    f"{REPORT_ROOT}/qualification_repair_amendment.json",
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class FamilyTransportPreflightError(ValueError):
    """Raised when the frozen family-transport design disagrees."""


def _at(value: Mapping[str, object], *path: str) -> object:
    current: object = value
    for component in path:
        if not isinstance(current, Mapping) or component not in current:
            raise FamilyTransportPreflightError(f"config lacks {'.'.join(path)}")
        current = current[component]
    return current


def _lower_hex(value: str, *, length: int, name: str) -> str:
    if len(value) != length or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise FamilyTransportPreflightError(f"{name} must be lowercase hexadecimal")
    return value


def _write_canonical(path: Path, value: Mapping[str, object]) -> None:
    raw = (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _validate_single_controller_schema(config: Mapping[str, object]) -> None:
    """Load the heavyweight runtime schema only at the explicit gate."""
    from nemo_rl.algorithms.single_controller_utils.config import (
        MasterConfig,
        validate_single_controller_config,
    )

    validate_single_controller_config(MasterConfig(**config))


def _validate_common_config(config: Mapping[str, object]) -> None:
    expected = {
        ("grpo", "max_num_steps"): 448,
        ("grpo", "num_prompts_per_step"): 4,
        ("grpo", "num_generations_per_prompt"): 8,
        ("grpo", "normalize_rewards"): True,
        ("grpo", "use_leave_one_out_baseline"): True,
        ("policy", "model_name"): "meta-llama/Llama-3.2-1B-Instruct",
        ("policy", "tokenizer", "name"): "meta-llama/Llama-3.2-1B-Instruct",
        ("policy", "train_global_batch_size"): 32,
        ("policy", "train_micro_batch_size"): 1,
        ("policy", "max_total_sequence_length"): 2048,
        ("policy", "make_sequence_length_divisible_by"): 1,
        ("policy", "dtensor_cfg", "enabled"): False,
        ("policy", "megatron_cfg", "enabled"): True,
        ("policy", "generation", "max_new_tokens"): 1792,
        ("policy", "generation", "vllm_cfg", "async_engine"): True,
        ("policy", "generation", "colocated", "enabled"): False,
        ("policy", "generation", "colocated", "resources", "gpus_per_node"): 1,
        ("policy", "generation", "colocated", "resources", "num_nodes"): 1,
        ("async_rl", "sampler", "name"): "windowed",
        ("async_rl", "sampler", "max_staleness_versions"): 1,
        ("async_rl", "sampler", "sample_freshest_first"): False,
        ("async_rl", "min_groups_for_streaming_train"): 4,
        ("async_rl", "max_inflight_prompts"): 16,
        ("async_rl", "max_buffered_rollouts"): 64,
        ("async_rl", "controlled_release_delay", "enabled"): True,
        ("async_rl", "gradient_opportunity_audit", "enabled"): True,
        ("loss_fn", "disable_ppo_ratio"): False,
        ("loss_fn", "positive_example_nll_weight"): 0.0,
        ("loss_fn", "sequence_level_importance_ratios"): False,
        ("loss_fn", "token_level_loss"): True,
        ("loss_fn", "use_cispo"): False,
        ("checkpointing", "enabled"): False,
        ("logger", "wandb_enabled"): False,
        ("logger", "tensorboard_enabled"): False,
        ("logger", "mlflow_enabled"): False,
        ("logger", "swanlab_enabled"): False,
        ("logger", "monitor_gpus"): False,
        ("data_plane", "enabled"): True,
        ("data_plane", "impl"): "transfer_queue",
        ("data_plane", "backend"): "simple",
        ("data", "default", "prompt_file"): "examples/prompts/cot.txt",
        ("data", "default", "processor"): "math_hf_data_processor",
        ("data", "default", "env_name"): "math",
        ("env", "math", "math_verify_impl"): "hf_math_verify",
        ("cluster", "gpus_per_node"): 2,
        ("cluster", "num_nodes"): 1,
    }
    for path, expected_value in expected.items():
        if _at(config, *path) != expected_value:
            raise FamilyTransportPreflightError(
                f"family-transport config field {'.'.join(path)} disagrees"
            )


def validate_acquisition_config(cell: str, config: Mapping[str, object]) -> None:
    """Validate one prospective acquisition without inspecting outcomes."""
    _validate_common_config(config)
    expected = {
        "llama3p2_1b_openmath": {
            "epochs": 1,
            "dataset": "OpenMathInstruct-2",
            "split_validation_size": 0.05,
            "seed": 20260929,
            "domain": "m4-opportunity-loss-llama3p2-1b-openmath-family-transport-v1",
            "root": "results/m4-llama3p2-1b-openmath-family-transport",
        },
        "llama3p2_1b_gsm8k": {
            "epochs": 2,
            "dataset": "gsm8k",
            "split_validation_size": 0.0,
            "seed": 20260930,
            "domain": "m4-opportunity-loss-llama3p2-1b-gsm8k-family-transport-v1",
            "root": "results/m4-llama3p2-1b-gsm8k-family-transport",
        },
    }
    if cell not in expected:
        raise FamilyTransportPreflightError("unknown prospective cell")
    item = expected[cell]
    checks = {
        ("grpo", "max_num_epochs"): item["epochs"],
        ("data", "train", "dataset_name"): item["dataset"],
        ("data", "train", "split_validation_size"): item["split_validation_size"],
        ("async_rl", "controlled_release_delay", "seed"): item["seed"],
        ("async_rl", "controlled_release_delay", "assignment_domain"): item["domain"],
        ("async_rl", "lifecycle_audit_path"): f"{item['root']}/lifecycle.jsonl",
        ("async_rl", "gradient_opportunity_audit", "output_path"): (
            f"{item['root']}/opportunity.jsonl"
        ),
        ("async_rl", "gradient_opportunity_audit", "observer_duty_path"): (
            f"{item['root']}/observer-duty.json"
        ),
        ("logger", "log_dir"): f"{item['root']}/metrics",
    }
    for path, expected_value in checks.items():
        if _at(config, *path) != expected_value:
            raise FamilyTransportPreflightError(
                f"{cell} config field {'.'.join(path)} disagrees"
            )
    if _at(config, "async_rl", "controlled_release_delay", "arms") != [
        {"label": "control", "delay_seconds": 0.0, "mass": 1},
        {"label": "d5", "delay_seconds": 5.0, "mass": 1},
    ]:
        raise FamilyTransportPreflightError(f"{cell} treatment arms disagree")
    if cell.endswith("gsm8k"):
        gsm8k = {
            ("data", "train", "subset"): "main",
            ("data", "train", "split"): "train",
            ("data", "train", "seed"): None,
            ("data", "train", "extract_answer"): True,
        }
        for path, expected_value in gsm8k.items():
            if _at(config, *path) != expected_value:
                raise FamilyTransportPreflightError(
                    f"{cell} config field {'.'.join(path)} disagrees"
                )
    _validate_single_controller_schema(config)


def validate_qualification_config(
    cell: str,
    qualification: Mapping[str, object],
    acquisition: Mapping[str, object],
) -> None:
    """Require a neutral, isolated 32-step qualification."""
    validate_acquisition_config(cell, acquisition)
    _validate_common_config(
        {**qualification, "grpo": {**qualification["grpo"], "max_num_steps": 448}}
    )  # type: ignore[index]
    if _at(qualification, "grpo", "max_num_steps") != 32:
        raise FamilyTransportPreflightError(f"{cell} qualification steps disagree")
    if _at(qualification, "async_rl", "controlled_release_delay", "arms") != [
        {"label": "neutral", "delay_seconds": 0.0, "mass": 1}
    ]:
        raise FamilyTransportPreflightError(f"{cell} qualification is not neutral")
    expected_identity = {
        "llama3p2_1b_openmath": (
            20261003,
            "m4-opportunity-loss-llama3p2-1b-openmath-neutral-qualification-v1",
            "results/m4-llama3p2-1b-openmath-neutral-qualification",
        ),
        "llama3p2_1b_gsm8k": (
            20261004,
            "m4-opportunity-loss-llama3p2-1b-gsm8k-neutral-qualification-v1",
            "results/m4-llama3p2-1b-gsm8k-neutral-qualification",
        ),
    }
    seed, domain, root = expected_identity[cell]
    checks = {
        ("async_rl", "controlled_release_delay", "seed"): seed,
        ("async_rl", "controlled_release_delay", "assignment_domain"): domain,
        ("async_rl", "lifecycle_audit_path"): f"{root}/lifecycle.jsonl",
        ("async_rl", "gradient_opportunity_audit", "output_path"): (
            f"{root}/opportunity.jsonl"
        ),
        ("async_rl", "gradient_opportunity_audit", "observer_duty_path"): (
            f"{root}/observer-duty.json"
        ),
        ("logger", "log_dir"): f"{root}/metrics",
    }
    for path, expected_value in checks.items():
        if _at(qualification, *path) != expected_value:
            raise FamilyTransportPreflightError(
                f"{cell} qualification field {'.'.join(path)} disagrees"
            )
    _validate_single_controller_schema(qualification)


def validate_power_plan(plan: Mapping[str, object]) -> None:
    """Recompute every outcome-blind precision and power value."""
    try:
        precision = plan["planning_precision"]
        alternatives = plan["planning_alternative"]
        tests = plan["power"]["tests"]  # type: ignore[index]
        minimum = float(plan["power"]["minimum_required_each_confirmatory_test"])  # type: ignore[index]
        if not all(
            isinstance(value, Mapping) for value in (precision, alternatives, tests)
        ):
            raise KeyError
        inflation = float(precision["transport_inflation_factor"])
        se_open = inflation * float(
            precision["openmath_hac_standard_error_proxy_before_inflation"]
        )
        se_gsm = inflation * float(
            precision["gsm8k_hac_standard_error_proxy_before_inflation"]
        )
        expected = {
            "openmath_materiality": (
                float(alternatives["llama_1p23b_openmath"]),
                0.2,
                se_open,
            ),
            "gsm8k_positive_effect": (
                float(alternatives["llama_1p23b_gsm8k"]),
                0.0,
                se_gsm,
            ),
            "openmath_minus_gsm8k": (
                float(alternatives["openmath_minus_gsm8k"]),
                0.0,
                math.hypot(se_open, se_gsm),
            ),
        }
        critical = NormalDist().inv_cdf(0.975)
        for name, (alternative, boundary, standard_error) in expected.items():
            record = tests[name]
            if not isinstance(record, Mapping):
                raise KeyError
            power = NormalDist().cdf(
                (alternative - boundary) / standard_error - critical
            )
            if (
                not math.isclose(
                    float(record["standard_error"]),
                    standard_error,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                or not math.isclose(
                    float(record["planned_power"]),
                    power,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                or power < minimum
            ):
                raise FamilyTransportPreflightError(
                    f"family-transport power gate disagrees: {name}"
                )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, FamilyTransportPreflightError):
            raise
        raise FamilyTransportPreflightError(
            "family-transport power plan is malformed"
        ) from error


def build_preflight_lock(
    *, repo: Path, source_commit: str, source_archive_sha256: str
) -> dict[str, object]:
    """Bind the frozen study while forbidding training and submission."""
    source_commit = _lower_hex(source_commit, length=40, name="source commit")
    source_archive_sha256 = _lower_hex(
        source_archive_sha256, length=64, name="source archive SHA256"
    )
    protocol_raw = (repo / PROTOCOL_PATH).read_bytes()
    protocol = json.loads(protocol_raw)
    if (
        protocol.get("protocol")
        != "m4-opportunity-loss-llama3p2-1b-family-transport-v1"
        or protocol.get("status")
        != "FROZEN_LOCAL_PROTOCOL_PENDING_NO_TRAINING_PREFLIGHT"
        or protocol.get("execution_lock")
        != {
            "analyze_only_after_both_terminal_artifacts_are_frozen": True,
            "both_acquisition_packages_must_be_frozen_before_first_submission": True,
            "failure_of_one_cell_permits_outcome_inspection": False,
            "outcome_conditioned_retry_or_extension": False,
            "submit_both_before_inspecting_either_causal_outcome": True,
        }
    ):
        raise FamilyTransportPreflightError(
            "family-transport protocol contract disagrees"
        )
    evidence = protocol.get("design_evidence")
    if not isinstance(evidence, Mapping):
        raise FamilyTransportPreflightError("design evidence is malformed")
    expected_hashes = {
        "analysis_tool_sha256": "tools/opportunity_loss_family_transport_analysis.py",
        "analysis_test_sha256": (
            "tests/unit/tools/test_opportunity_loss_family_transport_analysis.py"
        ),
        "completed_qwen_synthesis_sha256": HISTORICAL_SYNTHESIS_PATH,
        "design_sha256": DESIGN_PATH,
        "power_capacity_plan_sha256": POWER_PATH,
        "openmath_config_sha256": CONFIG_PATHS["llama3p2_1b_openmath"],
        "gsm8k_config_sha256": CONFIG_PATHS["llama3p2_1b_gsm8k"],
        "openmath_qualification_config_sha256": QUALIFICATION_CONFIG_PATHS[
            "llama3p2_1b_openmath"
        ],
        "gsm8k_qualification_config_sha256": QUALIFICATION_CONFIG_PATHS[
            "llama3p2_1b_gsm8k"
        ],
    }
    for key, name in expected_hashes.items():
        if evidence.get(key) != _sha((repo / name).read_bytes()):
            raise FamilyTransportPreflightError(f"frozen evidence moved: {name}")
    plan = json.loads((repo / POWER_PATH).read_bytes())
    validate_power_plan(plan)

    register_omegaconf_resolvers()
    acquisitions: dict[str, Mapping[str, object]] = {}
    qualifications: dict[str, Mapping[str, object]] = {}
    for cell in CONFIG_PATHS:
        acquisition = OmegaConf.to_container(
            load_config(repo / CONFIG_PATHS[cell]), resolve=True
        )
        qualification = OmegaConf.to_container(
            load_config(repo / QUALIFICATION_CONFIG_PATHS[cell]), resolve=True
        )
        if not isinstance(acquisition, Mapping) or not isinstance(
            qualification, Mapping
        ):
            raise FamilyTransportPreflightError(
                f"resolved family-transport config differs: {cell}"
            )
        validate_acquisition_config(cell, acquisition)
        validate_qualification_config(cell, qualification, acquisition)
        acquisitions[cell] = acquisition
        qualifications[cell] = qualification

    files: dict[str, dict[str, object]] = {}
    for name in PINNED_FILES:
        raw = (repo / name).read_bytes()
        files[name] = {"sha256": _sha(raw), "size": len(raw)}
    return {
        "schema": "m4-llama-family-transport-no-training-lock-v1",
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "files": files,
        "protocol": {"sha256": _sha(protocol_raw), "size": len(protocol_raw)},
        "image": {
            "path": IMAGE_PATH,
            "sha256": IMAGE_SHA256,
            "embedded_commit": IMAGE_COMMIT,
        },
        "model_access_probe": {
            "full_weight_download_allowed": False,
            "huggingface_id": "meta-llama/Llama-3.2-1B-Instruct",
            "required_files": [
                "config.json",
                "tokenizer.json_or_tokenizer_model",
                "tokenizer_config.json",
            ],
            "secret_value_logging_allowed": False,
        },
        "design": {
            "cells": ["llama3p2_1b_openmath", "llama3p2_1b_gsm8k"],
            "common_primary_versions": 400,
            "gpus_each_cell": 2,
            "minimum_primary_assignments_each_cell": 6900,
            "trainer_steps_each_cell": 448,
            "wall_clock_cap_hours_each_cell": 4.0,
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
    result = build_preflight_lock(
        repo=args.repo.resolve(),
        source_commit=args.source_commit,
        source_archive_sha256=args.source_archive_sha256,
    )
    _write_canonical(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
