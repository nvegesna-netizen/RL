# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Build the fail-closed no-training lock for paired M4 generalization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any

from omegaconf import OmegaConf

from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    validate_single_controller_config,
)
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from tools.m4_numinamath_pinned_dataset import (
    DATASET_REVISION,
    DATA_FILE_SHA256,
    DATA_FILE_SIZE,
    EXPECTED_FILTERED_ROWS,
    EXPECTED_SOURCE_ROWS,
    M4NuminaMathPinnedDataset,
)
from tools.opportunity_loss_workload_transport_preflight import (
    IMAGE_COMMIT,
    IMAGE_PATH,
    IMAGE_SHA256,
    WorkloadTransportPreflightError,
    _lower_hex,
    _write_canonical,
)
from tools.opportunity_loss_workload_transport_r2_preflight import (
    PINNED_FILES as R2_PINNED_FILES,
)


REPORT_ROOT = (
    "reports/auto_research/2026-09-07-m4-opportunity-loss-numinamath-generalization"
)
CONFIG_PATHS = {
    "qwen3_0p6b_numinamath": (
        "examples/configs/"
        "grpo_math_1B_megatron_single_controller_m4_qwen3_0p6b_numinamath_generalization.yaml"
    ),
    "qwen3_1p7b_numinamath": (
        "examples/configs/"
        "grpo_math_1B_megatron_single_controller_m4_qwen3_1p7b_numinamath_generalization.yaml"
    ),
}
PROTOCOL_PATH = f"{REPORT_ROOT}/protocol_config.json"
POWER_PATH = f"{REPORT_ROOT}/power_capacity_plan.json"
CANDIDATE_AUDIT_PATH = f"{REPORT_ROOT}/candidate_audit.json"
LOCAL_VALIDATION_PATH = f"{REPORT_ROOT}/local_validation.json"
EXPECTED_HASHES = {
    PROTOCOL_PATH: "bbe2c07211952e3e46d4511524f68e864606a7c63f271ae8e064263044dad0c8",
    POWER_PATH: "0066e02ad33c9d7141da019762318af18603464045cd6321b1c4e493c7dde89e",
    CANDIDATE_AUDIT_PATH: "8f9dcc0fb25387e43f8f4bc32bd41c074431eb37b51997bde1c6c5139d7f4433",
    LOCAL_VALIDATION_PATH: "8221a697dfd16a91a7404cc45e363a9a28d70d662727740d999d6fbb234b52ff",
    CONFIG_PATHS["qwen3_0p6b_numinamath"]: (
        "bbc134de9b6c590d76b2f6b7d63520435fae532793c16e6676eeb01aa7829e08"
    ),
    CONFIG_PATHS["qwen3_1p7b_numinamath"]: (
        "fc77637fe482262f5afb632297a1d4f7384b77d1ee8d5dc6bf7a597cbf36ea3c"
    ),
    "tools/m4_numinamath_pinned_dataset.py": (
        "5bb73e0736ccedc99f8f958c87edf48f47cf044556a9de457f10e1e7a5b26aaa"
    ),
}
PINNED_FILES = tuple(
    dict.fromkeys(
        (
            *R2_PINNED_FILES,
            *EXPECTED_HASHES,
            "tools/m4_numinamath_generalization_preflight.py",
            "tests/unit/tools/test_m4_numinamath_pinned_dataset.py",
            "tests/unit/tools/test_m4_numinamath_generalization_protocol.py",
            "tests/unit/tools/test_m4_numinamath_generalization_preflight.py",
            "tools/opportunity_loss_grid_analysis.py",
            "tests/unit/tools/test_opportunity_loss_grid_analysis.py",
        )
    )
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _changed_paths(
    left: Any, right: Any, prefix: tuple[str, ...] = ()
) -> set[tuple[str, ...]]:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return {prefix} if left != right else set()
    result: set[tuple[str, ...]] = set()
    for key in left.keys() | right.keys():
        path = (*prefix, key)
        if key not in left or key not in right:
            result.add(path)
        else:
            result.update(_changed_paths(left[key], right[key], path))
    return result


def validate_config_pair(configs: Mapping[str, Mapping[str, object]]) -> None:
    """Require identical paired geometry apart from model and fresh identity."""
    expected_cells = set(CONFIG_PATHS)
    if set(configs) != expected_cells:
        raise WorkloadTransportPreflightError("paired config cells disagree")
    expected = {
        "qwen3_0p6b_numinamath": (
            "Qwen/Qwen3-0.6B",
            20260925,
            "m4-opportunity-loss-qwen3-0p6b-numinamath-generalization-v1",
            "results/m4-qwen3-0p6b-numinamath-generalization",
        ),
        "qwen3_1p7b_numinamath": (
            "Qwen/Qwen3-1.7B",
            20260927,
            "m4-opportunity-loss-qwen3-1p7b-numinamath-generalization-v1",
            "results/m4-qwen3-1p7b-numinamath-generalization",
        ),
    }
    for cell, config in configs.items():
        model, seed, domain, output_root = expected[cell]
        try:
            policy = config["policy"]
            grpo = config["grpo"]
            data = config["data"]
            async_rl = config["async_rl"]
            cluster = config["cluster"]
            logger = config["logger"]
            if not all(
                isinstance(value, Mapping)
                for value in (policy, grpo, data, async_rl, cluster, logger)
            ):
                raise TypeError
            tokenizer = policy["tokenizer"]
            train = data["train"]
            default = data["default"]
            release = async_rl["controlled_release_delay"]
            audit = async_rl["gradient_opportunity_audit"]
            if not all(
                isinstance(value, Mapping)
                for value in (tokenizer, train, default, release, audit)
            ):
                raise TypeError
            if (
                policy["model_name"] != model
                or tokenizer["name"] != model
                or grpo["max_num_steps"] != 448
                or grpo["max_num_epochs"] != 1
                or grpo["num_prompts_per_step"] != 4
                or grpo["num_generations_per_prompt"] != 8
                or train["dataset_name"]
                != "tools.m4_numinamath_pinned_dataset.M4NuminaMathPinnedDataset"
                or data["validation"] is not None
                or default["processor"] != "math_hf_data_processor"
                or default["env_name"] != "math"
                or release["seed"] != seed
                or release["assignment_domain"] != domain
                or async_rl["lifecycle_audit_path"] != f"{output_root}/lifecycle.jsonl"
                or audit["output_path"] != f"{output_root}/opportunity.jsonl"
                or audit["observer_duty_path"] != f"{output_root}/observer-duty.json"
                or logger["log_dir"] != f"{output_root}/metrics"
                or cluster["gpus_per_node"] != 2
                or cluster["num_nodes"] != 1
            ):
                raise WorkloadTransportPreflightError(
                    f"paired config contract disagrees: {cell}"
                )
        except (AssertionError, KeyError, TypeError) as error:
            raise WorkloadTransportPreflightError(
                f"paired config is malformed: {cell}"
            ) from error
        validate_single_controller_config(MasterConfig(**config))
    if _changed_paths(
        dict(configs["qwen3_0p6b_numinamath"]),
        dict(configs["qwen3_1p7b_numinamath"]),
    ) != {
        ("async_rl", "controlled_release_delay", "assignment_domain"),
        ("async_rl", "controlled_release_delay", "seed"),
        ("async_rl", "gradient_opportunity_audit", "observer_duty_path"),
        ("async_rl", "gradient_opportunity_audit", "output_path"),
        ("async_rl", "lifecycle_audit_path"),
        ("logger", "log_dir"),
        ("policy", "model_name"),
        ("policy", "tokenizer", "name"),
    }:
        raise WorkloadTransportPreflightError("paired config delta disagrees")


def validate_power_plan(plan: Mapping[str, object]) -> None:
    """Recompute independent-cell precision and the frozen power gate."""
    try:
        precision = plan["planning_precision"]
        power = plan["power"]
        if not isinstance(precision, Mapping) or not isinstance(power, Mapping):
            raise KeyError
        fixed = precision["fixed_historical_gsm8k_hac_standard_errors"]
        proxies = precision[
            "new_numinamath_cell_proxy_hac_standard_errors_before_inflation"
        ]
        if not isinstance(fixed, Mapping) or not isinstance(proxies, Mapping):
            raise KeyError
        inflation = float(precision["transport_inflation_factor"])
        standard_error = math.sqrt(
            sum(float(value) ** 2 for value in fixed.values())
            + sum((inflation * float(value)) ** 2 for value in proxies.values())
        )
        effect = float(power["prior_interaction_magnitude"])
        critical = NormalDist().inv_cdf(0.975)
        planned_power = NormalDist().cdf(effect / standard_error - critical)
        planned_power += NormalDist().cdf(-effect / standard_error - critical)
        if (
            not math.isclose(
                standard_error,
                float(precision["combined_hac_standard_error"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or not math.isclose(
                planned_power,
                float(power["classification_power_at_prior_interaction_magnitude"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or planned_power < float(power["minimum_required"])
        ):
            raise WorkloadTransportPreflightError("paired power gate disagrees")
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, WorkloadTransportPreflightError):
            raise
        raise WorkloadTransportPreflightError(
            "paired power plan is malformed"
        ) from error


def validate_dataset_identity() -> None:
    """Load the immutable source and require the registered mapped cardinality."""
    dataset = M4NuminaMathPinnedDataset()
    if len(dataset.dataset) != EXPECTED_FILTERED_ROWS:
        raise WorkloadTransportPreflightError(
            "mapped NuminaMath-1.5 cardinality disagrees"
        )


def build_preflight_lock(
    *, repo: Path, source_commit: str, source_archive_sha256: str
) -> dict[str, object]:
    """Resolve and bind both cells without permitting any training path."""
    source_commit = _lower_hex(source_commit, length=40, name="source commit")
    source_archive_sha256 = _lower_hex(
        source_archive_sha256, length=64, name="source archive SHA256"
    )
    for name, expected_hash in EXPECTED_HASHES.items():
        if _sha((repo / name).read_bytes()) != expected_hash:
            raise WorkloadTransportPreflightError(f"frozen file moved: {name}")
    protocol = json.loads((repo / PROTOCOL_PATH).read_bytes())
    power = json.loads((repo / POWER_PATH).read_bytes())
    amendment = protocol.get("amendment")
    if (
        protocol.get("protocol")
        != "m4-opportunity-loss-numinamath-paired-generalization-v2"
        or protocol.get("status")
        != "FROZEN_REPAIRED_LOCAL_PROTOCOL_PENDING_NO_TRAINING_PREFLIGHT"
        or not isinstance(amendment, Mapping)
        or amendment.get("scientific_design_changed") is not False
        or amendment.get("causal_outcome_observed_before_amendment") is not False
        or protocol.get("paired_acquisition_lock")
        != {
            "analyze_only_after_both_terminal_artifacts_are_frozen": True,
            "both_packages_must_be_frozen_before_first_submission": True,
            "failure_of_one_cell_permits_outcome_inspection": False,
            "outcome_conditioned_retry_or_extension": False,
            "submit_both_before_inspecting_either_causal_outcome": True,
        }
    ):
        raise WorkloadTransportPreflightError("paired protocol contract disagrees")
    if not isinstance(power, Mapping):
        raise WorkloadTransportPreflightError("paired power plan must be an object")
    validate_power_plan(power)
    register_omegaconf_resolvers()
    configs: dict[str, Mapping[str, object]] = {}
    for cell, name in CONFIG_PATHS.items():
        resolved = OmegaConf.to_container(load_config(repo / name), resolve=True)
        if not isinstance(resolved, Mapping):
            raise WorkloadTransportPreflightError(f"resolved config differs: {cell}")
        configs[cell] = resolved
    validate_config_pair(configs)
    validate_dataset_identity()
    files: dict[str, dict[str, object]] = {}
    for name in PINNED_FILES:
        raw = (repo / name).read_bytes()
        files[name] = {"sha256": _sha(raw), "size": len(raw)}
    return {
        "schema": "m4-numinamath-paired-generalization-no-training-lock-v2",
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "files": files,
        "image": {
            "path": IMAGE_PATH,
            "sha256": IMAGE_SHA256,
            "embedded_commit": IMAGE_COMMIT,
        },
        "dataset": {
            "revision": DATASET_REVISION,
            "filtered_rows": EXPECTED_FILTERED_ROWS,
            "filtered_unique_problems": EXPECTED_FILTERED_ROWS,
            "shards": {
                name: {"sha256": digest, "size": DATA_FILE_SIZE[name]}
                for name, digest in DATA_FILE_SHA256.items()
            },
            "source_rows": EXPECTED_SOURCE_ROWS,
        },
        "design": {
            "models": ["Qwen3-0.6B", "Qwen3-1.7B"],
            "dataset": "NuminaMath-1.5",
            "arms": ["control", "d5"],
            "trainer_steps_each_cell": 448,
            "common_primary_versions": 400,
            "minimum_primary_assignments_each_cell": 6900,
            "gpus_each_cell": 2,
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
