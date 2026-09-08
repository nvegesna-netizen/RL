# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Static tests for the paired NuminaMath-1.5 M4 generalization design."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

from omegaconf import OmegaConf
import pytest

from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    validate_single_controller_config,
)
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from tools import m4_numinamath_pinned_dataset as pinned_dataset


_REPO = Path(__file__).resolve().parents[3]
_REPORT = (
    _REPO / "reports/auto_research/"
    "2026-09-07-m4-opportunity-loss-numinamath-generalization"
)
_PROTOCOL = _REPORT / "protocol_config.json"
_POWER = _REPORT / "power_capacity_plan.json"
_AUDIT = _REPORT / "candidate_audit.json"
_CONFIGS = {
    "qwen3_0p6b_numinamath": _REPO / "examples/configs/"
    "grpo_math_1B_megatron_single_controller_m4_qwen3_0p6b_numinamath_generalization.yaml",
    "qwen3_1p7b_numinamath": _REPO / "examples/configs/"
    "grpo_math_1B_megatron_single_controller_m4_qwen3_1p7b_numinamath_generalization.yaml",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def test_protocol_binds_candidate_dataset_configs_and_historical_inputs() -> None:
    protocol = json.loads(_PROTOCOL.read_bytes())
    evidence = protocol["design_evidence"]
    assert protocol["status"] == ("FROZEN_LOCAL_PROTOCOL_PENDING_NO_TRAINING_PREFLIGHT")
    assert evidence["candidate_audit_sha256"] == _sha256(_AUDIT)
    assert evidence["power_capacity_plan_sha256"] == _sha256(_POWER)
    assert evidence["pinned_dataset_loader_sha256"] == _sha256(
        _REPO / "tools/m4_numinamath_pinned_dataset.py"
    )
    for cell, path in _CONFIGS.items():
        assert evidence[f"{cell.removesuffix('_numinamath')}_config_sha256"] == (
            _sha256(path)
        )
    workload = protocol["workload"]
    audit = json.loads(_AUDIT.read_bytes())["candidates"]["numinamath_1p5"]
    assert workload["repository_revision"] == pinned_dataset.DATASET_REVISION
    assert {
        name: record["sha256"] for name, record in workload["shards"].items()
    } == pinned_dataset.DATA_FILE_SHA256
    assert workload["expected_source_rows"] == pinned_dataset.EXPECTED_SOURCE_ROWS
    assert workload["expected_filtered_rows"] == pinned_dataset.EXPECTED_FILTERED_ROWS
    assert (
        workload["expected_unique_filtered_problems"]
        == (audit["filtered_unique_problems"])
    )
    assert protocol["historical_fixed_inputs"]["common_primary_start_versions"] == [
        8,
        407,
    ]


def test_pair_resolves_and_differs_only_by_model_and_fresh_identity() -> None:
    register_omegaconf_resolvers()
    resolved = {
        cell: OmegaConf.to_container(load_config(path), resolve=True)
        for cell, path in _CONFIGS.items()
    }
    for config in resolved.values():
        assert isinstance(config, dict)
        assert config["grpo"]["max_num_steps"] == 448
        assert config["data"]["train"]["dataset_name"] == (
            "tools.m4_numinamath_pinned_dataset.M4NuminaMathPinnedDataset"
        )
        assert config["data"]["validation"] is None
        assert config["cluster"]["gpus_per_node"] == 2
        assert config["cluster"]["num_nodes"] == 1
        validate_single_controller_config(MasterConfig(**config))
    assert _changed_paths(
        resolved["qwen3_0p6b_numinamath"],
        resolved["qwen3_1p7b_numinamath"],
    ) == {
        ("async_rl", "controlled_release_delay", "assignment_domain"),
        ("async_rl", "controlled_release_delay", "seed"),
        ("async_rl", "gradient_opportunity_audit", "observer_duty_path"),
        ("async_rl", "gradient_opportunity_audit", "output_path"),
        ("async_rl", "lifecycle_audit_path"),
        ("logger", "log_dir"),
        ("policy", "model_name"),
        ("policy", "tokenizer", "name"),
    }


def test_power_plan_recomputes_from_independent_cell_variances() -> None:
    plan = json.loads(_POWER.read_bytes())
    precision = plan["planning_precision"]
    fixed = precision["fixed_historical_gsm8k_hac_standard_errors"]
    proxies = precision[
        "new_numinamath_cell_proxy_hac_standard_errors_before_inflation"
    ]
    inflation = precision["transport_inflation_factor"]
    standard_error = math.sqrt(
        sum(value**2 for value in fixed.values())
        + sum((inflation * value) ** 2 for value in proxies.values())
    )
    effect = plan["power"]["prior_interaction_magnitude"]
    critical = NormalDist().inv_cdf(0.975)
    power = NormalDist().cdf(effect / standard_error - critical) + NormalDist().cdf(
        -effect / standard_error - critical
    )
    assert standard_error == pytest.approx(precision["combined_hac_standard_error"])
    assert power == pytest.approx(
        plan["power"]["classification_power_at_prior_interaction_magnitude"]
    )
    assert power >= plan["power"]["minimum_required"]


def test_paired_lock_forbids_outcome_conditioned_stopping() -> None:
    protocol = json.loads(_PROTOCOL.read_bytes())
    lock = protocol["paired_acquisition_lock"]
    assert lock == {
        "analyze_only_after_both_terminal_artifacts_are_frozen": True,
        "both_packages_must_be_frozen_before_first_submission": True,
        "failure_of_one_cell_permits_outcome_inspection": False,
        "outcome_conditioned_retry_or_extension": False,
        "submit_both_before_inspecting_either_causal_outcome": True,
    }
    assert protocol["resource_caps_each_cell"]["automatic_retry"] is False
    assert protocol["resource_caps_each_cell"]["automatic_extension"] is False
