# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Static gates for the paired NuminaMath neutral qualifications."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    validate_single_controller_config,
)
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


_REPO = Path(__file__).resolve().parents[3]
_REPORT = (
    _REPO
    / "reports/auto_research/2026-09-07-m4-opportunity-loss-numinamath-generalization"
)
_PLAN = _REPORT / "qualification_plan.json"
_BASE = {
    scale: _REPO
    / "examples/configs"
    / (
        "grpo_math_1B_megatron_single_controller_m4_qwen3_"
        f"{scale}_numinamath_generalization.yaml"
    )
    for scale in ("0p6b", "1p7b")
}
_QUALIFICATION = {
    scale: _REPO
    / "examples/configs"
    / (
        "grpo_math_1B_megatron_single_controller_m4_qwen3_"
        f"{scale}_numinamath_neutral_qualification.yaml"
    )
    for scale in ("0p6b", "1p7b")
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


def _resolved(path: Path) -> dict[str, Any]:
    value = OmegaConf.to_container(load_config(path), resolve=True)
    assert isinstance(value, dict)
    return value


def test_each_qualification_is_a_neutral_operational_delta_only() -> None:
    register_omegaconf_resolvers()
    expected_delta = {
        ("async_rl", "controlled_release_delay", "arms"),
        ("async_rl", "controlled_release_delay", "assignment_domain"),
        ("async_rl", "controlled_release_delay", "seed"),
        ("async_rl", "gradient_opportunity_audit", "observer_duty_path"),
        ("async_rl", "gradient_opportunity_audit", "output_path"),
        ("async_rl", "lifecycle_audit_path"),
        ("grpo", "max_num_steps"),
        ("logger", "log_dir"),
    }
    for scale in _BASE:
        base = _resolved(_BASE[scale])
        qualification = _resolved(_QUALIFICATION[scale])
        assert _changed_paths(base, qualification) == expected_delta
        assert qualification["grpo"]["max_num_steps"] == 32
        assert qualification["data"]["train"]["dataset_name"] == (
            "tools.m4_numinamath_pinned_dataset.M4NuminaMathPinnedDataset"
        )
        assert qualification["cluster"]["gpus_per_node"] == 2
        assert qualification["cluster"]["num_nodes"] == 1
        release = qualification["async_rl"]["controlled_release_delay"]
        assert release["enabled"] is True
        assert release["arms"] == [
            {"label": "neutral", "delay_seconds": 0.0, "mass": 1}
        ]
        validate_single_controller_config(MasterConfig(**qualification))


def test_plan_binds_both_configs_and_forbids_causal_use() -> None:
    plan = json.loads(_PLAN.read_bytes())
    assert plan["schema"] == "m4-numinamath-paired-neutral-qualification-plan-v1"
    assert plan["release"]["authorized_submission_count"] == 2
    assert plan["release"]["submission_attempt_limit_each_cell"] == 1
    assert plan["release"]["automatic_retry"] is False
    assert plan["release"]["automatic_extension"] is False
    assert plan["release"]["scientific_acquisition_authorized"] is False
    assert (
        plan["decision_gates_each_cell"][
            "qualification_data_allowed_in_any_causal_estimator"
        ]
        is False
    )
    for scale, key in (
        ("0p6b", "qwen3_0p6b_numinamath"),
        ("1p7b", "qwen3_1p7b_numinamath"),
    ):
        assert plan["cells"][key]["config_sha256"] == _sha256(_QUALIFICATION[scale])
