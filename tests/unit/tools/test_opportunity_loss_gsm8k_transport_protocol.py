# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Static tests for the Qwen3-1.7B GSM8K M4 transport protocol."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
import pytest

from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    validate_single_controller_config,
)
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from tools.opportunity_loss_pipeline import (
    OpportunityLossPipelineError,
    _parse_protocol,
)
from tools.opportunity_loss_transport_pipeline import _validate_transport_contract
from tools.opportunity_loss_workload_transport_pipeline import (
    validate_workload_transport_contract,
)


_REPO = Path(__file__).resolve().parents[3]
_BASE_CONFIG = (
    _REPO / "examples/configs/"
    "grpo_math_1B_megatron_single_controller_m4_qwen3_1p7b_transport.yaml"
)
_CONFIG = (
    _REPO / "examples/configs/"
    "grpo_math_1B_megatron_single_controller_m4_qwen3_1p7b_gsm8k_transport.yaml"
)
_R2_CONFIG = (
    _REPO / "examples/configs/"
    "grpo_math_1B_megatron_single_controller_m4_qwen3_1p7b_gsm8k_transport_r2.yaml"
)
_QUALIFICATION_CONFIG = (
    _REPO / "examples/configs/"
    "grpo_math_1B_megatron_single_controller_m4_qwen3_1p7b_gsm8k_neutral_qualification.yaml"
)
_AUDIT = (
    _REPO / "reports/auto_research/2026-09-04-m4-opportunity-loss-workload-transport/"
    "qwen3-1p7b-gsm8k-compatibility-audit/audit.json"
)
_PROTOCOL = (
    _REPO / "reports/auto_research/2026-09-04-m4-opportunity-loss-workload-transport/"
    "qwen3-1p7b-gsm8k-confirmatory-design/protocol_config.json"
)
_R2_PROTOCOL = (
    _REPO / "reports/auto_research/2026-09-04-m4-opportunity-loss-workload-transport/"
    "qwen3-1p7b-gsm8k-confirmatory-design/protocol_config_r2.json"
)


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


def test_gsm8k_overlay_is_dataset_only_plus_fresh_identity() -> None:
    register_omegaconf_resolvers()
    baseline = OmegaConf.to_container(load_config(_BASE_CONFIG), resolve=True)
    candidate = OmegaConf.to_container(load_config(_CONFIG), resolve=True)
    assert isinstance(baseline, dict) and isinstance(candidate, dict)
    assert _changed_paths(baseline, candidate) == {
        ("async_rl", "controlled_release_delay", "assignment_domain"),
        ("async_rl", "controlled_release_delay", "seed"),
        ("async_rl", "gradient_opportunity_audit", "observer_duty_path"),
        ("async_rl", "gradient_opportunity_audit", "output_path"),
        ("async_rl", "lifecycle_audit_path"),
        ("data", "train", "dataset_name"),
        ("data", "train", "extract_answer"),
        ("data", "train", "seed"),
        ("data", "train", "split"),
        ("data", "train", "split_validation_size"),
        ("data", "train", "subset"),
        ("logger", "log_dir"),
    }
    assert candidate["data"]["train"] == {
        "dataset_name": "gsm8k",
        "extract_answer": True,
        "seed": None,
        "split": "train",
        "split_validation_size": 0.0,
        "subset": "main",
    }
    assert candidate["data"]["default"]["prompt_file"] == "examples/prompts/cot.txt"
    assert candidate["data"]["default"]["processor"] == "math_hf_data_processor"
    assert candidate["data"]["default"]["env_name"] == "math"
    assert candidate["env"]["math"]["math_verify_impl"] == "hf_math_verify"


def test_gsm8k_neutral_qualification_uses_supported_zero_dose_instrument() -> None:
    register_omegaconf_resolvers()
    candidate = OmegaConf.to_container(load_config(_QUALIFICATION_CONFIG), resolve=True)
    assert isinstance(candidate, dict)
    release = candidate["async_rl"]["controlled_release_delay"]
    assert release["enabled"] is True
    assert release["assignment_domain"] == (
        "m4-opportunity-loss-qwen3-1p7b-gsm8k-neutral-qualification-v1"
    )
    assert release["arms"] == [{"label": "neutral", "delay_seconds": 0.0, "mass": 1}]
    assert candidate["async_rl"]["gradient_opportunity_audit"]["enabled"] is True
    assert candidate["grpo"]["max_num_steps"] == 32
    assert candidate["cluster"]["gpus_per_node"] == 2
    assert candidate["cluster"]["num_nodes"] == 1
    validate_single_controller_config(MasterConfig(**candidate))


def test_gsm8k_protocol_has_frozen_geometry_and_provenance() -> None:
    raw, protocol, options = _parse_protocol(_PROTOCOL.read_bytes())
    validate_workload_transport_contract(raw, protocol, options)
    assert options["protocol_identity"] == (
        "m4-opportunity-loss-qwen3-1p7b-gsm8k-workload-transport-v1"
    )
    assert protocol.assignment_domain == "m4-opportunity-loss-qwen3-1p7b-gsm8k-v1"
    assert protocol.assignment_seed == 20260911
    assert protocol.primary_start_version == 8
    assert protocol.primary_end_version == 507
    assert [(arm.label, arm.delay_seconds, arm.mass) for arm in protocol.arms] == [
        ("control", 0.0, 1),
        ("d5", 5.0, 1),
    ]
    assert options["bootstrap_seed"] == 20260912
    assert raw["runtime"]["trainer_steps"] == 558
    assert raw["resource_caps"] == {
        "automatic_extension": False,
        "automatic_retry": False,
        "gpu_hour_cap": 8.0,
        "gpus": 2,
        "wall_clock_cap_hours": 4.0,
    }
    assert raw["workload"] == {
        "dataset_name": "gsm8k",
        "extract_answer": True,
        "huggingface_path": "openai/gsm8k",
        "processor": "math_hf_data_processor",
        "prompt_file": "examples/prompts/cot.txt",
        "reward_verifier": "hf_math_verify",
        "split": "train",
        "split_seed": None,
        "split_validation_size": 0.0,
        "subset": "main",
        "task_name": "gsm8k",
    }
    audit_sha256 = hashlib.sha256(_AUDIT.read_bytes()).hexdigest()
    assert raw["design_evidence"]["workload_compatibility_audit_sha256"] == audit_sha256
    assert raw["status"] == "FROZEN_LOCAL_PROTOCOL_PENDING_NO_TRAINING_PREFLIGHT"


def test_gsm8k_r2_is_capacity_only_plus_fresh_identity() -> None:
    register_omegaconf_resolvers()
    predecessor = OmegaConf.to_container(load_config(_CONFIG), resolve=True)
    candidate = OmegaConf.to_container(load_config(_R2_CONFIG), resolve=True)
    assert isinstance(predecessor, dict) and isinstance(candidate, dict)
    assert _changed_paths(predecessor, candidate) == {
        ("async_rl", "controlled_release_delay", "assignment_domain"),
        ("async_rl", "controlled_release_delay", "seed"),
        ("async_rl", "gradient_opportunity_audit", "observer_duty_path"),
        ("async_rl", "gradient_opportunity_audit", "output_path"),
        ("async_rl", "lifecycle_audit_path"),
        ("grpo", "max_num_epochs"),
        ("logger", "log_dir"),
    }
    assert candidate["grpo"]["max_num_epochs"] == 2
    assert candidate["grpo"]["max_num_steps"] == 558
    assert candidate["async_rl"]["controlled_release_delay"]["seed"] == 20260913
    assert (
        candidate["async_rl"]["controlled_release_delay"]["assignment_domain"]
        == "m4-opportunity-loss-qwen3-1p7b-gsm8k-r2-v1"
    )
    validate_single_controller_config(MasterConfig(**candidate))


def test_gsm8k_r2_protocol_preserves_inference_and_binds_capacity() -> None:
    predecessor = json.loads(_PROTOCOL.read_bytes())
    raw, protocol, options = _parse_protocol(_R2_PROTOCOL.read_bytes())
    validate_workload_transport_contract(raw, protocol, options)
    assert options["protocol_identity"] == (
        "m4-opportunity-loss-qwen3-1p7b-gsm8k-workload-transport-r2-v1"
    )
    assert raw["analysis"] == (
        predecessor["analysis"]
        | {
            "inference": predecessor["analysis"]["inference"]
            | {"bootstrap_seed": 20260914}
        }
    )
    assert raw["runtime"] == predecessor["runtime"] | {"max_num_epochs": 2}
    assert raw["windows"] == predecessor["windows"]
    assert raw["resource_caps"] == predecessor["resource_caps"]
    assert raw["workload"] == predecessor["workload"]
    assert raw["exposure"] == {
        "epoch_specific_group_instance_is_assignment_unit": True,
        "max_num_epochs": 2,
        "r1_observations_enter_estimator": False,
        "repeated_prompt_exposure_declared": True,
        "rollout_pump_cancelled_at_completed_step": 558,
    }
    assert protocol.assignment_domain == ("m4-opportunity-loss-qwen3-1p7b-gsm8k-r2-v1")
    assert protocol.assignment_seed == 20260913


def test_gsm8k_r2_rejects_one_epoch_capacity_regression() -> None:
    mutated = copy.deepcopy(json.loads(_R2_PROTOCOL.read_bytes()))
    mutated["runtime"]["max_num_epochs"] = 1
    raw, protocol, options = _parse_protocol(
        (json.dumps(mutated, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )
    with pytest.raises(OpportunityLossPipelineError, match="runtime"):
        validate_workload_transport_contract(raw, protocol, options)


def test_gsm8k_contract_rejects_dataset_mutation() -> None:
    mutated = copy.deepcopy(json.loads(_PROTOCOL.read_bytes()))
    mutated["workload"]["subset"] = "socratic"
    raw, protocol, options = _parse_protocol(
        (json.dumps(mutated, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )
    try:
        validate_workload_transport_contract(raw, protocol, options)
    except ValueError as error:
        assert "dataset" in str(error)
    else:
        raise AssertionError("workload mutation was accepted")


def test_gsm8k_protocol_is_rejected_by_openmath_transport_analyzer() -> None:
    raw, protocol, options = _parse_protocol(_PROTOCOL.read_bytes())
    with pytest.raises(OpportunityLossPipelineError):
        _validate_transport_contract(raw, protocol, options)
