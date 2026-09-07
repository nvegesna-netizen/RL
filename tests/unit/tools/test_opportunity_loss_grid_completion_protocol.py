# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Static tests for the Qwen3-0.6B/GSM8K M4 grid-completion design."""

from __future__ import annotations

import copy
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
from tools.opportunity_loss_pipeline import (
    OpportunityLossPipelineError,
    _parse_protocol,
)
from tools.opportunity_loss_grid_completion_preflight import (
    build_grid_completion_preflight_lock,
    validate_grid_completion_config,
    validate_power_capacity_plan,
)
from tools.opportunity_loss_workload_transport_preflight import (
    WorkloadTransportPreflightError,
)
from tools.opportunity_loss_workload_transport_pipeline import (
    GRID_SECONDARY_CONTRACT,
    validate_workload_transport_contract,
)


_REPO = Path(__file__).resolve().parents[3]
_PREDECESSOR_CONFIG = (
    _REPO / "examples/configs/"
    "grpo_math_1B_megatron_single_controller_m4_qwen3_1p7b_gsm8k_transport_r2.yaml"
)
_CONFIG = (
    _REPO / "examples/configs/"
    "grpo_math_1B_megatron_single_controller_m4_qwen3_0p6b_gsm8k_grid.yaml"
)
_PREDECESSOR_PROTOCOL = (
    _REPO / "reports/auto_research/2026-09-04-m4-opportunity-loss-workload-transport/"
    "qwen3-1p7b-gsm8k-confirmatory-design/protocol_config_r2.json"
)
_PROTOCOL = (
    _REPO / "reports/auto_research/2026-09-07-m4-opportunity-loss-grid-completion/"
    "protocol_config.json"
)
_POWER = (
    _REPO / "reports/auto_research/2026-09-07-m4-opportunity-loss-grid-completion/"
    "power_capacity_plan.json"
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


def test_grid_config_is_model_only_plus_fresh_identity() -> None:
    register_omegaconf_resolvers()
    predecessor = OmegaConf.to_container(load_config(_PREDECESSOR_CONFIG), resolve=True)
    candidate = OmegaConf.to_container(load_config(_CONFIG), resolve=True)
    assert isinstance(predecessor, dict) and isinstance(candidate, dict)
    assert _changed_paths(predecessor, candidate) == {
        ("async_rl", "controlled_release_delay", "assignment_domain"),
        ("async_rl", "controlled_release_delay", "seed"),
        ("async_rl", "gradient_opportunity_audit", "observer_duty_path"),
        ("async_rl", "gradient_opportunity_audit", "output_path"),
        ("async_rl", "lifecycle_audit_path"),
        ("logger", "log_dir"),
        ("policy", "model_name"),
        ("policy", "tokenizer", "name"),
    }
    assert candidate["policy"]["model_name"] == "Qwen/Qwen3-0.6B"
    assert candidate["policy"]["tokenizer"]["name"] == "Qwen/Qwen3-0.6B"
    assert candidate["grpo"]["max_num_steps"] == 558
    assert candidate["grpo"]["max_num_epochs"] == 2
    assert candidate["data"]["train"]["dataset_name"] == "gsm8k"
    validate_grid_completion_config(candidate)
    validate_single_controller_config(MasterConfig(**candidate))


def test_grid_no_training_lock_forbids_every_execution_path() -> None:
    lock = build_grid_completion_preflight_lock(
        repo=_REPO,
        source_commit="7" * 40,
        source_archive_sha256="8" * 64,
    )
    assert lock["training_allowed"] is False
    assert lock["qualification_submission_allowed"] is False
    assert lock["scientific_acquisition_allowed"] is False
    assert lock["eos_submission_allowed"] is False
    assert lock["automatic_retry"] is False
    assert lock["automatic_extension"] is False
    assert lock["design"]["model"] == "Qwen3-0.6B"
    assert lock["design"]["common_grid_versions"] == 400


def test_grid_protocol_preserves_primary_contract_and_freezes_secondary() -> None:
    predecessor = json.loads(_PREDECESSOR_PROTOCOL.read_bytes())
    raw, protocol, options = _parse_protocol(_PROTOCOL.read_bytes())
    validate_workload_transport_contract(raw, protocol, options)
    assert options["protocol_identity"] == (
        "m4-opportunity-loss-qwen3-0p6b-gsm8k-grid-completion-v1"
    )
    assert raw["analysis"] == predecessor["analysis"] | {
        "inference": predecessor["analysis"]["inference"] | {"bootstrap_seed": 20260916}
    }
    assert raw["arms"] == predecessor["arms"]
    assert raw["instrumentation"] == predecessor["instrumentation"]
    assert raw["mechanism_followup"] == predecessor["mechanism_followup"]
    assert raw["resource_caps"] == predecessor["resource_caps"]
    assert raw["windows"] == predecessor["windows"]
    assert raw["workload"] == predecessor["workload"]
    assert raw["runtime"] == predecessor["runtime"] | {"model": "Qwen3-0.6B"}
    assert raw["grid_secondary_analysis"] == GRID_SECONDARY_CONTRACT
    assert raw["exposure"]["prior_study_observations_enter_estimator"] is False
    assert protocol.assignment_domain == (
        "m4-opportunity-loss-qwen3-0p6b-gsm8k-grid-v1"
    )
    assert protocol.assignment_seed == 20260915


def test_grid_protocol_rejects_model_or_common_window_mutation() -> None:
    for path, value in (
        (("runtime", "model"), "Qwen3-1.7B"),
        (("grid_secondary_analysis", "common_primary_start_versions"), [8, 507]),
    ):
        mutated = copy.deepcopy(json.loads(_PROTOCOL.read_bytes()))
        target = mutated
        for component in path[:-1]:
            target = target[component]
        target[path[-1]] = value
        raw, protocol, options = _parse_protocol(
            (json.dumps(mutated, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
        with pytest.raises(OpportunityLossPipelineError):
            validate_workload_transport_contract(raw, protocol, options)


def test_power_plan_recomputes_and_passes_only_wide_margin_gate() -> None:
    plan = json.loads(_POWER.read_bytes())
    precision = plan["planning_precision"]
    power = plan["power"]
    standard_error = (
        precision["base_hac_standard_error"]
        * math.sqrt(
            precision["base_assignment_count"]
            / plan["capacity"]["minimum_primary_assignments"]
        )
        * precision["inflation_factor_for_model_transport"]
    )
    critical_value = NormalDist().inv_cdf(0.975)
    wide_margin_power = NormalDist().cdf(0.05 / standard_error - critical_value)
    near_margin_power = NormalDist().cdf(0.025 / standard_error - critical_value)
    assert standard_error == pytest.approx(precision["planned_hac_standard_error"])
    assert wide_margin_power == pytest.approx(
        power["classification_power_at_0p25_material"]
    )
    assert wide_margin_power == pytest.approx(
        power["classification_power_at_0p15_not_material"]
    )
    assert near_margin_power == pytest.approx(
        power["classification_power_at_0p225_material"]
    )
    assert wide_margin_power >= power["minimum_required_at_0p15_and_0p25"]
    assert near_margin_power < power["minimum_required_at_0p15_and_0p25"]
    validate_power_capacity_plan(plan)


def test_power_plan_rejects_optimistic_precision_mutation() -> None:
    plan = copy.deepcopy(json.loads(_POWER.read_bytes()))
    plan["planning_precision"]["inflation_factor_for_model_transport"] = 1.0
    with pytest.raises(WorkloadTransportPreflightError, match="power"):
        validate_power_capacity_plan(plan)
