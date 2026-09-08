# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the paired NuminaMath-1.5 no-training lock builder."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from omegaconf import OmegaConf
import pytest

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from tools import m4_numinamath_generalization_preflight as subject
from tools.opportunity_loss_workload_transport_preflight import (
    WorkloadTransportPreflightError,
)


_REPO = Path(__file__).resolve().parents[3]


def _configs():
    register_omegaconf_resolvers()
    return {
        cell: OmegaConf.to_container(load_config(_REPO / path), resolve=True)
        for cell, path in subject.CONFIG_PATHS.items()
    }


def test_pair_contract_and_power_pass() -> None:
    subject.validate_config_pair(_configs())
    plan = json.loads((_REPO / subject.POWER_PATH).read_bytes())
    subject.validate_power_plan(plan)


def test_pair_contract_rejects_geometry_mutation() -> None:
    configs = _configs()
    configs["qwen3_0p6b_numinamath"]["grpo"]["max_num_steps"] = 449
    with pytest.raises(WorkloadTransportPreflightError, match="contract"):
        subject.validate_config_pair(configs)


def test_power_rejects_optimistic_mutation() -> None:
    plan = json.loads((_REPO / subject.POWER_PATH).read_bytes())
    plan = copy.deepcopy(plan)
    plan["planning_precision"]["transport_inflation_factor"] = 1.0
    with pytest.raises(WorkloadTransportPreflightError, match="power"):
        subject.validate_power_plan(plan)


def test_lock_forbids_every_execution_path(monkeypatch) -> None:
    monkeypatch.setattr(subject, "validate_dataset_identity", lambda: None)
    lock = subject.build_preflight_lock(
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
    assert lock["design"]["models"] == ["Qwen3-0.6B", "Qwen3-1.7B"]
    assert lock["design"]["common_primary_versions"] == 400
    assert lock["dataset"]["source_rows"] == 896_215
    assert lock["dataset"]["filtered_rows"] == 680_786
    assert lock["dataset"]["filtered_unique_problems"] == 680_786
    assert len(lock["dataset"]["shards"]) == 3
