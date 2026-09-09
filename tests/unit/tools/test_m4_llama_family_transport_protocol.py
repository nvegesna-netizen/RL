# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Static checks for the prospective M4 Llama family-transport protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from omegaconf import OmegaConf
import pytest

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from tools import m4_llama_family_transport_preflight as subject


_REPO = Path(__file__).resolve().parents[3]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolved(path: str) -> dict[str, object]:
    value = OmegaConf.to_container(load_config(_REPO / path), resolve=True)
    assert isinstance(value, dict)
    return value


def test_protocol_binds_configs_power_and_historical_evidence() -> None:
    protocol = json.loads((_REPO / subject.PROTOCOL_PATH).read_bytes())
    evidence = protocol["design_evidence"]
    assert protocol["status"] == ("FROZEN_LOCAL_PROTOCOL_PENDING_NO_TRAINING_PREFLIGHT")
    expected = {
        "completed_qwen_synthesis_sha256": subject.HISTORICAL_SYNTHESIS_PATH,
        "design_sha256": subject.DESIGN_PATH,
        "power_capacity_plan_sha256": subject.POWER_PATH,
        "openmath_config_sha256": subject.CONFIG_PATHS["llama3p2_1b_openmath"],
        "gsm8k_config_sha256": subject.CONFIG_PATHS["llama3p2_1b_gsm8k"],
        "openmath_qualification_config_sha256": (
            subject.QUALIFICATION_CONFIG_PATHS["llama3p2_1b_openmath"]
        ),
        "gsm8k_qualification_config_sha256": (
            subject.QUALIFICATION_CONFIG_PATHS["llama3p2_1b_gsm8k"]
        ),
    }
    for key, path in expected.items():
        assert evidence[key] == _sha(_REPO / path)


def test_acquisition_and_qualification_configs_resolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "_validate_single_controller_schema", lambda _: None)
    register_omegaconf_resolvers()
    for cell in subject.CONFIG_PATHS:
        acquisition = _resolved(subject.CONFIG_PATHS[cell])
        qualification = _resolved(subject.QUALIFICATION_CONFIG_PATHS[cell])
        subject.validate_acquisition_config(cell, acquisition)
        subject.validate_qualification_config(cell, qualification, acquisition)
        assert qualification["grpo"]["max_num_steps"] == 32  # type: ignore[index]
        assert acquisition["grpo"]["max_num_steps"] == 448  # type: ignore[index]


def test_single_controller_runtime_schema_accepts_configs() -> None:
    if sys.platform == "darwin":
        pytest.skip("full runtime schema is verified by the Linux EOS preflight")
    register_omegaconf_resolvers()
    paths = (
        *subject.CONFIG_PATHS.values(),
        *subject.QUALIFICATION_CONFIG_PATHS.values(),
    )
    for path in paths:
        subject._validate_single_controller_schema(_resolved(path))


def test_paired_execution_lock_forbids_outcome_conditioning() -> None:
    protocol = json.loads((_REPO / subject.PROTOCOL_PATH).read_bytes())
    assert protocol["execution_lock"] == {
        "analyze_only_after_both_terminal_artifacts_are_frozen": True,
        "both_acquisition_packages_must_be_frozen_before_first_submission": True,
        "failure_of_one_cell_permits_outcome_inspection": False,
        "outcome_conditioned_retry_or_extension": False,
        "submit_both_before_inspecting_either_causal_outcome": True,
    }
    assert protocol["scope"]["pure_model_family_causal_effect_supported"] is False
    assert protocol["scope"]["family_wide_generalization_supported"] is False


def test_power_plan_recomputes() -> None:
    subject.validate_power_plan(json.loads((_REPO / subject.POWER_PATH).read_bytes()))
