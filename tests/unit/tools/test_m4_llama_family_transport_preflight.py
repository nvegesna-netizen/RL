# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fail-closed checks for the M4 Llama family-transport preflight lock."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from omegaconf import OmegaConf
import pytest

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from tools import m4_llama_family_transport_preflight as subject


_REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _lightweight_local_schema_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Full runtime-schema validation is repeated inside the Linux preflight."""
    monkeypatch.setattr(subject, "_validate_single_controller_schema", lambda _: None)


def _config(cell: str) -> dict[str, object]:
    register_omegaconf_resolvers()
    value = OmegaConf.to_container(
        load_config(_REPO / subject.CONFIG_PATHS[cell]), resolve=True
    )
    assert isinstance(value, dict)
    return value


def test_lock_forbids_every_execution_path() -> None:
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
    assert lock["model_access_probe"]["full_weight_download_allowed"] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("grpo", "max_num_steps"), 447),
        (("policy", "model_name"), "Qwen/Qwen3-1.7B"),
        (("policy", "make_sequence_length_divisible_by"), 8),
        (("async_rl", "controlled_release_delay", "seed"), 1),
        (("async_rl", "controlled_release_delay", "arms"), []),
        (("cluster", "gpus_per_node"), 4),
    ],
)
def test_acquisition_mutations_fail_closed(
    path: tuple[str, ...], value: object
) -> None:
    config = copy.deepcopy(_config("llama3p2_1b_openmath"))
    target = config
    for component in path[:-1]:
        target = target[component]  # type: ignore[index,assignment]
    target[path[-1]] = value
    with pytest.raises((subject.FamilyTransportPreflightError, ValueError)):
        subject.validate_acquisition_config("llama3p2_1b_openmath", config)


def test_power_mutation_fails_closed() -> None:
    plan = json.loads((_REPO / subject.POWER_PATH).read_bytes())
    plan["power"]["tests"]["openmath_materiality"]["planned_power"] = 0.8
    with pytest.raises(subject.FamilyTransportPreflightError, match="power"):
        subject.validate_power_plan(plan)


def test_malformed_source_identity_fails_closed() -> None:
    with pytest.raises(subject.FamilyTransportPreflightError, match="source commit"):
        subject.build_preflight_lock(
            repo=_REPO,
            source_commit="G" * 40,
            source_archive_sha256="8" * 64,
        )
