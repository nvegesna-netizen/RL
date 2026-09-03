# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from tools.opportunity_loss_transport_preflight import (
    CONFIG_PATH,
    TransportPreflightError,
    build_transport_preflight_lock,
    validate_neutral_qualification_config,
)

_REPO = Path(__file__).resolve().parents[3]


def _config() -> dict[str, object]:
    register_omegaconf_resolvers()
    value = OmegaConf.to_container(load_config(_REPO / CONFIG_PATH), resolve=True)
    assert isinstance(value, dict)
    return value


def test_neutral_config_resolves_and_lock_forbids_all_execution() -> None:
    validate_neutral_qualification_config(_config())
    lock = build_transport_preflight_lock(
        repo=_REPO,
        source_commit="7" * 40,
        source_archive_sha256="8" * 64,
    )

    assert lock["training_allowed"] is False
    assert lock["eos_submission_allowed"] is False
    assert lock["qualification_submission_allowed"] is False
    assert lock["scientific_acquisition_allowed"] is False
    assert lock["automatic_retry"] is False
    assert lock["qualification"] == {
        "model": "Qwen/Qwen3-1.7B",
        "dataset": "OpenMathInstruct-2",
        "arms": [{"label": "neutral", "delay_seconds": 0.0, "mass": 1}],
        "trainer_steps": 32,
        "candidate_gpus_per_node": 2,
        "scientific_acquisition": False,
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("grpo", "max_num_steps"), 31),
        (("policy", "model_name"), "Qwen/Qwen3-0.6B"),
        (("async_rl", "controlled_release_delay", "arms"), []),
        (
            ("async_rl", "controlled_release_delay", "arms"),
            [{"label": "d5", "delay_seconds": 5.0, "mass": 1}],
        ),
        (("loss_fn", "token_level_loss"), False),
        (("logger", "wandb_enabled"), True),
        (("cluster", "gpus_per_node"), 4),
    ],
)
def test_neutral_config_mutations_fail_closed(
    path: tuple[str, ...], value: object
) -> None:
    config = copy.deepcopy(_config())
    target = config
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    with pytest.raises((TransportPreflightError, ValueError)):
        validate_neutral_qualification_config(config)


def test_transport_lock_rejects_malformed_source_identity() -> None:
    with pytest.raises(TransportPreflightError, match="source commit"):
        build_transport_preflight_lock(
            repo=_REPO,
            source_commit="G" * 40,
            source_archive_sha256="8" * 64,
        )
