# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from tools.opportunity_loss_followup_preflight import (
    CONFIG_PATH,
    EMPIRICAL_POWER_SHA256,
    FollowupPreflightError,
    build_followup_preflight_lock,
    validate_followup_config,
)

_REPO = Path(__file__).resolve().parents[3]


def _config() -> dict[str, object]:
    register_omegaconf_resolvers()
    value = OmegaConf.to_container(load_config(_REPO / CONFIG_PATH), resolve=True)
    assert isinstance(value, dict)
    return value


def test_followup_config_resolves_and_lock_is_strictly_no_training() -> None:
    validate_followup_config(_config())
    lock = build_followup_preflight_lock(
        repo=_REPO,
        source_commit="7" * 40,
        source_archive_sha256="8" * 64,
    )

    assert lock["training_allowed"] is False
    assert lock["eos_submission_allowed"] is False
    assert lock["automatic_retry"] is False
    assert lock["design"] == {
        "arms": ["control", "d5"],
        "primary_versions": 400,
        "trainer_steps": 448,
        "empirical_power_at_0_25": 0.81555,
        "empirical_power_sha256": EMPIRICAL_POWER_SHA256,
    }
    assert EMPIRICAL_POWER_SHA256 in str(lock)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("grpo", "max_num_steps"), 447),
        (("async_rl", "controlled_release_delay", "seed"), 1),
        (("async_rl", "controlled_release_delay", "arms"), []),
        (("loss_fn", "token_level_loss"), False),
        (("logger", "wandb_enabled"), True),
    ],
)
def test_followup_config_mutations_fail_closed(
    path: tuple[str, ...], value: object
) -> None:
    config = copy.deepcopy(_config())
    target = config
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    with pytest.raises((FollowupPreflightError, ValueError)):
        validate_followup_config(config)


def test_followup_lock_rejects_malformed_source_identity() -> None:
    with pytest.raises(FollowupPreflightError, match="source commit"):
        build_followup_preflight_lock(
            repo=_REPO,
            source_commit="G" * 40,
            source_archive_sha256="8" * 64,
        )
