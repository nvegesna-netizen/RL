# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import copy
from pathlib import Path

from omegaconf import OmegaConf
import pytest

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from tools.opportunity_loss_workload_transport_preflight import (
    WorkloadTransportPreflightError,
)
from tools.opportunity_loss_workload_transport_r2_preflight import (
    CONFIG_PATH,
    build_r2_workload_transport_preflight_lock,
    validate_r2_workload_transport_config,
)


_REPO = Path(__file__).resolve().parents[3]


def _config() -> dict[str, object]:
    register_omegaconf_resolvers()
    value = OmegaConf.to_container(load_config(_REPO / CONFIG_PATH), resolve=True)
    assert isinstance(value, dict)
    return value


def test_r2_config_and_lock_forbid_execution() -> None:
    validate_r2_workload_transport_config(_config())
    lock = build_r2_workload_transport_preflight_lock(
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
    assert lock["design"]["max_num_epochs"] == 2
    assert lock["design"]["r1_observations_enter_estimator"] is False
    assert lock["design"]["epoch_specific_group_instance_is_assignment_unit"] is True


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("grpo", "max_num_epochs"), 1),
        (("grpo", "max_num_steps"), 557),
        (("async_rl", "controlled_release_delay", "seed"), 20260911),
        (("async_rl", "controlled_release_delay", "assignment_domain"), "wrong"),
        (("async_rl", "lifecycle_audit_path"), "results/r1/lifecycle.jsonl"),
        (("data", "train", "dataset_name"), "OpenMathInstruct-2"),
        (("cluster", "gpus_per_node"), 4),
    ],
)
def test_r2_config_mutations_fail_closed(path: tuple[str, ...], value: object) -> None:
    config = copy.deepcopy(_config())
    target = config
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    with pytest.raises((WorkloadTransportPreflightError, ValueError)):
        validate_r2_workload_transport_config(config)


def test_r2_lock_rejects_malformed_source_identity() -> None:
    with pytest.raises(WorkloadTransportPreflightError, match="source commit"):
        build_r2_workload_transport_preflight_lock(
            repo=_REPO,
            source_commit="G" * 40,
            source_archive_sha256="8" * 64,
        )
