# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from tools.opportunity_loss_preflight import (
    CONFIG_PATH,
    IMAGE_COMMIT,
    IMAGE_SHA256,
    OpportunityLossPreflightError,
    build_preflight_lock,
    validate_acquisition_config,
)

_REPO = Path(__file__).resolve().parents[3]
_PROTOCOL = (
    _REPO
    / "reports/auto_research/2026-09-01-m4-opportunity-loss/common-instrumentation-bounds/protocol_config.json"
)


def _config() -> dict[str, object]:
    register_omegaconf_resolvers()
    value = OmegaConf.to_container(load_config(_REPO / CONFIG_PATH), resolve=True)
    assert isinstance(value, dict)
    return value


def test_final_config_resolves_and_matches_registered_acquisition() -> None:
    validate_acquisition_config(_config())


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("grpo", "max_num_steps"), 223),
        (("loss_fn", "token_level_loss"), False),
        (("logger", "wandb_enabled"), True),
        (("data_plane", "impl"), "filesystem"),
        (("data", "train", "dataset_name"), "other"),
        (("async_rl", "controlled_release_delay", "seed"), 1),
        (("async_rl", "gradient_opportunity_audit", "observer_duty_path"), "same"),
    ],
)
def test_config_mutations_fail_closed(path: tuple[str, ...], value: object) -> None:
    config = copy.deepcopy(_config())
    target = config
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    if path[-1] == "observer_duty_path":
        target[path[-1]] = config["async_rl"]["lifecycle_audit_path"]

    with pytest.raises((OpportunityLossPreflightError, ValueError)):
        validate_acquisition_config(config)


def test_lock_is_no_training_and_binds_exact_image_and_pending_m4() -> None:
    result = build_preflight_lock(
        repo=_REPO,
        protocol=_PROTOCOL.read_bytes(),
        source_commit="7" * 40,
        source_archive_sha256="8" * 64,
    )

    assert result["training_allowed"] is False
    assert result["eos_submission_allowed"] is False
    assert result["image"]["sha256"] == IMAGE_SHA256
    assert result["image"]["embedded_commit"] == IMAGE_COMMIT
    assert result["m4_mechanism_replication"]["current_acquisition_status"] == "PENDING"
    encoded = json.dumps(result, sort_keys=True)
    assert "run_grpo_single_controller" not in encoded
    assert "jet submit" not in encoded


def test_lock_rejects_unbound_protocol_and_malformed_source_identity() -> None:
    protocol = json.loads(_PROTOCOL.read_bytes())
    protocol["m4_evidence_commit"] = "0" * 40

    with pytest.raises(OpportunityLossPreflightError, match="M4 evidence"):
        build_preflight_lock(
            repo=_REPO,
            protocol=(
                json.dumps(
                    protocol, allow_nan=False, separators=(",", ":"), sort_keys=True
                )
                + "\n"
            ).encode(),
            source_commit="7" * 40,
            source_archive_sha256="8" * 64,
        )
    with pytest.raises(OpportunityLossPreflightError, match="source commit"):
        build_preflight_lock(
            repo=_REPO,
            protocol=_PROTOCOL.read_bytes(),
            source_commit="G" * 40,
            source_archive_sha256="8" * 64,
        )


def test_lock_rejects_duplicate_key_protocol() -> None:
    raw = _PROTOCOL.read_bytes()
    duplicate = raw.replace(b'"protocol":', b'"protocol":"duplicate","protocol":', 1)
    with pytest.raises(OpportunityLossPreflightError, match="strictly valid"):
        build_preflight_lock(
            repo=_REPO,
            protocol=duplicate,
            source_commit="7" * 40,
            source_archive_sha256="8" * 64,
        )
