# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for supported-path validation."""

from types import SimpleNamespace

import pytest
from turn_level_credit.config import TurnCreditConfig
from turn_level_credit.validation import validate_supported_path
from turn_level_credit.verifier_credit import VerifierCreditTransformConfig


def _master_config(
    *,
    estimator_name="grpo",
    async_grpo=False,
    data_plane=False,
    nemo_gym=False,
    dynamic_sampling=False,
):
    return SimpleNamespace(
        grpo=SimpleNamespace(
            adv_estimator=SimpleNamespace(name=estimator_name),
            async_grpo=SimpleNamespace(enabled=async_grpo),
            use_dynamic_sampling=dynamic_sampling,
        ),
        data_plane={"enabled": data_plane},
        env={"should_use_nemo_gym": nemo_gym},
        policy={
            "generation": {
                "backend": "vllm",
                "vllm_cfg": {"async_engine": False},
            }
        },
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_supported_sync_path_passes_for_treatment_and_control(enabled):
    validate_supported_path(
        _master_config(),
        TurnCreditConfig(enabled=enabled),
    )


@pytest.mark.parametrize(
    "generation",
    [
        {"backend": "sglang", "use_async_rollouts": True},
        {"backend": "vllm", "vllm_cfg": {"async_engine": True}},
        {"backend": "trtllm", "trtllm_cfg": {"async_engine": True}},
        {
            "backend": "megatron",
            "mcore_generation_config": {"async_engine": True},
        },
    ],
)
def test_rejects_async_generation_backends(generation):
    master_config = _master_config()
    master_config.policy["generation"] = generation

    with pytest.raises(ValueError, match="synchronous native rollouts"):
        validate_supported_path(master_config, TurnCreditConfig(enabled=True))


@pytest.mark.parametrize(
    ("master_config", "message"),
    [
        (_master_config(async_grpo=True), "async GRPO"),
        (_master_config(data_plane=True), "data_plane"),
        (_master_config(nemo_gym=True), "NeMo Gym"),
    ],
)
def test_disabled_config_rejects_launcher_paths_it_cannot_dispatch(
    master_config,
    message,
):
    with pytest.raises(ValueError, match=message):
        validate_supported_path(
            master_config,
            TurnCreditConfig(enabled=False),
        )


@pytest.mark.parametrize(
    ("master_config", "message"),
    [
        (_master_config(estimator_name="gdpo"), "only.*grpo"),
        (_master_config(async_grpo=True), "async GRPO"),
        (_master_config(data_plane=True), "data_plane"),
        (_master_config(nemo_gym=True), "NeMo Gym"),
    ],
)
def test_enabled_config_rejects_unsupported_paths(master_config, message):
    with pytest.raises(ValueError, match=message):
        validate_supported_path(
            master_config,
            TurnCreditConfig(enabled=True),
        )


def test_verifier_transform_rejects_dynamic_sampling():
    with pytest.raises(ValueError, match="use_dynamic_sampling=false"):
        validate_supported_path(
            _master_config(dynamic_sampling=True),
            TurnCreditConfig(
                enabled=True,
                environment_component="reward/verifier_score",
                verifier_transform=VerifierCreditTransformConfig(),
            ),
        )
