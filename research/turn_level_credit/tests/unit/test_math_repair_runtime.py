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

"""Tests for the optional math-repair runtime integration."""

import pytest

pytest.importorskip("math_verify")

from nemo_rl.distributed import ray_actor_environment_registry
from nemo_rl.environments import utils as environment_utils
from turn_level_credit.math_repair_runtime import (
    MATH_REPAIR_ENVIRONMENT_FQN,
    install_math_repair_environment,
)


def test_math_repair_environment_installation_is_scoped():
    original_entry = environment_utils.ENV_REGISTRY["math"]
    original_fqn = original_entry["actor_class_fqn"]
    original_python = ray_actor_environment_registry.ACTOR_ENVIRONMENT_REGISTRY[
        original_fqn
    ]
    assert (
        MATH_REPAIR_ENVIRONMENT_FQN
        not in ray_actor_environment_registry.ACTOR_ENVIRONMENT_REGISTRY
    )

    with install_math_repair_environment():
        assert environment_utils.ENV_REGISTRY["math"] == {
            "actor_class_fqn": MATH_REPAIR_ENVIRONMENT_FQN
        }
        assert (
            ray_actor_environment_registry.ACTOR_ENVIRONMENT_REGISTRY[
                MATH_REPAIR_ENVIRONMENT_FQN
            ]
            == original_python
        )

    assert environment_utils.ENV_REGISTRY["math"] is original_entry
    assert (
        MATH_REPAIR_ENVIRONMENT_FQN
        not in ray_actor_environment_registry.ACTOR_ENVIRONMENT_REGISTRY
    )
