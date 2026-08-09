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

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_rl.algorithms.async_utils.controlled_release import (
    ControlledReleaseArmConfig,
    ControlledReleaseAssigner,
    ControlledReleaseDelayConfig,
)
from nemo_rl.algorithms.single_controller_utils.config import MasterConfig
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def test_exact_assignment_vector_and_ordinal_progression() -> None:
    config = ControlledReleaseDelayConfig()
    assigner = ControlledReleaseAssigner(config)

    assignments = [assigner.assign() for _ in range(16)]

    assert config.assignment_domain == "m3-release-v2"
    assert [assignment.global_ordinal for assignment in assignments] == list(range(16))
    assert [assignment.draw for assignment in assignments] == [
        5,
        1,
        3,
        1,
        3,
        2,
        4,
        5,
        1,
        1,
        4,
        5,
        0,
        4,
        3,
        0,
    ]
    assert assigner.next_ordinal == 16


def test_custom_assignment_domain_has_exact_distinct_draw_vector() -> None:
    arms = (
        ControlledReleaseArmConfig(label="control", delay_seconds=0, mass=5),
        ControlledReleaseArmConfig(label="d5", delay_seconds=5, mass=5),
        ControlledReleaseArmConfig(label="d10", delay_seconds=10, mass=2),
    )
    legacy_assigner = ControlledReleaseAssigner(
        ControlledReleaseDelayConfig(seed=20260810, arms=arms)
    )
    m5_assigner = ControlledReleaseAssigner(
        ControlledReleaseDelayConfig(
            seed=20260810,
            assignment_domain="m5a-opportunity-v1",
            arms=arms,
        )
    )

    legacy_draws = [legacy_assigner.assign().draw for _ in range(16)]
    m5_draws = [m5_assigner.assign().draw for _ in range(16)]

    assert legacy_draws == [8, 4, 11, 9, 8, 6, 1, 3, 2, 10, 10, 6, 8, 7, 8, 4]
    assert m5_draws == [5, 9, 0, 8, 8, 11, 8, 4, 4, 11, 9, 6, 1, 6, 5, 9]
    assert m5_draws != legacy_draws


def test_rejection_sampling_uses_next_nonce_and_configured_mapping() -> None:
    digests = iter(((2**256 - 1).to_bytes(32, "big"), (4).to_bytes(32, "big")))
    payloads: list[bytes] = []

    def digest_fn(payload: bytes) -> bytes:
        payloads.append(payload)
        return next(digests)

    assignment = ControlledReleaseAssigner(
        ControlledReleaseDelayConfig(), digest_fn=digest_fn
    ).assign()

    assert payloads == [
        b"m3-release-v2\x0020260808|0|0",
        b"m3-release-v2\x0020260808|0|1",
    ]
    assert assignment.nonce == 1
    assert assignment.draw == 4
    assert assignment.arm_label == "d30"
    assert assignment.delay_seconds == 30.0
    assert assignment.arm_mass == 1
    assert assignment.total_mass == 6


def test_fixed_seed_has_predeclared_m2_sized_arm_support() -> None:
    assigner = ControlledReleaseAssigner(ControlledReleaseDelayConfig())

    counts = Counter(assigner.assign().arm_label for _ in range(720))

    assert counts == {"control": 489, "d30": 111, "d60": 120}


@pytest.mark.parametrize(
    "arms",
    [
        (),
        (
            ControlledReleaseArmConfig(label="control", delay_seconds=0, mass=1),
            ControlledReleaseArmConfig(label="control", delay_seconds=1, mass=1),
        ),
        (ControlledReleaseArmConfig(label="dose", delay_seconds=1, mass=1),),
        (
            ControlledReleaseArmConfig(label="c0", delay_seconds=0, mass=1),
            ControlledReleaseArmConfig(label="c1", delay_seconds=0, mass=1),
        ),
        (
            ControlledReleaseArmConfig(label="control", delay_seconds=0, mass=2**256),
            ControlledReleaseArmConfig(label="dose", delay_seconds=1, mass=1),
        ),
    ],
)
def test_invalid_arm_schedules_are_rejected(
    arms: tuple[ControlledReleaseArmConfig, ...],
) -> None:
    with pytest.raises(ValidationError):
        ControlledReleaseDelayConfig(arms=arms)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"label": "", "delay_seconds": 0, "mass": 1},
        {"label": "délay", "delay_seconds": 0, "mass": 1},
        {"label": "delay", "delay_seconds": -1, "mass": 1},
        {"label": "delay", "delay_seconds": float("inf"), "mass": 1},
        {"label": "delay", "delay_seconds": 1, "mass": 0},
    ],
)
def test_invalid_arm_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ControlledReleaseArmConfig(**kwargs)


@pytest.mark.parametrize("assignment_domain", ["", "m5a-opportunité-v1", "m5a\0v1"])
def test_invalid_assignment_domain_is_rejected(assignment_domain: str) -> None:
    with pytest.raises(ValidationError, match="assignment domain"):
        ControlledReleaseDelayConfig(assignment_domain=assignment_domain)


def test_digest_must_be_exactly_32_bytes() -> None:
    assigner = ControlledReleaseAssigner(
        ControlledReleaseDelayConfig(), digest_fn=lambda _: b"short"
    )

    with pytest.raises(ValueError, match="32 bytes"):
        assigner.assign()


def test_single_controller_yaml_composes_with_enabled_cli_overrides() -> None:
    config_path = (
        Path(__file__).parents[3]
        / "examples"
        / "configs"
        / "grpo_math_1B_megatron_single_controller.yaml"
    )
    register_omegaconf_resolvers()
    config = load_config(config_path)
    config = parse_hydra_overrides(
        config,
        [
            "async_rl.controlled_release_delay.enabled=true",
            "+async_rl.lifecycle_audit_path=/tmp/m3-audit.jsonl",
        ],
    )
    master_config = MasterConfig(**OmegaConf.to_container(config, resolve=True))

    release = master_config.async_rl.controlled_release_delay
    assert release.enabled is True
    assert release.seed == 20260808
    assert release.assignment_domain == "m3-release-v2"
    assert [arm.label for arm in release.arms] == ["control", "d30", "d60"]
    assert [arm.mass for arm in release.arms] == [4, 1, 1]
    assert [arm.delay_seconds for arm in release.arms] == [0.0, 30.0, 60.0]
    assert master_config.async_rl.lifecycle_audit_path == "/tmp/m3-audit.jsonl"
