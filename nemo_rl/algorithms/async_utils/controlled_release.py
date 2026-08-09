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

"""Prospective assignment for controlled prompt-group release delays."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, Field, model_validator

_DIGEST_CARDINALITY = 1 << 256
DigestFunction = Callable[[bytes], bytes]


class ControlledReleaseArmConfig(BaseModel, frozen=True):
    """One arm in an exact integer-mass categorical assignment."""

    label: str
    delay_seconds: float
    mass: Annotated[int, Field(ge=1, strict=True)]

    @model_validator(mode="after")
    def validate_arm(self) -> ControlledReleaseArmConfig:
        if not self.label or not self.label.isascii():
            raise ValueError("controlled-release arm labels must be nonempty ASCII")
        if not math.isfinite(self.delay_seconds) or self.delay_seconds < 0:
            raise ValueError(
                "controlled-release arm delays must be finite and nonnegative"
            )
        return self


def _default_arms() -> tuple[ControlledReleaseArmConfig, ...]:
    return (
        ControlledReleaseArmConfig(label="control", delay_seconds=0.0, mass=4),
        ControlledReleaseArmConfig(label="d30", delay_seconds=30.0, mass=1),
        ControlledReleaseArmConfig(label="d60", delay_seconds=60.0, mass=1),
    )


class ControlledReleaseDelayConfig(BaseModel, frozen=True):
    """Default-off release-delay intervention configuration."""

    enabled: bool = False
    seed: Annotated[int, Field(ge=0)] = 20260808
    assignment_domain: str = "m3-release-v2"
    arms: tuple[ControlledReleaseArmConfig, ...] = Field(default_factory=_default_arms)

    @model_validator(mode="after")
    def validate_assignment(self) -> ControlledReleaseDelayConfig:
        if (
            not self.assignment_domain
            or not self.assignment_domain.isascii()
            or "\0" in self.assignment_domain
        ):
            raise ValueError(
                "controlled-release assignment domain must be nonempty ASCII "
                "without NUL"
            )
        if not self.arms:
            raise ValueError("controlled-release assignment requires at least one arm")
        labels = [arm.label for arm in self.arms]
        if len(labels) != len(set(labels)):
            raise ValueError("controlled-release arm labels must be unique")
        zero_arms = [arm for arm in self.arms if arm.delay_seconds == 0]
        if len(zero_arms) != 1:
            raise ValueError(
                "controlled-release assignment requires exactly one zero-dose control"
            )
        total_mass = sum(arm.mass for arm in self.arms)
        if not 1 <= total_mass <= _DIGEST_CARDINALITY:
            raise ValueError("controlled-release total mass must be in [1, 2^256]")
        return self

    @property
    def total_mass(self) -> int:
        return sum(arm.mass for arm in self.arms)


@dataclass(frozen=True)
class ControlledReleaseAssignment:
    """Immutable treatment assigned synchronously after buffer reservation."""

    arm_label: str
    delay_seconds: float
    arm_mass: int
    total_mass: int
    global_ordinal: int
    draw: int
    nonce: int


def sha256_digest(payload: bytes) -> bytes:
    """Return the assignment algorithm's 32-byte SHA-256 digest."""
    return hashlib.sha256(payload).digest()


class ControlledReleaseAssigner:
    """Counter-based independent categorical assigner with exact integer mass."""

    def __init__(
        self,
        config: ControlledReleaseDelayConfig,
        *,
        digest_fn: DigestFunction = sha256_digest,
    ) -> None:
        self._config = config
        self._digest_fn = digest_fn
        self._assignment_domain_prefix = (
            config.assignment_domain.encode("ascii") + b"\0"
        )
        self._next_ordinal = 0

    @property
    def next_ordinal(self) -> int:
        return self._next_ordinal

    def assign(self) -> ControlledReleaseAssignment:
        """Assign the next global reservation ordinal without awaiting or RNG state."""
        ordinal = self._next_ordinal
        self._next_ordinal += 1
        total_mass = self._config.total_mass
        limit = _DIGEST_CARDINALITY - (_DIGEST_CARDINALITY % total_mass)
        nonce = 0
        while True:
            payload = self._assignment_domain_prefix + (
                f"{self._config.seed}|{ordinal}|{nonce}"
            ).encode(
                "ascii",
            )
            digest = self._digest_fn(payload)
            if len(digest) != 32:
                raise ValueError(
                    "controlled-release digest function must return 32 bytes"
                )
            value = int.from_bytes(digest, byteorder="big", signed=False)
            if value < limit:
                break
            nonce += 1

        draw = value % total_mass
        cumulative_mass = 0
        for arm in self._config.arms:
            cumulative_mass += arm.mass
            if draw < cumulative_mass:
                return ControlledReleaseAssignment(
                    arm_label=arm.label,
                    delay_seconds=arm.delay_seconds,
                    arm_mass=arm.mass,
                    total_mass=total_mass,
                    global_ordinal=ordinal,
                    draw=draw,
                    nonce=nonce,
                )
        raise AssertionError("controlled-release cumulative mass did not cover draw")
