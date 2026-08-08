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

"""Low-overhead lifecycle events for SingleController rollout groups."""

from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class RolloutLifecycleStage(str, Enum):
    """Observable milestones emitted on the controller clock."""

    LEARNER_VERSION_ADVANCED = "learner_version_advanced"
    RESERVED = "reserved"
    RELEASE_DELAY_ASSIGNED = "release_delay_assigned"
    SIBLING_DONE = "sibling_done"
    RELEASE_DELAY_STARTED = "release_delay_started"
    RELEASE_DELAY_COMPLETED = "release_delay_completed"
    GROUP_COMPLETED = "group_completed"
    GROUP_READY = "group_ready"
    REMOVED = "removed"


class RolloutRemovalReason(str, Enum):
    """Why a rollout group left the replay-buffer metadata cache."""

    SELECTED = "selected"
    STALE_EVICTED = "stale_evicted"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BOUNDED_SHUTDOWN = "bounded_shutdown"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RolloutLifecycleEvent:
    """One prompt-group lifecycle event from a single monotonic clock domain."""

    schema_version: int
    run_id: str
    clock_domain_id: str
    sequence: int
    timestamp_ns: int
    group_id: str
    stage: RolloutLifecycleStage
    start_weight_version: int
    end_weight_version: Optional[int]
    target_step: Optional[int]
    learner_weight_version: Optional[int]
    mixed_generation_versions: Optional[bool]
    sample_ids: tuple[str, ...]
    removal_reason: Optional[RolloutRemovalReason]
    trajectory_id: Optional[str]
    sibling_idx: Optional[int]
    turn_count: Optional[int]
    assistant_tokens: Optional[int]
    env_tokens: Optional[int]
    reward: Optional[float]
    terminated: Optional[bool]
    truncated: Optional[bool]
    generation_duration_ns: Optional[int]
    environment_duration_ns: Optional[int]
    release_arm: Optional[str]
    release_delay_seconds: Optional[float]
    release_arm_mass: Optional[int]
    release_total_mass: Optional[int]
    release_global_ordinal: Optional[int]
    release_draw: Optional[int]
    release_nonce: Optional[int]
    generation_inflight: Optional[int]
    active_release_holds: Optional[int]
    reserved_buffer_occupancy: Optional[int]
    ready_buffer_depth: Optional[int]
    buffer_admission_stalls: Optional[int]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "clock_domain_id": self.clock_domain_id,
            "sequence": self.sequence,
            "timestamp_ns": self.timestamp_ns,
            "group_id": self.group_id,
            "stage": self.stage.value,
            "start_weight_version": self.start_weight_version,
            "end_weight_version": self.end_weight_version,
            "target_step": self.target_step,
            "learner_weight_version": self.learner_weight_version,
            "mixed_generation_versions": self.mixed_generation_versions,
            "sample_ids": list(self.sample_ids),
            "removal_reason": (
                self.removal_reason.value if self.removal_reason is not None else None
            ),
            "trajectory_id": self.trajectory_id,
            "sibling_idx": self.sibling_idx,
            "turn_count": self.turn_count,
            "assistant_tokens": self.assistant_tokens,
            "env_tokens": self.env_tokens,
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "generation_duration_ns": self.generation_duration_ns,
            "environment_duration_ns": self.environment_duration_ns,
            "release_arm": self.release_arm,
            "release_delay_seconds": self.release_delay_seconds,
            "release_arm_mass": self.release_arm_mass,
            "release_total_mass": self.release_total_mass,
            "release_global_ordinal": self.release_global_ordinal,
            "release_draw": self.release_draw,
            "release_nonce": self.release_nonce,
            "generation_inflight": self.generation_inflight,
            "active_release_holds": self.active_release_holds,
            "reserved_buffer_occupancy": self.reserved_buffer_occupancy,
            "ready_buffer_depth": self.ready_buffer_depth,
            "buffer_admission_stalls": self.buffer_admission_stalls,
        }


class RolloutLifecycleRecorder:
    """In-memory append-only event recorder for one controller actor.

    All timestamps come from the injected monotonic clock. Events from another
    recorder or Ray actor must not be subtracted from these timestamps.
    """

    def __init__(
        self,
        *,
        run_id: Optional[str] = None,
        clock_domain_id: Optional[str] = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.run_id = run_id or str(uuid.uuid4())
        self.clock_domain_id = clock_domain_id or str(uuid.uuid4())
        self._clock_ns = clock_ns
        self._events: list[RolloutLifecycleEvent] = []

    def record(
        self,
        *,
        group_id: str,
        stage: RolloutLifecycleStage,
        start_weight_version: int,
        end_weight_version: Optional[int] = None,
        target_step: Optional[int] = None,
        learner_weight_version: Optional[int] = None,
        sample_ids: Sequence[str] = (),
        removal_reason: Optional[RolloutRemovalReason] = None,
        trajectory_id: Optional[str] = None,
        sibling_idx: Optional[int] = None,
        turn_count: Optional[int] = None,
        assistant_tokens: Optional[int] = None,
        env_tokens: Optional[int] = None,
        reward: Optional[float] = None,
        terminated: Optional[bool] = None,
        truncated: Optional[bool] = None,
        generation_duration_ns: Optional[int] = None,
        environment_duration_ns: Optional[int] = None,
        release_arm: Optional[str] = None,
        release_delay_seconds: Optional[float] = None,
        release_arm_mass: Optional[int] = None,
        release_total_mass: Optional[int] = None,
        release_global_ordinal: Optional[int] = None,
        release_draw: Optional[int] = None,
        release_nonce: Optional[int] = None,
        generation_inflight: Optional[int] = None,
        active_release_holds: Optional[int] = None,
        reserved_buffer_occupancy: Optional[int] = None,
        ready_buffer_depth: Optional[int] = None,
        buffer_admission_stalls: Optional[int] = None,
    ) -> RolloutLifecycleEvent:
        """Append and return one lifecycle event."""
        if stage is RolloutLifecycleStage.REMOVED and removal_reason is None:
            raise ValueError("removed lifecycle events require removal_reason")
        if stage is not RolloutLifecycleStage.REMOVED and removal_reason is not None:
            raise ValueError("removal_reason is valid only for removed events")
        if stage is RolloutLifecycleStage.SIBLING_DONE:
            if trajectory_id is None or sibling_idx is None:
                raise ValueError(
                    "sibling_done lifecycle events require trajectory_id and sibling_idx"
                )
        elif trajectory_id is not None or sibling_idx is not None:
            raise ValueError(
                "trajectory_id and sibling_idx are valid only for sibling_done events"
            )

        release_stages = {
            RolloutLifecycleStage.RELEASE_DELAY_ASSIGNED,
            RolloutLifecycleStage.RELEASE_DELAY_STARTED,
            RolloutLifecycleStage.RELEASE_DELAY_COMPLETED,
        }
        release_fields = (
            release_arm,
            release_delay_seconds,
            release_arm_mass,
            release_total_mass,
            release_global_ordinal,
            release_draw,
            release_nonce,
        )
        if stage in release_stages and any(value is None for value in release_fields):
            raise ValueError(
                "release-delay lifecycle events require complete assignment fields"
            )
        if stage in release_stages:
            assert release_arm is not None
            assert release_delay_seconds is not None
            assert release_arm_mass is not None
            assert release_total_mass is not None
            assert release_global_ordinal is not None
            assert release_draw is not None
            assert release_nonce is not None
            if not release_arm or not release_arm.isascii():
                raise ValueError("release_arm must be nonempty ASCII")
            if not math.isfinite(release_delay_seconds) or release_delay_seconds < 0:
                raise ValueError("release_delay_seconds must be finite and nonnegative")
            if release_arm_mass <= 0 or release_total_mass <= 0:
                raise ValueError("release assignment masses must be positive")
            if not 0 <= release_draw < release_total_mass:
                raise ValueError("release_draw must fall within release_total_mass")
            if release_global_ordinal < 0 or release_nonce < 0:
                raise ValueError("release ordinal and nonce must be nonnegative")
            diagnostic_counts = (
                generation_inflight,
                active_release_holds,
                reserved_buffer_occupancy,
                ready_buffer_depth,
                buffer_admission_stalls,
            )
            if any(value is None for value in diagnostic_counts):
                raise ValueError(
                    "release-delay lifecycle events require complete diagnostics"
                )
            if any(value is not None and value < 0 for value in diagnostic_counts):
                raise ValueError("release-delay diagnostic counts must be nonnegative")
        if stage not in release_stages and any(
            value is not None
            for value in (
                *release_fields,
                generation_inflight,
                active_release_holds,
                reserved_buffer_occupancy,
                ready_buffer_depth,
                buffer_admission_stalls,
            )
        ):
            raise ValueError(
                "release-delay fields are valid only for release-delay events"
            )

        event = RolloutLifecycleEvent(
            schema_version=3,
            run_id=self.run_id,
            clock_domain_id=self.clock_domain_id,
            sequence=len(self._events),
            timestamp_ns=self._clock_ns(),
            group_id=group_id,
            stage=stage,
            start_weight_version=start_weight_version,
            end_weight_version=end_weight_version,
            target_step=target_step,
            learner_weight_version=learner_weight_version,
            mixed_generation_versions=(
                start_weight_version != end_weight_version
                if end_weight_version is not None
                else None
            ),
            sample_ids=tuple(sample_ids),
            removal_reason=removal_reason,
            trajectory_id=trajectory_id,
            sibling_idx=sibling_idx,
            turn_count=turn_count,
            assistant_tokens=assistant_tokens,
            env_tokens=env_tokens,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            generation_duration_ns=generation_duration_ns,
            environment_duration_ns=environment_duration_ns,
            release_arm=release_arm,
            release_delay_seconds=release_delay_seconds,
            release_arm_mass=release_arm_mass,
            release_total_mass=release_total_mass,
            release_global_ordinal=release_global_ordinal,
            release_draw=release_draw,
            release_nonce=release_nonce,
            generation_inflight=generation_inflight,
            active_release_holds=active_release_holds,
            reserved_buffer_occupancy=reserved_buffer_occupancy,
            ready_buffer_depth=ready_buffer_depth,
            buffer_admission_stalls=buffer_admission_stalls,
        )
        self._events.append(event)
        return event

    def snapshot(self) -> tuple[RolloutLifecycleEvent, ...]:
        """Return an immutable snapshot in event order."""
        return tuple(self._events)

    def record_learner_version_advanced(
        self,
        *,
        previous_version: int,
        learner_weight_version: int,
    ) -> RolloutLifecycleEvent:
        """Record a successful learner-version transition on this clock."""
        if learner_weight_version != previous_version + 1:
            raise ValueError(
                "learner version must advance by exactly one: "
                f"{previous_version} -> {learner_weight_version}"
            )
        return self.record(
            group_id="__learner__",
            stage=RolloutLifecycleStage.LEARNER_VERSION_ADVANCED,
            start_weight_version=previous_version,
            learner_weight_version=learner_weight_version,
        )

    def flush_jsonl(self, output_path: str | Path) -> None:
        """Write the current snapshot as JSON Lines.

        Args:
            output_path: Destination path on the controller host.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as output:
            for event in self._events:
                output.write(json.dumps(event.to_dict(), sort_keys=True))
                output.write("\n")
