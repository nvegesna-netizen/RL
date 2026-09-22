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

import asyncio
import math
import statistics
import threading as _threading
import time
import uuid
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import ray

from nemo_rl.algorithms.async_utils.interfaces import ReplayBufferProtocol
from nemo_rl.algorithms.async_utils.rollout_lifecycle import (
    RolloutLifecycleRecorder,
    RolloutLifecycleStage,
    RolloutRemovalReason,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import ROUTED_EXPERTS_FIELD
from nemo_rl.experience.interfaces import PromptGroupRecord
from nemo_rl.experience.payload import pack_payload, record_to_train_batch
from nemo_rl.utils.r3_trace import trace_rollout_payload


def _tensor_payload_state(fields: Any) -> tuple[tuple[Any, ...], ...]:
    """Return cheap mutation metadata for a packed TensorDict payload."""
    state: list[tuple[Any, ...]] = []
    for key, value in fields.items():
        state.append(
            (
                str(key),
                id(value),
                getattr(value, "_version", None),
                tuple(value.shape),
                str(value.dtype),
                str(value.device),
            )
        )
    return tuple(sorted(state))


@dataclass(frozen=True)
class PreparedTQCommit:
    """Owner-bound, single-use capability for one definitive packed payload."""

    group_id: str
    partition_id: str
    start_weight_version: int
    sample_ids: tuple[str, ...]
    fields: Any = field(repr=False)
    tags: tuple[dict[str, Any], ...]
    tag_state: tuple[tuple[tuple[str, Any], ...], ...] = field(repr=False)
    sequence_lengths: tuple[int, ...]
    observer_metadata: tuple[tuple[str, Any], ...]
    payload_state: tuple[tuple[Any, ...], ...] = field(repr=False)
    _owner: object = field(repr=False)
    _capability: object = field(repr=False)


# Classes with @ray.remote can't be inherited from, so we split the implementation out.
class ReplayBufferImpl(ReplayBufferProtocol):
    """Replay buffer storing per-prompt groups.

    A single entry corresponds to 1 prompt repeated by
    grpo.num_generations_per_prompt (required to compute per-prompt advantages).
    """

    def __init__(self, max_size: int):
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self.max_size = max_size
        self.trajectories = []  # List[dict[str, Any]]
        # If trajectory_version is 1 and target_weight_version is 4 it means that weight version 1 was used for generating a trajectory and this trajectory will be used for training when weight version is 4.
        self.trajectory_versions = []  # it is the weight-version used for generation of a trajectory
        self.target_weight_versions = []  # it is the weight-version of the trainer where this trajectory will be used.

        self.last_target_weight_already_generated = -1
        self._lock = _threading.Lock()

    @staticmethod
    def _rollout_metrics_turn_count_for_diagnostics(
        rm: dict[str, Any],
    ) -> Optional[float]:
        """One scalar turn-depth per buffered trajectory for starvation diagnostics.

        Supports sync multi-turn rollouts (`max_turns_per_sample` / `avg_turns_per_sample`)
        and NeMo Gym (`turns_per_sample/max` / `turns_per_sample/mean`).
        """
        if "max_turns_per_sample" in rm:
            return float(rm["max_turns_per_sample"])
        if "avg_turns_per_sample" in rm:
            return float(rm["avg_turns_per_sample"])
        if "turns_per_sample/max" in rm:
            return float(rm["turns_per_sample/max"])
        if "turns_per_sample/mean" in rm:
            return float(rm["turns_per_sample/mean"])
        return None

    def add(
        self,
        trajectory: dict[str, Any],
        weight_version: int,
        target_weight_version: int,
    ) -> str:
        """Add a per-prompt trajectory group with metadata.

        Args:
            trajectory: data dict
            weight_version: version of the model weights used for generation
            target_weight_version: version of the model weights this trajectory is intended for training
        """
        with self._lock:
            if len(self.trajectories) >= self.max_size:
                return "full"

            print("🔍 ReplayBuffer.add: Adding trajectory")
            self.trajectories.append(trajectory)
            self.trajectory_versions.append(weight_version)
            self.target_weight_versions.append(target_weight_version)
            # Do not advance last_target_weight_already_generated here. A target
            # is only safe to skip once training consumes a complete batch for it.
            print(
                f"ReplayBuffer state: {len(self.trajectories)} groups, versions={self.trajectory_versions}, targets={self.target_weight_versions}, last_target_weight_already_generated={self.last_target_weight_already_generated}"
            )
            return "success"

    def get_debug_info(self) -> dict:
        """Get debug information about buffer state."""
        info: dict[str, Any] = {
            "total_trajectories": len(self.trajectories),
            "trajectory_versions": self.trajectory_versions,
            "target_weight_versions": self.target_weight_versions,
            "max_size": self.max_size,
        }
        if self.trajectories:
            durations = []
            max_gen_tokens_per_turn_list = []
            turn_counts_list = []
            for t in self.trajectories:
                rm = t.get("rollout_metrics", {})
                if "trajectory_duration_s" in rm:
                    durations.append(rm["trajectory_duration_s"])
                if "max_gen_tokens_per_turn/max" in rm:
                    max_gen_tokens_per_turn_list.append(
                        rm["max_gen_tokens_per_turn/max"]
                    )
                elif "max_gen_tokens_per_turn" in rm:
                    max_gen_tokens_per_turn_list.append(rm["max_gen_tokens_per_turn"])
                tc = self._rollout_metrics_turn_count_for_diagnostics(rm)
                if tc is not None:
                    turn_counts_list.append(tc)

            def _pct(values: list[float], p: float) -> float:
                if not values:
                    return 0.0
                sorted_v = sorted(values)
                idx = min(int(len(sorted_v) * p / 100), len(sorted_v) - 1)
                return float(sorted_v[idx])

            info["starvation_diagnostics"] = {
                "trajectory_duration_s": {
                    "mean": sum(durations) / len(durations) if durations else 0,
                    "median": statistics.median(durations) if durations else 0,
                    "max": max(durations) if durations else 0,
                    "p95": _pct(durations, 95),
                },
                "max_gen_tokens_per_turn_in_buffer": {
                    "mean": sum(max_gen_tokens_per_turn_list)
                    / len(max_gen_tokens_per_turn_list)
                    if max_gen_tokens_per_turn_list
                    else 0,
                    "median": statistics.median(max_gen_tokens_per_turn_list)
                    if max_gen_tokens_per_turn_list
                    else 0,
                    "max": max(max_gen_tokens_per_turn_list)
                    if max_gen_tokens_per_turn_list
                    else 0,
                    "p95": _pct(max_gen_tokens_per_turn_list, 95),
                },
                "turns_per_sample_in_buffer": {
                    "mean": sum(turn_counts_list) / len(turn_counts_list)
                    if turn_counts_list
                    else 0,
                    "median": statistics.median(turn_counts_list)
                    if turn_counts_list
                    else 0,
                    "max": max(turn_counts_list) if turn_counts_list else 0,
                    "p95": _pct(turn_counts_list, 95),
                },
                "num_trajectories_sampled": len(self.trajectories),
            }
        return info

    def get_last_target_weight_already_generated(self) -> int:
        with self._lock:
            return self.last_target_weight_already_generated

    def get_existing_target_weights(self) -> set[int]:
        """Get set of target weight versions that already have trajectories."""
        with self._lock:
            return set(self.target_weight_versions)

    def _remove_indices(self, indices: Iterable[int]) -> None:
        """Remove trajectories at the given indices."""
        for idx in sorted(indices, reverse=True):
            self.trajectory_versions.pop(idx)
            self.target_weight_versions.pop(idx)
            self.trajectories.pop(idx)

    def sample(
        self,
        num_prompt_groups: int,
        current_weight_version: int,
        max_age_steps: int,
    ) -> Optional[dict[str, Any]]:
        """Sample per-prompt trajectory groups intended for the current training step.

        Only returns trajectories with target_weight_version == current_weight_version.
        If insufficient trajectories are available, returns None to stall training
        until the remaining trajectories are generated. This ensures no trajectory
        loses its last chance to be used for its intended training step.

        Returns:
            Dictionary with 'trajectories' and 'avg_trajectory_age' keys, or None if insufficient data
        """
        with self._lock:
            if not self.trajectories:
                return None

            total_trajectories = len(self.trajectories)
            print("🔍 ReplayBuffer sampling debug:")
            print(f"   {current_weight_version=}, {max_age_steps=}")
            print(f"   {self.trajectory_versions=}")

            # For debugging: check for unexpected old trajectories
            version_counts = Counter(self.trajectory_versions)
            print(f"   {version_counts=}")

            # Compute minimum valid version based on age window
            # max_age_steps=1 means trajectories from the last 1 step are valid
            min_valid_version = max(0, current_weight_version - max_age_steps)
            print(f"   {min_valid_version=}")

            # Evict old trajectories that are beyond the age window. This can
            # happen after checkpoint restore when old trajectories remain.
            old_indices = [
                i
                for i, v in enumerate(self.trajectory_versions)
                if v < min_valid_version
            ]
            if old_indices:
                print(
                    f"   Evicting {len(old_indices)} stale trajectories "
                    f"(version < {min_valid_version})"
                )
                self._remove_indices(old_indices)
                total_trajectories = len(self.trajectories)

            # Filter for valid trajectories without modifying the buffer
            valid_indices = [
                i
                for i, v in enumerate(self.trajectory_versions)
                if min_valid_version <= v <= current_weight_version
            ]
            print(
                f"   valid_indices: {len(valid_indices)}/{total_trajectories} trajectories within age window"
            )
            if not valid_indices:
                print("No trajectories available for sampling.")
                return None

            # Enforce exact number of groups if available; otherwise, signal to wait
            if len(valid_indices) < num_prompt_groups:
                print(
                    f"Insufficient valid groups: have {len(valid_indices)}, need {num_prompt_groups}. Waiting for buffer to fill."
                )
                return None

            # Only select trajectories intended for the current training step
            # This ensures no trajectory loses its "last chance" to be used for its intended step
            intended_indices = [
                i
                for i in valid_indices
                if self.target_weight_versions[i] == current_weight_version
            ]

            print(
                f"   🎯 Found {len(intended_indices)} trajectories intended for current step {current_weight_version}"
            )

            # Stall training if we don't have enough trajectories intended for this step
            if len(intended_indices) < num_prompt_groups:
                print(
                    f"   ⏸️ STALLING: Need {num_prompt_groups} trajectories for step {current_weight_version}, but only {len(intended_indices)} are ready"
                )
                print(
                    f"   ⏸️ Training will wait for remaining {num_prompt_groups - len(intended_indices)} trajectories to be generated"
                )
                return None

            # Select exactly the trajectories intended for this step (FIFO within same target)
            selected: list[int] = intended_indices[:num_prompt_groups]
            print(
                f"   ✅ Selected {len(selected)} trajectories all intended for step {current_weight_version}"
            )

            sampled_weights = [self.trajectory_versions[i] for i in selected]
            avg_trajectory_age = current_weight_version - sum(sampled_weights) / len(
                sampled_weights
            )
            print(
                f"✅ Selected counts by generation weight-version: {Counter(sampled_weights)}"
            )
            print(f"📊 Average trajectory age: {avg_trajectory_age:.2f} steps")
            print(
                f"🎯 All selected trajectories target step {current_weight_version} (100% target match)"
            )

            # Remove selected items in reverse order to maintain correct indices
            sampled_items = [self.trajectories[i] for i in selected]
            self._remove_indices(selected)

            old_last_target = self.last_target_weight_already_generated
            self.last_target_weight_already_generated = max(
                self.last_target_weight_already_generated,
                current_weight_version,
            )
            if self.last_target_weight_already_generated > old_last_target:
                print(
                    "Advanced last_target_weight_already_generated: "
                    f"{old_last_target} -> "
                    f"{self.last_target_weight_already_generated} "
                    f"(consumed batch for step {current_weight_version})"
                )

            print(
                f"🗑️ Consumed and removed {len(selected)} groups from buffer, old buffer size: {total_trajectories}, new buffer size: {len(self.trajectories)}, new target weight versions {self.target_weight_versions}"
            )

            return {
                "trajectories": sampled_items,
                "avg_trajectory_age": avg_trajectory_age,
            }

    def size(self) -> int:
        """Return current buffer size."""
        with self._lock:
            return len(self.trajectories)

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self.trajectories.clear()
            self.trajectory_versions.clear()
            self.target_weight_versions.clear()

    def state_dict(self) -> dict[str, Any]:
        """Return serializable state for checkpointing."""
        with self._lock:
            return {
                "trajectories": list(self.trajectories),
                "trajectory_versions": list(self.trajectory_versions),
                "target_weight_versions": list(self.target_weight_versions),
                "last_target_weight_already_generated": (
                    self.last_target_weight_already_generated
                ),
                "max_size": self.max_size,
            }

    def load_state_dict(
        self,
        state: dict[str, Any],
        num_prompts_per_step: int | None = None,
        current_training_step: int | None = None,
        max_age_steps: int | None = None,
    ) -> None:
        """Restore replay buffer state from a checkpoint.

        Args:
            state: State returned by ``state_dict``.
            num_prompts_per_step: Number of prompt groups required for one
                training step. When provided, incomplete target steps can be
                removed or prepared for gap filling.
            current_training_step: Step being resumed. When provided with
                ``num_prompts_per_step``, past target steps are dropped and
                incomplete current/future target steps are kept for gap filling.
            max_age_steps: Maximum allowed age for restored trajectories. When
                provided, stale trajectories are removed during restore.

        Raises:
            ValueError: If the checkpoint is missing required fields or has
                inconsistent parallel list lengths.
        """
        with self._lock:
            required_keys = {
                "trajectories",
                "trajectory_versions",
                "target_weight_versions",
                "last_target_weight_already_generated",
            }
            missing_keys = required_keys - set(state)
            if missing_keys:
                raise ValueError(f"Checkpoint missing required keys: {missing_keys}")

            trajectories = list(state["trajectories"])
            trajectory_versions = list(state["trajectory_versions"])
            target_weight_versions = list(state["target_weight_versions"])
            if not (
                len(trajectories)
                == len(trajectory_versions)
                == len(target_weight_versions)
            ):
                raise ValueError(
                    "Checkpoint has inconsistent replay buffer lengths: "
                    f"trajectories={len(trajectories)}, "
                    f"trajectory_versions={len(trajectory_versions)}, "
                    f"target_weight_versions={len(target_weight_versions)}"
                )

            if "max_size" in state and state["max_size"] != self.max_size:
                print(
                    "ReplayBuffer max_size changed: "
                    f"checkpoint={state['max_size']}, current={self.max_size}. "
                    "Using current config value."
                )

            self.trajectories = trajectories
            self.trajectory_versions = trajectory_versions
            self.target_weight_versions = target_weight_versions
            self.last_target_weight_already_generated = state[
                "last_target_weight_already_generated"
            ]

            if current_training_step is not None and num_prompts_per_step is not None:
                self._prepare_for_training_step(
                    current_step=current_training_step,
                    num_prompts_per_step=num_prompts_per_step,
                )
            elif num_prompts_per_step is not None and self.trajectories:
                self._remove_incomplete_target_steps(num_prompts_per_step)

            if max_age_steps is not None and self.trajectories:
                self._remove_stale_trajectories(max_age_steps)
                if current_training_step is None and num_prompts_per_step is not None:
                    self._remove_incomplete_target_steps(num_prompts_per_step)

            self._truncate_to_max_size(current_training_step)

            print(
                f"ReplayBuffer restored: {len(self.trajectories)} trajectories, "
                "last_target_weight_already_generated="
                f"{self.last_target_weight_already_generated}"
            )

    def _prepare_for_training_step(
        self, current_step: int, num_prompts_per_step: int
    ) -> None:
        """Prepare restored state so training can resume at ``current_step``."""
        print(f"   Preparing replay buffer for training step {current_step}...")

        original_count = len(self.trajectories)
        indices_to_keep = [
            i
            for i, target in enumerate(self.target_weight_versions)
            if target >= current_step
        ]

        if len(indices_to_keep) < original_count:
            removed_past = original_count - len(indices_to_keep)
            self.trajectories = [self.trajectories[i] for i in indices_to_keep]
            self.trajectory_versions = [
                self.trajectory_versions[i] for i in indices_to_keep
            ]
            self.target_weight_versions = [
                self.target_weight_versions[i] for i in indices_to_keep
            ]
            print(
                f"   Removed {removed_past} trajectories for past steps "
                f"(target < {current_step})"
            )

        if not self.trajectories:
            self.last_target_weight_already_generated = current_step - 1
            print(
                "   No restored trajectories remain; collector will generate "
                f"from step {current_step}"
            )
            return

        target_counts = Counter(self.target_weight_versions)
        complete_targets = {
            target
            for target, count in target_counts.items()
            if count >= num_prompts_per_step
        }
        incomplete_targets = {
            target
            for target, count in target_counts.items()
            if count < num_prompts_per_step
        }

        print(
            "   Complete targets: "
            f"{sorted(complete_targets) if complete_targets else 'none'}"
        )
        for target in sorted(incomplete_targets):
            print(
                f"   Incomplete target {target}: "
                f"{target_counts[target]}/{num_prompts_per_step}"
            )

        # Let the collector ask each target from current_step onward how many
        # trajectories are still needed, so incomplete restored batches can be
        # gap-filled and complete batches can be skipped.
        self.last_target_weight_already_generated = current_step - 1

    @staticmethod
    def _is_valid_for_target(
        trajectory_version: int, target_step: int, max_age_steps: int | None
    ) -> bool:
        if max_age_steps is None:
            return True
        min_valid_version = max(0, target_step - max_age_steps)
        return min_valid_version <= trajectory_version <= target_step

    def _remove_stale_trajectories(self, max_age_steps: int) -> None:
        """Remove restored trajectories that are stale for their target step.

        Must be called while holding ``self._lock``.
        """
        indices_to_remove = [
            i
            for i, (trajectory_version, target) in enumerate(
                zip(self.trajectory_versions, self.target_weight_versions)
            )
            if not self._is_valid_for_target(trajectory_version, target, max_age_steps)
        ]
        if not indices_to_remove:
            return

        print(
            f"   Removing {len(indices_to_remove)} stale restored trajectories "
            f"(max_age_steps={max_age_steps})"
        )
        self._remove_indices(indices_to_remove)

    def _count_for_target(
        self, target_step: int, max_age_steps: int | None = None
    ) -> int:
        """Count trajectories usable for ``target_step``.

        Must be called while holding ``self._lock``.
        """
        return sum(
            1
            for trajectory_version, target in zip(
                self.trajectory_versions, self.target_weight_versions
            )
            if target == target_step
            and self._is_valid_for_target(
                trajectory_version, target_step, max_age_steps
            )
        )

    def _truncate_to_max_size(self, current_training_step: int | None = None) -> None:
        """Truncate restored state to ``max_size`` after resume cleanup.

        Must be called while holding ``self._lock``.
        """
        if len(self.trajectories) <= self.max_size:
            return

        print(
            f"Truncating restored buffer from {len(self.trajectories)} "
            f"to max_size={self.max_size}"
        )
        if current_training_step is None:
            indices_to_keep = list(
                range(len(self.trajectories) - self.max_size, len(self.trajectories))
            )
        else:
            prioritized_indices = sorted(
                range(len(self.trajectories)),
                key=lambda i: (self.target_weight_versions[i], i),
            )
            indices_to_keep = sorted(prioritized_indices[: self.max_size])

        self.trajectories = [self.trajectories[i] for i in indices_to_keep]
        self.trajectory_versions = [
            self.trajectory_versions[i] for i in indices_to_keep
        ]
        self.target_weight_versions = [
            self.target_weight_versions[i] for i in indices_to_keep
        ]

    def get_trajectories_needed(
        self,
        target_step: int,
        num_prompts_per_step: int,
        max_age_steps: int | None = None,
    ) -> int:
        """Return additional trajectories needed for ``target_step``."""
        with self._lock:
            current_count = self._count_for_target(target_step, max_age_steps)
            return max(0, num_prompts_per_step - current_count)

    def has_complete_batch(
        self,
        target_step: int,
        num_prompts_per_step: int,
        max_age_steps: int | None = None,
    ) -> bool:
        """Return whether ``target_step`` has enough trajectories to train."""
        with self._lock:
            current_count = self._count_for_target(target_step, max_age_steps)
            return current_count >= num_prompts_per_step

    def _remove_incomplete_target_steps(self, num_prompts_per_step: int) -> None:
        """Remove target steps without a complete batch.

        Must be called while holding ``self._lock``.
        """
        target_counts = Counter(self.target_weight_versions)
        incomplete_targets = {
            target
            for target, count in target_counts.items()
            if count < num_prompts_per_step
        }
        if not incomplete_targets:
            print(f"   All target steps have complete batches ({num_prompts_per_step})")
            return

        print(f"   Removing incomplete target steps: {sorted(incomplete_targets)}")
        original_count = len(self.trajectories)
        indices_to_keep = [
            i
            for i, target in enumerate(self.target_weight_versions)
            if target not in incomplete_targets
        ]
        self.trajectories = [self.trajectories[i] for i in indices_to_keep]
        self.trajectory_versions = [
            self.trajectory_versions[i] for i in indices_to_keep
        ]
        self.target_weight_versions = [
            self.target_weight_versions[i] for i in indices_to_keep
        ]
        print(
            f"   Removed {original_count - len(self.trajectories)} trajectories "
            "from incomplete target steps"
        )

        if self.target_weight_versions:
            first_remaining_target = min(self.target_weight_versions)
            self.last_target_weight_already_generated = min(
                self.last_target_weight_already_generated,
                first_remaining_target - 1,
            )
        else:
            self.last_target_weight_already_generated = -1


@ray.remote  # pragma: no cover
class ReplayBuffer(ReplayBufferImpl):
    pass


class TQReplayBuffer:
    """Meta cache + TQ writer with reserve-then-commit slot semantics.

    Metadata, weight, readiness, ready-timestamp, and group-ID lists are
    parallel; a slot stays ready=False until commit fills it.
    """

    def __init__(
        self,
        dp_client: Any,
        partition_id: str,
        *,
        pad_value_dict: Mapping[str, int],
        require_routed_experts: bool = False,
        lifecycle_recorder: Optional[RolloutLifecycleRecorder] = None,
    ):
        self._dp_client = dp_client
        self._partition_id = partition_id
        self._pad_value_dict = dict(pad_value_dict)
        self._require_routed_experts = require_routed_experts
        self._lifecycle_recorder = lifecycle_recorder
        self._owner_capability = object()
        self._prepared_capabilities: dict[str, object] = {}
        self._prepare_observer: Optional[Any] = None
        self.meta_list: list[Optional[KVBatchMeta]] = []
        self.start_weight_list: list[int] = []
        self.end_weight_list: list[int] = []
        # Per-slot target training step (set when force_in_order=True, else None).
        self.target_step_list: list[Optional[int]] = []
        self.ready_list: list[bool] = []
        # Controller-local monotonic time when a committed group became ready.
        # Kept parallel to the other slot lists so observation-only schedulers
        # can compare ready age without parsing the lifecycle ledger.
        self.ready_timestamp_ns_list: list[Optional[int]] = []
        self._group_ids: list[str] = []

    def reserve(
        self,
        *,
        weight_version: int,
        target_step: Optional[int] = None,
        group_id: Optional[str] = None,
    ) -> str:
        """Append an unready slot tagged with weight_version.

        Args:
            weight_version: Weight version stamped on the slot.
            target_step: Training step this slot targets; only consulted by StalenessSampler.force_in_order.
            group_id: Per-group sample_id prefix; defaults to a fresh uuid4.

        Returns:
            group_id used by the matching commit.
        """
        if group_id is None:
            group_id = str(uuid.uuid4())
        if group_id in self._group_ids:
            raise ValueError(f"duplicate live group_id={group_id!r}")
        self.meta_list.append(None)
        self.start_weight_list.append(weight_version)
        self.end_weight_list.append(-1)
        self.target_step_list.append(target_step)
        self.ready_list.append(False)
        self.ready_timestamp_ns_list.append(None)
        self._group_ids.append(group_id)
        if self._lifecycle_recorder is not None:
            self._lifecycle_recorder.record(
                group_id=group_id,
                stage=RolloutLifecycleStage.RESERVED,
                start_weight_version=weight_version,
                target_step=target_step,
            )
        return group_id

    def set_lifecycle_recorder(
        self, recorder: Optional[RolloutLifecycleRecorder]
    ) -> None:
        """Attach or detach the controller-local lifecycle recorder."""
        self._lifecycle_recorder = recorder

    def set_prepare_observer(self, observer: Optional[Any]) -> None:
        """Attach an observer that may return scalar-only per-group metadata."""
        self._prepare_observer = observer

    @property
    def prepare_observer_enabled(self) -> bool:
        """Whether controlled release must prepare its payload before the hold."""
        return self._prepare_observer is not None

    def prepare_commit(
        self,
        group_id: str,
        record: PromptGroupRecord,
        *,
        start_weight_version: int,
    ) -> PreparedTQCommit:
        """Synchronously mint the definitive packed payload for one reserved slot."""
        try:
            idx = self._group_ids.index(group_id)
        except ValueError as error:
            raise ValueError(
                f"prepare called with unknown group_id={group_id!r}"
            ) from error
        if self.ready_list[idx] or self.meta_list[idx] is not None:
            raise ValueError(f"group_id={group_id!r} is already ready")
        if self.start_weight_list[idx] != start_weight_version:
            raise ValueError(
                f"start version mismatch for group_id={group_id!r}: "
                f"reserved={self.start_weight_list[idx]} prepared={start_weight_version}"
            )
        if group_id in self._prepared_capabilities:
            raise ValueError(f"group_id={group_id!r} already has a prepared payload")

        train_batch = record_to_train_batch(record, pad_value_dict=self._pad_value_dict)
        sample_ids, fields, tags = pack_payload(
            train_batch, weight_version=start_weight_version, group_id=group_id
        )
        if self._require_routed_experts and ROUTED_EXPERTS_FIELD not in fields:
            raise RuntimeError(
                "policy.router_replay.enabled=true requires routed_experts in "
                "the SingleController rollout payload, but payload packing did "
                "not produce that field. Check vLLM routed-expert capture and "
                "the async message-log flattening path."
            )
        trace_rollout_payload(keys=sample_ids, data=train_batch)
        payload_state = _tensor_payload_state(fields)
        observer_metadata: tuple[tuple[str, Any], ...] = ()
        if self._prepare_observer is not None:
            raw_observer_metadata = self._prepare_observer(
                group_id=group_id,
                record=record,
                train_batch=train_batch,
                sample_ids=tuple(sample_ids),
                start_weight_version=start_weight_version,
            )
            if _tensor_payload_state(fields) != payload_state:
                raise RuntimeError(
                    "prepare observer mutated the packed training payload"
                )
            if raw_observer_metadata is not None:
                if not isinstance(raw_observer_metadata, Mapping):
                    raise TypeError(
                        "prepare observer metadata must be a mapping or None"
                    )
                frozen_metadata: list[tuple[str, Any]] = []
                for key, value in raw_observer_metadata.items():
                    if not isinstance(key, str) or not key:
                        raise TypeError(
                            "prepare observer metadata keys must be strings"
                        )
                    if (
                        not isinstance(value, (str, bool, int, float))
                        and value is not None
                    ):
                        raise TypeError(
                            "prepare observer metadata values must be scalar JSON values"
                        )
                    if isinstance(value, float) and not math.isfinite(value):
                        raise ValueError(
                            "prepare observer metadata floats must be finite"
                        )
                    frozen_metadata.append((key, value))
                observer_metadata = tuple(sorted(frozen_metadata))

        capability = object()
        self._prepared_capabilities[group_id] = capability
        lengths = train_batch["input_lengths"]
        return PreparedTQCommit(
            group_id=group_id,
            partition_id=self._partition_id,
            start_weight_version=start_weight_version,
            sample_ids=tuple(sample_ids),
            fields=fields,
            tags=tuple(dict(tag) for tag in tags),
            tag_state=tuple(tuple(sorted(tag.items())) for tag in tags),
            sequence_lengths=tuple(int(value) for value in lengths.tolist()),
            observer_metadata=observer_metadata,
            payload_state=payload_state,
            _owner=self._owner_capability,
            _capability=capability,
        )

    async def commit(
        self,
        group_id: str,
        record: PromptGroupRecord,
        start_weight_version: int,
        end_weight_version: int,
    ) -> KVBatchMeta:
        """Tensorize record, write N rows to TQ, and mark the slot ready.

        Args:
            group_id: group_id returned by the matching reserve call.
            record: PromptGroupRecord to tensorize.
            start_weight_version: Weight version stamped on the slot before rollout.
                The same as the one from reserve, passed again to avoid race condition when lookup.
            end_weight_version: Weight version stamped on the slot after rollout.

        Returns:
            KVBatchMeta for the committed group.

        Raises:
            ValueError: group_id has no live slot (removed or never reserved).
            RuntimeError: router replay is enabled but the payload has no routes.
        """
        prepared = self.prepare_commit(
            group_id,
            record,
            start_weight_version=start_weight_version,
        )
        return await self.commit_prepared(
            prepared, end_weight_version=end_weight_version
        )

    async def commit_prepared(
        self,
        prepared: PreparedTQCommit,
        *,
        end_weight_version: int,
    ) -> KVBatchMeta:
        """Transactionally publish one owner-bound prepared payload exactly once."""
        if prepared._owner is not self._owner_capability:
            raise ValueError("prepared payload belongs to another replay buffer")
        if prepared.partition_id != self._partition_id:
            raise ValueError("prepared payload partition no longer matches its owner")
        capability = self._prepared_capabilities.get(prepared.group_id)
        if capability is not prepared._capability:
            raise ValueError("prepared payload is stale, unknown, or already consumed")
        try:
            idx = self._group_ids.index(prepared.group_id)
        except ValueError as error:
            raise ValueError(
                f"prepared payload has unknown group_id={prepared.group_id!r}"
            ) from error
        if self.start_weight_list[idx] != prepared.start_weight_version:
            raise ValueError(
                "prepared payload start version no longer matches its slot"
            )
        if self.ready_list[idx] or self.meta_list[idx] is not None:
            raise ValueError("prepared payload slot is already ready")
        if _tensor_payload_state(prepared.fields) != prepared.payload_state:
            raise ValueError("prepared payload was mutated after preparation")
        if tuple(tuple(sorted(tag.items())) for tag in prepared.tags) != (
            prepared.tag_state
        ):
            raise ValueError("prepared payload tags were mutated after preparation")

        # Consume before the first side effect: a failed transaction is never retried.
        del self._prepared_capabilities[prepared.group_id]
        sample_ids = list(prepared.sample_ids)
        tags = [dict(tag) for tag in prepared.tags]
        try:
            await self._call_dp(
                "put_samples",
                sample_ids=sample_ids,
                partition_id=self._partition_id,
                fields=prepared.fields,
                tags=tags,
            )

            meta = KVBatchMeta(
                partition_id=self._partition_id,
                task_name="train",
                sample_ids=list(sample_ids),
                fields=list(prepared.fields.keys()),
                sequence_lengths=list(prepared.sequence_lengths),
                extra_info=dict(prepared.observer_metadata),
                tags=[dict(t) for t in tags],
            )

            if self._lifecycle_recorder is not None:
                self._lifecycle_recorder.record(
                    group_id=prepared.group_id,
                    stage=RolloutLifecycleStage.GROUP_READY,
                    start_weight_version=prepared.start_weight_version,
                    end_weight_version=end_weight_version,
                    target_step=self.target_step_list[idx],
                    sample_ids=sample_ids,
                )
            self.meta_list[idx] = meta
            self.end_weight_list[idx] = end_weight_version
            self.ready_timestamp_ns_list[idx] = time.perf_counter_ns()
            self.ready_list[idx] = True
            return meta
        except BaseException as commit_error:
            self.meta_list[idx] = None
            self.end_weight_list[idx] = -1
            self.ready_timestamp_ns_list[idx] = None
            self.ready_list[idx] = False
            try:
                await self._call_dp(
                    "clear_samples",
                    sample_ids=list(sample_ids),
                    partition_id=self._partition_id,
                )
            except BaseException as rollback_error:
                if isinstance(commit_error, asyncio.CancelledError):
                    raise commit_error from rollback_error
                raise BaseExceptionGroup(
                    "commit and rollback both failed for "
                    f"group_id={prepared.group_id!r}",
                    [commit_error, rollback_error],
                )
            raise

    async def remove_group(
        self,
        group_id: str,
        *,
        remove_in_dp: bool = False,
        reason: RolloutRemovalReason = RolloutRemovalReason.UNKNOWN,
        learner_weight_version: Optional[int] = None,
    ) -> int:
        """Remove the live slot identified by ``group_id``.

        Args:
            group_id: Group identifier returned by :meth:`reserve`.
            remove_in_dp: Whether to clear rows referenced by a committed slot.
            reason: Why the group is being removed.
            learner_weight_version: Learner version at removal, when applicable.

        Returns:
            Number of removed slots (always one on success).

        Raises:
            ValueError: ``group_id`` has no live slot.
        """
        try:
            idx = self._group_ids.index(group_id)
        except ValueError as error:
            raise ValueError(f"unknown group_id={group_id!r}") from error
        return await self.remove(
            [idx],
            remove_in_dp=remove_in_dp,
            reason=reason,
            learner_weight_version=learner_weight_version,
        )

    async def remove(
        self,
        idxs: list[int],
        remove_in_dp: bool,
        *,
        reason: RolloutRemovalReason = RolloutRemovalReason.UNKNOWN,
        learner_weight_version: Optional[int] = None,
    ) -> int:
        """Drop entries at the given indices and optionally clear them from DataPlane.

        Args:
            idxs: Entry indices to drop. Must be within [0, size).
            remove_in_dp: If True, also clear the dropped rows from DataPlane.
            reason: Why the groups are being removed.
            learner_weight_version: Learner version at removal, when applicable.

        Returns:
            Number of group entries removed from the buffer.
        """
        if len(idxs) == 0:
            return 0

        drop_idxs = sorted(idxs, reverse=True)
        if drop_idxs[0] >= len(self.meta_list):
            raise IndexError(
                f"TQReplayBuffer.remove: indices out of range: {drop_idxs[0]}; "
                f"size={len(self.meta_list)}"
            )

        dropped_sample_ids: list[str] = []
        removed_events: list[
            tuple[str, int, Optional[int], Optional[int], tuple[str, ...]]
        ] = []
        for i in sorted(idxs):
            meta = self.meta_list[i]
            sample_ids = tuple(meta.sample_ids) if meta is not None else ()
            if meta is not None:
                dropped_sample_ids.extend(meta.sample_ids)
            removed_events.append(
                (
                    self._group_ids[i],
                    self.start_weight_list[i],
                    (self.end_weight_list[i] if self.end_weight_list[i] >= 0 else None),
                    self.target_step_list[i],
                    sample_ids,
                )
            )
            self._prepared_capabilities.pop(self._group_ids[i], None)
        for i in drop_idxs:
            del self.meta_list[i]
            del self.start_weight_list[i]
            del self.end_weight_list[i]
            del self.target_step_list[i]
            del self.ready_list[i]
            del self.ready_timestamp_ns_list[i]
            del self._group_ids[i]

        if remove_in_dp:
            await self._call_dp(
                "clear_samples",
                sample_ids=dropped_sample_ids,
                partition_id=self._partition_id,
            )

        if self._lifecycle_recorder is not None:
            for (
                group_id,
                start_weight_version,
                end_weight_version,
                target_step,
                sample_ids,
            ) in removed_events:
                self._lifecycle_recorder.record(
                    group_id=group_id,
                    stage=RolloutLifecycleStage.REMOVED,
                    start_weight_version=start_weight_version,
                    end_weight_version=end_weight_version,
                    target_step=target_step,
                    learner_weight_version=learner_weight_version,
                    sample_ids=sample_ids,
                    removal_reason=reason,
                )

        return len(drop_idxs)

    def size(self) -> int:
        """Return the number of prompt-group entries currently held."""
        return len(self.meta_list)

    def ready_size(self) -> int:
        """Return the number of committed prompt groups currently ready."""
        return sum(self.ready_list)

    def has_group(self, group_id: str) -> bool:
        """Return whether a prompt group still owns a live buffer slot."""
        return group_id in self._group_ids

    def __len__(self) -> int:
        return len(self.meta_list)

    async def _call_dp(self, method_name: str, **kwargs: Any) -> Any:
        """Call a DataPlaneClient method, awaiting Ray remotes if needed."""
        method = getattr(self._dp_client, method_name)
        remote = getattr(method, "remote", None)
        if remote is not None:
            return await remote(**kwargs)
        result = method(**kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result
