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

"""SingleController: asyncio orchestrator for the RL training loop.

CPU-only Ray actor that runs two concurrent pumps and coordinates the
other actors via lightweight RPCs. SC sends control signals and reads
metadata only — model tensors still move through DataPlane or NCCL.

Data flow:
  _rollout_pump  → gen.generate_and_push(prompt, dp_client) ← RPC to GenWorker
                     GenWorker → dp_client.put_samples(...)
  _train_pump    → sampler.evict/select against TQReplayBuffer
                 → _advantage_stage(meta) → dp_client.get_samples(...)
                                        → adv_estimator.compute_advantage(...)
                                        → dp_client.put_samples(...)
                 → trainer.begin/train_microbatches/finish_train_step (split API,
                     driver-side TQPolicy via asyncio.to_thread)
                     Trainer → dp_client.get_samples(...)   (via its own client)
                 → dp_client.clear_samples(...)             ← SC clears after train
  _sync_weights  → WeightSynchronizer.sync_weights()
"""

from __future__ import annotations

import asyncio
import time
from functools import partial
from typing import Any, Optional, Union

import ray
import torch

from nemo_rl.algorithms.advantage_estimator import GRPOAdvantageEstimator
from nemo_rl.algorithms.async_utils.gradient_opportunity import (
    GRPOOpportunityInputs,
    GradientOpportunityRecorder,
    compute_grpo_gradient_opportunity,
)
from nemo_rl.algorithms.async_utils.observer_duty import CommonObserverDutyMeter
from nemo_rl.algorithms.async_utils.staleness_sampler import create_sampler
from nemo_rl.algorithms.async_utils.rollout_lifecycle import (
    ControllerEventSequencer,
    RolloutLifecycleRecorder,
    RolloutRemovalReason,
)
from nemo_rl.algorithms.loss import ClippedPGLossFn
from nemo_rl.algorithms.single_controller_utils.config import (
    AdvantageConfig,
    MasterConfig,
    validate_sampler_buffer_capacity,
    validate_single_controller_config,
)
from nemo_rl.algorithms.single_controller_utils.setup import SingleControllerActorArgs
from nemo_rl.algorithms.single_controller_utils.utils import (
    aggregate_step_metrics,
    compute_advantages_from_data,
    fields_for_put,
    prepare_advantage_inputs_from_data,
    reduce_advantage_pump_metrics,
)
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import DP_CALIB_INPUT_FIELDS
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.policy.tq_policy import TQPolicy
from nemo_rl.utils.logger import Logger
from nemo_rl.utils.timer import Timer

Generation = Union[VllmGeneration, SGLangGeneration]


@ray.remote(num_cpus=1, num_gpus=0)  # pragma: no cover
class SingleControllerActor:
    """CPU-only Ray actor that orchestrates the RL training loop.

    Owns two concurrent asyncio tasks:
      - _rollout_pump: dispatches prompts to GenerationWorkerActor
      - _train_pump:   claims DataPlane meta, trains, clears consumed rows,
                       then runs _sync_weights (drain gate + weight
                       synchronization) inline after each optimizer step

    All other actors are passive — they expose methods and wait to be called.
    """

    def __init__(
        self,
        master_config: MasterConfig,
        actor_args: SingleControllerActorArgs,
    ) -> None:
        """Initialize the SingleController actor.

        Args:
            master_config: SC MasterConfig.
            actor_args: Pre-built actor args from setup_single_controller.
        """
        validate_single_controller_config(master_config)

        self._advantage_cfg = AdvantageConfig()
        self._partition_id: str = actor_args.partition_id

        self._master_config = master_config
        self._async_cfg = master_config.async_rl
        self._policy_logprobs_required = not (
            master_config.loss_fn.force_on_policy_ratio
            and master_config.grpo.seq_logprob_error_threshold is None
        )
        self._reference_logprobs_required = not bool(
            master_config.grpo.skip_reference_policy_logprobs_calculation
        )
        self._dp_client = actor_args.dp_client
        self._gen: Generation = actor_args.gen_handle
        self._trainer: TQPolicy = actor_args.trainer_handle
        self._dataloader = actor_args.dataloader
        self._weight_synchronizer = actor_args.weight_synchronizer
        self._advantage_estimator = actor_args.advantage_estimator
        self._loss_fn = actor_args.loss_fn
        self._buffer = actor_args.tq_buffer
        self._rollout_manager = actor_args.rollout_manager
        # Rebind so writer and sampler share one buffer instance even
        # when Ray deserializes rollout_manager and tq_buffer separately.
        self._rollout_manager._tq_buffer = self._buffer
        self._lifecycle_recorder: Optional[RolloutLifecycleRecorder] = None
        self._opportunity_recorder: Optional[GradientOpportunityRecorder] = None
        self._observer_duty_meter: Optional[CommonObserverDutyMeter] = None
        if self._async_cfg.lifecycle_audit_path is not None:
            opportunity_enabled = self._async_cfg.gradient_opportunity_audit.enabled
            controller_sequencer = (
                ControllerEventSequencer() if opportunity_enabled else None
            )
            self._lifecycle_recorder = RolloutLifecycleRecorder(
                controller_sequencer=controller_sequencer
            )
            self._buffer.set_lifecycle_recorder(self._lifecycle_recorder)
            self._rollout_manager.set_lifecycle_recorder(self._lifecycle_recorder)
            if opportunity_enabled:
                assert controller_sequencer is not None
                if not isinstance(
                    self._advantage_estimator, GRPOAdvantageEstimator
                ) or not isinstance(self._loss_fn, ClippedPGLossFn):
                    raise TypeError(
                        "gradient opportunity audit requires native GRPO advantages "
                        "and ClippedPGLossFn"
                    )
                self._opportunity_recorder = GradientOpportunityRecorder(
                    run_id=self._lifecycle_recorder.run_id,
                    clock_domain_id=self._lifecycle_recorder.clock_domain_id,
                    sequencer=controller_sequencer,
                )
                self._observer_duty_meter = CommonObserverDutyMeter(
                    assignment_domain=(
                        self._async_cfg.controlled_release_delay.assignment_domain
                    ),
                    release_arm_labels=tuple(
                        arm.label
                        for arm in self._async_cfg.controlled_release_delay.arms
                    ),
                )
                self._opportunity_recorder.append_header(
                    estimator_name=type(self._advantage_estimator).__name__,
                    estimator_settings={
                        "baseline_algorithm": "nemo_rl_grpo_v1",
                        "normalize_rewards": self._advantage_estimator.normalize_rewards,
                        "normalization_epsilon": 1e-6,
                        "reduction_dtype": "float32",
                        "reward_dtype": "float32",
                        "use_leave_one_out_baseline": (
                            self._advantage_estimator.use_leave_one_out_baseline
                        ),
                    },
                    loss_settings={
                        "disable_ppo_ratio": master_config.loss_fn.disable_ppo_ratio,
                        "positive_example_nll_weight": (
                            master_config.loss_fn.positive_example_nll_weight
                        ),
                        "sequence_level_importance_ratios": (
                            master_config.loss_fn.sequence_level_importance_ratios
                        ),
                        "token_level_loss": master_config.loss_fn.token_level_loss,
                        "use_cispo": master_config.loss_fn.use_cispo,
                    },
                )

                def _record_prepared_opportunity(
                    *,
                    group_id: str,
                    record: Any,
                    train_batch: Any,
                    sample_ids: tuple[str, ...],
                    start_weight_version: int,
                ) -> None:
                    assert self._observer_duty_meter is not None
                    with self._observer_duty_meter.observe():
                        prepared_inputs = prepare_advantage_inputs_from_data(
                            train_batch,
                            advantage_config=self._advantage_cfg,
                            policy_logprobs_required=False,
                            reference_logprobs_required=False,
                        )

                        def _use_prepared_inputs(_: Any) -> GRPOOpportunityInputs:
                            return GRPOOpportunityInputs(
                                prompt_ids=prepared_inputs.prompt_ids,
                                rewards=prepared_inputs.rewards,
                                actor_mask=prepared_inputs.actor_mask,
                                repeated_batch=prepared_inputs.repeated_batch,
                                estimator_kwargs=prepared_inputs.estimator_kwargs,
                            )

                        summary = compute_grpo_gradient_opportunity(
                            train_batch,
                            group_id=group_id,
                            sample_ids=sample_ids,
                            start_weight_version=start_weight_version,
                            truncation=tuple(
                                bool(completion.truncated)
                                for completion in record.completions
                            ),
                            estimator=self._advantage_estimator,
                            prepare_inputs=_use_prepared_inputs,
                        )
                        assert self._opportunity_recorder is not None
                        self._opportunity_recorder.append_group(summary)

                self._buffer.set_prepare_observer(_record_prepared_opportunity)

        # Built here, not on the driver: Logger backends (wandb/tb/...) hold
        # _thread.lock that Ray can't cloudpickle into the actor.
        self._logger = Logger(master_config.logger)  # type: ignore
        self._logger.log_hyperparams(master_config.model_dump())
        self._timer = Timer()

        # Pin clusters so RayVirtualCluster.__del__ doesn't remove the PGs.
        self._train_cluster = actor_args.train_cluster
        self._inference_cluster = actor_args.inference_cluster

        num_prompts_per_step = self._master_config.grpo.num_prompts_per_step
        self._sampler = create_sampler(
            self._buffer,
            self._async_cfg.sampler,
        )
        required_capacity = self._sampler.required_buffer_capacity(num_prompts_per_step)
        validate_sampler_buffer_capacity(
            self._async_cfg,
            required_capacity=required_capacity,
            sampler_name=type(self._sampler).__name__,
        )

        # ── asyncio state ──────────────────────────────────────────────────
        # Gate: cleared during _sync_weights, set when generation may proceed
        self._rollout_permitted: asyncio.Event = asyncio.Event()
        self._rollout_permitted.set()

        # Set only after _rollout_pump exhausts its configured epochs and all
        # dispatched tasks finish successfully. Rollout failures propagate
        # through run() instead of being reported as normal exhaustion.
        self._rollout_exhausted: asyncio.Event = asyncio.Event()

        # Count of in-flight generate_and_push calls
        self._inflight_rollouts: int = 0
        # Cumulative attempts that found the controlled-release buffer full.
        self._buffer_admission_stalls: int = 0

        # Cancellation handles for in-flight rollout dispatches.
        self._dispatched_rollouts: set[asyncio.Task[None]] = set()

        # Backpressure valve: max unconsumed rollout groups allowed in DataPlane.
        # Acquired before each rollout dispatch; released when the buffer
        # drops a group (sampler.evict or post-train buffer.remove).
        self._buffer_capacity: asyncio.Semaphore = asyncio.Semaphore(
            self._async_cfg.max_buffered_rollouts
        )

        self._trainer_version: int = 0
        self._train_steps: int = 0
        self._current_epoch: int = 0
        self._step_log_dict: dict[str, list] = {
            "rewards": [],
            "masked_advantages": [],
            "sequence_lengths": [],
        }

        print(
            f"SingleControllerActor: "
            f"sampler={self._async_cfg.sampler.name} "
            f"buffer={self._async_cfg.max_buffered_rollouts} "
            f"inflight={self._async_cfg.max_inflight_prompts} "
            f"weight_sync={type(self._weight_synchronizer).__name__}",
            flush=True,
        )

    # ── public API ─────────────────────────────────────────────────────────

    async def run(self) -> dict[str, Any]:
        """Main entry point. Runs until max_train_steps is reached."""
        # Synchronize weights before starting the pumps
        await self._sync_weights()
        if self._observer_duty_meter is not None:
            self._observer_duty_meter.begin_active_window()

        # Start the rollout and train pumps
        rollout_task = asyncio.create_task(self._rollout_pump())
        train_task = asyncio.create_task(self._train_pump())
        cleanup_reason = RolloutRemovalReason.CANCELLED
        try:
            done, _ = await asyncio.wait(
                {rollout_task, train_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if rollout_task in done:
                # Propagate rollout failures immediately. A normally exhausted
                # rollout pump leaves the train pump to drain committed groups.
                await rollout_task
            await train_task
            expected_steps = self._master_config.grpo.max_num_steps
            completed_bounded_run = (
                self._train_steps == expected_steps
                and self._trainer_version == expected_steps
            )
            if (
                self._async_cfg.controlled_release_delay.enabled
                and not completed_bounded_run
            ):
                raise RuntimeError(
                    "controlled-release run ended before the configured "
                    "train-step/version boundary: "
                    f"steps={self._train_steps} "
                    f"version={self._trainer_version} "
                    f"expected={expected_steps}"
                )
            if (
                self._async_cfg.controlled_release_delay.enabled
                and completed_bounded_run
            ):
                cleanup_reason = RolloutRemovalReason.BOUNDED_SHUTDOWN
        finally:
            set_cancellation_reason = getattr(
                self._rollout_manager, "set_cancellation_removal_reason", None
            )
            if set_cancellation_reason is not None:
                set_cancellation_reason(cleanup_reason)
            rollout_task.cancel()
            train_task.cancel()
            await asyncio.gather(rollout_task, train_task, return_exceptions=True)
            try:
                await self._cancel_residual_buffer_groups(reason=cleanup_reason)
            finally:
                if self._observer_duty_meter is not None:
                    self._observer_duty_meter.end_active_window()
                try:
                    if self._lifecycle_recorder is not None:
                        assert self._async_cfg.lifecycle_audit_path is not None
                        self._lifecycle_recorder.flush_jsonl(
                            self._async_cfg.lifecycle_audit_path
                        )
                finally:
                    try:
                        if self._opportunity_recorder is not None:
                            opportunity_path = (
                                self._async_cfg.gradient_opportunity_audit.output_path
                            )
                            assert opportunity_path is not None
                            self._opportunity_recorder.flush_jsonl(opportunity_path)
                    finally:
                        try:
                            if self._observer_duty_meter is not None:
                                duty_path = self._async_cfg.gradient_opportunity_audit.observer_duty_path
                                assert duty_path is not None
                                self._observer_duty_meter.flush_json(duty_path)
                        finally:
                            self._logger.finish()

        return {
            "train_steps": self._train_steps,
            "trainer_version": self._trainer_version,
        }

    async def ping(self) -> dict[str, Any]:
        """Liveness check — returns immediately if event loop is running."""
        return {
            "alive": True,
            "trainer_version": self._trainer_version,
            "train_steps": self._train_steps,
            "inflight_rollouts": self._inflight_rollouts,
            "rollout_permitted": self._rollout_permitted.is_set(),
            "epoch": self._current_epoch,
        }

    async def _cancel_residual_buffer_groups(
        self, *, reason: RolloutRemovalReason
    ) -> None:
        """Clear and terminally record groups left at bounded-run shutdown."""
        residual_groups = len(self._buffer)
        if residual_groups == 0:
            return
        await self._buffer.remove(
            list(range(residual_groups)),
            remove_in_dp=True,
            reason=reason,
            learner_weight_version=self._trainer_version,
        )

    def _advance_trainer_version(self) -> None:
        """Advance and audit the learner version after a successful train step."""
        previous_version = self._trainer_version
        self._trainer_version += 1
        if self._lifecycle_recorder is not None:
            self._lifecycle_recorder.record_learner_version_advanced(
                previous_version=previous_version,
                learner_weight_version=self._trainer_version,
            )

    # ── internal helpers ───────────────────────────────────────────────────

    async def _ray_get(self, obj_ref: Any) -> Any:
        """Await a Ray ObjectRef without blocking the asyncio event loop."""
        return await obj_ref

    async def _call_dp(self, method_name: str, **kwargs) -> Any:
        """Call a DataPlaneClient method or a Ray actor exposing that method."""
        method = getattr(self._dp_client, method_name)
        remote = getattr(method, "remote", None)
        if remote is not None:
            return await self._ray_get(remote(**kwargs))
        result = method(**kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    # ── the three pumps + the inline advantage stage ───────────────────────

    async def _rollout_pump(self) -> None:
        """Continuously dispatch rollout tasks until cancellation.

        Per batch:
          0. await sampler.admit(...) to wait until the batch may dispatch and
             obtain its target_step stamp.

        Per prompt:
          1. Acquire _buffer_capacity slot (backpressure)
          2. Acquire sem (cap concurrent in-flight rollouts)
          3. Wait for _rollout_permitted (paused during weight sync)
          4. Call rollout_manager.generate_and_push(prompt) — local async
             RolloutManager reserves a slot, runs the rollout, then commits the
             group via TQReplayBuffer (→ dp_client.put_samples + mark ready)
          5. Decrement _inflight_rollouts
        """
        sem = asyncio.Semaphore(self._async_cfg.max_inflight_prompts)
        controlled_release_enabled = bool(
            getattr(self._rollout_manager, "controlled_release_enabled", False)
        )
        if not hasattr(self, "_buffer_admission_stalls"):
            self._buffer_admission_stalls = 0
        self._rollout_exhausted.clear()
        print("rollout_pump: starting", flush=True)

        async def _dispatch_one_prompt(
            prompt: DatumSpec,
            target_step: Optional[int],
            task_started_event: asyncio.Event,
        ) -> None:
            task_started_event.set()
            self._inflight_rollouts += 1
            if controlled_release_enabled:
                pending = None
                try:
                    try:
                        pending = await self._rollout_manager.generate_pending(
                            prompt,
                            target_step=target_step,
                            generation_inflight=self._inflight_rollouts,
                            buffer_admission_stalls=self._buffer_admission_stalls,
                        )
                    finally:
                        # A controlled hold owns its reserved buffer slot but no
                        # longer consumes generation admission.
                        self._inflight_rollouts -= 1
                        sem.release()
                    await self._rollout_manager.release_and_commit(
                        pending,
                        generation_inflight_fn=lambda: self._inflight_rollouts,
                        buffer_admission_stalls_fn=(
                            lambda: self._buffer_admission_stalls
                        ),
                    )
                except BaseException as error:
                    if pending is not None:
                        await self._rollout_manager.abort_pending(
                            pending,
                            reason=(
                                RolloutRemovalReason.CANCELLED
                                if isinstance(error, asyncio.CancelledError)
                                else RolloutRemovalReason.FAILED
                            ),
                        )
                    self._buffer_capacity.release()
                    raise
            else:
                try:
                    await self._rollout_manager.generate_and_push(
                        prompt, target_step=target_step
                    )
                except BaseException:
                    # On success ownership transfers to the train pump, which
                    # releases this permit after consuming the committed group.
                    self._buffer_capacity.release()
                    raise
                finally:
                    self._inflight_rollouts -= 1
                    sem.release()

            if self._async_cfg.diagnostics:
                content = ""
                for i in range(len(prompt["message_log"])):
                    if prompt["message_log"][i]["role"] == "user":
                        content = prompt["message_log"][i]["content"]
                        break
                print(f"  rollout done for prompt='{content[:20]}...'", flush=True)

        def _release_permits_if_task_not_started(
            _: asyncio.Task[Any],
            *,
            task_started_event: asyncio.Event,
        ) -> None:
            if not task_started_event.is_set():
                self._buffer_capacity.release()
                sem.release()

        max_epochs = self._master_config.grpo.max_num_epochs
        async with asyncio.TaskGroup() as rollout_tasks:
            while max_epochs is None or self._current_epoch < max_epochs:
                for prompt_batch in self._dataloader:
                    target_step = await self._sampler.admit(
                        trainer_version_fn=lambda: self._trainer_version
                    )

                    for prompt_idx in range(prompt_batch.size):
                        prompt: DatumSpec = {  # type: ignore
                            k: v[prompt_idx] for k, v in prompt_batch.items()
                        }

                        # Check if buffer is full. This diagnostic is evaluated
                        # only on the enabled research path.
                        if (
                            controlled_release_enabled
                            and self._buffer_capacity.locked()
                        ):
                            self._buffer_admission_stalls += 1
                        await self._buffer_capacity.acquire()
                        # check if inflight rollouts is full
                        await sem.acquire()
                        # wait for rollout to be permitted
                        await self._rollout_permitted.wait()

                        task_started_event = asyncio.Event()
                        # dispatch rollout
                        task = rollout_tasks.create_task(
                            _dispatch_one_prompt(
                                prompt, target_step, task_started_event
                            )
                        )
                        self._dispatched_rollouts.add(task)
                        task.add_done_callback(self._dispatched_rollouts.discard)
                        task.add_done_callback(
                            partial(
                                _release_permits_if_task_not_started,
                                task_started_event=task_started_event,
                            )
                        )

                self._current_epoch += 1

        # Drain in-flight so return implies "all rollouts in TQ".
        inflight = list(self._dispatched_rollouts)
        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)

        self._rollout_exhausted.set()
        print(f"rollout_pump: completed {self._current_epoch} epoch(s)", flush=True)

    async def _train_pump(self) -> None:
        """Per-prompt-group streaming train loop.

        Per step:
          1. sampler.evict drops stale groups from the buffer and clears their TQ rows.
          2. sampler.select returns K prompt groups (or None) and drops them from the
             buffer; DP rows survive so the trainer can read them. Already trainable —
             buffer wrote training-shaped rows at rollout time.
          3. _advantage_stage(train_meta).
          4. trainer.train_microbatches_from_meta + finish_train_step.
          5. dp_client.clear_samples on consumed sample_ids; release _buffer_capacity
             per dropped group, then sync.
        """
        grpo_cfg = self._master_config.grpo

        while self._train_steps < grpo_cfg.max_num_steps:
            version_during_step = self._trainer_version
            groups_dispatched = 0
            min_sample_version = None
            step_open = False
            completed_step_sample_ids: list[str] = []
            calibration_batches: list[BatchedDataDict[Any]] = []

            with self._timer.time("total_step_time"):
                while groups_dispatched < grpo_cfg.num_prompts_per_step:
                    # Wait for a selectable batch
                    with self._timer.time("exposed_generation"):
                        await asyncio.sleep(0)

                        # Evict stale groups
                        evicted = await self._sampler.evict(
                            current_train_weight=self._trainer_version,
                        )
                        if evicted:
                            print(
                                f"  evicted {evicted} stale prompt group(s)",
                                flush=True,
                            )
                            for _ in range(evicted):
                                self._buffer_capacity.release()

                        # Select a batch
                        max_prompt_groups = (
                            grpo_cfg.num_prompts_per_step - groups_dispatched
                        )
                        min_prompt_groups = min(
                            self._async_cfg.min_groups_for_streaming_train,
                            max_prompt_groups,
                        )
                        train_meta, num_groups = await self._sampler.select(
                            current_train_weight=self._trainer_version,
                            min_prompt_groups=min_prompt_groups,
                            max_prompt_groups=max_prompt_groups,
                        )

                        # If no batch is selectable, sleep and retry
                        if train_meta is None:
                            if self._rollout_exhausted.is_set():
                                buffered_groups = len(self._buffer)
                                if groups_dispatched == 0 and buffered_groups == 0:
                                    print(
                                        "train_pump: rollout exhausted and "
                                        "buffer drained",
                                        flush=True,
                                    )
                                    return
                                raise RuntimeError(
                                    "rollout exhausted before a complete training "
                                    f"step was assembled: dispatched "
                                    f"{groups_dispatched}/"
                                    f"{grpo_cfg.num_prompts_per_step} prompt "
                                    f"groups with {buffered_groups} group(s) "
                                    f"remaining in the buffer"
                                )
                            await asyncio.sleep(0.005)
                            continue

                        # Release buffer capacity
                        for _ in range(num_groups):
                            self._buffer_capacity.release()

                    # Compute prev_logprobs / ref_logprobs
                    if (
                        self._policy_logprobs_required
                        or self._reference_logprobs_required
                    ):
                        with self._timer.time("logprob_inference_prep"):
                            await asyncio.to_thread(
                                self._trainer.prepare_for_lp_inference
                            )
                        with self._timer.time("policy_and_reference_logprobs"):
                            if self._policy_logprobs_required:
                                await asyncio.to_thread(
                                    self._trainer.get_logprobs_from_meta, train_meta
                                )
                            if self._reference_logprobs_required:
                                await asyncio.to_thread(
                                    self._trainer.get_reference_policy_logprobs_from_meta,
                                    train_meta,
                                )

                    # Compute advantages
                    with self._timer.time("advantage_calculation"):
                        train_meta = await self._advantage_stage(train_meta)

                    # Train
                    with self._timer.time("training_prep"):
                        await asyncio.to_thread(self._trainer.prepare_for_training)
                    with self._timer.time("policy_training"):
                        if not step_open:
                            await asyncio.to_thread(
                                self._trainer.begin_train_step,
                                self._loss_fn,
                            )
                            step_open = True
                        await asyncio.to_thread(
                            self._trainer.train_microbatches_from_meta,
                            train_meta,
                        )
                        completed_step_sample_ids.extend(train_meta.sample_ids)

                    if train_meta.sequence_lengths:
                        self._step_log_dict["sequence_lengths"].extend(
                            int(s) for s in train_meta.sequence_lengths
                        )

                    if getattr(self._gen, "requires_kv_scale_sync", False):
                        calibration_fields = [
                            field
                            for field in (train_meta.fields or [])
                            if field in DP_CALIB_INPUT_FIELDS
                        ]
                        calibration_batches.append(
                            await asyncio.to_thread(
                                self._trainer.read_from_dataplane,
                                train_meta,
                                select_fields=calibration_fields,
                            )
                        )

                    # Refresh min_sample_version
                    curr_min_sample_version = min(
                        t["weight_version"]
                        for t in train_meta.tags  # type: ignore
                    )
                    if min_sample_version is not None:
                        min_sample_version = min(
                            min_sample_version, curr_min_sample_version
                        )
                    else:
                        min_sample_version = curr_min_sample_version

                    # Remove consumed sample_ids from the buffer
                    await self._call_dp(
                        "clear_samples",
                        sample_ids=list(train_meta.sample_ids),
                        partition_id=self._partition_id,
                    )

                    groups_dispatched += num_groups

                with self._timer.time("policy_training"):
                    result = await asyncio.to_thread(self._trainer.finish_train_step)

                step_metrics = aggregate_step_metrics(result)
                step_metrics.update(
                    reduce_advantage_pump_metrics(**self._step_log_dict)
                )
                self._step_log_dict = {k: [] for k in self._step_log_dict}

                previous_trainer_version = self._trainer_version
                self._advance_trainer_version()
                if self._opportunity_recorder is not None:
                    self._opportunity_recorder.append_train_step_completed(
                        previous_learner_version=previous_trainer_version,
                        learner_version=self._trainer_version,
                        sample_ids=completed_step_sample_ids,
                    )
                self._train_steps += 1
                with self._timer.time("weight_sync"):
                    calibration_data = (
                        BatchedDataDict.from_batches(calibration_batches)
                        if calibration_batches
                        else None
                    )
                    await self._sync_weights(calibration_data=calibration_data)

            timing_metrics: dict[str, float] = self._timer.get_timing_metrics(
                reduction_op="sum"
            )  # type: ignore

            total_time = timing_metrics.get("total_step_time", 0.0)
            total_num_gpus = int(ray.cluster_resources().get("GPU", 0))
            if (
                total_time > 0
                and total_num_gpus > 0
                and "global_valid_toks" in step_metrics
            ):
                timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                    step_metrics["global_valid_toks"] / total_time / total_num_gpus
                )

            print("\n⏱️  Timing:")
            print(f"  • Total step time: {total_time:.2f}s")
            for k, v in sorted(
                timing_metrics.items(), key=lambda item: item[1], reverse=True
            ):
                if k == "total_step_time":
                    continue
                percent = (v / total_time * 100) if total_time > 0 else 0.0
                print(f"  • {k}: {v:.2f}s ({percent:.1f}%)")

            # TODO: checkpointing (save_period/top-k metric_name,
            #   policy.save_checkpoint, dataloader state, TQReplayBuffer state).
            # TODO: per-step train_data jsonl dump, vllm metrics logger,
            #   histogram log, rollout_metrics, seq_logprob_error_metrics,
            #   pretty-print "Training Results" block, print_performance_metrics.
            print(f"step_metrics={step_metrics}", flush=True)
            self._logger.log_metrics(
                step_metrics, step=self._train_steps, prefix="train"
            )
            self._logger.log_metrics(
                timing_metrics, step=self._train_steps, prefix="timing/train"
            )
            self._timer.reset()

            # min sample version refers to the version each consumed sample was
            # generated with; lag = training version - oldest sample version.
            lag = version_during_step - min_sample_version  # type: ignore
            print(
                f"train step {self._train_steps}/{grpo_cfg.max_num_steps}  "
                f"trainer_v={self._trainer_version}  "
                f"lag={lag}  ",
                flush=True,
            )

    async def _sync_weights(
        self,
        *,
        calibration_data: Optional[BatchedDataDict[Any]] = None,
    ) -> None:
        """Pause new rollout dispatches, synchronize weights, resume.

        SC owns the pause gate; in-flight generations continue through the
        refit — vLLM V1 async engine supports weight updates during pending
        requests.

        Flow:
          1. _rollout_permitted.clear()  — no new dispatches
          2. Optionally calibrate FP8 KV-cache scales.
          3. weight_synchronizer.sync_weights(kv_scales=...)
          4. _rollout_permitted.set()   — resume
        """
        self._rollout_permitted.clear()

        # TODO(#2625): Add drain-gate support during refit.

        t0 = time.monotonic()
        kv_scales = None
        if (
            getattr(self._gen, "requires_kv_scale_sync", False)
            and calibration_data is not None
        ):
            print("▶ Computing KV cache scales...", flush=True)
            calibration_result = await asyncio.to_thread(
                self._trainer.calibrate_qkv_fp8_scales,
                calibration_data,
                include_q=True,
            )
            kv_scales = calibration_result["layers"]

        await asyncio.to_thread(
            self._weight_synchronizer.sync_weights,
            kv_scales=kv_scales,
        )
        if self._async_cfg.recompute_kv_cache_after_weight_updates:
            self._gen.invalidate_kv_cache()
        elapsed = time.monotonic() - t0

        print(f"  _sync_weights: sync done in {elapsed:.3f}s", flush=True)
        self._rollout_manager.set_weight_version(self._trainer_version)
        self._rollout_permitted.set()

    async def _advantage_stage(self, meta: KVBatchMeta) -> KVBatchMeta:
        """Fetch advantage inputs, compute advantages, and write them back.

        SC owns the prompt-group-scoped advantage stage because the selected
        ``KVBatchMeta`` still contains complete prompt groups before trainer
        DP sharding. Tensor payloads still move through DataPlane: SC fetches
        only the configured advantage input columns and writes the computed
        ``advantages`` column back under the same ``sample_ids``.
        """
        if self._advantage_estimator is None:
            return meta
        adv_cfg = self._advantage_cfg

        data = await self._call_dp(
            "get_samples",
            sample_ids=meta.sample_ids,
            partition_id=meta.partition_id,
            select_fields=self._advantage_input_fields(),
        )

        computed = compute_advantages_from_data(
            data,
            advantage_config=adv_cfg,
            advantage_estimator=self._advantage_estimator,
            policy_logprobs_required=self._policy_logprobs_required,
            reference_logprobs_required=self._reference_logprobs_required,
        )
        advantages = computed.advantages
        response_advantages = torch.masked_select(
            advantages, computed.actor_mask.bool()
        )
        self._step_log_dict["rewards"].append(computed.rewards.detach().cpu())
        self._step_log_dict["masked_advantages"].append(
            response_advantages.detach().cpu()
        )

        await self._call_dp(
            "put_samples",
            sample_ids=meta.sample_ids,
            partition_id=meta.partition_id,
            fields=fields_for_put(
                meta,
                {adv_cfg.output_field: advantages},
            ),
        )
        return meta.with_fields([adv_cfg.output_field])

    # ── utility helpers ────────────────────────────────────────────────────

    def _advantage_input_fields(self) -> list[str]:
        adv_cfg = self._advantage_cfg
        fields = [
            adv_cfg.prompt_ids_field,
            adv_cfg.reward_field,
            adv_cfg.token_mask_field,
            adv_cfg.sample_mask_field,
            *adv_cfg.repeated_batch_fields,
        ]
        if self._policy_logprobs_required:
            fields.append(adv_cfg.policy_logprobs_field)
        if self._reference_logprobs_required:
            fields.append(adv_cfg.reference_logprobs_field)
        return list(dict.fromkeys(fields))
