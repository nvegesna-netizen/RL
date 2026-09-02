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
import hashlib
import json
import time
import uuid
from functools import partial
from typing import Any, Optional, Union

import ray
import torch

from nemo_rl.algorithms.async_utils.fixed_pool import (
    FixedPoolManifest,
)
from nemo_rl.algorithms.async_utils.staleness_sampler import create_sampler
from nemo_rl.algorithms.async_utils.scheduler_trace import (
    JsonlSchedulerTraceSink,
    NoopSchedulerTraceSink,
    SchedulerEventType,
)
from nemo_rl.algorithms.single_controller_utils.config import (
    AdvantageConfig,
    MasterConfig,
    validate_sampler_buffer_capacity,
    validate_single_controller_config,
)
from nemo_rl.algorithms.single_controller_utils.setup import SingleControllerActorArgs
from nemo_rl.algorithms.single_controller_utils.utils import (
    aggregate_step_metrics,
    fields_for_put,
    reduce_advantage_pump_metrics,
    squeeze_trailing_unit_dim,
    tensor_field,
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


def _resolved_generation_trace_summaries(
    master_config: MasterConfig,
) -> dict[str, str | int | float | bool]:
    """Return provenance-critical values from the resolved runtime config."""
    generation = master_config.policy.get("generation")
    if generation is None:
        raise ValueError("policy.generation is required for scheduler tracing")
    backend = generation["backend"]
    if backend == "vllm":
        context_length = generation["vllm_cfg"]["max_model_len"]
    elif backend == "sglang":
        context_length = generation["sglang_cfg"]["context_length"]
    else:
        context_length = generation["max_new_tokens"]
    stop_token_ids = generation.get("stop_token_ids") or []
    stop_strings = generation.get("stop_strings") or []
    tokenizer_eos_token_id = generation.get("_tokenizer_eos_token_id")
    stop_token_ids_json = json.dumps(stop_token_ids, separators=(",", ":"))
    stop_strings_json = json.dumps(stop_strings, separators=(",", ":"))
    speculative_config = (
        generation.get("vllm_kwargs", {}).get("speculative_config")
        if backend == "vllm"
        else None
    )
    speculative_config_json = json.dumps(
        speculative_config, separators=(",", ":"), sort_keys=True
    )
    tokenizer_name_json = json.dumps(
        master_config.policy["tokenizer"]["name"], separators=(",", ":")
    )
    model_name_json = json.dumps(
        master_config.policy["model_name"], separators=(",", ":")
    )
    return {
        "generation_backend": backend,
        "max_total_sequence_length": master_config.policy["max_total_sequence_length"],
        "configured_max_new_tokens": generation["max_new_tokens"],
        "generation_context_length": context_length,
        "tokenizer_eos_token_present": isinstance(tokenizer_eos_token_id, int),
        "tokenizer_eos_token_id": (
            tokenizer_eos_token_id if isinstance(tokenizer_eos_token_id, int) else -1
        ),
        "effective_stop_token_count": len(stop_token_ids),
        "effective_stop_token_ids_sha256": hashlib.sha256(
            stop_token_ids_json.encode("utf-8")
        ).hexdigest(),
        "effective_single_stop_token_id": (
            stop_token_ids[0]
            if len(stop_token_ids) == 1 and isinstance(stop_token_ids[0], int)
            else -1
        ),
        "effective_stop_string_count": len(stop_strings),
        "effective_stop_strings_sha256": hashlib.sha256(
            stop_strings_json.encode("utf-8")
        ).hexdigest(),
        "vllm_skip_tokenizer_init": (
            bool(generation["vllm_cfg"].get("skip_tokenizer_init", False))
            if backend == "vllm"
            else False
        ),
        "generation_ignore_eos": bool(generation.get("ignore_eos", False)),
        "generation_temperature": generation["temperature"],
        "generation_top_p": generation["top_p"],
        "generation_top_k": (
            generation.get("top_k") if generation.get("top_k") is not None else -1
        ),
        "generation_use_async_rollouts": (
            bool(generation["vllm_cfg"].get("async_engine", False))
            if backend == "vllm"
            else bool(generation.get("use_async_rollouts", False))
        ),
        "generation_study_seed": (
            int(study_seed)
            if backend == "vllm"
            and isinstance(
                (study_seed := generation["vllm_cfg"].get("study_seed")), int
            )
            else -1
        ),
        "generation_speculative_config_sha256": hashlib.sha256(
            speculative_config_json.encode("utf-8")
        ).hexdigest(),
        "policy_tokenizer_name_sha256": hashlib.sha256(
            tokenizer_name_json.encode("utf-8")
        ).hexdigest(),
        "policy_model_name_sha256": hashlib.sha256(
            model_name_json.encode("utf-8")
        ).hexdigest(),
        "grpo_seed": master_config.grpo.seed,
        "num_generations_per_prompt": master_config.grpo.num_generations_per_prompt,
        "max_rollout_turns": master_config.grpo.max_rollout_turns,
        "num_prompts_per_step": master_config.grpo.num_prompts_per_step,
        "max_inflight_prompts": master_config.async_rl.max_inflight_prompts,
        "max_buffered_rollouts": master_config.async_rl.max_buffered_rollouts,
        "vllm_include_stop_str_in_output": True,
        "finish_reason_code_schema_version": 1,
    }


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
        self._fixed_pool_manifest: Optional[FixedPoolManifest] = (
            actor_args.fixed_pool_manifest
        )
        if self._async_cfg.fixed_pool.enabled != (
            self._fixed_pool_manifest is not None
        ):
            raise ValueError(
                "fixed-pool config and setup manifest must be enabled together"
            )
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

        # Built here, not on the driver: Logger backends (wandb/tb/...) hold
        # _thread.lock that Ray can't cloudpickle into the actor.
        self._logger = Logger(master_config.logger)  # type: ignore
        self._logger.log_hyperparams(master_config.model_dump())
        self._timer = Timer()

        # Pin clusters so RayVirtualCluster.__del__ doesn't remove the PGs.
        self._train_cluster = actor_args.train_cluster
        self._inference_cluster = actor_args.inference_cluster

        num_prompts_per_step = self._master_config.grpo.num_prompts_per_step
        self._sampler: Any = (
            None
            if self._fixed_pool_manifest is not None
            else create_sampler(self._buffer, self._async_cfg.sampler)
        )
        trace_cfg = self._async_cfg.scheduler_trace
        self._trace_enabled = trace_cfg.enabled
        fingerprint_record = (
            {
                "mode": "fixed_pool",
                "pool_id": self._fixed_pool_manifest.pool_id,
                "manifest_sha256": self._fixed_pool_manifest.manifest_sha256,
            }
            if self._fixed_pool_manifest is not None
            else self._async_cfg.sampler.model_dump()
        )
        self._sampler_fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_record,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        self._scheduler_trace = (
            JsonlSchedulerTraceSink(
                trace_cfg.path,  # type: ignore[arg-type]
                max_queue_events=trace_cfg.max_queue_events,
                flush_every=trace_cfg.flush_every,
            )
            if trace_cfg.enabled
            else NoopSchedulerTraceSink()
        )
        if self._trace_enabled:
            self._rollout_manager.set_scheduler_trace_sink(self._scheduler_trace)
        if self._sampler is not None:
            required_capacity = self._sampler.required_buffer_capacity(
                num_prompts_per_step
            )
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
        self._collected_groups: int = 0
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
        run_error: Optional[BaseException] = None
        trace_started = False
        try:
            await self._scheduler_trace.start()
            trace_started = True
            if getattr(self, "_trace_enabled", False):
                self._scheduler_trace.emit(
                    SchedulerEventType.RUN_STARTED,
                    sampler_name=(
                        "fixed_pool"
                        if self._fixed_pool_manifest is not None
                        else self._async_cfg.sampler.name
                    ),
                    sampler_fingerprint=self._sampler_fingerprint,
                    trainer_version=self._trainer_version,
                    run_mode=(
                        "fixed_pool"
                        if self._fixed_pool_manifest is not None
                        else "training"
                    ),
                    pool_id=(
                        self._fixed_pool_manifest.pool_id
                        if self._fixed_pool_manifest is not None
                        else None
                    ),
                    pool_manifest_sha256=(
                        self._fixed_pool_manifest.manifest_sha256
                        if self._fixed_pool_manifest is not None
                        else None
                    ),
                    model_revision=(
                        self._fixed_pool_manifest.model_revision
                        if self._fixed_pool_manifest is not None
                        else None
                    ),
                    model_weights_sha256=(
                        self._fixed_pool_manifest.model_weights_sha256
                        if self._fixed_pool_manifest is not None
                        else None
                    ),
                    scalar_summaries={
                        "planned_train_steps": self._master_config.grpo.max_num_steps,
                        "planned_prompt_groups": (
                            len(self._fixed_pool_manifest.items)
                            if self._fixed_pool_manifest is not None
                            else 0
                        ),
                        "fixed_pool_design_id": (
                            self._async_cfg.fixed_pool.design_id
                            if self._fixed_pool_manifest is not None
                            else "none"
                        ),
                        **_resolved_generation_trace_summaries(self._master_config),
                    },
                )
            # Synchronize weights before starting the pumps
            await self._sync_weights()

            if self._fixed_pool_manifest is not None:
                await self._collect_fixed_pool()
            else:
                # Start the rollout and train pumps
                rollout_task = asyncio.create_task(self._rollout_pump())
                train_task = asyncio.create_task(self._train_pump())
                done, _ = await asyncio.wait(
                    {rollout_task, train_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if rollout_task in done:
                    # Propagate rollout failures immediately. A normally exhausted
                    # rollout pump leaves the train pump to drain committed groups.
                    await rollout_task
                await train_task
        except BaseException as error:
            run_error = error
            raise
        finally:
            if "rollout_task" in locals():
                rollout_task.cancel()
                train_task.cancel()
                await asyncio.gather(rollout_task, train_task, return_exceptions=True)
            trace_error: Optional[BaseException] = None
            try:
                if trace_started and getattr(self, "_trace_enabled", False):
                    self._scheduler_trace.emit(
                        SchedulerEventType.RUN_ENDED,
                        sampler_name=(
                            "fixed_pool"
                            if self._fixed_pool_manifest is not None
                            else self._async_cfg.sampler.name
                        ),
                        sampler_fingerprint=self._sampler_fingerprint,
                        trainer_version=self._trainer_version,
                        live_logical_group_ids=tuple(self._buffer.group_ids),
                        run_mode=(
                            "fixed_pool"
                            if self._fixed_pool_manifest is not None
                            else "training"
                        ),
                        pool_id=(
                            self._fixed_pool_manifest.pool_id
                            if self._fixed_pool_manifest is not None
                            else None
                        ),
                        pool_manifest_sha256=(
                            self._fixed_pool_manifest.manifest_sha256
                            if self._fixed_pool_manifest is not None
                            else None
                        ),
                        model_revision=(
                            self._fixed_pool_manifest.model_revision
                            if self._fixed_pool_manifest is not None
                            else None
                        ),
                        model_weights_sha256=(
                            self._fixed_pool_manifest.model_weights_sha256
                            if self._fixed_pool_manifest is not None
                            else None
                        ),
                        terminal_reason=(
                            "fixed_pool_complete"
                            if run_error is None
                            and self._fixed_pool_manifest is not None
                            and self._collected_groups
                            == len(self._fixed_pool_manifest.items)
                            else "max_train_steps"
                            if run_error is None
                            and self._train_steps
                            >= self._master_config.grpo.max_num_steps
                            else "rollout_exhausted"
                            if run_error is None
                            else "cancelled"
                            if isinstance(run_error, asyncio.CancelledError)
                            else "error"
                        ),
                        exception_class=(
                            type(run_error).__name__ if run_error is not None else None
                        ),
                        scalar_summaries={
                            "completed_train_steps": self._train_steps,
                            "collected_prompt_groups": self._collected_groups,
                        },
                    )
            except BaseException as error:
                trace_error = error
            finally:
                if trace_started:
                    try:
                        await self._scheduler_trace.close()
                    except BaseException as close_error:
                        if trace_error is None:
                            trace_error = close_error
                        else:
                            trace_error = BaseExceptionGroup(
                                "scheduler trace final event and close both failed",
                                [trace_error, close_error],
                            )
            try:
                if trace_error is not None:
                    if run_error is None:
                        raise trace_error
                    run_error.add_note(
                        f"scheduler trace shutdown also failed: {trace_error!r}"
                    )
            finally:
                self._logger.finish()

        return {
            "train_steps": self._train_steps,
            "trainer_version": self._trainer_version,
            "collected_prompt_groups": self._collected_groups,
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

    # ── internal helpers ───────────────────────────────────────────────────

    async def _collect_fixed_pool(self) -> None:
        """Generate and archive every predeclared group without training."""
        manifest = self._fixed_pool_manifest
        if manifest is None:
            raise RuntimeError("fixed-pool collection requires a loaded manifest")

        semaphore = asyncio.Semaphore(self._async_cfg.max_inflight_prompts)
        tasks: list[asyncio.Task[None]] = []

        async def collect_one(
            prompt: DatumSpec,
            admission_id: str,
            dispatch_started_event: asyncio.Event,
        ) -> None:
            buffer_acquired = False
            semaphore_acquired = False
            try:
                await self._buffer_capacity.acquire()
                buffer_acquired = True
                await semaphore.acquire()
                semaphore_acquired = True
                handle = await self._rollout_manager.generate_and_push(
                    prompt,
                    target_step=prompt["dispatch_cohort"],
                    admission_id=admission_id,
                    dispatch_started_event=dispatch_started_event,
                )
                if handle is None:
                    raise RuntimeError(
                        "fixed-pool collection requires scheduler tracing"
                    )
                await self._rollout_manager.archive_fixed_pool_group(handle)
                self._collected_groups += 1
            finally:
                dispatch_started_event.set()
                if semaphore_acquired:
                    semaphore.release()
                if buffer_acquired:
                    self._buffer_capacity.release()

        try:
            for prompt_batch in self._dataloader:
                cohorts = tuple(prompt_batch["dispatch_cohort"])
                if not cohorts or len(set(cohorts)) != 1:
                    raise RuntimeError(
                        "a fixed-pool dataloader batch must contain exactly one cohort"
                    )
                cohort = int(cohorts[0])
                admission_id = f"fixed-pool-cohort-{cohort}"
                self._scheduler_trace.emit(
                    SchedulerEventType.ADMISSION_GRANTED,
                    admission_id=admission_id,
                    sampler_name="fixed_pool",
                    sampler_fingerprint=self._sampler_fingerprint,
                    trainer_version=self._trainer_version,
                    sampler_dispatch_index=cohort,
                    target_step=cohort,
                    pool_id=manifest.pool_id,
                    pool_manifest_sha256=manifest.manifest_sha256,
                    run_mode="fixed_pool",
                    scalar_summaries={"expected_prompt_groups": prompt_batch.size},
                )
                for prompt_offset in range(prompt_batch.size):
                    prompt: DatumSpec = {  # type: ignore[assignment]
                        key: value[prompt_offset] for key, value in prompt_batch.items()
                    }
                    dispatch_started_event = asyncio.Event()
                    task = asyncio.create_task(
                        collect_one(prompt, admission_id, dispatch_started_event),
                        name=f"fixed-pool-{prompt['source_pool_ordinal']}",
                    )
                    tasks.append(task)
                    await dispatch_started_event.wait()
                    if task.done() and (error := task.exception()) is not None:
                        raise error

            results = await asyncio.gather(*tasks, return_exceptions=True)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise BaseExceptionGroup("fixed-pool collection failed", failures)
        if self._collected_groups != len(manifest.items):
            raise RuntimeError(
                "fixed-pool dataloader coverage mismatch: collected "
                f"{self._collected_groups}/{len(manifest.items)} groups"
            )

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
        self._rollout_exhausted.clear()
        print("rollout_pump: starting", flush=True)

        async def _dispatch_one_prompt(
            prompt: DatumSpec,
            target_step: Optional[int],
            admission_id: Optional[str],
            task_started_event: asyncio.Event,
        ) -> None:
            task_started_event.set()
            self._inflight_rollouts += 1
            try:
                if getattr(self, "_trace_enabled", False):
                    await self._rollout_manager.generate_and_push(
                        prompt,
                        target_step=target_step,
                        admission_id=admission_id,
                    )
                else:
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
                    admission_id = (
                        str(uuid.uuid4())
                        if getattr(self, "_trace_enabled", False)
                        else None
                    )
                    if getattr(self, "_trace_enabled", False):
                        self._scheduler_trace.emit(
                            SchedulerEventType.ADMISSION_GRANTED,
                            admission_id=admission_id,
                            sampler_name=self._async_cfg.sampler.name,
                            sampler_fingerprint=self._sampler_fingerprint,
                            trainer_version=self._trainer_version,
                            sampler_dispatch_index=self._sampler.dispatch_index,
                            target_step=target_step,
                            scalar_summaries={
                                "expected_prompt_groups": prompt_batch.size
                            },
                        )

                    for prompt_idx in range(prompt_batch.size):
                        prompt: DatumSpec = {  # type: ignore
                            k: v[prompt_idx] for k, v in prompt_batch.items()
                        }

                        # check if buffer is full
                        await self._buffer_capacity.acquire()
                        # check if inflight rollouts is full
                        await sem.acquire()
                        # wait for rollout to be permitted
                        await self._rollout_permitted.wait()

                        task_started_event = asyncio.Event()
                        # dispatch rollout
                        task = rollout_tasks.create_task(
                            _dispatch_one_prompt(
                                prompt,
                                target_step,
                                admission_id,
                                task_started_event,
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
                        if getattr(self, "_trace_enabled", False):
                            evicted_group_ids = (
                                self._sampler.take_last_evicted_group_ids()
                            )
                            for group_id in evicted_group_ids:
                                self._scheduler_trace.emit(
                                    SchedulerEventType.GROUP_EVICTED,
                                    logical_group_id=group_id,
                                    sampler_name=self._async_cfg.sampler.name,
                                    sampler_fingerprint=self._sampler_fingerprint,
                                    trainer_version=self._trainer_version,
                                    terminal_reason="ready_group_weight_window",
                                )
                            if evicted != len(evicted_group_ids):
                                raise RuntimeError(
                                    "sampler eviction count does not match removed group IDs"
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
                        if getattr(self, "_trace_enabled", False):
                            buffered_before_select = len(self._buffer)
                            ready_before_select = sum(self._buffer.ready_list)
                            eligible_group_ids = self._sampler.eligible_group_ids(
                                current_train_weight=self._trainer_version
                            )
                        train_meta, num_groups = await self._sampler.select(
                            current_train_weight=self._trainer_version,
                            min_prompt_groups=min_prompt_groups,
                            max_prompt_groups=max_prompt_groups,
                        )
                        if getattr(self, "_trace_enabled", False):
                            selected_group_ids = (
                                self._sampler.take_last_selected_group_ids()
                            )
                            self._scheduler_trace.emit(
                                SchedulerEventType.SELECT_DECISION,
                                sampler_name=self._async_cfg.sampler.name,
                                sampler_fingerprint=self._sampler_fingerprint,
                                trainer_version=self._trainer_version,
                                sampler_dispatch_index=self._sampler.dispatch_index,
                                min_prompt_groups=min_prompt_groups,
                                max_prompt_groups=max_prompt_groups,
                                ready_prompt_groups=ready_before_select,
                                eligible_prompt_groups=len(eligible_group_ids),
                                eligible_logical_group_ids=eligible_group_ids,
                                selected_logical_group_ids=selected_group_ids,
                                scalar_summaries={
                                    "buffered_prompt_groups": buffered_before_select,
                                    "selected_prompt_groups": num_groups,
                                },
                            )
                            if num_groups != len(selected_group_ids):
                                raise RuntimeError(
                                    "sampler selection count does not match removed group IDs"
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

                self._trainer_version += 1
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

        prompt_ids = tensor_field(data, adv_cfg.prompt_ids_field)
        rewards = squeeze_trailing_unit_dim(
            tensor_field(data, adv_cfg.reward_field)
        ).float()
        token_mask = tensor_field(data, adv_cfg.token_mask_field).float()
        sample_mask = squeeze_trailing_unit_dim(
            tensor_field(data, adv_cfg.sample_mask_field)
        ).float()
        mask = token_mask * sample_mask.unsqueeze(-1)

        repeated_batch: dict[str, torch.Tensor] = {
            "total_reward": rewards,
        }
        for field_name in adv_cfg.repeated_batch_fields:
            repeated_batch[field_name] = squeeze_trailing_unit_dim(
                tensor_field(data, field_name)
            )

        kwargs: dict[str, torch.Tensor] = {}
        if self._policy_logprobs_required:
            kwargs["logprobs_policy"] = tensor_field(
                data,
                adv_cfg.policy_logprobs_field,
            )
        if self._reference_logprobs_required:
            kwargs["logprobs_reference"] = tensor_field(
                data,
                adv_cfg.reference_logprobs_field,
            )

        advantages = self._advantage_estimator.compute_advantage(
            prompt_ids=prompt_ids,
            rewards=rewards,
            mask=mask,
            repeated_batch=repeated_batch,
            **kwargs,
        )
        response_advantages = torch.masked_select(advantages, mask.bool())
        self._step_log_dict["rewards"].append(rewards.detach().cpu())
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
