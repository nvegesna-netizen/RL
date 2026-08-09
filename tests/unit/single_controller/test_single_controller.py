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

"""Tests for SingleController initialization and pump lifecycle."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

import nemo_rl.algorithms.single_controller as single_controller
from nemo_rl.algorithms.grpo import GRPOConfig
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.algorithms.single_controller_utils.config import (
    AdvantageConfig,
    AsyncRLConfig,
    GradientOpportunityAuditConfig,
    MasterConfig,
    validate_single_controller_config,
)
from nemo_rl.algorithms.async_utils.controlled_release import (
    ControlledReleaseDelayConfig,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.utils.timer import Timer


class FakeWeightSynchronizer:
    pass


def _controlled_release_master_config(
    *, lifecycle_audit_path: str | None, use_nemo_gym: bool = False
) -> MasterConfig:
    return MasterConfig.model_construct(
        policy={"train_global_batch_size": 4},
        grpo=GRPOConfig.model_construct(
            num_prompts_per_step=1,
            num_generations_per_prompt=4,
            skip_reference_policy_logprobs_calculation=True,
        ),
        loss_fn=ClippedPGLossConfig(reference_policy_kl_penalty=0),
        async_rl=AsyncRLConfig(
            min_groups_for_streaming_train=1,
            lifecycle_audit_path=lifecycle_audit_path,
            controlled_release_delay=ControlledReleaseDelayConfig(enabled=True),
        ),
        env={"should_use_nemo_gym": use_nemo_gym},
    )


@pytest.mark.parametrize("lifecycle_audit_path", [None, ""])
def test_controlled_release_requires_lifecycle_audit(
    lifecycle_audit_path: str | None,
) -> None:
    config = _controlled_release_master_config(
        lifecycle_audit_path=lifecycle_audit_path
    )

    with pytest.raises(ValueError, match="requires async_rl.lifecycle_audit_path"):
        validate_single_controller_config(config)


def test_controlled_release_rejects_nemo_gym() -> None:
    config = _controlled_release_master_config(
        lifecycle_audit_path="audit.jsonl", use_nemo_gym=True
    )

    with pytest.raises(ValueError, match="only by native async rollouts"):
        validate_single_controller_config(config)


def test_gradient_opportunity_audit_requires_controlled_release() -> None:
    config = _controlled_release_master_config(lifecycle_audit_path="lifecycle.jsonl")
    config.async_rl.controlled_release_delay = ControlledReleaseDelayConfig()
    config.async_rl.gradient_opportunity_audit = GradientOpportunityAuditConfig(
        enabled=True,
        output_path="opportunity.jsonl",
    )

    with pytest.raises(ValueError, match="requires.*controlled_release_delay"):
        validate_single_controller_config(config)


@pytest.mark.parametrize("output_path", [None, ""])
def test_gradient_opportunity_audit_requires_output_path(
    output_path: str | None,
) -> None:
    config = _controlled_release_master_config(lifecycle_audit_path="lifecycle.jsonl")
    config.async_rl.gradient_opportunity_audit = GradientOpportunityAuditConfig(
        enabled=True,
        output_path=output_path,
    )

    with pytest.raises(ValueError, match="requires.*output_path"):
        validate_single_controller_config(config)


def test_gradient_opportunity_audit_requires_distinct_resolved_path() -> None:
    config = _controlled_release_master_config(lifecycle_audit_path="audit.jsonl")
    config.async_rl.gradient_opportunity_audit = GradientOpportunityAuditConfig(
        enabled=True,
        output_path="./audit.jsonl",
    )

    with pytest.raises(ValueError, match="distinct paths"):
        validate_single_controller_config(config)


def test_gradient_opportunity_audit_accepts_supported_configuration() -> None:
    config = _controlled_release_master_config(lifecycle_audit_path="lifecycle.jsonl")
    config.async_rl.gradient_opportunity_audit = GradientOpportunityAuditConfig(
        enabled=True,
        output_path="opportunity.jsonl",
    )

    validate_single_controller_config(config)


def test_gradient_opportunity_audit_rejects_non_grpo_estimator() -> None:
    config = _controlled_release_master_config(lifecycle_audit_path="lifecycle.jsonl")
    config.async_rl.gradient_opportunity_audit = GradientOpportunityAuditConfig(
        enabled=True,
        output_path="opportunity.jsonl",
    )
    config.grpo.adv_estimator.name = "gdpo"

    with pytest.raises(ValueError, match="native GRPO"):
        validate_single_controller_config(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("disable_ppo_ratio", True),
        ("token_level_loss", False),
        ("sequence_level_importance_ratios", True),
        ("use_cispo", True),
        ("positive_example_nll_weight", 0.1),
    ],
)
def test_gradient_opportunity_audit_rejects_unsupported_loss(
    field: str,
    value: object,
) -> None:
    config = _controlled_release_master_config(lifecycle_audit_path="lifecycle.jsonl")
    config.async_rl.gradient_opportunity_audit = GradientOpportunityAuditConfig(
        enabled=True,
        output_path="opportunity.jsonl",
    )
    setattr(config.loss_fn, field, value)

    with pytest.raises(ValueError, match="ordinary token-level clipped PG"):
        validate_single_controller_config(config)


def test_rejects_multiple_optimizer_steps_per_rl_step(monkeypatch) -> None:
    monkeypatch.setattr(single_controller, "Logger", lambda _: object())
    master_config = MasterConfig.model_construct(
        policy={"train_global_batch_size": 4},
        grpo=GRPOConfig.model_construct(
            num_prompts_per_step=2,
            num_generations_per_prompt=4,
        ),
        async_rl=AsyncRLConfig(min_groups_for_streaming_train=1),
        logger={},
    )
    actor_args = SimpleNamespace(
        partition_id="rollout_data",
        dp_client=None,
        gen_handle=None,
        trainer_handle=None,
        dataloader=None,
        weight_synchronizer=None,
        advantage_estimator=None,
        loss_fn=None,
        tq_buffer=None,
        rollout_manager=SimpleNamespace(_tq_buffer=None),
        train_cluster=None,
        inference_cluster=None,
    )
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class

    with pytest.raises(
        ValueError,
        match=(
            r"num_prompts_per_step \* num_generations_per_prompt \(8\) "
            r"must equal policy.train_global_batch_size \(4\)"
        ),
    ):
        controller_cls(
            master_config=master_config,
            actor_args=actor_args,
        )


def test_logs_hyperparameters_and_concrete_weight_synchronizer(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = MagicMock()
    monkeypatch.setattr(single_controller, "Logger", lambda _: logger)
    master_config = MasterConfig.model_construct(
        policy={"train_global_batch_size": 8},
        grpo=GRPOConfig.model_construct(
            num_prompts_per_step=2,
            num_generations_per_prompt=4,
        ),
        loss_fn=ClippedPGLossConfig(force_on_policy_ratio=False),
        async_rl=AsyncRLConfig(
            min_groups_for_streaming_train=1,
            max_buffered_rollouts=4,
        ),
        logger={},
    )
    actor_args = SimpleNamespace(
        partition_id="rollout_data",
        dp_client=None,
        gen_handle=None,
        trainer_handle=None,
        dataloader=None,
        weight_synchronizer=FakeWeightSynchronizer(),
        advantage_estimator=None,
        loss_fn=None,
        tq_buffer=None,
        rollout_manager=SimpleNamespace(_tq_buffer=None),
        train_cluster=None,
        inference_cluster=None,
    )
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class

    controller_cls(
        master_config=master_config,
        actor_args=actor_args,
    )

    logger.log_hyperparams.assert_called_once_with(master_config.model_dump())
    output = capsys.readouterr().out
    assert "weight_sync=FakeWeightSynchronizer" in output
    assert "transport=stub" not in output


@pytest.mark.parametrize(
    ("recompute_kv_cache", "expected_invalidation_calls"),
    [(False, 0), (True, 1)],
)
def test_sync_weights_honors_recompute_kv_cache_config(
    recompute_kv_cache: bool,
    expected_invalidation_calls: int,
) -> None:
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._async_cfg = AsyncRLConfig(
        recompute_kv_cache_after_weight_updates=recompute_kv_cache
    )
    ctrl._rollout_permitted = asyncio.Event()
    ctrl._rollout_permitted.set()
    ctrl._weight_synchronizer = SimpleNamespace(sync_weights=MagicMock())
    ctrl._gen = SimpleNamespace(
        invalidate_kv_cache=MagicMock(),
        requires_kv_scale_sync=False,
    )
    ctrl._rollout_manager = SimpleNamespace(set_weight_version=MagicMock())
    ctrl._trainer_version = 3

    asyncio.run(ctrl._sync_weights())

    ctrl._weight_synchronizer.sync_weights.assert_called_once_with(kv_scales=None)
    assert ctrl._gen.invalidate_kv_cache.call_count == expected_invalidation_calls
    ctrl._rollout_manager.set_weight_version.assert_called_once_with(3)
    assert ctrl._rollout_permitted.is_set()


def test_sync_weights_calibrates_and_forwards_fp8_kv_scales() -> None:
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._async_cfg = AsyncRLConfig()
    ctrl._rollout_permitted = asyncio.Event()
    ctrl._rollout_permitted.set()
    ctrl._weight_synchronizer = SimpleNamespace(sync_weights=MagicMock())
    ctrl._gen = SimpleNamespace(
        invalidate_kv_cache=MagicMock(),
        requires_kv_scale_sync=True,
    )
    ctrl._trainer = SimpleNamespace(
        calibrate_qkv_fp8_scales=MagicMock(return_value={"layers": {"layer.0": 0.5}})
    )
    ctrl._rollout_manager = SimpleNamespace(set_weight_version=MagicMock())
    ctrl._trainer_version = 3
    calibration_data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2]]),
            "input_lengths": torch.tensor([2]),
        }
    )

    asyncio.run(ctrl._sync_weights(calibration_data=calibration_data))

    ctrl._trainer.calibrate_qkv_fp8_scales.assert_called_once_with(
        calibration_data,
        include_q=True,
    )
    ctrl._weight_synchronizer.sync_weights.assert_called_once_with(
        kv_scales={"layer.0": 0.5}
    )


class _EmptySampler:
    async def evict(self, *, current_train_weight: int) -> int:
        del current_train_weight
        return 0

    async def select(self, **kwargs):
        del kwargs
        return None, 0


class _OneThenEmptySampler(_EmptySampler):
    def __init__(self, meta: KVBatchMeta) -> None:
        self._meta: KVBatchMeta | None = meta

    async def select(self, **kwargs):
        del kwargs
        if self._meta is None:
            return None, 0
        meta = self._meta
        self._meta = None
        return meta, 1


class _FiniteSampler(_EmptySampler):
    def __init__(self, metas: list[KVBatchMeta]) -> None:
        self._metas = iter(metas)

    async def select(self, **kwargs):
        del kwargs
        try:
            return next(self._metas), 1
        except StopIteration:
            return None, 0


class _EmptyBuffer:
    def __len__(self) -> int:
        return 0


class _ResidualBuffer:
    def __init__(self, size: int) -> None:
        self.size = size
        self.remove_calls: list[dict[str, object]] = []

    def __len__(self) -> int:
        return self.size

    async def remove(
        self, idxs: list[int], remove_in_dp: bool, **kwargs: object
    ) -> int:
        self.remove_calls.append({"idxs": idxs, "remove_in_dp": remove_in_dp, **kwargs})
        removed = self.size
        self.size = 0
        return removed


class _NoOpTrainer:
    def prepare_for_lp_inference(self) -> None:
        pass

    def prepare_for_training(self) -> None:
        pass

    def begin_train_step(self, loss_fn) -> None:
        del loss_fn

    def train_microbatches_from_meta(self, meta: KVBatchMeta) -> None:
        del meta

    def finish_train_step(self) -> dict[str, float]:
        return {}


class _NoOpDataPlane:
    def clear_samples(self, **kwargs) -> None:
        del kwargs


def _train_pump_controller(*, sampler) -> object:
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._master_config = SimpleNamespace(
        grpo=GRPOConfig.model_construct(
            num_prompts_per_step=2,
            max_num_steps=1,
        )
    )
    ctrl._async_cfg = SimpleNamespace(min_groups_for_streaming_train=1)
    ctrl._advantage_cfg = AdvantageConfig()
    ctrl._policy_logprobs_required = False
    ctrl._reference_logprobs_required = False
    ctrl._advantage_estimator = None
    ctrl._partition_id = "rollout_data"
    ctrl._sampler = sampler
    ctrl._buffer = _EmptyBuffer()
    ctrl._buffer_capacity = asyncio.Semaphore(2)
    ctrl._rollout_exhausted = asyncio.Event()
    ctrl._rollout_exhausted.set()
    ctrl._trainer = _NoOpTrainer()
    ctrl._gen = SimpleNamespace(requires_kv_scale_sync=False)
    ctrl._loss_fn = None
    ctrl._dp_client = _NoOpDataPlane()
    ctrl._timer = Timer()
    ctrl._trainer_version = 0
    ctrl._train_steps = 0
    ctrl._lifecycle_recorder = None
    ctrl._opportunity_recorder = None
    ctrl._sync_weights = AsyncMock()
    ctrl._logger = MagicMock()
    ctrl._step_log_dict = {
        "rewards": [],
        "masked_advantages": [],
        "sequence_lengths": [],
    }
    return ctrl


def test_bounded_run_cleanup_removes_all_residual_buffer_groups() -> None:
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._buffer = _ResidualBuffer(3)
    ctrl._trainer_version = 48

    asyncio.run(
        ctrl._cancel_residual_buffer_groups(
            reason=single_controller.RolloutRemovalReason.BOUNDED_SHUTDOWN
        )
    )

    assert len(ctrl._buffer) == 0
    assert ctrl._buffer.remove_calls == [
        {
            "idxs": [0, 1, 2],
            "remove_in_dp": True,
            "reason": single_controller.RolloutRemovalReason.BOUNDED_SHUTDOWN,
            "learner_weight_version": 48,
        }
    ]


def test_successful_train_step_advances_audited_learner_version() -> None:
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._trainer_version = 4
    ctrl._lifecycle_recorder = MagicMock()
    ctrl._opportunity_recorder = None
    ctrl._rollout_manager = MagicMock()

    ctrl._advance_trainer_version()

    assert ctrl._trainer_version == 5
    ctrl._lifecycle_recorder.record_learner_version_advanced.assert_called_once_with(
        previous_version=4,
        learner_weight_version=5,
    )


def _run_lifecycle_controller(
    *, train_steps: int = 128, trainer_version: int = 128
) -> object:
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._sync_weights = AsyncMock()
    ctrl._rollout_pump = AsyncMock()
    ctrl._train_pump = AsyncMock()
    ctrl._cancel_residual_buffer_groups = AsyncMock()
    ctrl._lifecycle_recorder = MagicMock()
    ctrl._opportunity_recorder = None
    ctrl._rollout_manager = MagicMock()
    ctrl._master_config = SimpleNamespace(grpo=SimpleNamespace(max_num_steps=128))
    ctrl._async_cfg = SimpleNamespace(
        lifecycle_audit_path="audit.jsonl",
        controlled_release_delay=SimpleNamespace(enabled=True),
        gradient_opportunity_audit=SimpleNamespace(output_path=None),
    )
    ctrl._logger = MagicMock()
    ctrl._train_steps = train_steps
    ctrl._trainer_version = trainer_version
    return ctrl


def test_run_uses_bounded_shutdown_only_at_configured_boundary() -> None:
    ctrl = _run_lifecycle_controller()

    result = asyncio.run(ctrl.run())

    assert result == {"train_steps": 128, "trainer_version": 128}
    ctrl._cancel_residual_buffer_groups.assert_awaited_once_with(
        reason=single_controller.RolloutRemovalReason.BOUNDED_SHUTDOWN
    )


@pytest.mark.parametrize(
    ("train_steps", "trainer_version"),
    [(127, 127), (128, 127), (127, 128)],
)
def test_controlled_run_rejects_premature_or_mismatched_boundary(
    train_steps: int, trainer_version: int
) -> None:
    ctrl = _run_lifecycle_controller(
        train_steps=train_steps, trainer_version=trainer_version
    )

    with pytest.raises(RuntimeError, match="ended before the configured"):
        asyncio.run(ctrl.run())

    ctrl._cancel_residual_buffer_groups.assert_awaited_once_with(
        reason=single_controller.RolloutRemovalReason.CANCELLED
    )


def test_disabled_run_preserves_early_completion_behavior() -> None:
    ctrl = _run_lifecycle_controller(train_steps=127, trainer_version=127)
    ctrl._async_cfg.controlled_release_delay.enabled = False

    result = asyncio.run(ctrl.run())

    assert result == {"train_steps": 127, "trainer_version": 127}
    ctrl._cancel_residual_buffer_groups.assert_awaited_once_with(
        reason=single_controller.RolloutRemovalReason.CANCELLED
    )


def test_disabled_run_preserves_exact_boundary_cleanup_reason() -> None:
    ctrl = _run_lifecycle_controller()
    ctrl._async_cfg.controlled_release_delay.enabled = False

    result = asyncio.run(ctrl.run())

    assert result == {"train_steps": 128, "trainer_version": 128}
    ctrl._cancel_residual_buffer_groups.assert_awaited_once_with(
        reason=single_controller.RolloutRemovalReason.CANCELLED
    )


def test_run_failure_retains_cancelled_cleanup_reason() -> None:
    ctrl = _run_lifecycle_controller()
    ctrl._train_pump = AsyncMock(side_effect=RuntimeError("train failed"))

    with pytest.raises(RuntimeError, match="train failed"):
        asyncio.run(ctrl.run())

    ctrl._cancel_residual_buffer_groups.assert_awaited_once_with(
        reason=single_controller.RolloutRemovalReason.CANCELLED
    )


def test_rollout_failure_retains_cancelled_cleanup_reason() -> None:
    ctrl = _run_lifecycle_controller()
    ctrl._rollout_pump = AsyncMock(side_effect=RuntimeError("rollout failed"))

    with pytest.raises(RuntimeError, match="rollout failed"):
        asyncio.run(ctrl.run())

    ctrl._cancel_residual_buffer_groups.assert_awaited_once_with(
        reason=single_controller.RolloutRemovalReason.CANCELLED
    )


def test_external_run_cancellation_retains_cancelled_cleanup_reason() -> None:
    ctrl = _run_lifecycle_controller()

    async def cancel_running_controller() -> None:
        pump_started = asyncio.Event()
        pump_block = asyncio.Event()

        async def blocked_pump() -> None:
            pump_started.set()
            await pump_block.wait()

        ctrl._rollout_pump = blocked_pump
        ctrl._train_pump = blocked_pump
        run_task = asyncio.create_task(ctrl.run())
        await pump_started.wait()
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task

    asyncio.run(cancel_running_controller())

    ctrl._cancel_residual_buffer_groups.assert_awaited_once_with(
        reason=single_controller.RolloutRemovalReason.CANCELLED
    )


def test_run_flushes_audit_when_residual_cleanup_fails() -> None:
    ctrl = _run_lifecycle_controller()
    ctrl._cancel_residual_buffer_groups = AsyncMock(
        side_effect=RuntimeError("cleanup failed")
    )

    with pytest.raises(RuntimeError, match="cleanup failed"):
        asyncio.run(ctrl.run())

    ctrl._cancel_residual_buffer_groups.assert_awaited_once_with(
        reason=single_controller.RolloutRemovalReason.BOUNDED_SHUTDOWN
    )
    ctrl._lifecycle_recorder.flush_jsonl.assert_called_once_with("audit.jsonl")
    ctrl._logger.finish.assert_called_once_with()


def test_run_finishes_logger_when_audit_flush_fails() -> None:
    ctrl = _run_lifecycle_controller()
    ctrl._lifecycle_recorder.flush_jsonl.side_effect = RuntimeError("flush failed")

    with pytest.raises(RuntimeError, match="flush failed"):
        asyncio.run(ctrl.run())

    ctrl._logger.finish.assert_called_once_with()


def test_run_atomically_flushes_opportunity_ledger() -> None:
    ctrl = _run_lifecycle_controller()
    ctrl._opportunity_recorder = MagicMock()
    ctrl._async_cfg.gradient_opportunity_audit.output_path = "opportunity.jsonl"

    asyncio.run(ctrl.run())

    ctrl._opportunity_recorder.flush_jsonl.assert_called_once_with("opportunity.jsonl")
    ctrl._lifecycle_recorder.flush_jsonl.assert_called_once_with("audit.jsonl")


def test_train_pump_stops_after_rollout_exhaustion_and_buffer_drain() -> None:
    ctrl = _train_pump_controller(sampler=_EmptySampler())

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    assert ctrl._train_steps == 0


def test_train_pump_fails_if_rollout_exhausts_during_partial_step() -> None:
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["sample-0"],
        fields=[],
        sequence_lengths=[1],
        tags=[{"weight_version": 0}],
    )
    ctrl = _train_pump_controller(sampler=_OneThenEmptySampler(meta))

    with pytest.raises(
        RuntimeError,
        match=(
            r"rollout exhausted before a complete training step was assembled: "
            r"dispatched 1/2 prompt groups"
        ),
    ):
        asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))


def _training_meta(sample_id: str) -> KVBatchMeta:
    return KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=[sample_id],
        fields=[],
        sequence_lengths=[1],
        tags=[{"weight_version": 0}],
    )


def test_completed_step_ledger_is_emitted_after_version_advance(monkeypatch) -> None:
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})
    ctrl = _train_pump_controller(
        sampler=_FiniteSampler([_training_meta("sample-0"), _training_meta("sample-1")])
    )
    ctrl._opportunity_recorder = MagicMock()

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    ctrl._opportunity_recorder.append_train_step_completed.assert_called_once_with(
        previous_learner_version=0,
        learner_version=1,
        sample_ids=["sample-0", "sample-1"],
    )
    assert ctrl._train_steps == 1


@pytest.mark.parametrize(
    "failure_stage",
    [
        "begin_train_step",
        "train_microbatches_from_meta",
        "finish_train_step",
        "advance",
    ],
)
def test_completed_step_ledger_is_not_emitted_for_failed_step(
    failure_stage: str,
) -> None:
    ctrl = _train_pump_controller(
        sampler=_FiniteSampler([_training_meta("sample-0"), _training_meta("sample-1")])
    )
    ctrl._opportunity_recorder = MagicMock()
    failure = RuntimeError(f"injected {failure_stage} failure")
    if failure_stage == "advance":
        ctrl._advance_trainer_version = MagicMock(side_effect=failure)
    else:
        setattr(ctrl._trainer, failure_stage, MagicMock(side_effect=failure))

    with pytest.raises(RuntimeError, match=f"injected {failure_stage} failure"):
        asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    ctrl._opportunity_recorder.append_train_step_completed.assert_not_called()
