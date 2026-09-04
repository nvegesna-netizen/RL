# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the controlled zero-update live scheduler assay."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from nemo_rl.algorithms.async_utils.scheduler_assay import (
    SchedulerAssayPlan,
    compute_scheduler_assay_plan_id,
    load_scheduler_assay_plan,
)
from nemo_rl.algorithms.async_utils.scheduler_trace import SchedulerEventType
from nemo_rl.algorithms.single_controller import SingleControllerActor


def _plan_record() -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": 1,
        "analysis_status": "controlled_zero_update_live_scheduler_assay",
        "calibration_only": True,
        "confirmatory_eligible": False,
        "natural_latency_claim_authorized": False,
        "replay_of_natural_workloads_authorized": False,
        "training_authorized": False,
        "analysis_code_commit": "a" * 40,
        "source_design_id": "ready_bias_v1",
        "pools": [
            {
                "order_seed": 42001,
                "pool_id": "186215e9b80685ce292408db438319ffb5cf4e281b35ce78e197c0b11d12a88a",
                "manifest_sha256": "340b2cb322e35ebbcbf6c25a320c846bd797765ddfae67be4b480623087a7294",
            },
            {
                "order_seed": 42002,
                "pool_id": "6825c7ba315a6887708192b46f80e9b88f447063a946eaadab9fb24a1c340cf9",
                "manifest_sha256": "bc0afd679cc0fe8445bbfa27dc1484eabe2c522c12d0a58f8400f875e36d0817",
            },
            {
                "order_seed": 42003,
                "pool_id": "3cb47d86319e7dae1a8bd00a220ecea912870b80550ea0e4384dfece51c01414",
                "manifest_sha256": "4f0b14cb21963341991470241442cb7ed68fe17288f4348ba9d5c11fd5e0ef42",
            },
        ],
        "arms": [
            {
                "arm_id": "ready_first_aime_delayed",
                "sampler": "ready_first",
                "delayed_task": "AIME2024",
            },
            {
                "arm_id": "ready_first_gsm8k_delayed",
                "sampler": "ready_first",
                "delayed_task": "gsm8k",
            },
            {
                "arm_id": "in_order_aime_delayed",
                "sampler": "in_order",
                "delayed_task": "AIME2024",
            },
            {
                "arm_id": "in_order_gsm8k_delayed",
                "sampler": "in_order",
                "delayed_task": "gsm8k",
            },
        ],
        "prompt_groups": 48,
        "dispatch_cohorts": 12,
        "groups_per_cohort": 4,
        "groups_per_task_per_cohort": 2,
        "completions_per_group": 2,
        "max_inflight_prompts": 4,
        "max_buffered_rollouts": 16,
        "sampler_lookahead_versions": 3,
        "release_delay_seconds": 30.0,
        "primary_horizon": 8,
        "replication_order": [42001, 42002, 42003],
        "thresholds": {
            "minimum_delayed_hold_seconds": 29.5,
            "maximum_undelayed_commit_seconds": 5.0,
            "ready_first_undelayed_share_min": 0.875,
            "in_order_undelayed_share": 0.5,
            "ready_first_minus_in_order_min": 0.375,
        },
        "plan_id": "0" * 64,
    }
    draft = SchedulerAssayPlan.model_construct(**record)
    record["plan_id"] = compute_scheduler_assay_plan_id(draft)
    return record


def test_load_scheduler_assay_plan_verifies_canonical_identity(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_plan_record()))

    plan = load_scheduler_assay_plan(plan_path)

    assert plan.plan_id == compute_scheduler_assay_plan_id(plan)
    assert plan.pool(42001).order_seed == 42001
    assert plan.arm("ready_first_aime_delayed").sampler == "ready_first"


def test_scheduler_assay_pump_advances_only_logical_clock() -> None:
    class _Buffer:
        def __init__(self) -> None:
            self.group_ids = [f"group-{index}" for index in range(8)]
            self.ready_list = [True] * 8

        def __len__(self) -> int:
            return len(self.group_ids)

    class _Sampler:
        dispatch_index = 1

        def __init__(self, buffer: _Buffer) -> None:
            self.buffer = buffer
            self.selected: tuple[str, ...] = ()

        async def evict(self, *, current_train_weight: int) -> int:
            del current_train_weight
            return 0

        def take_last_evicted_group_ids(self) -> tuple[str, ...]:
            return ()

        def eligible_group_ids(self, *, current_train_weight: int) -> tuple[str, ...]:
            del current_train_weight
            return tuple(self.buffer.group_ids)

        async def select(
            self,
            *,
            current_train_weight: int,
            min_prompt_groups: int,
            max_prompt_groups: int,
        ) -> tuple[SimpleNamespace, int]:
            del current_train_weight
            assert min_prompt_groups == max_prompt_groups == 4
            self.selected = tuple(self.buffer.group_ids[:4])
            del self.buffer.group_ids[:4]
            del self.buffer.ready_list[:4]
            return SimpleNamespace(sample_ids=[f"sample-{i}" for i in range(8)]), 4

        def take_last_selected_group_ids(self) -> tuple[str, ...]:
            selected, self.selected = self.selected, ()
            return selected

    class _Trace:
        def __init__(self) -> None:
            self.events: list[tuple[SchedulerEventType, dict[str, object]]] = []

        def emit(self, event_type: SchedulerEventType, **fields: object) -> None:
            self.events.append((event_type, fields))

    async def _main() -> None:
        controller_cls = SingleControllerActor.__ray_metadata__.modified_class
        ctrl = object.__new__(controller_cls)
        ctrl._buffer = _Buffer()
        ctrl._sampler = _Sampler(ctrl._buffer)
        ctrl._scheduler_trace = _Trace()
        ctrl._fixed_pool_manifest = SimpleNamespace(items=tuple(range(8)))
        ctrl._scheduler_assay_enabled = True
        ctrl._trace_enabled = True
        ctrl._master_config = SimpleNamespace(
            grpo=SimpleNamespace(num_prompts_per_step=4)
        )
        ctrl._async_cfg = SimpleNamespace(sampler=SimpleNamespace(name="ready_first"))
        ctrl._sampler_fingerprint = "fingerprint"
        ctrl._partition_id = "partition"
        ctrl._buffer_capacity = asyncio.Semaphore(0)
        ctrl._rollout_exhausted = asyncio.Event()
        ctrl._trainer_version = 0
        ctrl._train_steps = 0
        ctrl._assay_scheduler_step = 0
        ctrl._assay_selection_steps = 0
        ctrl._assay_selected_groups = 0
        clear_calls: list[dict[str, object]] = []

        async def _call_dp(method_name: str, **kwargs: object) -> None:
            assert method_name == "clear_samples"
            clear_calls.append(kwargs)

        ctrl._call_dp = _call_dp
        await ctrl._scheduler_assay_pump()

        assert ctrl._trainer_version == 0
        assert ctrl._train_steps == 0
        assert ctrl._assay_scheduler_step == 2
        assert ctrl._assay_selection_steps == 2
        assert ctrl._assay_selected_groups == 8
        assert len(clear_calls) == 2
        assert [event for event, _ in ctrl._scheduler_trace.events] == [
            SchedulerEventType.SELECT_DECISION,
            SchedulerEventType.SELECT_DECISION,
        ]

    asyncio.run(_main())
