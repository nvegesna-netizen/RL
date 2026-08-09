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

"""Unit tests for TQReplayBuffer (plain SC-process buffer + TQ proxy)."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest
import torch

import nemo_rl.algorithms.async_utils.replay_buffer as _replay_buffer_module
from nemo_rl.algorithms.async_utils.replay_buffer import TQReplayBuffer
from nemo_rl.algorithms.async_utils.rollout_lifecycle import (
    RolloutLifecycleRecorder,
    RolloutLifecycleStage,
    RolloutRemovalReason,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.interfaces import PromptGroupRecord

# Each record yields _N_GENS training rows.
_N_GENS = 2


def _stub_record_to_train_batch(
    record: PromptGroupRecord, *, pad_value_dict: Any
) -> BatchedDataDict[Any]:
    del record, pad_value_dict
    return BatchedDataDict[Any](
        {
            "input_ids": torch.ones((_N_GENS, 3), dtype=torch.long),
            "input_lengths": torch.full((_N_GENS,), 3, dtype=torch.long),
            "total_reward": torch.zeros(_N_GENS, dtype=torch.float32),
        }
    )


@pytest.fixture(autouse=True)
def _patch_converter(monkeypatch):
    """Bypass the real ``record_to_train_batch`` so tests can use empty records."""
    monkeypatch.setattr(
        _replay_buffer_module,
        "record_to_train_batch",
        _stub_record_to_train_batch,
    )


class FakeDataPlaneClient:
    """Sync in-memory DataPlaneClient stub used by TQReplayBuffer tests."""

    def __init__(self, partition_id: str = "rollout_data") -> None:
        self._partition_id = partition_id
        self._rows: dict[str, dict[str, Any]] = {}
        self.put_calls: list[dict[str, Any]] = []
        self.clear_calls: list[list[str]] = []

    def put_samples(
        self,
        sample_ids: list[str],
        partition_id: str,
        fields: Any = None,
        tags: list[dict[str, Any]] | None = None,
    ) -> KVBatchMeta:
        assert partition_id == self._partition_id
        self.put_calls.append(
            {
                "sample_ids": list(sample_ids),
                "fields": fields,
                "tags": [dict(t) for t in tags] if tags is not None else None,
            }
        )
        for i, sid in enumerate(sample_ids):
            self._rows[sid] = {
                "tag": dict(tags[i]) if tags is not None else {},
            }
        return KVBatchMeta(
            partition_id=partition_id,
            task_name=None,
            sample_ids=list(sample_ids),
            fields=None,
            tags=[dict(t) for t in tags] if tags is not None else None,
        )

    def clear_samples(self, sample_ids: list[str] | None, partition_id: str) -> None:
        assert partition_id == self._partition_id
        ids = list(sample_ids) if sample_ids is not None else list(self._rows)
        self.clear_calls.append(list(ids))
        for sid in ids:
            self._rows.pop(sid, None)

    def depth(self) -> int:
        return len(self._rows)


class FailAfterPutDataPlaneClient(FakeDataPlaneClient):
    """Write all rows, then fail to simulate a partial-success RPC."""

    def put_samples(
        self,
        sample_ids: list[str],
        partition_id: str,
        fields: Any = None,
        tags: list[dict[str, Any]] | None = None,
    ) -> KVBatchMeta:
        super().put_samples(sample_ids, partition_id, fields, tags)
        raise RuntimeError("injected put failure")


class FailPutAndClearDataPlaneClient(FailAfterPutDataPlaneClient):
    def clear_samples(self, sample_ids: list[str] | None, partition_id: str) -> None:
        del sample_ids, partition_id
        raise RuntimeError("injected clear failure")


class FailOnReadyRecorder(RolloutLifecycleRecorder):
    def record(self, *, stage: RolloutLifecycleStage, **kwargs: Any):
        if stage is RolloutLifecycleStage.GROUP_READY:
            raise RuntimeError("injected lifecycle failure")
        return super().record(stage=stage, **kwargs)


def _run(coro):
    return asyncio.run(coro)


def _make_record() -> PromptGroupRecord:
    """Opaque PromptGroupRecord — converter is stubbed, so contents are unused."""
    return PromptGroupRecord(
        prompt_idx=0,
        prompt=[],
        extra_env_info=None,
        metadata={},
        completions=[],
        rollout_metrics={},
    )


def _make_buffer(
    dp: FakeDataPlaneClient,
    *,
    require_routed_experts: bool = False,
    lifecycle_recorder: RolloutLifecycleRecorder | None = None,
) -> TQReplayBuffer:
    return TQReplayBuffer(
        dp,
        partition_id="rollout_data",
        pad_value_dict={"token_ids": 0},
        require_routed_experts=require_routed_experts,
        lifecycle_recorder=lifecycle_recorder,
    )


def _add_group(
    buf: TQReplayBuffer, weight: int, end_weight: int | None = None
) -> KVBatchMeta:
    if end_weight is None:
        end_weight = weight
    group_id = buf.reserve(weight_version=weight)
    return _run(
        buf.commit(
            group_id,
            _make_record(),
            start_weight_version=weight,
            end_weight_version=end_weight,
        )
    )


class TestTQReplayBufferReserveCommit:
    def test_prepare_tensorizes_and_packs_exactly_once(self, monkeypatch):
        calls = {"tensorize": 0, "pack": 0}
        original_pack = _replay_buffer_module.pack_payload

        def _count_tensorize(*args, **kwargs):
            calls["tensorize"] += 1
            return _stub_record_to_train_batch(*args, **kwargs)

        def _count_pack(*args, **kwargs):
            calls["pack"] += 1
            return original_pack(*args, **kwargs)

        monkeypatch.setattr(
            _replay_buffer_module, "record_to_train_batch", _count_tensorize
        )
        monkeypatch.setattr(_replay_buffer_module, "pack_payload", _count_pack)
        buf = _make_buffer(FakeDataPlaneClient())
        group_id = buf.reserve(weight_version=3)

        prepared = buf.prepare_commit(group_id, _make_record(), start_weight_version=3)
        _run(buf.commit_prepared(prepared, end_weight_version=4))

        assert calls == {"tensorize": 1, "pack": 1}

    def test_reserve_rejects_duplicate_live_group_id(self):
        buf = _make_buffer(FakeDataPlaneClient())
        buf.reserve(weight_version=1, group_id="fixed")

        with pytest.raises(ValueError, match="duplicate live group_id"):
            buf.reserve(weight_version=1, group_id="fixed")

    def test_prepare_observer_runs_before_single_use_commit(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        observed: list[tuple[str, tuple[str, ...], int]] = []
        buf.set_prepare_observer(
            lambda **kwargs: observed.append(
                (
                    kwargs["group_id"],
                    kwargs["sample_ids"],
                    kwargs["start_weight_version"],
                )
            )
        )
        group_id = buf.reserve(weight_version=3)

        prepared = buf.prepare_commit(
            group_id,
            _make_record(),
            start_weight_version=3,
        )

        assert observed == [(group_id, prepared.sample_ids, 3)]
        assert dp.put_calls == []
        meta = _run(buf.commit_prepared(prepared, end_weight_version=4))
        assert meta.sample_ids == list(prepared.sample_ids)
        with pytest.raises(ValueError, match="already consumed"):
            _run(buf.commit_prepared(prepared, end_weight_version=4))

    def test_prepared_payload_is_byte_equal_at_put_after_async_hold(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        observer_snapshots: list[dict[str, torch.Tensor]] = []
        buf.set_prepare_observer(
            lambda **kwargs: observer_snapshots.append(
                {
                    name: value.detach().clone()
                    for name, value in kwargs["train_batch"].items()
                    if isinstance(value, torch.Tensor)
                }
            )
        )
        group_id = buf.reserve(weight_version=3)
        prepared = buf.prepare_commit(group_id, _make_record(), start_weight_version=3)
        packed_before_hold = {
            name: value.detach().clone() for name, value in prepared.fields.items()
        }
        tags_before_hold = tuple(dict(tag) for tag in prepared.tags)

        async def _hold_then_commit() -> None:
            await asyncio.sleep(0)
            await buf.commit_prepared(prepared, end_weight_version=4)

        _run(_hold_then_commit())

        assert len(observer_snapshots) == 1
        assert dp.put_calls[0]["fields"] is prepared.fields
        assert tuple(dp.put_calls[0]["tags"]) == tags_before_hold
        assert set(dp.put_calls[0]["fields"].keys()) == set(packed_before_hold)

        def _row_bytes(value: torch.Tensor) -> tuple[bytes, ...]:
            rows = value.unbind() if value.is_nested else (value,)
            return tuple(
                row.detach().cpu().contiguous().numpy().tobytes() for row in rows
            )

        for name, expected in packed_before_hold.items():
            actual = dp.put_calls[0]["fields"][name]
            assert actual.dtype == expected.dtype
            assert actual.shape == expected.shape
            assert _row_bytes(actual) == _row_bytes(expected)

    def test_prepare_rejects_unknown_ready_and_version_mismatch(self):
        buf = _make_buffer(FakeDataPlaneClient())
        group_id = buf.reserve(weight_version=3)

        with pytest.raises(ValueError, match="unknown group_id"):
            buf.prepare_commit("missing", _make_record(), start_weight_version=3)
        with pytest.raises(ValueError, match="start version mismatch"):
            buf.prepare_commit(group_id, _make_record(), start_weight_version=2)
        prepared = buf.prepare_commit(group_id, _make_record(), start_weight_version=3)
        _run(buf.commit_prepared(prepared, end_weight_version=3))
        with pytest.raises(ValueError, match="already ready"):
            buf.prepare_commit(group_id, _make_record(), start_weight_version=3)

    def test_prepared_payload_rejects_wrong_owner_and_mutation(self):
        first = _make_buffer(FakeDataPlaneClient())
        second = _make_buffer(FakeDataPlaneClient())
        group_id = first.reserve(weight_version=3)
        prepared = first.prepare_commit(
            group_id, _make_record(), start_weight_version=3
        )

        with pytest.raises(ValueError, match="another replay buffer"):
            _run(second.commit_prepared(prepared, end_weight_version=3))

        first_tensor = next(iter(prepared.fields.values()))
        first_tensor.add_(1)
        with pytest.raises(ValueError, match="mutated"):
            _run(first.commit_prepared(prepared, end_weight_version=3))

    def test_prepared_payload_rejects_partition_group_and_stale_capability(self):
        buf = _make_buffer(FakeDataPlaneClient())
        group_id = buf.reserve(weight_version=3)
        prepared = buf.prepare_commit(group_id, _make_record(), start_weight_version=3)

        with pytest.raises(ValueError, match="partition"):
            _run(
                buf.commit_prepared(
                    replace(prepared, partition_id="other"), end_weight_version=3
                )
            )
        with pytest.raises(ValueError, match="stale, unknown"):
            _run(
                buf.commit_prepared(
                    replace(prepared, group_id="other"), end_weight_version=3
                )
            )

        _run(
            buf.remove(
                [0],
                remove_in_dp=True,
                reason=RolloutRemovalReason.FAILED,
                learner_weight_version=3,
            )
        )
        with pytest.raises(ValueError, match="stale, unknown"):
            _run(buf.commit_prepared(prepared, end_weight_version=3))

    def test_failed_prepared_commit_restores_slot_and_cannot_retry(self):
        dp = FailAfterPutDataPlaneClient()
        buf = _make_buffer(dp)
        group_id = buf.reserve(weight_version=3)
        prepared = buf.prepare_commit(group_id, _make_record(), start_weight_version=3)

        with pytest.raises(RuntimeError, match="injected put failure"):
            _run(buf.commit_prepared(prepared, end_weight_version=4))

        assert buf.meta_list == [None]
        assert buf.end_weight_list == [-1]
        assert buf.ready_list == [False]
        assert dp.depth() == 0
        with pytest.raises(ValueError, match="already consumed"):
            _run(buf.commit_prepared(prepared, end_weight_version=4))

    def test_lifecycle_failure_rolls_back_but_preserves_prepared_observation(self):
        observed: list[str] = []
        dp = FakeDataPlaneClient()
        recorder = FailOnReadyRecorder(clock_ns=lambda: 0)
        buf = _make_buffer(dp, lifecycle_recorder=recorder)
        buf.set_prepare_observer(lambda **kwargs: observed.append(kwargs["group_id"]))
        group_id = buf.reserve(weight_version=3)
        prepared = buf.prepare_commit(group_id, _make_record(), start_weight_version=3)

        with pytest.raises(RuntimeError, match="injected lifecycle failure"):
            _run(buf.commit_prepared(prepared, end_weight_version=4))

        assert observed == [group_id]
        assert dp.depth() == 0
        assert buf.meta_list == [None]
        assert buf.end_weight_list == [-1]
        assert buf.ready_list == [False]
        assert all(
            event.stage is not RolloutLifecycleStage.GROUP_READY
            for event in recorder.snapshot()
        )

    def test_commit_and_rollback_failure_restores_local_slot_and_groups_errors(self):
        dp = FailPutAndClearDataPlaneClient()
        buf = _make_buffer(dp)
        group_id = buf.reserve(weight_version=3)
        prepared = buf.prepare_commit(group_id, _make_record(), start_weight_version=3)

        with pytest.raises(BaseExceptionGroup) as error:
            _run(buf.commit_prepared(prepared, end_weight_version=4))

        assert [str(exc) for exc in error.value.exceptions] == [
            "injected put failure",
            "injected clear failure",
        ]
        assert buf.meta_list == [None]
        assert buf.end_weight_list == [-1]
        assert buf.ready_list == [False]

    def test_lifecycle_records_reserve_ready_and_selected(self):
        timestamps = iter((10, 20, 30))
        recorder = RolloutLifecycleRecorder(
            run_id="run",
            clock_domain_id="controller",
            clock_ns=lambda: next(timestamps),
        )
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp, lifecycle_recorder=recorder)

        group_id = buf.reserve(weight_version=3, target_step=5)
        meta = _run(
            buf.commit(
                group_id,
                _make_record(),
                start_weight_version=3,
                end_weight_version=4,
            )
        )
        _run(
            buf.remove(
                [0],
                remove_in_dp=False,
                reason=RolloutRemovalReason.SELECTED,
                learner_weight_version=5,
            )
        )

        events = recorder.snapshot()
        assert [event.stage for event in events] == [
            RolloutLifecycleStage.RESERVED,
            RolloutLifecycleStage.GROUP_READY,
            RolloutLifecycleStage.REMOVED,
        ]
        assert events[1].mixed_generation_versions is True
        assert events[1].sample_ids == tuple(meta.sample_ids)
        assert events[2].removal_reason is RolloutRemovalReason.SELECTED
        assert events[2].learner_weight_version == 5

    def test_commit_clears_rows_when_put_raises_after_writing(self):
        dp = FailAfterPutDataPlaneClient()
        buf = _make_buffer(dp)
        group_id = buf.reserve(weight_version=3)

        with pytest.raises(RuntimeError, match="injected put failure"):
            _run(
                buf.commit(
                    group_id,
                    _make_record(),
                    start_weight_version=3,
                    end_weight_version=3,
                )
            )

        assert dp.depth() == 0
        assert dp.clear_calls == [dp.put_calls[0]["sample_ids"]]
        # commit() rolls back DataPlane rows; generate_and_push() owns removal
        # of the reserved buffer slot.
        assert buf.size() == 1
        assert buf.ready_list == [False]
        assert buf.meta_list == [None]

    def test_reserve_appends_placeholder_unready(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)

        group_id = buf.reserve(weight_version=3)

        assert isinstance(group_id, str) and group_id
        assert buf.size() == 1
        assert buf.start_weight_list == [3]
        assert buf.end_weight_list == [-1]
        assert buf.ready_list == [False]
        assert buf.meta_list == [None]
        assert dp.depth() == 0
        assert dp.put_calls == []

    def test_commit_writes_tq_then_fills_meta(self, monkeypatch):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        trace_calls = []
        monkeypatch.setattr(
            _replay_buffer_module,
            "trace_rollout_payload",
            lambda **kwargs: trace_calls.append(kwargs),
        )

        group_id = buf.reserve(weight_version=3)
        meta = _run(
            buf.commit(
                group_id,
                _make_record(),
                start_weight_version=3,
                end_weight_version=4,
            )
        )

        # pack_payload stamps sample_ids as ``{group_uuid}_g{i}``.
        assert len(meta.sample_ids) == _N_GENS
        head, _, idx = meta.sample_ids[0].rpartition("_g")
        assert head == group_id and idx == "0"
        assert all(sid.startswith(group_id + "_g") for sid in meta.sample_ids)
        assert dp.depth() == _N_GENS
        assert buf.size() == 1
        assert buf.start_weight_list == [3]
        assert buf.end_weight_list == [4]
        assert buf.ready_list == [True]
        assert buf.meta_list[0].sample_ids == meta.sample_ids
        # TQ tag uses start_weight_version (dispatch time).
        assert meta.tags == [{"weight_version": 3}] * _N_GENS
        assert len(dp.put_calls) == 1
        assert len(trace_calls) == 1
        assert trace_calls[0]["keys"] == meta.sample_ids
        assert trace_calls[0]["data"]["input_lengths"].tolist() == [3, 3]

    def test_commit_requires_routed_experts_before_tq_write(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp, require_routed_experts=True)
        group_id = buf.reserve(weight_version=3)

        with pytest.raises(
            RuntimeError,
            match="router_replay.enabled=true requires routed_experts",
        ):
            _run(
                buf.commit(
                    group_id,
                    _make_record(),
                    start_weight_version=3,
                    end_weight_version=3,
                )
            )

        assert dp.put_calls == []
        assert dp.depth() == 0
        assert buf.ready_list == [False]

    def test_commit_raises_for_unknown_group_id(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        buf.reserve(weight_version=3)

        with pytest.raises(ValueError):
            _run(
                buf.commit(
                    "not-a-real-id",
                    _make_record(),
                    start_weight_version=3,
                    end_weight_version=3,
                )
            )

        # No orphan rows in DataPlane: commit must validate group_id before writing.
        assert dp.depth() == 0
        assert dp.put_calls == []

    def test_reserve_then_commit_preserves_dispatch_order(self):
        """Reserve in dispatch order, commit out of order; insertion order holds."""
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)

        weights = (1, 2, 3)
        gids = [buf.reserve(weight_version=w) for w in weights]
        # Commit out of order: 2, 0, 1 — buffer order must still match reserve order.
        for i in (2, 0, 1):
            _run(
                buf.commit(
                    gids[i],
                    _make_record(),
                    start_weight_version=weights[i],
                    end_weight_version=weights[i],
                )
            )

        assert buf.size() == 3
        assert buf.start_weight_list == [1, 2, 3]
        assert buf.end_weight_list == [1, 2, 3]
        assert buf.ready_list == [True, True, True]
        # sample_id head equals reserved group_id at each slot.
        for i, gid in enumerate(gids):
            assert buf.meta_list[i] is not None
            assert buf.meta_list[i].sample_ids[0].startswith(gid + "_g")

    def test_commit_appends_multiple_records_in_order(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)

        metas = [_add_group(buf, weight=w) for w in (1, 2, 3)]

        assert buf.size() == 3
        assert buf.start_weight_list == [1, 2, 3]
        assert buf.end_weight_list == [1, 2, 3]
        assert [m.sample_ids for m in buf.meta_list] == [
            list(metas[0].sample_ids),
            list(metas[1].sample_ids),
            list(metas[2].sample_ids),
        ]


class TestTQReplayBufferRemove:
    def test_remove_drops_indices_and_clears_dp_when_requested(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        metas = [_add_group(buf, weight=g) for g in range(3)]

        n = _run(buf.remove([0, 2], remove_in_dp=True))

        assert n == 2
        assert buf.size() == 1
        assert buf.start_weight_list == [1]
        assert buf.end_weight_list == [1]
        assert buf.meta_list[0].sample_ids == list(metas[1].sample_ids)
        assert dp.depth() == _N_GENS
        assert set(dp._rows) == set(metas[1].sample_ids)

    def test_remove_without_dp_keeps_rows(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        metas = [_add_group(buf, weight=g) for g in range(2)]

        n = _run(buf.remove([0], remove_in_dp=False))

        assert n == 1
        assert buf.size() == 1
        assert buf.start_weight_list == [1]
        assert buf.end_weight_list == [1]
        assert buf.meta_list[0].sample_ids == list(metas[1].sample_ids)
        assert dp.clear_calls == []
        assert dp.depth() == 2 * _N_GENS

    def test_remove_rejects_out_of_range_before_mutating(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        metas = [_add_group(buf, weight=g) for g in range(2)]

        with pytest.raises(IndexError, match=r"out of range: 5; size=2"):
            _run(buf.remove([0, 5], remove_in_dp=True))

        assert buf.size() == 2
        assert [m.sample_ids for m in buf.meta_list] == [
            list(metas[0].sample_ids),
            list(metas[1].sample_ids),
        ]
        assert dp.depth() == 2 * _N_GENS
        assert dp.clear_calls == []

    def test_remove_empty_is_noop(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        _add_group(buf, weight=0)
        _add_group(buf, weight=0)

        n = _run(buf.remove([], remove_in_dp=True))

        assert n == 0
        assert buf.size() == 2
        assert dp.depth() == 2 * _N_GENS
        assert dp.clear_calls == []


class TestTQReplayBufferSize:
    def test_size_and_len(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        assert buf.size() == 0
        assert len(buf) == 0

        _add_group(buf, weight=0)
        assert buf.size() == 1
        assert len(buf) == 1

        _add_group(buf, weight=0)
        assert buf.size() == 2
        assert len(buf) == 2

        _run(buf.remove([0], remove_in_dp=True))
        assert buf.size() == 1
        assert len(buf) == 1
