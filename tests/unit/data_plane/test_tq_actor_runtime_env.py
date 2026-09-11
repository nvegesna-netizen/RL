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

from typing import cast

import pytest

pytest.importorskip("transfer_queue")

from nemo_rl.data_plane.adapters import transfer_queue as tq_adapter
from nemo_rl.data_plane.interfaces import DataPlaneConfig


def _config(mode: str) -> DataPlaneConfig:
    return cast(DataPlaneConfig, {"actor_runtime_env_mode": mode})


def test_pip_mode_injects_actor_runtime_environment(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        tq_adapter, "_patch_tq_actor_runtime_env", lambda: calls.append("patched")
    )

    tq_adapter._configure_tq_actor_runtime_env(_config("pip"))

    assert calls == ["patched"]


def test_baked_mode_accepts_exactly_one_live_node(monkeypatch) -> None:
    monkeypatch.setattr(
        tq_adapter.ray,
        "nodes",
        lambda: [{"Alive": True}, {"Alive": False}],
    )
    monkeypatch.setattr(
        tq_adapter,
        "_patch_tq_actor_runtime_env",
        lambda: pytest.fail("baked mode must not inject a pip runtime environment"),
    )

    tq_adapter._configure_tq_actor_runtime_env(_config("inherit_baked_single_node"))


@pytest.mark.parametrize("alive_node_count", [0, 2])
def test_baked_mode_rejects_non_single_node_ray_clusters(
    monkeypatch, alive_node_count: int
) -> None:
    monkeypatch.setattr(
        tq_adapter.ray,
        "nodes",
        lambda: [{"Alive": True} for _ in range(alive_node_count)],
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "inherit_baked_single_node requires exactly one live Ray node; "
            f"found {alive_node_count}"
        ),
    ):
        tq_adapter._configure_tq_actor_runtime_env(_config("inherit_baked_single_node"))


def test_unknown_actor_runtime_environment_mode_fails_loudly() -> None:
    with pytest.raises(
        ValueError, match="unknown TQ actor runtime environment mode: 'unknown'"
    ):
        tq_adapter._configure_tq_actor_runtime_env(_config("unknown"))
