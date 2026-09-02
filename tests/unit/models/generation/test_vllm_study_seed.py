# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import pytest

from nemo_rl.models.generation.vllm.vllm_worker import resolve_vllm_engine_seed


def test_study_seed_offsets_topology_seed() -> None:
    assert resolve_vllm_engine_seed(3, 51001) == 51004
    assert resolve_vllm_engine_seed(None, 51001) == 51001
    assert resolve_vllm_engine_seed(3, None) == 3


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_study_seed_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        resolve_vllm_engine_seed(0, value)
