# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import json

import pytest
import torch

from nemo_rl.algorithms.async_utils.conditional_m4_capsule import (
    CapsuleGroup,
    ConditionalM4Selection,
    select_conditional_m4_contrast,
    write_capsule,
)
from nemo_rl.algorithms.async_utils.opportunity_at_risk import (
    OPPORTUNITY_GROUP_ID_KEY,
)
from nemo_rl.algorithms.async_utils.opportunity_at_risk_v2 import OARSV2Candidate
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import DP_TRAIN_FIELDS
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _candidate(group_id: str, *, m4: float, variance: float) -> OARSV2Candidate:
    return OARSV2Candidate(
        group_id=group_id,
        start_weight_version=0,
        ready_timestamp_ns=1,
        l1=m4,
        l2=m4,
        valid_actor_tokens=100,
        reward_mean=0.5,
        reward_variance=variance,
    )


def test_conditional_contrast_preserves_comparator_utility_and_token_band() -> None:
    candidates = [
        _candidate(letter, m4=10.0 if letter < "e" else 100.0, variance=0.25)
        for letter in "abcdefgh"
    ]
    result = select_conditional_m4_contrast(
        candidates,
        coefficients=(0.0, 0.0, 0.0, 1.0),
        intercept=0.0,
        minimum_service_multiplier=0.98,
        maximum_service_multiplier=1.02,
        batch_groups=4,
    )
    assert result.comparator_group_ids == ("a", "b", "c", "d")
    assert result.treatment_group_ids == ("e", "f", "g", "h")
    assert result.relative_predicted_norm_gain > 1.0
    assert result.treatment_reward_variance >= result.comparator_reward_variance
    assert result.treatment_tokens == result.comparator_tokens


def _capsule_group(group_id: str, index: int) -> CapsuleGroup:
    fields = BatchedDataDict(
        {
            field: torch.full((2, 3), index, dtype=torch.float32)
            for field in DP_TRAIN_FIELDS
        }
    )
    fields["input_ids"] = torch.full((2, 3), index, dtype=torch.int64)
    fields["input_lengths"] = torch.tensor([3, 3], dtype=torch.int64)
    return CapsuleGroup(
        meta=KVBatchMeta(
            partition_id="train",
            task_name="train",
            sample_ids=[f"{group_id}_g0", f"{group_id}_g1"],
            extra_info={OPPORTUNITY_GROUP_ID_KEY: group_id},
        ),
        fields=fields,
    )


def _selection() -> ConditionalM4Selection:
    return ConditionalM4Selection(
        comparator_group_ids=("a", "b", "c", "d"),
        treatment_group_ids=("e", "f", "g", "h"),
        comparator_predicted_norm_sum=4.0,
        treatment_predicted_norm_sum=8.0,
        relative_predicted_norm_gain=1.0,
        comparator_reward_variance=1.0,
        treatment_reward_variance=1.0,
        comparator_tokens=400,
        treatment_tokens=400,
        minimum_tokens=392,
        maximum_tokens=408,
        feasible_combination_count=70,
    )


def test_capsule_writer_authenticates_raw_tensors_and_roles(tmp_path) -> None:
    groups = [
        _capsule_group(letter, index) for index, letter in enumerate("abcdefghijkl")
    ]
    result = write_capsule(
        tmp_path / "capsule",
        frontier=groups[:8],
        heldout=groups[8:],
        selection=_selection(),
        parameter_sha256_before="a" * 64,
        parameter_sha256_after="a" * 64,
        frontiers_examined=3,
    )
    manifest = json.loads((tmp_path / "capsule" / "manifest.json").read_text())
    assert result["status"] == "QUALIFIED_NO_UPDATE_CAPSULE"
    assert manifest["parameter_hash_unchanged"] is True
    assert manifest["learning_outcomes_opened"] is False
    assert len(manifest["groups"]) == 12
    assert manifest["groups"][0]["roles"] == ["comparator"]
    assert manifest["groups"][4]["roles"] == ["treatment"]
    assert manifest["groups"][8]["roles"] == ["heldout"]
    restored = torch.load(
        tmp_path / "capsule" / manifest["groups"][0]["raw_file"],
        weights_only=True,
    )
    assert tuple(restored) == DP_TRAIN_FIELDS


def test_capsule_writer_refuses_parameter_change(tmp_path) -> None:
    groups = [
        _capsule_group(letter, index) for index, letter in enumerate("abcdefghijkl")
    ]
    with pytest.raises(RuntimeError, match="parameters changed"):
        write_capsule(
            tmp_path / "capsule",
            frontier=groups[:8],
            heldout=groups[8:],
            selection=_selection(),
            parameter_sha256_before="a" * 64,
            parameter_sha256_after="b" * 64,
            frontiers_examined=1,
        )
