# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import json

import pytest
import torch

from nemo_rl.algorithms.async_utils.conditional_m4_capsule import CapsuleGroup
from nemo_rl.algorithms.async_utils.eligible_live_decision_capsule import (
    write_eligible_live_decision_capsule,
)
from nemo_rl.algorithms.async_utils.opportunity_at_risk import (
    OPPORTUNITY_GROUP_ID_KEY,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import DP_TRAIN_FIELDS
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _group(group_id: str, value: int) -> CapsuleGroup:
    fields = BatchedDataDict(
        {
            field: torch.full((2, 3), value, dtype=torch.float32)
            for field in DP_TRAIN_FIELDS
        }
    )
    fields["input_ids"] = torch.full((2, 3), value, dtype=torch.int64)
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


def _decision() -> dict:
    common = {
        "status": "proposed",
        "search_strategy": "exact",
        "proposed_reward_variance_sum": 4.0,
        "proposed_imminent_reward_variance_sum": 4.0,
    }
    return {
        "actuation_scorer": "m4_shield",
        "actual_selected_group_ids": ["c", "d", "g", "h"],
        "baseline_valid_actor_tokens": 400,
        "candidate_group_count": 8,
        "proposal_matches_actual": True,
        "skip_reason": None,
        "proposals": {
            "reward_variance_risk": common
            | {
                "proposed_group_ids": ["a", "b", "c", "d"],
                "proposed_imminent_l1": 10.0,
            },
            "m4_shield": common
            | {
                "proposed_group_ids": ["c", "d", "g", "h"],
                "proposed_imminent_l1": 20.0,
                "proposed_valid_actor_tokens": 400,
            },
        },
    }


def test_eligible_capsule_authenticates_checkpoint_tensors_and_embargo(
    tmp_path,
) -> None:
    groups = [_group(letter, index) for index, letter in enumerate("abcdefghijkl")]
    output_dir = tmp_path / "capsule"
    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "state.bin").write_bytes(b"checkpoint-state")

    result = write_eligible_live_decision_capsule(
        output_dir,
        frontier=groups[:8],
        heldout=groups[8:],
        decision=_decision(),
        base_group_ids=("a", "b", "c", "d"),
        shield_group_ids=("c", "d", "g", "h"),
        learner_version=7,
        train_steps_before_capture=7,
        parameter_sha256_before="a" * 64,
        parameter_sha256_after="a" * 64,
        checkpoint_dir=checkpoint_dir,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert result["status"] == "QUALIFIED_OUTCOME_EXCLUDED_ELIGIBLE_CAPTURE"
    assert manifest["captured_update_executed"] is False
    assert manifest["post_update_outcomes_opened"] is False
    assert manifest["checkpoint_files"][0]["path"] == "state.bin"
    assert manifest["groups"][2]["roles"] == ["base", "shield"]
    assert manifest["groups"][8]["roles"] == ["heldout"]
    restored = torch.load(
        output_dir / manifest["groups"][0]["raw_file"], weights_only=True
    )
    assert tuple(restored) == DP_TRAIN_FIELDS


def test_eligible_capsule_rejects_noneligible_action(tmp_path) -> None:
    groups = [_group(letter, index) for index, letter in enumerate("abcdefghijkl")]
    output_dir = tmp_path / "capsule"
    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "state.bin").write_bytes(b"checkpoint-state")
    decision = _decision()
    decision["proposals"]["m4_shield"]["proposed_reward_variance_sum"] = 3.0

    with pytest.raises(ValueError, match="changed proposed_reward_variance_sum"):
        write_eligible_live_decision_capsule(
            output_dir,
            frontier=groups[:8],
            heldout=groups[8:],
            decision=decision,
            base_group_ids=("a", "b", "c", "d"),
            shield_group_ids=("c", "d", "g", "h"),
            learner_version=7,
            train_steps_before_capture=7,
            parameter_sha256_before="a" * 64,
            parameter_sha256_after="a" * 64,
            checkpoint_dir=checkpoint_dir,
        )
