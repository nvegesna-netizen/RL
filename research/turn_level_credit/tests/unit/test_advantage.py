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

"""Tests for turn-level GRPO advantage composition."""

import pytest
import torch
from turn_level_credit.advantage import TurnLevelGRPOAdvantageEstimator
from turn_level_credit.config import TurnCreditConfig
from turn_level_credit.trace import TurnBatch, attach_turn_batch
from turn_level_credit.verifier_credit import (
    VerifierCreditTransformConfig,
    VerifierScoreBatch,
    compute_verifier_credit,
)


class _FixedBaseEstimator:
    def __init__(self, output):
        self.output = output

    def compute_advantage(self, **_kwargs):
        return self.output


def _repeated_batch():
    batch = {}
    attach_turn_batch(
        batch,
        TurnBatch(
            rewards=torch.tensor([[1.5, 0.75]]),
            credit_rewards=torch.tensor([[0.5, -0.25]]),
            mask=torch.tensor([[True, True]]),
            trainable_mask=torch.tensor([[True, True]]),
            assistant_spans=torch.tensor([[[1, 3], [4, 5]]]),
            terminateds=torch.tensor([[False, True]]),
        ),
    )
    return batch


def _verifier_repeated_batch(scores: torch.Tensor):
    batch_size, turns = scores.shape
    batch = {}
    spans = torch.arange(
        turns,
        dtype=torch.int64,
        device=scores.device,
    ).repeat(batch_size, 1)
    terminateds = torch.zeros_like(scores, dtype=torch.bool)
    terminateds[:, -1] = True
    attach_turn_batch(
        batch,
        TurnBatch(
            rewards=scores.clone(),
            credit_rewards=scores.clone(),
            mask=torch.ones_like(scores, dtype=torch.bool),
            trainable_mask=torch.ones_like(scores, dtype=torch.bool),
            assistant_spans=torch.stack((spans, spans + 1), dim=-1),
            terminateds=terminateds,
        ),
    )
    return batch


def test_zero_turn_weight_is_bitwise_base_equivalent():
    base_output = torch.tensor([[7.0, 7.0, 7.0, 7.0, 7.0]])
    estimator = TurnLevelGRPOAdvantageEstimator(
        base_estimator=_FixedBaseEstimator(base_output),
        config=TurnCreditConfig(enabled=True, turn_weight=0.0),
    )

    actual = estimator.compute_advantage(
        prompt_ids=torch.tensor([[1]]),
        rewards=torch.tensor([1.0]),
        mask=torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0]]),
        repeated_batch=_repeated_batch(),
    )

    assert actual.data_ptr() == base_output.data_ptr()
    assert torch.equal(actual, base_output)


def test_zero_turn_weight_still_validates_trace_transport():
    estimator = TurnLevelGRPOAdvantageEstimator(
        base_estimator=_FixedBaseEstimator(torch.ones((1, 5))),
        config=TurnCreditConfig(enabled=True, turn_weight=0.0),
    )

    with pytest.raises(ValueError, match="missing fields"):
        estimator.compute_advantage(
            prompt_ids=torch.tensor([[1]]),
            rewards=torch.tensor([1.0]),
            mask=torch.ones((1, 5)),
            repeated_batch={},
        )


def test_zero_turn_weight_with_verifier_config_is_bitwise_base_equivalent():
    base_output = torch.tensor([[3.0, 5.0]])
    estimator = TurnLevelGRPOAdvantageEstimator(
        base_estimator=_FixedBaseEstimator(base_output),
        config=TurnCreditConfig(
            enabled=True,
            environment_component="reward/verifier_score",
            turn_weight=0.0,
            verifier_transform=VerifierCreditTransformConfig(
                mode="retrospective_hindsight",
                normalization="group_zscore",
            ),
        ),
    )

    actual = estimator.compute_advantage(
        prompt_ids=torch.tensor([[1, 2]]),
        rewards=torch.tensor([0.0]),
        mask=torch.ones((1, 2)),
        repeated_batch=_verifier_repeated_batch(torch.tensor([[0.0, 1.0]])),
    )

    assert actual.data_ptr() == base_output.data_ptr()
    assert torch.equal(actual, base_output)


def test_composes_macro_and_turn_credit_on_generated_spans():
    estimator = TurnLevelGRPOAdvantageEstimator(
        base_estimator=_FixedBaseEstimator(torch.ones((1, 5))),
        config=TurnCreditConfig(
            enabled=True,
            macro_weight=1.0,
            turn_weight=2.0,
        ),
    )

    actual = estimator.compute_advantage(
        prompt_ids=torch.tensor([[1]]),
        rewards=torch.tensor([1.0]),
        mask=torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0]]),
        repeated_batch=_repeated_batch(),
    )

    assert actual.tolist() == [[0.0, 2.0, 2.0, 0.0, 0.5]]


def test_fractional_sample_multiplier_is_applied_only_by_the_loss():
    estimator = TurnLevelGRPOAdvantageEstimator(
        base_estimator=_FixedBaseEstimator(torch.ones((1, 5))),
        config=TurnCreditConfig(
            enabled=True,
            macro_weight=1.0,
            turn_weight=2.0,
        ),
    )

    actual = estimator.compute_advantage(
        prompt_ids=torch.tensor([[1]]),
        rewards=torch.tensor([1.0]),
        mask=torch.tensor([[0.0, 0.5, 0.5, 0.0, 0.5]]),
        repeated_batch=_repeated_batch(),
    )

    assert actual.tolist() == [[0.0, 2.0, 2.0, 0.0, 0.5]]


def test_return_to_go_changes_earlier_turn_only():
    estimator = TurnLevelGRPOAdvantageEstimator(
        base_estimator=_FixedBaseEstimator(torch.zeros((1, 5))),
        config=TurnCreditConfig(
            enabled=True,
            environment_mode="return_to_go",
            discount=0.5,
            turn_weight=1.0,
        ),
    )

    actual = estimator.compute_advantage(
        prompt_ids=torch.tensor([[1]]),
        rewards=torch.tensor([1.0]),
        mask=torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0]]),
        repeated_batch=_repeated_batch(),
    )

    assert actual.tolist() == [[0.0, 0.375, 0.375, 0.0, -0.25]]


def test_verifier_normalization_uses_exact_nonadjacent_prompt_rows():
    scores = torch.tensor([[0.0], [0.5], [1.0], [0.5]])
    estimator = TurnLevelGRPOAdvantageEstimator(
        base_estimator=_FixedBaseEstimator(torch.zeros((4, 1))),
        config=TurnCreditConfig(
            enabled=True,
            environment_component="reward/verifier_score",
            macro_weight=0.0,
            turn_weight=1.0,
            verifier_transform=VerifierCreditTransformConfig(
                mode="raw",
                normalization="group_zscore",
                normalization_epsilon=1.0e-12,
            ),
        ),
    )

    actual = estimator.compute_advantage(
        prompt_ids=torch.tensor([[1, 2], [9, 9], [1, 2], [9, 9]]),
        rewards=torch.zeros(4),
        mask=torch.ones((4, 1)),
        repeated_batch=_verifier_repeated_batch(scores),
    )

    torch.testing.assert_close(
        actual,
        torch.tensor([[-1.0], [0.0], [1.0], [0.0]]),
    )


@pytest.mark.parametrize(
    "mode",
    ["raw", "adjacent_delta", "retrospective", "retrospective_hindsight"],
)
def test_verifier_modes_flow_through_turn_span_scatter(mode):
    scores = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    transform = VerifierCreditTransformConfig(mode=mode)
    estimator = TurnLevelGRPOAdvantageEstimator(
        base_estimator=_FixedBaseEstimator(torch.zeros((2, 3))),
        config=TurnCreditConfig(
            enabled=True,
            environment_component="reward/verifier_score",
            macro_weight=0.0,
            turn_weight=1.0,
            verifier_transform=transform,
        ),
    )

    actual = estimator.compute_advantage(
        prompt_ids=torch.tensor([[4, 2], [4, 2]]),
        rewards=torch.zeros(2),
        mask=torch.ones((2, 3)),
        repeated_batch=_verifier_repeated_batch(scores),
    )
    expected = compute_verifier_credit(
        VerifierScoreBatch(
            scores=scores,
            mask=torch.ones_like(scores, dtype=torch.bool),
            prompt_group_ids=torch.zeros(2, dtype=torch.int64),
        ),
        config=transform,
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("prompt_ids", "error_type", "message"),
    [
        (torch.tensor([1, 1]), ValueError, r"shape \[turn-credit batch"),
        (torch.empty((2, 0), dtype=torch.int64), ValueError, "prompt tokens"),
        (torch.tensor([[1.0], [1.0]]), TypeError, "integer token dtype"),
        (torch.tensor([[True], [True]]), TypeError, "integer token dtype"),
    ],
)
def test_verifier_rejects_malformed_prompt_group_inputs(
    prompt_ids,
    error_type,
    message,
):
    estimator = TurnLevelGRPOAdvantageEstimator(
        base_estimator=_FixedBaseEstimator(torch.zeros((2, 1))),
        config=TurnCreditConfig(
            enabled=True,
            environment_component="reward/verifier_score",
            macro_weight=0.0,
            turn_weight=1.0,
            verifier_transform=VerifierCreditTransformConfig(),
        ),
    )

    with pytest.raises(error_type, match=message):
        estimator.compute_advantage(
            prompt_ids=prompt_ids,
            rewards=torch.zeros(2),
            mask=torch.ones((2, 1)),
            repeated_batch=_verifier_repeated_batch(torch.zeros((2, 1))),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_verifier_moves_cpu_prompt_groups_to_score_device():
    scores = torch.tensor([[0.0], [1.0]], device="cuda")
    estimator = TurnLevelGRPOAdvantageEstimator(
        base_estimator=_FixedBaseEstimator(torch.zeros((2, 1), device="cuda")),
        config=TurnCreditConfig(
            enabled=True,
            environment_component="reward/verifier_score",
            macro_weight=0.0,
            turn_weight=1.0,
            verifier_transform=VerifierCreditTransformConfig(mode="raw"),
        ),
    )

    actual = estimator.compute_advantage(
        prompt_ids=torch.tensor([[7], [7]], device="cpu"),
        rewards=torch.zeros(2, device="cuda"),
        mask=torch.ones((2, 1), device="cuda"),
        repeated_batch=_verifier_repeated_batch(scores),
    )

    assert actual.device.type == "cuda"
    torch.testing.assert_close(actual, scores)
