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

"""Tests for pure verifier score-to-credit transforms."""

import pytest
import torch
from pydantic import ValidationError
from turn_level_credit.verifier_credit import (
    VerifierCreditTransformConfig,
    VerifierScoreBatch,
    adjacent_score_delta_credit,
    compute_verifier_credit,
    hindsight_leave_one_out_credit,
    normalize_credit_within_prompt_turn,
    postprocess_verifier_credit,
    raw_verifier_credit,
    retrospective_verifier_credit,
)


def _batch(
    scores: list[list[float]],
    *,
    mask: list[list[bool]] | None = None,
    group_ids: list[int] | None = None,
    dtype: torch.dtype = torch.float32,
) -> VerifierScoreBatch:
    score_tensor = torch.tensor(scores, dtype=dtype)
    if mask is None:
        mask = [[True] * score_tensor.shape[1] for _ in scores]
    if group_ids is None:
        group_ids = list(range(score_tensor.shape[0]))
    return VerifierScoreBatch(
        scores=score_tensor,
        mask=torch.tensor(mask, dtype=torch.bool),
        prompt_group_ids=torch.tensor(group_ids, dtype=torch.int64),
    )


def _assert_values(
    actual: torch.Tensor,
    expected: list[list[float | int]],
) -> None:
    torch.testing.assert_close(
        actual,
        torch.tensor(expected, dtype=actual.dtype, device=actual.device),
    )


def test_raw_credit_preserves_dtype_and_zeros_padding():
    batch = _batch(
        [[0.25, 0.75, 99.0], [1.0, 99.0, 99.0]],
        mask=[[True, True, False], [True, False, False]],
        dtype=torch.float64,
    )

    credit = raw_verifier_credit(batch)

    assert credit.dtype == torch.float64
    assert credit.tolist() == [[0.25, 0.75, 0.0], [1.0, 0.0, 0.0]]


def test_adjacent_delta_uses_zero_initial_score_and_zeros_padding():
    batch = _batch(
        [[0.25, 0.75, 0.5], [1.0, 0.0, 99.0]],
        mask=[[True, True, True], [True, True, False]],
    )

    credit = adjacent_score_delta_credit(batch)

    _assert_values(credit, [[0.25, 0.5, -0.25], [1.0, -1.0, 0.0]])


def test_retrospective_separates_progress_preservation_and_regression():
    batch = _batch([[0.2, 1.0, 1.0, 0.5, 0.75]])
    config = VerifierCreditTransformConfig(
        mode="retrospective",
        progress_weight=2.0,
        preservation_weight=0.25,
        regression_weight=3.0,
    )

    result = retrospective_verifier_credit(batch, config=config)

    _assert_values(result.best_previous_score, [[0.0, 0.2, 1.0, 1.0, 1.0]])
    _assert_values(result.improvement, [[0.2, 0.8, 0, 0, 0]])
    assert result.preservation_mask.tolist() == [[False, False, True, False, False]]
    _assert_values(result.regression, [[0, 0, 0, 0.5, 0.25]])
    _assert_values(result.credit, [[0.4, 1.6, 0.25, -1.5, -0.75]])


def test_retrospective_non_improving_pre_success_turn_has_zero_credit():
    batch = _batch([[0.25, 0.1, 0.25]])
    result = retrospective_verifier_credit(
        batch,
        config=VerifierCreditTransformConfig(mode="retrospective"),
    )

    _assert_values(result.credit, [[0.25, 0.0, 0.0]])
    assert not result.preservation_mask.any()
    assert not result.regression.any()


def test_hindsight_uses_eligible_same_prompt_same_turn_leave_one_out_peers():
    batch = _batch(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0],
        ],
        group_ids=[7, 7, 7, 9],
    )

    result = hindsight_leave_one_out_credit(batch, success_threshold=1.0)

    _assert_values(
        result.future_best_score,
        [[1, 1, 1], [0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]],
    )
    assert result.eligibility_mask.tolist() == [
        [True, True, False],
        [True, True, True],
        [True, False, True],
        [True, True, False],
    ]
    assert result.reference_count.tolist() == [
        [2, 1, 0],
        [2, 1, 1],
        [2, 0, 1],
        [0, 0, 0],
    ]
    _assert_values(
        result.leave_one_out_baseline,
        [[0.25, 0.0, 0.0], [0.75, 1.0, 0.5], [0.5, 0.0, 0.0], [0, 0, 0]],
    )
    _assert_values(
        result.credit,
        [[0.75, 1.0, 0.0], [-0.75, -1.0, -0.5], [0.0, 0.0, 0.5], [0, 0, 0]],
    )


def test_hindsight_excludes_post_success_keep_and_regression():
    batch = _batch(
        [[1.0, 1.0, 0.0], [1.0, 0.5, 0.0]],
        group_ids=[0, 0],
    )

    result = hindsight_leave_one_out_credit(batch, success_threshold=1.0)

    assert result.eligibility_mask.tolist() == [
        [False, False, False],
        [False, False, False],
    ]
    assert not result.credit.any()


def test_hindsight_respects_variable_length_prefix_masks():
    batch = _batch(
        [[0.0, 0.0, 1.0], [0.0, 0.5, 99.0]],
        mask=[[True, True, True], [True, True, False]],
        group_ids=[3, 3],
    )

    result = hindsight_leave_one_out_credit(batch, success_threshold=1.0)

    _assert_values(result.future_best_score, [[1.0, 1.0, 1.0], [0.5, 0.5, 0.0]])
    assert result.eligibility_mask.tolist() == [
        [True, True, False],
        [True, False, False],
    ]
    assert result.reference_count.tolist() == [[1, 0, 0], [1, 0, 0]]
    _assert_values(result.credit, [[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])


def test_hindsight_ignores_nonfinite_padding_across_variable_horizons():
    batch = _batch(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, float("nan")],
            [0.0, float("inf"), float("nan")],
        ],
        mask=[
            [True, True, True],
            [True, True, False],
            [True, False, False],
        ],
        group_ids=[-3, -3, -3],
    )

    result = hindsight_leave_one_out_credit(batch, success_threshold=1.0)

    _assert_values(
        result.future_best_score,
        [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )
    assert result.reference_count.tolist() == [
        [2, 1, 0],
        [2, 1, 0],
        [2, 0, 0],
    ]
    _assert_values(
        result.credit,
        [[1.0, 1.0, 0.0], [-0.5, -1.0, 0.0], [-0.5, 0.0, 0.0]],
    )


def test_hindsight_isolates_arbitrary_prompt_groups():
    batch = _batch(
        [
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.25],
            [0.0, 0.75],
        ],
        group_ids=[101, 101, -8, -8],
    )

    result = hindsight_leave_one_out_credit(batch, success_threshold=1.0)

    _assert_values(
        result.credit,
        [[1.0, 0.0], [-1.0, 0.0], [-0.5, 0.0], [0.5, 0.0]],
    )


def test_hindsight_singleton_eligible_set_has_zero_fallback():
    batch = _batch(
        [[0.0, 0.0], [0.5, 0.75]],
        group_ids=[4, 4],
    )

    result = hindsight_leave_one_out_credit(batch, success_threshold=1.0)

    assert result.eligibility_mask.tolist() == [
        [True, True],
        [False, False],
    ]
    assert not result.reference_count.any()
    assert not result.leave_one_out_baseline.any()
    assert not result.credit.any()


def test_combined_dispatch_adds_weighted_hindsight_to_retrospective():
    batch = _batch(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        group_ids=[5, 5],
    )
    config = VerifierCreditTransformConfig(
        mode="retrospective_hindsight",
        hindsight_weight=0.5,
    )

    credit = compute_verifier_credit(batch, config=config)

    _assert_values(credit, [[0.5, 0.5, 1.0], [-0.5, -0.5, 0.0]])


def test_prompt_turn_normalization_isolates_groups_and_centers_each_turn():
    batch = _batch(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        group_ids=[7, 7, -2, -2],
    )
    raw_credit = torch.tensor(
        [[1.0, 4.0], [3.0, 8.0], [10.0, -2.0], [14.0, 2.0]],
    )

    normalized = normalize_credit_within_prompt_turn(
        raw_credit,
        batch,
        epsilon=1.0e-12,
    )

    _assert_values(
        normalized,
        [[-1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [1.0, 1.0]],
    )


def test_prompt_turn_normalization_skips_singletons_and_zeros_padding():
    batch = _batch(
        [[0.0, 0.0], [0.0, 99.0], [0.0, 0.0]],
        mask=[[True, True], [True, False], [True, True]],
        group_ids=[5, 5, 9],
    )
    raw_credit = torch.tensor(
        [[1.0, 3.0], [3.0, float("nan")], [7.0, 11.0]],
        dtype=torch.float64,
    )

    normalized = normalize_credit_within_prompt_turn(
        raw_credit,
        batch,
        epsilon=1.0e-12,
    )

    assert normalized.dtype == torch.float64
    _assert_values(normalized, [[-1.0, 3.0], [1.0, 0.0], [7.0, 11.0]])


def test_prompt_turn_normalization_maps_constant_multi_sample_group_to_zero():
    batch = _batch([[0.0], [0.0]], group_ids=[0, 0])

    normalized = normalize_credit_within_prompt_turn(
        torch.tensor([[2.5], [2.5]]),
        batch,
        epsilon=1.0e-6,
    )

    assert not normalized.any()


def test_compute_dispatch_applies_normalization_then_early_turn_discount():
    batch = _batch(
        [[0.0, 0.25, 0.75], [1.0, 0.75, 0.25]],
        group_ids=[3, 3],
    )
    config = VerifierCreditTransformConfig(
        mode="raw",
        normalization="group_zscore",
        normalization_epsilon=1.0e-12,
        early_turn_discount=0.5,
    )

    credit = compute_verifier_credit(batch, config=config)

    _assert_values(credit, [[-1.0, -0.5, 0.25], [1.0, 0.5, -0.25]])


def test_postprocessing_without_normalization_only_applies_early_turn_discount():
    batch = _batch([[0.0, 0.0, 0.0]])
    config = VerifierCreditTransformConfig(early_turn_discount=0.5)

    credit = postprocess_verifier_credit(
        torch.ones_like(batch.scores),
        batch,
        config=config,
    )

    _assert_values(credit, [[1.0, 0.5, 0.25]])


@pytest.mark.parametrize(
    ("batch", "error_type", "message"),
    [
        (
            VerifierScoreBatch(
                scores=torch.tensor([[0]], dtype=torch.int64),
                mask=torch.tensor([[True]]),
                prompt_group_ids=torch.tensor([0]),
            ),
            TypeError,
            "floating-point",
        ),
        (
            VerifierScoreBatch(
                scores=torch.tensor([[0.0]]),
                mask=torch.tensor([[1]], dtype=torch.int64),
                prompt_group_ids=torch.tensor([0]),
            ),
            ValueError,
            "mask must be boolean",
        ),
        (_batch([[0.0, 0.0]], mask=[[False, True]]), ValueError, "prefix-contiguous"),
        (_batch([[0.0]], mask=[[False]]), ValueError, "observed turn"),
        (
            VerifierScoreBatch(
                scores=torch.tensor([[0.0], [0.0]]),
                mask=torch.tensor([[True], [True]]),
                prompt_group_ids=torch.tensor([[0], [1]]),
            ),
            ValueError,
            r"shape \[batch\]",
        ),
        (
            VerifierScoreBatch(
                scores=torch.tensor([[0.0]]),
                mask=torch.tensor([[True]]),
                prompt_group_ids=torch.tensor([0.0]),
            ),
            TypeError,
            "integer dtype",
        ),
        (_batch([[float("nan")]]), ValueError, "finite"),
        (_batch([[1.01]]), ValueError, r"in \[0, 1\]"),
    ],
)
def test_score_batch_validation_rejects_malformed_inputs(batch, error_type, message):
    with pytest.raises(error_type, match=message):
        raw_verifier_credit(batch)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"success_threshold": 0.0},
        {"success_threshold": 1.1},
        {"progress_weight": -1.0},
        {"preservation_weight": -1.0},
        {"regression_weight": -1.0},
        {"hindsight_weight": -1.0},
        {"normalization_epsilon": 0.0},
        {"early_turn_discount": 0.0},
        {"early_turn_discount": 1.1},
        {"unknown_option": 1},
    ],
)
def test_transform_config_rejects_invalid_or_unknown_values(kwargs):
    with pytest.raises(ValidationError):
        VerifierCreditTransformConfig(**kwargs)


@pytest.mark.parametrize(
    ("batch", "error_type", "message"),
    [
        (
            VerifierScoreBatch(
                scores=torch.empty((0, 1)),
                mask=torch.empty((0, 1), dtype=torch.bool),
                prompt_group_ids=torch.empty((0,), dtype=torch.int64),
            ),
            ValueError,
            "non-empty shape",
        ),
        (
            VerifierScoreBatch(
                scores=torch.zeros((2, 2)),
                mask=torch.ones((2, 1), dtype=torch.bool),
                prompt_group_ids=torch.arange(2),
            ),
            ValueError,
            "score tensor shape",
        ),
    ],
)
def test_score_batch_validation_rejects_empty_and_mismatched_shapes(
    batch,
    error_type,
    message,
):
    with pytest.raises(error_type, match=message):
        raw_verifier_credit(batch)


@pytest.mark.parametrize(
    "transform",
    [
        raw_verifier_credit,
        adjacent_score_delta_credit,
    ],
)
def test_simple_transforms_ignore_nonfinite_padding(transform):
    batch = _batch(
        [[0.25, float("nan")], [0.75, float("inf")]],
        mask=[[True, False], [True, False]],
    )

    credit = transform(batch)

    assert torch.isfinite(credit).all()
    _assert_values(credit, [[0.25, 0.0], [0.75, 0.0]])


@pytest.mark.parametrize("epsilon", [0.0, -1.0, float("nan"), float("inf")])
def test_prompt_turn_normalization_rejects_invalid_epsilon(epsilon):
    batch = _batch([[0.0], [0.0]], group_ids=[0, 0])

    with pytest.raises(ValueError, match="finite and positive"):
        normalize_credit_within_prompt_turn(
            torch.zeros_like(batch.scores),
            batch,
            epsilon=epsilon,
        )


@pytest.mark.parametrize(
    ("credit", "error_type", "message"),
    [
        (torch.zeros((2, 2)), ValueError, "score tensor shape"),
        (torch.zeros((2, 1), dtype=torch.int64), TypeError, "floating-point"),
        (torch.tensor([[0.0], [float("nan")]]), ValueError, "finite"),
    ],
)
def test_prompt_turn_normalization_rejects_malformed_credit(
    credit,
    error_type,
    message,
):
    batch = _batch([[0.0], [0.0]], group_ids=[0, 0])

    with pytest.raises(error_type, match=message):
        normalize_credit_within_prompt_turn(
            credit,
            batch,
            epsilon=1.0e-6,
        )
