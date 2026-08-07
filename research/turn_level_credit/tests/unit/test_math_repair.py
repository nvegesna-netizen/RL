"""Tests for fixed-horizon math-repair state transitions."""

import pytest
import torch
from pydantic import ValidationError
from nemo_rl.distributed import ray_actor_environment_registry
from nemo_rl.environments import utils as environment_utils
from turn_level_credit.math_repair import (
    MATH_REPAIR_ENVIRONMENT_FQN,
    TERMINAL_SUCCESS_KEY,
    VERIFIER_SCORE_KEY,
    MathRepairConfig,
    _latest_assistant_responses,
    advance_math_repair_turns,
    install_math_repair_environment,
)


def _metadata(**overrides):
    metadata = {"ground_truth": "42"}
    metadata.update(overrides)
    return metadata


def test_fixed_horizon_continues_after_success_and_emits_outcome_once():
    config = MathRepairConfig(max_turns=3)

    first = advance_math_repair_turns([1.0], [_metadata()], config=config)
    second = advance_math_repair_turns(
        [0.0],
        first.metadata,
        config=config,
    )
    third = advance_math_repair_turns(
        [1.0],
        second.metadata,
        config=config,
    )

    assert first.metadata[0] == {
        "ground_truth": "42",
        "turn_index": 1,
        "first_success_turn": 1,
        "best_score": 1.0,
    }
    assert second.metadata[0]["first_success_turn"] == 1
    assert third.metadata[0]["first_success_turn"] == 1
    assert first.terminateds.tolist() == [False]
    assert second.terminateds.tolist() == [False]
    assert third.terminateds.tolist() == [True]
    assert first.terminal_success.tolist() == [0.0]
    assert second.terminal_success.tolist() == [0.0]
    assert third.terminal_success.tolist() == [1.0]
    assert first.verifier_scores.tolist() == [1.0]
    assert second.verifier_scores.tolist() == [0.0]
    assert third.verifier_scores.tolist() == [1.0]
    assert "correct" in first.observations[0]["content"]
    assert "incorrect" in second.observations[0]["content"]


def test_success_on_final_turn_counts_and_all_failure_does_not():
    config = MathRepairConfig(max_turns=2)
    initial = [_metadata(), _metadata(ground_truth="7")]

    first = advance_math_repair_turns([0.0, 0.0], initial, config=config)
    final = advance_math_repair_turns([1.0, 0.0], first.metadata, config=config)

    assert final.terminateds.tolist() == [True, True]
    assert final.terminal_success.tolist() == [1.0, 0.0]
    assert final.metadata[0]["first_success_turn"] == 2
    assert final.metadata[1]["first_success_turn"] is None


def test_turn_batch_uses_stable_tensor_dtypes():
    result = advance_math_repair_turns(
        [0.25, 1.0],
        [_metadata(), _metadata()],
        config=MathRepairConfig(max_turns=3, success_threshold=0.75),
    )

    assert result.verifier_scores.dtype == torch.float32
    assert result.terminal_success.dtype == torch.float32
    assert result.terminateds.dtype == torch.bool
    assert VERIFIER_SCORE_KEY != TERMINAL_SUCCESS_KEY


def test_latest_response_ignores_prior_candidates():
    responses = _latest_assistant_responses(
        [
            [
                {"role": "user", "content": "problem"},
                {"role": "assistant", "content": "old answer"},
                {"role": "environment", "content": "incorrect"},
                {"role": "assistant", "content": "new answer"},
            ]
        ]
    )

    assert responses == ["new answer"]


def test_math_repair_environment_installation_is_scoped():
    original_entry = environment_utils.ENV_REGISTRY["math"]
    original_fqn = original_entry["actor_class_fqn"]
    original_python = ray_actor_environment_registry.ACTOR_ENVIRONMENT_REGISTRY[
        original_fqn
    ]
    assert (
        MATH_REPAIR_ENVIRONMENT_FQN
        not in ray_actor_environment_registry.ACTOR_ENVIRONMENT_REGISTRY
    )

    with install_math_repair_environment():
        assert environment_utils.ENV_REGISTRY["math"] == {
            "actor_class_fqn": MATH_REPAIR_ENVIRONMENT_FQN
        }
        assert (
            ray_actor_environment_registry.ACTOR_ENVIRONMENT_REGISTRY[
                MATH_REPAIR_ENVIRONMENT_FQN
            ]
            == original_python
        )

    assert environment_utils.ENV_REGISTRY["math"] is original_entry
    assert (
        MATH_REPAIR_ENVIRONMENT_FQN
        not in ray_actor_environment_registry.ACTOR_ENVIRONMENT_REGISTRY
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_workers": 0},
        {"max_turns": 1},
        {"success_threshold": 0.0},
        {"success_threshold": 1.1},
        {"unknown_option": True},
    ],
)
def test_math_repair_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValidationError):
        MathRepairConfig(**kwargs)


@pytest.mark.parametrize(
    ("scores", "metadata", "error_type", "message"),
    [
        ([], [], ValueError, "empty batch"),
        ([0.0], [], ValueError, "batches must align"),
        ([float("nan")], [_metadata()], ValueError, "finite and in"),
        ([1.1], [_metadata()], ValueError, "finite and in"),
        ([True], [_metadata()], TypeError, "score must be numeric"),
        ([0.0], [{}], ValueError, "non-empty ground_truth"),
        ([0.0], [_metadata(turn_index=True)], TypeError, "must be an integer"),
        (
            [0.0],
            [_metadata(turn_index=3)],
            ValueError,
            r"must be in \[0, 3\)",
        ),
        (
            [0.0],
            [_metadata(turn_index=1, first_success_turn=2, best_score=1.0)],
            ValueError,
            "must describe a prior turn",
        ),
        (
            [0.0],
            [_metadata(turn_index=1, first_success_turn=1, best_score=0.0)],
            ValueError,
            "without a successful best_score",
        ),
        (
            [0.0],
            [_metadata(turn_index=1, first_success_turn=None, best_score=1.0)],
            ValueError,
            "successful best_score without a first_success_turn",
        ),
    ],
)
def test_advance_rejects_malformed_scores_and_metadata(
    scores,
    metadata,
    error_type,
    message,
):
    with pytest.raises(error_type, match=message):
        advance_math_repair_turns(
            scores,
            metadata,
            config=MathRepairConfig(),
        )


@pytest.mark.parametrize(
    ("conversation", "error_type", "message"),
    [
        ([{"role": "user", "content": "problem"}], ValueError, "no assistant"),
        (
            [{"role": "assistant", "content": torch.tensor([1])}],
            TypeError,
            "content must be a string",
        ),
    ],
)
def test_latest_response_rejects_missing_or_non_text_candidates(
    conversation,
    error_type,
    message,
):
    with pytest.raises(error_type, match=message):
        _latest_assistant_responses([conversation])
