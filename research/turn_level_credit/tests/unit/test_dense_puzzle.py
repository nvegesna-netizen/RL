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

"""Tests for potential-based dense sliding-puzzle rewards."""

import copy
import random

import pytest
from turn_level_credit.dense_puzzle import (
    PROGRESS_REWARD_KEY,
    SUCCESS_REWARD_KEY,
    DensePuzzleConfig,
    DenseSlidingPuzzleRunner,
    _generate_initial_state,
    _sample_seed,
    manhattan_distance,
    normalized_manhattan_potential,
)


def _solved_state() -> dict:
    return {
        "size": 2,
        "grid": [[1, 2], [3, 0]],
        "solution": [[1, 2], [3, 0]],
        "empty_pos": (1, 1),
        "commands": {},
    }


def _one_move_state() -> dict:
    state = _solved_state()
    state["grid"] = [[1, 2], [0, 3]]
    state["empty_pos"] = (1, 0)
    return state


def _metadata(state: dict, *, moves: int = 0, max_moves: int = 4) -> dict:
    return {
        "game_state": copy.deepcopy(state),
        "num_moves": moves,
        "max_moves": max_moves,
    }


def _message(action: str) -> list[dict]:
    return [{"role": "assistant", "content": f"<action>{action}</action>"}]


def _config() -> DensePuzzleConfig:
    return DensePuzzleConfig(
        size=2,
        shuffle_moves=3,
        minimum_manhattan_distance=1,
        max_generation_attempts=10,
        max_moves=4,
        progress_scale=1.0,
    )


def test_manhattan_potential_is_normalized_and_exact_at_goal():
    assert manhattan_distance(_solved_state()) == 0
    assert normalized_manhattan_potential(_solved_state()) == 1.0
    assert manhattan_distance(_one_move_state()) == 1
    assert normalized_manhattan_potential(_one_move_state()) == pytest.approx(5 / 6)


def test_progress_telescopes_and_inverse_move_cannot_inflate_reward():
    start = _one_move_state()
    solved = _solved_state()
    start_potential = normalized_manhattan_potential(start)
    solved_potential = normalized_manhattan_potential(solved)

    forward = solved_potential - start_potential
    inverse = start_potential - solved_potential

    assert forward > 0
    assert forward + inverse == pytest.approx(0.0)
    assert forward == pytest.approx(solved_potential - start_potential)


def test_runner_emits_progress_and_success_on_solving_move():
    runner = DenseSlidingPuzzleRunner(_config())

    observation, rewards, terminated, _, next_metadata, _ = runner.process_turn(
        _message("left"),
        _metadata(_one_move_state()),
    )

    assert "Congratulations" in observation["content"]
    assert rewards[PROGRESS_REWARD_KEY] == pytest.approx(1 / 6)
    assert rewards[SUCCESS_REWARD_KEY] == 1.0
    assert terminated
    assert next_metadata is None


def test_invalid_action_preserves_potential_and_has_no_success_reward():
    runner = DenseSlidingPuzzleRunner(_config())

    _, rewards, terminated, _, next_metadata, _ = runner.process_turn(
        _message("up"),
        _metadata(_one_move_state()),
    )

    assert rewards == {PROGRESS_REWARD_KEY: 0.0, SUCCESS_REWARD_KEY: 0.0}
    assert not terminated
    assert next_metadata is not None


def test_max_move_termination_preserves_potential():
    runner = DenseSlidingPuzzleRunner(_config())

    _, rewards, terminated, _, next_metadata, _ = runner.process_turn(
        _message("left"),
        _metadata(_one_move_state(), moves=4, max_moves=4),
    )

    assert rewards == {PROGRESS_REWARD_KEY: 0.0, SUCCESS_REWARD_KEY: 0.0}
    assert terminated
    assert next_metadata is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("size", 1, "size must be at least 2"),
        ("minimum_manhattan_distance", 0, "must be positive"),
        ("max_generation_attempts", 0, "must be positive"),
        ("progress_scale", 0.0, "must be positive"),
    ],
)
def test_config_rejects_invalid_ranges(field, value, message):
    config = _config().model_dump()
    config[field] = value

    with pytest.raises(ValueError, match=message):
        DensePuzzleConfig.model_validate(config)


def test_manhattan_distance_rejects_duplicate_tiles():
    state = _solved_state()
    state["grid"] = [[1, 1], [3, 0]]

    with pytest.raises(ValueError, match="each tile exactly once"):
        manhattan_distance(state)


def test_dataset_state_is_deterministic_per_split_and_sample():
    config = _config()
    global_rng_state = random.getstate()

    first = _generate_initial_state(config, split_seed=7, sample_index=11)
    repeated = _generate_initial_state(config, split_seed=7, sample_index=11)

    assert first == repeated
    assert _sample_seed(7, 11) != _sample_seed(8, 11)
    assert random.getstate() == global_rng_state


def test_sample_seed_pairing_is_injective_for_test_grid():
    seeds = {
        _sample_seed(split_seed, sample_index)
        for split_seed in range(10)
        for sample_index in range(10)
    }

    assert len(seeds) == 100


def test_config_requires_disjoint_nonnegative_split_seeds():
    with pytest.raises(ValueError, match="must be non-negative"):
        DensePuzzleConfig(train_seed=-1)
    with pytest.raises(ValueError, match="must differ"):
        DensePuzzleConfig(train_seed=7, validation_seed=7)
