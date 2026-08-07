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

"""Research-only sliding puzzle with potential-based dense rewards."""

import itertools
import random
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import ray
import torch
from pydantic import BaseModel, ConfigDict, model_validator
from torch.utils.data import IterableDataset

from examples import run_grpo_sliding_puzzle
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType, TokenizerType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.games.sliding_puzzle import (
    SlidingPuzzleGameLogic,
    SlidingPuzzleMetadata,
    SlidingPuzzleRunner,
)
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

PROGRESS_REWARD_KEY = "reward/progress"
SUCCESS_REWARD_KEY = "reward/success"


class DensePuzzleConfig(BaseModel):
    """User-facing configuration for the dense puzzle experiment."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    size: int = 3
    shuffle_moves: int = 11
    minimum_manhattan_distance: int = 2
    max_generation_attempts: int = 100
    max_moves: int = 12
    progress_scale: float = 1.0
    train_seed: int = 42_000
    validation_seed: int = 43_000

    @model_validator(mode="after")
    def _validate_ranges(self) -> "DensePuzzleConfig":
        if self.size < 2:
            raise ValueError("dense puzzle size must be at least 2")
        if self.shuffle_moves < 1:
            raise ValueError("dense puzzle shuffle_moves must be positive")
        if self.minimum_manhattan_distance < 1:
            raise ValueError("dense puzzle minimum_manhattan_distance must be positive")
        max_distance = (self.size * self.size - 1) * 2 * (self.size - 1)
        if self.minimum_manhattan_distance > max_distance:
            raise ValueError(
                "dense puzzle minimum_manhattan_distance exceeds its safe upper bound"
            )
        if self.max_generation_attempts < 1:
            raise ValueError("dense puzzle max_generation_attempts must be positive")
        if self.max_moves < 1:
            raise ValueError("dense puzzle max_moves must be positive")
        if self.progress_scale <= 0.0:
            raise ValueError("dense puzzle progress_scale must be positive")
        if self.train_seed < 0 or self.validation_seed < 0:
            raise ValueError("dense puzzle dataset seeds must be non-negative")
        if self.train_seed == self.validation_seed:
            raise ValueError("dense puzzle train and validation seeds must differ")
        return self


def manhattan_distance(game_state: dict[str, Any]) -> int:
    """Return numbered-tile Manhattan distance from the configured solution."""
    size = int(game_state["size"])
    grid = game_state["grid"]
    solution = game_state["solution"]
    if size < 2:
        raise ValueError("sliding puzzle state size must be at least 2")
    if len(grid) != size or len(solution) != size:
        raise ValueError("sliding puzzle grid and solution must match state size")
    if any(len(row) != size for row in grid) or any(
        len(row) != size for row in solution
    ):
        raise ValueError("sliding puzzle grid and solution must be square")

    grid_tiles = [int(tile) for row in grid for tile in row]
    solution_tiles = [int(tile) for row in solution for tile in row]
    expected_tiles = set(range(size * size))
    if set(grid_tiles) != expected_tiles or len(set(grid_tiles)) != size * size:
        raise ValueError("sliding puzzle grid must contain each tile exactly once")
    if set(solution_tiles) != expected_tiles or len(set(solution_tiles)) != size * size:
        raise ValueError("sliding puzzle solution must contain each tile exactly once")

    goal_positions = {
        int(tile): (row_index, column_index)
        for row_index, row in enumerate(solution)
        for column_index, tile in enumerate(row)
    }
    distance = 0
    for row_index, row in enumerate(grid):
        for column_index, raw_tile in enumerate(row):
            tile = int(raw_tile)
            if tile == 0:
                continue
            goal_row, goal_column = goal_positions[tile]
            distance += abs(row_index - goal_row) + abs(column_index - goal_column)
    return distance


def normalized_manhattan_potential(game_state: dict[str, Any]) -> float:
    """Return a bounded potential where one means the puzzle is solved."""
    size = int(game_state["size"])
    safe_max_distance = (size * size - 1) * 2 * (size - 1)
    distance = manhattan_distance(game_state)
    potential = 1.0 - distance / safe_max_distance
    if not 0.0 <= potential <= 1.0:
        raise ValueError("normalized Manhattan potential must be in [0, 1]")
    return potential


class DenseSlidingPuzzleRunner:
    """Delegate puzzle transitions and convert state progress into rewards."""

    def __init__(self, config: DensePuzzleConfig) -> None:
        self._config = config
        self._base_runner = SlidingPuzzleRunner()

    def process_turn(
        self,
        message_log: LLMMessageLogType,
        metadata: SlidingPuzzleMetadata,
    ) -> tuple[
        dict[str, str],
        dict[str, float],
        bool,
        list[str] | None,
        SlidingPuzzleMetadata | None,
        str | None,
    ]:
        """Run one transition and emit progress plus terminal success."""
        before_potential = normalized_manhattan_potential(metadata["game_state"])
        (
            observation,
            sparse_reward,
            terminated,
            stop_strings,
            next_metadata,
            runner_answers,
        ) = self._base_runner.process_turn(message_log, metadata)

        if sparse_reward == 1.0:
            after_potential = 1.0
        elif next_metadata is not None:
            after_potential = normalized_manhattan_potential(
                next_metadata["game_state"]
            )
        else:
            # Invalid formatting and max-move termination preserve the state.
            after_potential = before_potential

        rewards = {
            PROGRESS_REWARD_KEY: self._config.progress_scale
            * (after_potential - before_potential),
            SUCCESS_REWARD_KEY: float(sparse_reward == 1.0),
        }
        return (
            observation,
            rewards,
            terminated,
            stop_strings,
            next_metadata,
            runner_answers[0] if runner_answers else None,
        )


@ray.remote  # pragma: no cover
class DenseSlidingPuzzleEnv(EnvironmentInterface[SlidingPuzzleMetadata | None]):
    """Batched Ray environment for the dense puzzle experiment."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._config = DensePuzzleConfig.model_validate(cfg)
        self._runner = DenseSlidingPuzzleRunner(self._config)

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[SlidingPuzzleMetadata | None],
    ) -> EnvironmentReturn[SlidingPuzzleMetadata | None]:
        """Process a batch and retain named reward components."""
        if len(message_log_batch) != len(metadata):
            raise ValueError("dense puzzle message and metadata batches must align")
        results = []
        for row, (message_log, sample_metadata) in enumerate(
            zip(message_log_batch, metadata, strict=True)
        ):
            if sample_metadata is None:
                raise ValueError(
                    f"dense puzzle row {row} has no metadata before termination"
                )
            results.append(self._runner.process_turn(message_log, sample_metadata))
        return EnvironmentReturn(
            observations=[result[0] for result in results],
            rewards={
                reward_name: torch.tensor(
                    [result[1][reward_name] for result in results],
                    dtype=torch.float32,
                )
                for reward_name in (PROGRESS_REWARD_KEY, SUCCESS_REWARD_KEY)
            },
            terminateds=torch.tensor(
                [result[2] for result in results], dtype=torch.bool
            ),
            next_stop_strings=[result[3] for result in results],
            metadata=[result[4] for result in results],
            answers=[result[5] for result in results],
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict[str, float]]:
        """Report success independently from the shaped trajectory reward."""
        if SUCCESS_REWARD_KEY not in batch or PROGRESS_REWARD_KEY not in batch:
            raise ValueError("dense puzzle batch is missing named reward components")
        success = batch[SUCCESS_REWARD_KEY].float()
        progress = batch[PROGRESS_REWARD_KEY].float()
        return batch, {
            "dense_puzzle_success_rate": float(success.mean().item()),
            "dense_puzzle_progress/mean": float(progress.mean().item()),
            "dense_puzzle_progress/std": float(progress.std(unbiased=False).item()),
        }

    def shutdown(self) -> None:
        """Release environment resources."""


def _sample_seed(split_seed: int, sample_index: int) -> int:
    """Map a non-negative split seed and sample index to a unique integer."""
    if sample_index < 0:
        raise ValueError("dense puzzle sample index must be non-negative")
    diagonal = split_seed + sample_index
    return diagonal * (diagonal + 1) // 2 + sample_index


def _generate_puzzle_state(
    *,
    size: int,
    shuffle_moves: int,
    rng: random.Random,
) -> dict[str, Any]:
    """Generate one puzzle without reading or mutating process-global RNG state."""
    grid = [[row * size + column + 1 for column in range(size)] for row in range(size)]
    grid[-1][-1] = 0
    solution = [row[:] for row in grid]
    empty_row, empty_column = size - 1, size - 1
    for _ in range(shuffle_moves):
        valid_positions = [
            (row, column)
            for row, column in (
                (empty_row, empty_column + 1),
                (empty_row + 1, empty_column),
                (empty_row, empty_column - 1),
                (empty_row - 1, empty_column),
            )
            if 0 <= row < size and 0 <= column < size
        ]
        next_row, next_column = rng.choice(valid_positions)
        grid[empty_row][empty_column], grid[next_row][next_column] = (
            grid[next_row][next_column],
            grid[empty_row][empty_column],
        )
        empty_row, empty_column = next_row, next_column
    return {
        "size": size,
        "grid": grid,
        "solution": solution,
        "empty_pos": (empty_row, empty_column),
        "commands": {
            "up": "Slide tile below empty space up",
            "down": "Slide tile above empty space down",
            "left": "Slide tile to the right of empty space left",
            "right": "Slide tile to the left of empty space right",
            "view": "View the current state of the board",
        },
    }


def _generate_initial_state(
    config: DensePuzzleConfig,
    *,
    split_seed: int,
    sample_index: int,
) -> dict[str, Any]:
    """Generate a deterministic nonsolved state satisfying the distance gate."""
    rng = random.Random(_sample_seed(split_seed, sample_index))
    for _attempt in range(config.max_generation_attempts):
        game_state = _generate_puzzle_state(
            size=config.size,
            shuffle_moves=config.shuffle_moves,
            rng=rng,
        )
        if manhattan_distance(game_state) >= config.minimum_manhattan_distance:
            return game_state
    raise RuntimeError(
        "failed to generate a sliding puzzle satisfying "
        "minimum_manhattan_distance within max_generation_attempts"
    )


def generate_dense_puzzle_datum(
    *,
    tokenizer: TokenizerType,
    config: DensePuzzleConfig,
    task_name: str,
    idx: int,
    split_seed: int,
    add_system_prompt: bool,
) -> DatumSpec:
    """Generate one fixed-size dense-puzzle training datum."""
    initial_game_state = _generate_initial_state(
        config,
        split_seed=split_seed,
        sample_index=idx,
    )
    initial_render = SlidingPuzzleGameLogic.render(initial_game_state)
    welcome_message = SlidingPuzzleGameLogic.init(initial_game_state)
    prompt_instructions = (
        f"{welcome_message}\n\n"
        f"Current Board State:\n{initial_render}\n\n"
        f"Reach the goal state where numbers are ordered 1 through "
        f"{config.size**2 - 1} with the empty space (0) at the bottom right.\n"
        "Valid actions: 'up', 'down', 'left', 'right', or 'slide row col' "
        "(e.g., 'slide 1 2').\n"
        "Respond with exactly one action tag and no other text, for example "
        "<action>up</action>.\n"
    )
    initial_prompt_content = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_instructions}],
        tokenize=False,
        add_system_prompt=add_system_prompt,
        add_generation_prompt=True,
        add_special_tokens=False,
    ).strip()
    tokenized_prompt = tokenizer(
        initial_prompt_content,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]
    message_log: LLMMessageLogType = [
        {
            "role": "user",
            "content": initial_prompt_content,
            "token_ids": tokenized_prompt,
        }
    ]
    metadata = SlidingPuzzleMetadata(
        game_state=initial_game_state,
        num_moves=0,
        max_moves=config.max_moves,
    )
    return {
        "message_log": message_log,
        "length": len(tokenized_prompt),
        "extra_env_info": dict(metadata),
        "loss_multiplier": 1.0,
        "idx": idx,
        "task_name": task_name,
        "stop_strings": ["</action>"],
    }


class DensePuzzleDataset(IterableDataset):
    """Generate deterministic-shape puzzle samples indefinitely."""

    def __init__(
        self,
        *,
        tokenizer: TokenizerType,
        config: DensePuzzleConfig,
        task_name: str,
        add_system_prompt: bool,
        length: int,
        split_seed: int,
    ) -> None:
        super().__init__()
        self._tokenizer = tokenizer
        self._config = config
        self._task_name = task_name
        self._add_system_prompt = add_system_prompt
        self._length = length
        self._split_seed = split_seed

    def __iter__(self) -> Iterator[DatumSpec]:
        for idx in itertools.count():
            yield generate_dense_puzzle_datum(
                tokenizer=self._tokenizer,
                config=self._config,
                task_name=self._task_name,
                idx=idx,
                split_seed=self._split_seed,
                add_system_prompt=self._add_system_prompt,
            )

    def __len__(self) -> int:
        return self._length


def setup_dense_puzzle_data(
    tokenizer: TokenizerType,
    env_cfg: dict[str, Any],
    task_name: str,
    length: int,
    val_length: int,
    add_system_prompt: bool,
) -> tuple[IterableDataset, IterableDataset, dict[str, Any], dict[str, Any]]:
    """Build datasets and environment for the research-only dense task."""
    if task_name not in env_cfg:
        raise ValueError(f"missing environment configuration for {task_name!r}")
    task_config = env_cfg[task_name]
    if "cfg" not in task_config:
        raise ValueError(f"environment {task_name!r} is missing cfg")
    config = DensePuzzleConfig.model_validate(task_config["cfg"])
    env = DenseSlidingPuzzleEnv.options(  # type: ignore # decorated with @ray.remote
        num_gpus=0
    ).remote(cfg=config.model_dump())
    training_dataset = DensePuzzleDataset(
        tokenizer=tokenizer,
        config=config,
        task_name=task_name,
        add_system_prompt=add_system_prompt,
        length=length,
        split_seed=config.train_seed,
    )
    validation_dataset = DensePuzzleDataset(
        tokenizer=tokenizer,
        config=config,
        task_name=task_name,
        add_system_prompt=add_system_prompt,
        length=val_length,
        split_seed=config.validation_seed,
    )
    task_to_env = {task_name: env}
    return training_dataset, validation_dataset, task_to_env, task_to_env


@contextmanager
def install_dense_puzzle_runtime() -> Iterator[None]:
    """Scope the upstream runner's data factory replacement to one run."""
    original_setup_puzzle_data = run_grpo_sliding_puzzle.setup_puzzle_data
    run_grpo_sliding_puzzle.setup_puzzle_data = setup_dense_puzzle_data
    try:
        yield
    finally:
        run_grpo_sliding_puzzle.setup_puzzle_data = original_setup_puzzle_data
