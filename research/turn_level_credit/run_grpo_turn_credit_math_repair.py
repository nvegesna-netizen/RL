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

"""Run frozen-policy fixed-horizon math-repair calibration."""

import copy
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import run_grpo_turn_credit
from run_grpo_turn_credit import load_master_and_turn_credit_config, parse_args
from torch.utils.data import Dataset
from turn_level_credit.math_repair import (
    TERMINAL_SUCCESS_KEY,
    VERIFIER_SCORE_KEY,
    MathRepairConfig,
)
from turn_level_credit.math_repair_runtime import install_math_repair_environment

CALIBRATION_GENERATIONS_PER_PROMPT = 8
CALIBRATION_PROMPT_COUNT = 32


class _RepeatedValidationDataset(Dataset):
    """Repeat each validation prompt contiguously for grouped credit analysis."""

    def __init__(self, dataset: Any, *, repeats: int) -> None:
        if repeats < 2:
            raise ValueError("math-repair validation repeats must be at least two")
        self._dataset = dataset
        self._repeats = repeats

    def __len__(self) -> int:
        return len(self._dataset) * self._repeats

    def __getitem__(self, index: int) -> Any:
        if not 0 <= index < len(self):
            raise IndexError(index)
        datum = copy.deepcopy(self._dataset[index // self._repeats])
        if not isinstance(datum, dict):
            raise TypeError("math-repair validation datum must be a dictionary")
        datum["idx"] = index
        return datum


@contextmanager
def install_repeated_math_validation() -> Iterator[None]:
    """Scope validation repetition to this research entrypoint."""
    original_setup = run_grpo_turn_credit.setup_response_data

    def setup_with_repeated_validation(*args: Any, **kwargs: Any) -> Any:
        result = original_setup(*args, **kwargs)
        if len(result) != 4:
            raise RuntimeError("math-repair setup requires native environments")
        dataset, val_dataset, task_to_env, val_task_to_env = result
        if val_dataset is None:
            raise ValueError("math-repair calibration requires validation data")
        repeated_validation = _RepeatedValidationDataset(
            val_dataset,
            repeats=CALIBRATION_GENERATIONS_PER_PROMPT,
        )
        return dataset, repeated_validation, task_to_env, val_task_to_env

    run_grpo_turn_credit.setup_response_data = setup_with_repeated_validation
    try:
        yield
    finally:
        run_grpo_turn_credit.setup_response_data = original_setup


def main() -> None:
    """Validate the calibration protocol, install the environment, and run."""
    args, overrides = parse_args()
    config_path = args.config
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__),
            "configs",
            "grpo_math_repair_calibration.yaml",
        )
        sys.argv[1:1] = ["--config", config_path]

    master_config, turn_credit_config = load_master_and_turn_credit_config(
        config_path,
        overrides,
    )
    repair_config = MathRepairConfig.model_validate(master_config.env["math"])
    if master_config.grpo.max_num_steps != 0:
        raise ValueError("math-repair calibration must use a frozen policy")
    if master_config.grpo.max_rollout_turns != repair_config.max_turns:
        raise ValueError(
            "grpo.max_rollout_turns must equal the math-repair fixed horizon"
        )
    if not master_config.grpo.val_at_start:
        raise ValueError("math-repair calibration requires validation at start")
    minimum_validation_samples = (
        CALIBRATION_PROMPT_COUNT * CALIBRATION_GENERATIONS_PER_PROMPT
    )
    if (
        master_config.grpo.max_val_samples is None
        or master_config.grpo.max_val_samples < minimum_validation_samples
    ):
        raise ValueError(
            "math-repair calibration requires at least 256 repeated validation samples"
        )
    if turn_credit_config.environment_component != VERIFIER_SCORE_KEY:
        raise ValueError("math-repair calibration must capture verifier scores")
    if turn_credit_config.evaluation_environment_component != TERMINAL_SUCCESS_KEY:
        raise ValueError("math-repair calibration must evaluate terminal success")
    print(
        "TURN_CREDIT_MATH_REPAIR_CALIBRATION_CONFIG "
        f"max_num_steps={master_config.grpo.max_num_steps} "
        f"max_turns={repair_config.max_turns} "
        f"max_val_prompts={CALIBRATION_PROMPT_COUNT} "
        "generations_per_prompt="
        f"{CALIBRATION_GENERATIONS_PER_PROMPT} "
        f"seed={master_config.grpo.seed} "
        f"success_threshold={repair_config.success_threshold}",
        flush=True,
    )
    with install_math_repair_environment(), install_repeated_math_validation():
        run_grpo_turn_credit.main()


if __name__ == "__main__":
    main()
