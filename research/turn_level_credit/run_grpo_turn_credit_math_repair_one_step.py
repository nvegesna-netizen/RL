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

"""Run the one-step-only online verifier-credit engineering gate."""

import os
import sys
from typing import TYPE_CHECKING

import run_grpo_turn_credit
from run_grpo_turn_credit import load_master_and_turn_credit_config, parse_args
from run_grpo_turn_credit_math_repair import (
    CALIBRATION_GENERATIONS_PER_PROMPT,
    CALIBRATION_PROMPT_COUNT,
    install_repeated_math_validation,
)
from turn_level_credit.config import TurnCreditConfig
from turn_level_credit.math_repair import (
    TERMINAL_SUCCESS_KEY,
    VERIFIER_SCORE_KEY,
    MathRepairConfig,
)

if TYPE_CHECKING:
    from nemo_rl.algorithms.grpo import MasterConfig


def validate_one_step_protocol(
    master_config: "MasterConfig",
    turn_credit_config: TurnCreditConfig,
) -> None:
    """Reject any drift from the preregistered engineering-only protocol."""
    repair_config = MathRepairConfig.model_validate(master_config.env["math"])
    if master_config.grpo.max_num_steps != 1:
        raise ValueError("math-repair engineering gate requires exactly one step")
    if master_config.grpo.max_rollout_turns != repair_config.max_turns:
        raise ValueError(
            "grpo.max_rollout_turns must equal the math-repair fixed horizon"
        )
    if master_config.grpo.num_generations_per_prompt != 8:
        raise ValueError("math-repair engineering gate requires eight-way groups")
    if master_config.grpo.use_dynamic_sampling:
        raise ValueError("math-repair engineering gate disables dynamic sampling")
    if not master_config.grpo.val_at_start or not master_config.grpo.val_at_end:
        raise ValueError("math-repair engineering gate requires start/end validation")
    expected_validation_samples = (
        CALIBRATION_PROMPT_COUNT * CALIBRATION_GENERATIONS_PER_PROMPT
    )
    if master_config.grpo.max_val_samples != expected_validation_samples:
        raise ValueError("math-repair engineering gate requires 256 validation samples")
    if turn_credit_config.environment_component != VERIFIER_SCORE_KEY:
        raise ValueError("math-repair engineering gate must capture verifier scores")
    if turn_credit_config.evaluation_environment_component != TERMINAL_SUCCESS_KEY:
        raise ValueError("math-repair engineering gate must evaluate terminal success")
    transform = turn_credit_config.verifier_transform
    if transform is None:
        raise ValueError("math-repair engineering gate requires a verifier transform")
    if transform.mode != "retrospective_hindsight":
        raise ValueError(
            "math-repair engineering gate requires retrospective_hindsight mode"
        )
    if transform.normalization != "group_zscore":
        raise ValueError("math-repair engineering gate requires group_zscore")


def main() -> None:
    """Validate, install the fixed-horizon environment, and run one update."""
    from turn_level_credit.math_repair_runtime import install_math_repair_environment

    args, overrides = parse_args()
    config_path = args.config
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__),
            "configs",
            "grpo_math_repair_one_step.yaml",
        )
        sys.argv[1:1] = ["--config", config_path]

    master_config, turn_credit_config = load_master_and_turn_credit_config(
        config_path,
        overrides,
    )
    validate_one_step_protocol(master_config, turn_credit_config)
    transform = turn_credit_config.verifier_transform
    assert transform is not None
    print(
        "TURN_CREDIT_MATH_REPAIR_ONE_STEP_CONFIG "
        f"max_num_steps={master_config.grpo.max_num_steps} "
        f"max_turns={master_config.grpo.max_rollout_turns} "
        f"generations_per_prompt={master_config.grpo.num_generations_per_prompt} "
        f"max_val_samples={master_config.grpo.max_val_samples} "
        f"seed={master_config.grpo.seed} "
        f"turn_weight={turn_credit_config.turn_weight} "
        f"mode={transform.mode} "
        f"normalization={transform.normalization}",
        flush=True,
    )
    with install_math_repair_environment(), install_repeated_math_validation():
        run_grpo_turn_credit.main()


if __name__ == "__main__":
    main()
