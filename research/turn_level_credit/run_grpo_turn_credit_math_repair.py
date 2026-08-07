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

import os
import sys

import run_grpo_turn_credit
from run_grpo_turn_credit import load_master_and_turn_credit_config, parse_args
from turn_level_credit.math_repair import (
    TERMINAL_SUCCESS_KEY,
    VERIFIER_SCORE_KEY,
    MathRepairConfig,
    install_math_repair_environment,
)


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
    if master_config.grpo.max_val_samples < 32:
        raise ValueError("math-repair calibration requires at least 32 prompts")
    if master_config.grpo.val_num_generations_per_prompt < 4:
        raise ValueError(
            "math-repair calibration requires at least four generations per prompt"
        )
    if turn_credit_config.environment_component != VERIFIER_SCORE_KEY:
        raise ValueError("math-repair calibration must capture verifier scores")
    if turn_credit_config.evaluation_environment_component != TERMINAL_SUCCESS_KEY:
        raise ValueError("math-repair calibration must evaluate terminal success")
    print(
        "TURN_CREDIT_MATH_REPAIR_CALIBRATION_CONFIG "
        f"max_num_steps={master_config.grpo.max_num_steps} "
        f"max_turns={repair_config.max_turns} "
        f"max_val_prompts={master_config.grpo.max_val_samples} "
        "generations_per_prompt="
        f"{master_config.grpo.val_num_generations_per_prompt} "
        f"seed={master_config.grpo.seed} "
        f"success_threshold={repair_config.success_threshold}",
        flush=True,
    )
    with install_math_repair_environment():
        run_grpo_turn_credit.main()


if __name__ == "__main__":
    main()
