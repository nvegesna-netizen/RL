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

"""Run the potential-shaped sliding-puzzle turn-credit experiment."""

import hashlib
import json
import os
import sys

from examples import run_grpo_sliding_puzzle
from run_grpo_turn_credit import load_master_and_turn_credit_config, parse_args
from turn_level_credit.dense_puzzle import install_dense_puzzle_runtime
from turn_level_credit.integration import install_turn_credit_runtime

from nemo_rl.algorithms.grpo import MasterConfig


def _paired_config_sha256(master_config: MasterConfig) -> str:
    """Fingerprint all resolved settings except the intended paired differences."""
    paired_config = master_config.model_dump(mode="json")
    turn_credit = paired_config.get("turn_credit")
    if isinstance(turn_credit, dict):
        turn_credit.pop("turn_weight", None)
    logger = paired_config.get("logger")
    if isinstance(logger, dict):
        logger.pop("log_dir", None)
    canonical = json.dumps(
        paired_config,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def main() -> None:
    """Install research runtimes and delegate to the upstream trainer."""
    args, overrides = parse_args()
    config_path = args.config
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__),
            "configs",
            "grpo_dense_sliding_puzzle_trajectory.yaml",
        )
        sys.argv[1:1] = ["--config", config_path]

    master_config, turn_credit_config = load_master_and_turn_credit_config(
        config_path,
        overrides,
    )
    if master_config.grpo.max_num_steps == 0:
        puzzle_config = master_config.env["sliding_puzzle_game"]["cfg"]
        print(
            "TURN_CREDIT_DENSE_CALIBRATION_CONFIG "
            f"max_num_steps={master_config.grpo.max_num_steps} "
            f"max_val_samples={master_config.grpo.max_val_samples} "
            f"seed={master_config.grpo.seed} "
            f"turn_weight={turn_credit_config.turn_weight} "
            f"validation_seed={puzzle_config['validation_seed']} "
            f"puzzle_size={puzzle_config['size']} "
            f"shuffle_moves={puzzle_config['shuffle_moves']} "
            f"max_moves={puzzle_config['max_moves']} "
            f"randomized_solution={int(puzzle_config['randomize_solution'])} "
            "credit_uses_progress="
            f"{int(turn_credit_config.environment_component == 'reward/progress')} "
            "training_uses_progress="
            f"{int(turn_credit_config.macro_environment_component == 'reward/progress')} "
            "evaluation_uses_success="
            f"{int(turn_credit_config.evaluation_environment_component == 'reward/success')} "
            f"paired_config_sha256={_paired_config_sha256(master_config)}",
            flush=True,
        )
    with (
        install_dense_puzzle_runtime(),
        install_turn_credit_runtime(turn_credit_config),
    ):
        run_grpo_sliding_puzzle.main()


if __name__ == "__main__":
    main()
