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

"""Tests for turn-level credit configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError
from run_grpo_turn_credit import load_master_and_turn_credit_config
from run_grpo_turn_credit_dense_puzzle import _paired_config_sha256
from turn_level_credit.config import TurnCreditConfig


def test_config_defaults_are_macro_only():
    config = TurnCreditConfig()

    assert not config.enabled
    assert config.source == "environment"
    assert config.environment_component is None
    assert config.macro_environment_component is None
    assert config.evaluation_environment_component is None
    assert config.environment_mode == "immediate"
    assert config.macro_weight == 1.0
    assert config.turn_weight == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("discount", -0.1),
        ("discount", 1.1),
        ("macro_weight", -1.0),
        ("turn_weight", -1.0),
        ("raw_reward_atol", -1.0),
    ],
)
def test_config_rejects_invalid_numeric_ranges(field, value):
    with pytest.raises(ValidationError):
        TurnCreditConfig.model_validate({field: value})


def test_config_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        TurnCreditConfig.model_validate({"silent_fallback": True})


@pytest.mark.parametrize(
    "field",
    [
        "environment_component",
        "macro_environment_component",
        "evaluation_environment_component",
    ],
)
def test_config_rejects_blank_component_names(field):
    with pytest.raises(ValidationError):
        TurnCreditConfig.model_validate({field: "  "})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("discount", float("nan")),
        ("discount", float("inf")),
        ("discount", float("-inf")),
        ("macro_weight", float("nan")),
        ("macro_weight", float("inf")),
        ("macro_weight", float("-inf")),
        ("turn_weight", float("nan")),
        ("turn_weight", float("inf")),
        ("turn_weight", float("-inf")),
        ("raw_reward_atol", float("nan")),
        ("raw_reward_atol", float("inf")),
        ("raw_reward_atol", float("-inf")),
    ],
)
def test_config_rejects_non_finite_numbers(field, value):
    with pytest.raises(ValidationError):
        TurnCreditConfig.model_validate({field: value})


def test_sliding_puzzle_pilot_is_multi_turn_and_macro_only_by_default():
    config_path = (
        Path(__file__).parents[2] / "configs" / "grpo_sliding_puzzle_turn_credit.yaml"
    )

    master_config, turn_credit_config = load_master_and_turn_credit_config(
        str(config_path),
        [],
    )

    assert master_config.grpo.max_rollout_turns == 6
    assert master_config.grpo.max_num_steps == 10
    assert master_config.policy["train_global_batch_size"] == 16
    assert turn_credit_config.enabled
    assert turn_credit_config.turn_weight == 0.0


def test_dense_puzzle_control_separates_objectives_and_localized_signal():
    config_path = (
        Path(__file__).parents[2]
        / "configs"
        / "grpo_dense_sliding_puzzle_trajectory.yaml"
    )

    master_config, turn_credit_config = load_master_and_turn_credit_config(
        str(config_path),
        [],
    )

    assert master_config.grpo.max_rollout_turns == 12
    puzzle_config = master_config.env["sliding_puzzle_game"]["cfg"]
    assert puzzle_config["size"] == 3
    assert puzzle_config["shuffle_moves"] == 3
    assert puzzle_config["max_moves"] == 6
    assert puzzle_config["randomize_solution"] is True
    assert turn_credit_config.environment_component == "reward/progress"
    assert turn_credit_config.macro_environment_component == "reward/progress"
    assert turn_credit_config.evaluation_environment_component == "reward/success"
    assert turn_credit_config.turn_weight == 0.0


def test_dense_puzzle_calibration_is_frozen_policy_only():
    config_path = (
        Path(__file__).parents[2]
        / "configs"
        / "grpo_dense_sliding_puzzle_calibration.yaml"
    )

    master_config, turn_credit_config = load_master_and_turn_credit_config(
        str(config_path),
        [],
    )

    assert master_config.grpo.max_num_steps == 0
    assert master_config.grpo.val_at_start
    assert not master_config.grpo.val_at_end
    assert master_config.grpo.max_val_samples == 256
    assert turn_credit_config.turn_weight == 0.0


def test_dense_calibration_fingerprint_excludes_only_intended_pair_differences():
    config_path = (
        Path(__file__).parents[2]
        / "configs"
        / "grpo_dense_sliding_puzzle_calibration.yaml"
    )
    control, _ = load_master_and_turn_credit_config(str(config_path), [])
    treatment, _ = load_master_and_turn_credit_config(
        str(config_path),
        ["turn_credit.turn_weight=0.2", "logger.log_dir=other-log-dir"],
    )
    different_seed, _ = load_master_and_turn_credit_config(
        str(config_path),
        ["grpo.seed=43"],
    )

    assert _paired_config_sha256(control) == _paired_config_sha256(treatment)
    assert _paired_config_sha256(control) != _paired_config_sha256(different_seed)
