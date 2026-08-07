"""Tests for the version-independent math-repair calibration launcher."""

from pathlib import Path

import run_grpo_turn_credit
from run_grpo_turn_credit import load_master_and_turn_credit_config
from run_grpo_turn_credit_math_repair import (
    CALIBRATION_GENERATIONS_PER_PROMPT,
    CALIBRATION_PROMPT_COUNT,
    _RepeatedValidationDataset,
    install_repeated_math_validation,
)


class _Dataset:
    def __init__(self, length=2):
        self._rows = [{"idx": index, "payload": [index]} for index in range(length)]

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        return self._rows[index]


def test_repeated_validation_is_contiguous_and_copies_rows():
    source = _Dataset()
    repeated = _RepeatedValidationDataset(source, repeats=3)

    assert len(repeated) == 6
    assert [repeated[index]["payload"] for index in range(6)] == [
        [0],
        [0],
        [0],
        [1],
        [1],
        [1],
    ]
    assert [repeated[index]["idx"] for index in range(6)] == list(range(6))
    first = repeated[0]
    first["payload"].append(99)
    assert repeated[1]["payload"] == [0]
    assert source[0]["payload"] == [0]


def test_repeated_validation_setup_is_scoped():
    original = run_grpo_turn_credit.setup_response_data

    def fake_setup(*args, **kwargs):
        del args, kwargs
        return _Dataset(), _Dataset(), {"task": "env"}, {"task": "val-env"}

    run_grpo_turn_credit.setup_response_data = fake_setup
    try:
        with install_repeated_math_validation():
            result = run_grpo_turn_credit.setup_response_data(None, {}, {})
            assert len(result[1]) == 2 * CALIBRATION_GENERATIONS_PER_PROMPT
        assert run_grpo_turn_credit.setup_response_data is fake_setup
    finally:
        run_grpo_turn_credit.setup_response_data = original


def test_checked_in_math_repair_calibration_is_frozen_and_grouped_explicitly():
    config_path = (
        Path(__file__).parents[2] / "configs" / "grpo_math_repair_calibration.yaml"
    )
    master_config, turn_credit_config = load_master_and_turn_credit_config(
        str(config_path),
        [],
    )

    assert master_config.grpo.max_num_steps == 0
    assert master_config.grpo.max_rollout_turns == 3
    assert master_config.grpo.max_val_samples == (
        CALIBRATION_PROMPT_COUNT * CALIBRATION_GENERATIONS_PER_PROMPT
    )
    assert turn_credit_config.environment_component == "reward/verifier_score"
    assert turn_credit_config.evaluation_environment_component == (
        "reward/terminal_success"
    )
