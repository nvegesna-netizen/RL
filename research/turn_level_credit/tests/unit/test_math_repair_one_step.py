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

"""Tests for the one-step-only verifier-credit engineering protocol."""

from pathlib import Path

import pytest
from run_grpo_turn_credit import load_master_and_turn_credit_config
from run_grpo_turn_credit_math_repair_one_step import validate_one_step_protocol


CONFIG_PATH = Path(__file__).parents[2] / "configs" / "grpo_math_repair_one_step.yaml"


def _load(overrides: list[str] | None = None):
    return load_master_and_turn_credit_config(
        str(CONFIG_PATH),
        [] if overrides is None else overrides,
    )


def test_checked_in_one_step_protocol_is_exact_and_online():
    master_config, turn_credit_config = _load()

    validate_one_step_protocol(master_config, turn_credit_config)
    assert master_config.grpo.max_num_steps == 1
    assert master_config.grpo.val_at_start
    assert master_config.grpo.val_at_end
    assert not master_config.grpo.use_dynamic_sampling
    assert turn_credit_config.turn_weight == 0.0
    assert turn_credit_config.verifier_transform is not None
    assert turn_credit_config.verifier_transform.mode == "retrospective_hindsight"
    assert turn_credit_config.verifier_transform.normalization == "group_zscore"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("grpo.max_num_steps=2", "exactly one step"),
        ("grpo.num_generations_per_prompt=4", "eight-way groups"),
        ("grpo.val_at_end=false", "start/end validation"),
        ("grpo.max_val_samples=255", "256 validation samples"),
        (
            "turn_credit.environment_component=reward/other",
            "capture verifier scores",
        ),
        (
            "turn_credit.verifier_transform.mode=raw",
            "retrospective_hindsight mode",
        ),
        (
            "turn_credit.verifier_transform.normalization=none",
            "group_zscore",
        ),
    ],
)
def test_one_step_protocol_rejects_drift(override, message):
    master_config, turn_credit_config = _load([override])

    with pytest.raises(ValueError, match=message):
        validate_one_step_protocol(master_config, turn_credit_config)
