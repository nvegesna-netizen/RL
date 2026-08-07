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

"""Tests for the frozen-policy dense calibration gate."""

import pytest
from turn_level_credit.dense_calibration import evaluate_dense_calibration


def _log(*, turn_weight: float, success_rate: float = 0.25) -> str:
    metric_values = {
        "turn_credit/sample_count": 256,
        "turn_credit/observed_turn_count": 1024,
        "turn_credit/gate/two_plus_trainable_turns_fraction": 0.9,
        "turn_credit/gate/nonzero_intermediate_source_reward_fraction": 0.4,
        "turn_credit/source_reward/positive_fraction": 0.2,
        "turn_credit/source_reward/zero_fraction": 0.6,
        "turn_credit/source_reward/negative_fraction": 0.2,
        "turn_credit/objective_reward/mean": success_rate,
    }
    metric_payload = " ".join(f"{key}={value}" for key, value in metric_values.items())
    return (
        "TURN_CREDIT_DENSE_CALIBRATION_CONFIG "
        f"max_num_steps=0 max_val_samples=256 seed=42 turn_weight={turn_weight} "
        "validation_seed=43000 puzzle_size=4 shuffle_moves=20 max_moves=20 "
        "randomized_solution=1 "
        "credit_uses_progress=1 training_uses_progress=1 "
        "evaluation_uses_success=1 paired_config_sha256=abc123\n"
        f"TURN_CREDIT_ROLLOUT_METRICS {metric_payload}\n"
    )


def test_dense_calibration_passes_all_predeclared_gates():
    result = evaluate_dense_calibration(
        control_log_text=_log(turn_weight=0.0),
        treatment_log_text=_log(turn_weight=0.2),
        control_rollouts=b"same rollouts",
        treatment_rollouts=b"same rollouts",
    )

    assert result["passed"]
    assert all(result["checks"].values())


@pytest.mark.parametrize(
    ("success_rate", "rollouts", "failed_check"),
    [
        (0.01, b"same rollouts", "control_success_not_floor_or_ceiling"),
        (0.90, b"same rollouts", "control_success_not_floor_or_ceiling"),
        (0.25, b"different rollouts", "pre_update_rollouts_match_exactly"),
    ],
)
def test_dense_calibration_rejects_failed_signal_or_pairing(
    success_rate, rollouts, failed_check
):
    result = evaluate_dense_calibration(
        control_log_text=_log(turn_weight=0.0, success_rate=success_rate),
        treatment_log_text=_log(turn_weight=0.2, success_rate=success_rate),
        control_rollouts=b"same rollouts",
        treatment_rollouts=rollouts,
    )

    assert not result["passed"]
    assert not result["checks"][failed_check]


@pytest.mark.parametrize(
    ("control_weight", "treatment_weight", "failed_check"),
    [
        (0.1, 0.2, "control_turn_weight_is_zero"),
        (0.0, 0.0, "treatment_turn_weight_is_positive"),
    ],
)
def test_dense_calibration_requires_control_and_treatment_roles(
    control_weight, treatment_weight, failed_check
):
    result = evaluate_dense_calibration(
        control_log_text=_log(turn_weight=control_weight),
        treatment_log_text=_log(turn_weight=treatment_weight),
        control_rollouts=b"same rollouts",
        treatment_rollouts=b"same rollouts",
    )

    assert not result["passed"]
    assert not result["checks"][failed_check]


def test_dense_calibration_requires_declared_reward_components():
    treatment_log = _log(turn_weight=0.2).replace(
        "evaluation_uses_success=1",
        "evaluation_uses_success=0",
    )
    result = evaluate_dense_calibration(
        control_log_text=_log(turn_weight=0.0),
        treatment_log_text=treatment_log,
        control_rollouts=b"same rollouts",
        treatment_rollouts=b"same rollouts",
    )

    assert not result["passed"]
    assert not result["checks"]["configured_components_are_declared"]


def test_dense_calibration_requires_full_paired_config_match():
    treatment_log = _log(turn_weight=0.2).replace(
        "paired_config_sha256=abc123",
        "paired_config_sha256=def456",
    )
    result = evaluate_dense_calibration(
        control_log_text=_log(turn_weight=0.0),
        treatment_log_text=treatment_log,
        control_rollouts=b"same rollouts",
        treatment_rollouts=b"same rollouts",
    )

    assert not result["passed"]
    assert not result["checks"]["calibration_config_fields_match"]


def test_dense_calibration_rejects_missing_paired_config_fingerprint():
    control_log = _log(turn_weight=0.0).replace(
        " paired_config_sha256=abc123",
        "",
    )
    treatment_log = _log(turn_weight=0.2).replace(
        " paired_config_sha256=abc123",
        "",
    )
    result = evaluate_dense_calibration(
        control_log_text=control_log,
        treatment_log_text=treatment_log,
        control_rollouts=b"same rollouts",
        treatment_rollouts=b"same rollouts",
    )

    assert not result["passed"]
    assert not result["checks"]["calibration_config_fields_match"]


def test_dense_calibration_requires_markers():
    with pytest.raises(ValueError, match="configuration marker"):
        evaluate_dense_calibration(
            control_log_text="",
            treatment_log_text=_log(turn_weight=0.2),
            control_rollouts=b"same",
            treatment_rollouts=b"same",
        )


def test_dense_calibration_rejects_nonfinite_metrics():
    with pytest.raises(ValueError, match="Non-finite"):
        evaluate_dense_calibration(
            control_log_text=_log(turn_weight=0.0).replace(
                "turn_credit/objective_reward/mean=0.25",
                "turn_credit/objective_reward/mean=nan",
            ),
            treatment_log_text=_log(turn_weight=0.2),
            control_rollouts=b"same",
            treatment_rollouts=b"same",
        )
