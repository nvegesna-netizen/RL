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

"""Mechanical acceptance gates for frozen-policy dense-puzzle calibration."""

from dataclasses import dataclass
from math import isfinite
from typing import Any

METRIC_MARKER = "TURN_CREDIT_ROLLOUT_METRICS "
CONFIG_MARKER = "TURN_CREDIT_DENSE_CALIBRATION_CONFIG "


@dataclass(frozen=True)
class DenseCalibrationThresholds:
    """Predeclared dense-signal acceptance thresholds."""

    minimum_samples: int = 256
    minimum_two_plus_trainable_turns_fraction: float = 0.80
    minimum_nonzero_intermediate_fraction: float = 0.30


def _parse_key_values(payload: str, *, marker: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in payload.split():
        if "=" not in item:
            raise ValueError(f"Malformed {marker.strip()} item: {item!r}")
        key, raw_value = item.split("=", 1)
        try:
            value = float(raw_value)
        except ValueError as error:
            raise ValueError(
                f"Non-numeric {marker.strip()} value for {key!r}: {raw_value!r}"
            ) from error
        if not isfinite(value):
            raise ValueError(
                f"Non-finite {marker.strip()} value for {key!r}: {raw_value!r}"
            )
        values[key] = value
    return values


def parse_calibration_log(
    log_text: str,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Parse one calibration config marker and its rollout metric records."""
    config_records = []
    metric_records = []
    for line in log_text.splitlines():
        if CONFIG_MARKER in line:
            config_records.append(
                _parse_key_values(
                    line.split(CONFIG_MARKER, 1)[1],
                    marker=CONFIG_MARKER,
                )
            )
        if METRIC_MARKER in line:
            metric_records.append(
                _parse_key_values(
                    line.split(METRIC_MARKER, 1)[1],
                    marker=METRIC_MARKER,
                )
            )
    if len(config_records) != 1:
        raise ValueError(
            "Calibration log must contain exactly one configuration marker, "
            f"found {len(config_records)}"
        )
    if not metric_records:
        raise ValueError("Calibration log contains no rollout metric records")
    return config_records[0], metric_records


def _weighted_mean(
    records: list[dict[str, float]],
    value_key: str,
    weight_key: str,
) -> float:
    missing = [
        index
        for index, record in enumerate(records)
        if value_key not in record or weight_key not in record
    ]
    if missing:
        raise ValueError(
            f"Calibration metric {value_key!r} or weight {weight_key!r} "
            f"is missing from records {missing}"
        )
    total_weight = sum(record[weight_key] for record in records)
    if total_weight <= 0:
        raise ValueError(f"Calibration weight {weight_key!r} must sum to positive")
    return (
        sum(record[value_key] * record[weight_key] for record in records) / total_weight
    )


def summarize_calibration_records(
    records: list[dict[str, float]],
) -> dict[str, float]:
    """Aggregate batch metrics with their correct sample or turn weights."""
    sample_count = sum(
        record.get("turn_credit/sample_count", 0.0) for record in records
    )
    observed_turn_count = sum(
        record.get("turn_credit/observed_turn_count", 0.0) for record in records
    )
    return {
        "sample_count": sample_count,
        "observed_turn_count": observed_turn_count,
        "two_plus_trainable_turns_fraction": _weighted_mean(
            records,
            "turn_credit/gate/two_plus_trainable_turns_fraction",
            "turn_credit/sample_count",
        ),
        "nonzero_intermediate_source_reward_fraction": _weighted_mean(
            records,
            "turn_credit/gate/nonzero_intermediate_source_reward_fraction",
            "turn_credit/sample_count",
        ),
        "positive_source_reward_fraction": _weighted_mean(
            records,
            "turn_credit/source_reward/positive_fraction",
            "turn_credit/observed_turn_count",
        ),
        "zero_source_reward_fraction": _weighted_mean(
            records,
            "turn_credit/source_reward/zero_fraction",
            "turn_credit/observed_turn_count",
        ),
        "negative_source_reward_fraction": _weighted_mean(
            records,
            "turn_credit/source_reward/negative_fraction",
            "turn_credit/observed_turn_count",
        ),
        "success_rate": _weighted_mean(
            records,
            "turn_credit/objective_reward/mean",
            "turn_credit/sample_count",
        ),
    }


def evaluate_dense_calibration(
    *,
    control_log_text: str,
    treatment_log_text: str,
    control_rollouts: bytes,
    treatment_rollouts: bytes,
    thresholds: DenseCalibrationThresholds = DenseCalibrationThresholds(),
) -> dict[str, Any]:
    """Evaluate both frozen-policy arms and exact pre-update rollout matching."""
    control_config, control_records = parse_calibration_log(control_log_text)
    treatment_config, treatment_records = parse_calibration_log(treatment_log_text)
    control = summarize_calibration_records(control_records)
    treatment = summarize_calibration_records(treatment_records)

    config_match_fields = ("max_num_steps", "max_val_samples", "seed")
    config_fields_match = all(
        control_config.get(field) == treatment_config.get(field)
        for field in config_match_fields
    )
    checks = {
        "control_is_frozen_policy": control_config.get("max_num_steps") == 0,
        "treatment_is_frozen_policy": treatment_config.get("max_num_steps") == 0,
        "calibration_config_fields_match": config_fields_match,
        "turn_weights_differ": control_config.get("turn_weight")
        != treatment_config.get("turn_weight"),
        "control_sample_count": control["sample_count"] >= thresholds.minimum_samples,
        "treatment_sample_count": treatment["sample_count"]
        >= thresholds.minimum_samples,
        "control_two_plus_trainable_turns": control["two_plus_trainable_turns_fraction"]
        >= thresholds.minimum_two_plus_trainable_turns_fraction,
        "treatment_two_plus_trainable_turns": treatment[
            "two_plus_trainable_turns_fraction"
        ]
        >= thresholds.minimum_two_plus_trainable_turns_fraction,
        "control_nonzero_intermediate": control[
            "nonzero_intermediate_source_reward_fraction"
        ]
        >= thresholds.minimum_nonzero_intermediate_fraction,
        "treatment_nonzero_intermediate": treatment[
            "nonzero_intermediate_source_reward_fraction"
        ]
        >= thresholds.minimum_nonzero_intermediate_fraction,
        "control_all_reward_signs": all(
            control[key] > 0
            for key in (
                "positive_source_reward_fraction",
                "zero_source_reward_fraction",
                "negative_source_reward_fraction",
            )
        ),
        "treatment_all_reward_signs": all(
            treatment[key] > 0
            for key in (
                "positive_source_reward_fraction",
                "zero_source_reward_fraction",
                "negative_source_reward_fraction",
            )
        ),
        "control_success_not_floor_or_ceiling": 0.0 < control["success_rate"] < 1.0,
        "treatment_success_not_floor_or_ceiling": 0.0 < treatment["success_rate"] < 1.0,
        "pre_update_rollouts_match_exactly": control_rollouts == treatment_rollouts,
    }
    return {
        "passed": all(checks.values()),
        "thresholds": {
            "minimum_samples": thresholds.minimum_samples,
            "minimum_two_plus_trainable_turns_fraction": thresholds.minimum_two_plus_trainable_turns_fraction,
            "minimum_nonzero_intermediate_fraction": thresholds.minimum_nonzero_intermediate_fraction,
        },
        "control": control,
        "treatment": treatment,
        "checks": checks,
    }
