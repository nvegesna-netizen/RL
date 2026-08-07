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

"""Tests for frozen fixed-horizon math-repair calibration gates."""

import pytest
from turn_level_credit.math_repair_calibration import (
    evaluate_math_repair_calibration,
    parse_math_repair_validation_records,
)


def _conversation(scores, prompt):
    messages = [{"role": "user", "content": prompt}]
    for score in scores:
        messages.extend(
            [
                {"role": "assistant", "content": "answer"},
                {
                    "role": "environment",
                    "content": (
                        "Verifier: correct. Re-check the solution."
                        if score
                        else "Verifier: incorrect. Recompute carefully."
                    ),
                },
            ]
        )
    return messages


def _records(patterns=None):
    if patterns is None:
        patterns = (
            [0, 1, 1],
            [0, 0, 0],
            [1, 1, 1],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        )
    rows = []
    for prompt in range(32):
        for scores in patterns:
            rows.append(
                {
                    "idx": len(rows),
                    "content": [_conversation(scores, f"problem {prompt}")],
                    "rewards": [float(any(scores))],
                }
            )
    return rows


def test_informative_fixed_horizon_calibration_passes_all_gates():
    result = evaluate_math_repair_calibration(_records())

    assert result["decision"] == "pass"
    assert result["failed_checks"] == []
    assert result["metrics"]["sample_count"] == 256
    assert result["metrics"]["observed_turns"] == 3
    assert result["metrics"]["repair_after_initial_failure_fraction"] > 0
    assert result["metrics"]["regression_after_success_transition_count"] > 0
    assert result["checks"]["regression_is_observed"]


def test_calibration_fails_when_no_failed_answer_is_repaired():
    patterns = ([1, 1, 1],) * 2 + ([0, 0, 0],) * 6
    result = evaluate_math_repair_calibration(_records(patterns))

    assert result["decision"] == "fail"
    assert "repair_is_observed" in result["failed_checks"]
    assert "hindsight_is_nonzero" in result["failed_checks"]


@pytest.mark.parametrize(
    ("mutate", "error_type", "message"),
    [
        (lambda records: records.clear(), ValueError, "no validation records"),
        (
            lambda records: records[1].update(idx=17),
            ValueError,
            "indices must be contiguous",
        ),
        (
            lambda records: records[0]["rewards"].append(0.0),
            ValueError,
            "must contain one logged sample",
        ),
        (
            lambda records: records[0]["content"][0].pop(),
            ValueError,
            "inconsistent horizons",
        ),
        (
            lambda records: records[0]["rewards"].__setitem__(0, 0.0),
            ValueError,
            "does not match",
        ),
        (
            lambda records: records[8]["content"][0][0].update(content="problem 0"),
            ValueError,
            "do not match generations_per_prompt",
        ),
    ],
)
def test_calibration_parser_rejects_malformed_artifacts(mutate, error_type, message):
    records = _records()
    mutate(records)

    with pytest.raises(error_type, match=message):
        parse_math_repair_validation_records(records, generations_per_prompt=8)
