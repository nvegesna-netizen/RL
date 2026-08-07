"""Tests for frozen fixed-horizon math-repair calibration gates."""

import pytest
from turn_level_credit.math_repair_calibration import (
    evaluate_math_repair_calibration,
    parse_math_repair_validation_records,
)


def _conversation(scores):
    messages = [{"role": "user", "content": "problem"}]
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
    for _prompt in range(32):
        for scores in patterns:
            rows.append(
                {
                    "idx": len(rows),
                    "content": [_conversation(scores)],
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
    ],
)
def test_calibration_parser_rejects_malformed_artifacts(mutate, error_type, message):
    records = _records()
    mutate(records)

    with pytest.raises(error_type, match=message):
        parse_math_repair_validation_records(records, generations_per_prompt=8)
