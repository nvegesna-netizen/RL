# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Tests for independent observer-duty assessment."""

from __future__ import annotations

import copy
import json

import pytest

from tools.observer_duty_analysis import (
    ObserverDutyAnalysisError,
    assess_observer_duty,
    load_observer_duty,
)


def _raw(*, corrected_ns: int = 8, corrected_duty: float = 0.008):
    return {
        "schema_version": 1,
        "measurement_scope": "pre_delay_prepare_observer_callback",
        "assignment_domain": "study-v1",
        "release_arm_labels": ["control", "d5", "d10"],
        "arm_dependent_branch_forbidden": True,
        "calibration_pairs": 64,
        "paired_clock_overhead_ns": 1,
        "observation_count": 2,
        "active_window_ns": 1000,
        "raw_observer_ns": 10,
        "corrected_observer_ns": corrected_ns,
        "raw_observer_duty": 0.01,
        "corrected_observer_duty": corrected_duty,
    }


def _assess(raw):
    return assess_observer_duty(
        raw,
        assignment_domain="study-v1",
        release_arm_labels=("control", "d5", "d10"),
        expected_observation_count=2,
    )


def test_supported_and_amber_are_portability_only() -> None:
    assert _assess(_raw()).conclusion == "SUPPORTED"
    amber = _raw(corrected_ns=12, corrected_duty=0.012)
    amber["raw_observer_ns"] = 14
    amber["raw_observer_duty"] = 0.014
    assert _assess(amber).conclusion == "AMBER_OBSERVER_DUTY"


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_key",
        "domain",
        "arms",
        "arm_branch",
        "count",
        "corrected",
        "raw_ratio",
        "corrected_ratio",
    ],
)
def test_mutated_or_self_reported_duty_fails_closed(mutation: str) -> None:
    raw = copy.deepcopy(_raw())
    if mutation == "extra_key":
        raw["extra"] = 1
    elif mutation == "domain":
        raw["assignment_domain"] = "other"
    elif mutation == "arms":
        raw["release_arm_labels"] = ["d5", "control", "d10"]
    elif mutation == "arm_branch":
        raw["arm_dependent_branch_forbidden"] = False
    elif mutation == "count":
        raw["observation_count"] = 1
    elif mutation == "corrected":
        raw["corrected_observer_ns"] = 9
    elif mutation == "raw_ratio":
        raw["raw_observer_duty"] = 0.1
    elif mutation == "corrected_ratio":
        raw["corrected_observer_duty"] = 0.1

    with pytest.raises(ObserverDutyAnalysisError):
        _assess(raw)


def test_loader_requires_canonical_single_line(tmp_path) -> None:
    canonical = (
        json.dumps(_raw(), separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    path = tmp_path / "duty.json"
    path.write_bytes(canonical)
    assert load_observer_duty(path) == _raw()

    path.write_bytes(canonical + b"\n")
    with pytest.raises(ObserverDutyAnalysisError, match="one line"):
        load_observer_duty(path)

    path.write_bytes(json.dumps(_raw()).encode() + b"\n")
    with pytest.raises(ObserverDutyAnalysisError, match="canonical"):
        load_observer_duty(path)
