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

"""Tests for common-arm observer-duty measurement."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from nemo_rl.algorithms.async_utils.observer_duty import CommonObserverDutyMeter


def _clock(values: Iterator[int]):
    return lambda: next(values)


def test_corrected_duty_uses_one_common_active_window() -> None:
    # Calibration pairs are 2ns and 3ns, active window is [10, 110].
    meter = CommonObserverDutyMeter(
        assignment_domain="study-v1",
        release_arm_labels=("control", "d5", "d10"),
        clock_ns=_clock(iter((0, 2, 4, 7, 10, 20, 30, 50, 65, 110))),
        calibration_pairs=2,
    )
    meter.begin_active_window()
    with meter.observe():
        pass
    with meter.observe():
        pass
    meter.end_active_window()

    summary = meter.snapshot()

    assert summary.release_arm_labels == ("control", "d5", "d10")
    assert summary.arm_dependent_branch_forbidden
    assert summary.paired_clock_overhead_ns == 2
    assert summary.observation_count == 2
    assert summary.active_window_ns == 100
    assert summary.raw_observer_ns == 25
    assert summary.corrected_observer_ns == 21
    assert summary.raw_observer_duty == 0.25
    assert summary.corrected_observer_duty == 0.21


def test_failed_observer_is_still_measured() -> None:
    meter = CommonObserverDutyMeter(
        assignment_domain="study-v1",
        release_arm_labels=("control", "d5"),
        clock_ns=_clock(iter((0, 1, 2, 3, 9, 12))),
        calibration_pairs=1,
    )
    meter.begin_active_window()

    with pytest.raises(RuntimeError, match="callback"):
        with meter.observe():
            raise RuntimeError("callback failed")
    meter.end_active_window()

    assert meter.snapshot().observation_count == 1
    assert meter.snapshot().raw_observer_ns == 6


def test_nested_or_out_of_window_measurement_fails_closed() -> None:
    meter = CommonObserverDutyMeter(
        assignment_domain="study-v1",
        release_arm_labels=("control", "d5"),
        clock_ns=_clock(iter((0, 1, 2, 3, 4, 5))),
        calibration_pairs=1,
    )
    with pytest.raises(RuntimeError, match="outside"):
        with meter.observe():
            pass
    meter.begin_active_window()
    with meter.observe():
        with pytest.raises(RuntimeError, match="overlap"):
            with meter.observe():
                pass
    meter.end_active_window()
    with pytest.raises(RuntimeError, match="outside"):
        with meter.observe():
            pass


def test_canonical_atomic_summary_is_deterministic(tmp_path: Path) -> None:
    outputs: list[bytes] = []
    for name in ("first.json", "second.json"):
        meter = CommonObserverDutyMeter(
            assignment_domain="study-v1",
            release_arm_labels=("control", "d5", "d10"),
            clock_ns=_clock(iter((0, 1, 2, 3, 6, 10))),
            calibration_pairs=1,
        )
        meter.begin_active_window()
        with meter.observe():
            pass
        meter.end_active_window()
        path = tmp_path / name
        meter.flush_json(path)
        outputs.append(path.read_bytes())
        assert not list(tmp_path.glob(f".{name}.*.tmp"))

    assert outputs[0] == outputs[1]
    decoded = json.loads(outputs[0])
    assert decoded["measurement_scope"] == "pre_delay_prepare_observer_callback"
    assert decoded["corrected_observer_duty"] == 0.25


def test_custom_measurement_scope_is_preserved() -> None:
    meter = CommonObserverDutyMeter(
        assignment_domain="study-v1",
        release_arm_labels=("control", "d5"),
        measurement_scope="all_synchronous_lifecycle_record_calls",
        clock_ns=_clock(iter((0, 1, 2, 3, 4, 8))),
        calibration_pairs=1,
    )
    meter.begin_active_window()
    with meter.observe():
        pass
    meter.end_active_window()

    assert (
        meter.snapshot().measurement_scope == "all_synchronous_lifecycle_record_calls"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"assignment_domain": "", "release_arm_labels": ("control",)},
        {"assignment_domain": "x", "release_arm_labels": ()},
        {"assignment_domain": "x", "release_arm_labels": ("d5", "d5")},
        {
            "assignment_domain": "x",
            "release_arm_labels": ("control",),
            "calibration_pairs": 0,
        },
    ],
)
def test_invalid_contract_fails_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        CommonObserverDutyMeter(**kwargs)  # type: ignore[arg-type]
