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

"""Independent validation of common-arm observer-duty evidence."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

ObserverDutyConclusion = Literal["SUPPORTED", "AMBER_OBSERVER_DUTY"]


class ObserverDutyAnalysisError(ValueError):
    """Raised when observer-duty evidence violates the frozen schema."""


@dataclass(frozen=True)
class ObserverDutyAssessment:
    """Verified common-instrumentation portability assessment."""

    corrected_observer_duty: float
    maximum_corrected_observer_duty: float
    observation_count: int
    active_window_ns: int
    conclusion: ObserverDutyConclusion

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible result."""
        return asdict(self)


_KEYSET = {
    "schema_version",
    "measurement_scope",
    "assignment_domain",
    "release_arm_labels",
    "arm_dependent_branch_forbidden",
    "calibration_pairs",
    "paired_clock_overhead_ns",
    "observation_count",
    "active_window_ns",
    "raw_observer_ns",
    "corrected_observer_ns",
    "raw_observer_duty",
    "corrected_observer_duty",
}


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ObserverDutyAnalysisError(f"{name} must be an integer")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ObserverDutyAnalysisError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ObserverDutyAnalysisError(f"{name} must be finite")
    return result


def assess_observer_duty(
    raw: Mapping[str, object],
    *,
    assignment_domain: str,
    release_arm_labels: Sequence[str],
    expected_observation_count: int,
    maximum_corrected_observer_duty: float = 0.01,
) -> ObserverDutyAssessment:
    """Recompute duty and classify only its external-validity qualifier."""
    if set(raw) != _KEYSET:
        raise ObserverDutyAnalysisError("observer-duty keyset disagrees")
    if raw.get("schema_version") != 1 or raw.get("measurement_scope") != (
        "pre_delay_prepare_observer_callback"
    ):
        raise ObserverDutyAnalysisError("observer-duty schema or scope disagrees")
    if raw.get("assignment_domain") != assignment_domain:
        raise ObserverDutyAnalysisError("observer-duty assignment domain disagrees")
    observed_labels = raw.get("release_arm_labels")
    if not isinstance(observed_labels, list) or tuple(observed_labels) != tuple(
        release_arm_labels
    ):
        raise ObserverDutyAnalysisError("observer-duty release arms disagree")
    if raw.get("arm_dependent_branch_forbidden") is not True:
        raise ObserverDutyAnalysisError("observer-duty arm-blind contract disagrees")
    calibration_pairs = _integer(raw.get("calibration_pairs"), name="calibration_pairs")
    overhead = _integer(
        raw.get("paired_clock_overhead_ns"), name="paired_clock_overhead_ns"
    )
    count = _integer(raw.get("observation_count"), name="observation_count")
    active = _integer(raw.get("active_window_ns"), name="active_window_ns")
    observed_raw = _integer(raw.get("raw_observer_ns"), name="raw_observer_ns")
    corrected = _integer(raw.get("corrected_observer_ns"), name="corrected_observer_ns")
    if (
        calibration_pairs <= 0
        or overhead < 0
        or count != expected_observation_count
        or count < 0
        or active <= 0
        or observed_raw < 0
        or observed_raw > active
    ):
        raise ObserverDutyAnalysisError("observer-duty counters violate contract")
    expected_corrected = max(0, observed_raw - count * overhead)
    if corrected != expected_corrected:
        raise ObserverDutyAnalysisError("corrected observer nanoseconds disagree")
    raw_duty = _number(raw.get("raw_observer_duty"), name="raw_observer_duty")
    corrected_duty = _number(
        raw.get("corrected_observer_duty"), name="corrected_observer_duty"
    )
    if raw_duty != observed_raw / active or corrected_duty != corrected / active:
        raise ObserverDutyAnalysisError("observer-duty ratios disagree")
    if (
        not math.isfinite(maximum_corrected_observer_duty)
        or not 0.0 <= maximum_corrected_observer_duty <= 1.0
    ):
        raise ObserverDutyAnalysisError("observer-duty threshold is invalid")
    conclusion: ObserverDutyConclusion = (
        "SUPPORTED"
        if corrected_duty <= maximum_corrected_observer_duty
        else "AMBER_OBSERVER_DUTY"
    )
    return ObserverDutyAssessment(
        corrected_observer_duty=corrected_duty,
        maximum_corrected_observer_duty=maximum_corrected_observer_duty,
        observation_count=count,
        active_window_ns=active,
        conclusion=conclusion,
    )


def load_observer_duty(path: str | Path) -> Mapping[str, object]:
    """Load one canonical single-line observer-duty artifact."""
    data = Path(path).read_bytes()
    if not data.endswith(b"\n") or data.count(b"\n") != 1:
        raise ObserverDutyAnalysisError("observer-duty artifact must be one line")
    try:
        parsed = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ObserverDutyAnalysisError(
            "observer-duty artifact is invalid JSON"
        ) from error
    if not isinstance(parsed, dict):
        raise ObserverDutyAnalysisError("observer-duty artifact must contain an object")
    canonical = (
        json.dumps(
            parsed,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if canonical != data:
        raise ObserverDutyAnalysisError("observer-duty artifact is not canonical")
    return parsed
