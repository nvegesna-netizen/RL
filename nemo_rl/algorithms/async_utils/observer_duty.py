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

"""Common-arm synchronous observer-duty measurement."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ObserverDutySummary:
    """Value-free timing summary for one common-instrumentation interval."""

    schema_version: int
    measurement_scope: str
    assignment_domain: str
    release_arm_labels: tuple[str, ...]
    arm_dependent_branch_forbidden: bool
    calibration_pairs: int
    paired_clock_overhead_ns: int
    observation_count: int
    active_window_ns: int
    raw_observer_ns: int
    corrected_observer_ns: int
    raw_observer_duty: float
    corrected_observer_duty: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible detached value."""
        result = asdict(self)
        result["release_arm_labels"] = list(self.release_arm_labels)
        return result


class CommonObserverDutyMeter:
    """Measure one identical synchronous callback path across randomized arms."""

    SCHEMA_VERSION = 1
    MEASUREMENT_SCOPE = "pre_delay_prepare_observer_callback"

    def __init__(
        self,
        *,
        assignment_domain: str,
        release_arm_labels: Sequence[str],
        measurement_scope: str = MEASUREMENT_SCOPE,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        calibration_pairs: int = 64,
    ) -> None:
        labels = tuple(release_arm_labels)
        if (
            not assignment_domain
            or not labels
            or len(labels) != len(set(labels))
            or any(not label for label in labels)
        ):
            raise ValueError("observer duty requires a domain and unique arms")
        if (
            not measurement_scope
            or not measurement_scope.isascii()
            or "\0" in measurement_scope
        ):
            raise ValueError("observer duty measurement_scope must be nonempty ASCII")
        if (
            isinstance(calibration_pairs, bool)
            or not isinstance(calibration_pairs, int)
            or calibration_pairs <= 0
        ):
            raise ValueError("calibration_pairs must be a positive integer")
        self._assignment_domain = assignment_domain
        self._release_arm_labels = labels
        self._measurement_scope = measurement_scope
        self._clock_ns = clock_ns
        self._calibration_pairs = calibration_pairs
        calibration: list[int] = []
        for _ in range(calibration_pairs):
            started = self._read_clock()
            ended = self._read_clock()
            if ended < started:
                raise ValueError("observer duty clock moved backwards")
            calibration.append(ended - started)
        self._paired_clock_overhead_ns = min(calibration)
        self._active_start_ns: int | None = None
        self._active_end_ns: int | None = None
        self._observation_start_ns: int | None = None
        self._durations_ns: list[int] = []

    def _read_clock(self) -> int:
        value = self._clock_ns()
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("observer duty clock must return nonnegative integers")
        return value

    def begin_active_window(self) -> None:
        """Start the common acquisition interval before either pump runs."""
        if self._active_start_ns is not None:
            raise RuntimeError("observer duty active window already started")
        self._active_start_ns = self._read_clock()

    @contextmanager
    def observe(self) -> Iterator[None]:
        """Measure one pre-delay observer callback, including failed callbacks."""
        if self._active_start_ns is None or self._active_end_ns is not None:
            raise RuntimeError("observer callback lies outside the active window")
        if self._observation_start_ns is not None:
            raise RuntimeError("observer duty measurements cannot overlap")
        self._observation_start_ns = self._read_clock()
        try:
            yield
        finally:
            ended = self._read_clock()
            started = self._observation_start_ns
            self._observation_start_ns = None
            assert started is not None
            if ended < started:
                raise ValueError("observer duty clock moved backwards")
            self._durations_ns.append(ended - started)

    def end_active_window(self) -> None:
        """Close the common acquisition interval after the pumps stop."""
        if self._active_start_ns is None or self._active_end_ns is not None:
            raise RuntimeError("observer duty active window is not open")
        if self._observation_start_ns is not None:
            raise RuntimeError("cannot end duty window during an observation")
        ended = self._read_clock()
        if ended <= self._active_start_ns:
            raise ValueError("observer duty active window must be positive")
        self._active_end_ns = ended

    def snapshot(self) -> ObserverDutySummary:
        """Return the immutable corrected and uncorrected duty summary."""
        if self._active_start_ns is None or self._active_end_ns is None:
            raise RuntimeError("observer duty active window is incomplete")
        active_window = self._active_end_ns - self._active_start_ns
        raw = sum(self._durations_ns)
        corrected = max(
            0,
            raw - len(self._durations_ns) * self._paired_clock_overhead_ns,
        )
        raw_duty = raw / active_window
        corrected_duty = corrected / active_window
        if not all(math.isfinite(value) for value in (raw_duty, corrected_duty)):
            raise ValueError("observer duty is nonfinite")
        return ObserverDutySummary(
            schema_version=self.SCHEMA_VERSION,
            measurement_scope=self._measurement_scope,
            assignment_domain=self._assignment_domain,
            release_arm_labels=self._release_arm_labels,
            arm_dependent_branch_forbidden=True,
            calibration_pairs=self._calibration_pairs,
            paired_clock_overhead_ns=self._paired_clock_overhead_ns,
            observation_count=len(self._durations_ns),
            active_window_ns=active_window,
            raw_observer_ns=raw,
            corrected_observer_ns=corrected,
            raw_observer_duty=raw_duty,
            corrected_observer_duty=corrected_duty,
        )

    def flush_json(self, output_path: str | Path) -> None:
        """Atomically publish one canonical summary after the window closes."""
        raw = (
            json.dumps(
                self.snapshot().to_dict(),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "wb") as output:
                output.write(raw)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
