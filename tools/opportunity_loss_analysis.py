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

"""Bound terminal missingness for the randomized opportunity-loss estimand.

The opportunity value must be observed before release for every assignment.
Only the later binary delivery disposition may be missing. A missing disposition
is bounded as either delivered or undelivered, so the result never relies on a
complete-case assumption.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

Conclusion = Literal[
    "MATERIAL",
    "NOT_MATERIAL",
    "INCONCLUSIVE",
    "INSUFFICIENT_TERMINAL_COVERAGE",
]


class OpportunityLossAnalysisError(ValueError):
    """Raised when an input violates the registered analysis contract."""


@dataclass(frozen=True)
class OpportunityAssignment:
    """One eligible randomized assignment with pre-release opportunity."""

    assignment_id: str
    arm: str
    opportunity: float
    delivered: bool | None


@dataclass(frozen=True)
class ArmTerminalSummary:
    """Observed opportunity and bounded undelivered opportunity for one arm."""

    arm: str
    assignments: int
    missing_terminals: int
    missing_fraction: float
    total_opportunity: float
    missing_opportunity: float
    missing_opportunity_fraction: float
    lost_opportunity_lower_mean: float
    lost_opportunity_upper_mean: float


@dataclass(frozen=True)
class OpportunityLossBounds:
    """Finite-sample bounds for the registered normalized loss contrast."""

    control: ArmTerminalSummary
    treatment: ArmTerminalSummary
    control_mean_opportunity: float
    delta_l_lower: float
    delta_l_upper: float
    complete_data_delta_l: float | None
    material_threshold: float
    max_missing_fraction: float
    coverage_gate_passed: bool
    conclusion: Conclusion

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible result."""
        return asdict(self)


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpportunityLossAnalysisError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise OpportunityLossAnalysisError(f"{name} must be finite and nonnegative")
    return result


def _validate_assignments(
    assignments: Iterable[OpportunityAssignment],
) -> tuple[OpportunityAssignment, ...]:
    rows = tuple(assignments)
    if not rows:
        raise OpportunityLossAnalysisError("at least one assignment is required")
    seen: set[str] = set()
    validated: list[OpportunityAssignment] = []
    for row in rows:
        if not isinstance(row.assignment_id, str) or not row.assignment_id:
            raise OpportunityLossAnalysisError("assignment_id must be nonempty")
        if row.assignment_id in seen:
            raise OpportunityLossAnalysisError(
                f"duplicate assignment_id={row.assignment_id!r}"
            )
        if not isinstance(row.arm, str) or not row.arm:
            raise OpportunityLossAnalysisError("arm must be nonempty")
        opportunity = _finite_nonnegative(
            row.opportunity, name=f"opportunity for {row.assignment_id}"
        )
        if row.delivered is not None and not isinstance(row.delivered, bool):
            raise OpportunityLossAnalysisError(
                f"delivered for {row.assignment_id} must be boolean or null"
            )
        seen.add(row.assignment_id)
        validated.append(
            OpportunityAssignment(
                assignment_id=row.assignment_id,
                arm=row.arm,
                opportunity=opportunity,
                delivered=row.delivered,
            )
        )
    return tuple(validated)


def _summarize_arm(
    assignments: Sequence[OpportunityAssignment], *, arm: str
) -> ArmTerminalSummary:
    rows = tuple(row for row in assignments if row.arm == arm)
    if not rows:
        raise OpportunityLossAnalysisError(f"arm {arm!r} has no assignments")
    total_opportunity = math.fsum(row.opportunity for row in rows)
    missing = tuple(row for row in rows if row.delivered is None)
    missing_opportunity = math.fsum(row.opportunity for row in missing)
    observed_lost = math.fsum(row.opportunity for row in rows if row.delivered is False)
    count = len(rows)
    return ArmTerminalSummary(
        arm=arm,
        assignments=count,
        missing_terminals=len(missing),
        missing_fraction=len(missing) / count,
        total_opportunity=total_opportunity,
        missing_opportunity=missing_opportunity,
        missing_opportunity_fraction=(
            missing_opportunity / total_opportunity if total_opportunity > 0.0 else 0.0
        ),
        lost_opportunity_lower_mean=observed_lost / count,
        lost_opportunity_upper_mean=(observed_lost + missing_opportunity) / count,
    )


def bound_opportunity_loss(
    assignments: Iterable[OpportunityAssignment],
    *,
    control_arm: str = "control",
    treatment_arm: str = "d5",
    material_threshold: float = 0.2,
    max_missing_fraction: float = 0.01,
) -> OpportunityLossBounds:
    """Bound the d5-control normalized opportunity-loss contrast.

    For arm ``a``, the finite-sample loss mean is ``mean(Q * D | A=a)``.
    Missing terminal ``D`` values contribute ``0`` to the lower arm mean and
    ``Q`` to its upper arm mean. The contrast bounds pair the treatment lower
    with the control upper, and vice versa, then normalize by mean control Q.
    """
    if not control_arm or not treatment_arm or control_arm == treatment_arm:
        raise OpportunityLossAnalysisError(
            "control_arm and treatment_arm must be distinct and nonempty"
        )
    threshold = _finite_nonnegative(material_threshold, name="material_threshold")
    missing_limit = _finite_nonnegative(
        max_missing_fraction, name="max_missing_fraction"
    )
    if missing_limit > 1.0:
        raise OpportunityLossAnalysisError("max_missing_fraction cannot exceed one")

    rows = _validate_assignments(assignments)
    control = _summarize_arm(rows, arm=control_arm)
    treatment = _summarize_arm(rows, arm=treatment_arm)
    control_mean_q = control.total_opportunity / control.assignments
    if control_mean_q <= 0.0:
        raise OpportunityLossAnalysisError(
            "mean control opportunity must be strictly positive"
        )

    lower = (
        treatment.lost_opportunity_lower_mean - control.lost_opportunity_upper_mean
    ) / control_mean_q
    upper = (
        treatment.lost_opportunity_upper_mean - control.lost_opportunity_lower_mean
    ) / control_mean_q
    if lower > upper:
        raise OpportunityLossAnalysisError("internal bound ordering failure")

    complete = (
        lower if control.missing_terminals == treatment.missing_terminals == 0 else None
    )
    coverage_passed = (
        control.missing_fraction <= missing_limit
        and treatment.missing_fraction <= missing_limit
    )
    if not coverage_passed:
        conclusion: Conclusion = "INSUFFICIENT_TERMINAL_COVERAGE"
    elif lower > threshold:
        conclusion = "MATERIAL"
    elif upper <= threshold:
        conclusion = "NOT_MATERIAL"
    else:
        conclusion = "INCONCLUSIVE"

    return OpportunityLossBounds(
        control=control,
        treatment=treatment,
        control_mean_opportunity=control_mean_q,
        delta_l_lower=lower,
        delta_l_upper=upper,
        complete_data_delta_l=complete,
        material_threshold=threshold,
        max_missing_fraction=missing_limit,
        coverage_gate_passed=coverage_passed,
        conclusion=conclusion,
    )


def load_assignments(path: Path) -> tuple[OpportunityAssignment, ...]:
    """Load strict canonical-shape JSONL assignment rows."""
    expected_keys = {"assignment_id", "arm", "opportunity", "delivered"}
    rows: list[OpportunityAssignment] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise OpportunityLossAnalysisError(f"cannot read {path}: {error}") from error
    if not lines:
        raise OpportunityLossAnalysisError(f"{path} is empty")
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise OpportunityLossAnalysisError(
                f"{path}:{line_number}: invalid JSON: {error}"
            ) from error
        if not isinstance(value, dict) or set(value) != expected_keys:
            raise OpportunityLossAnalysisError(
                f"{path}:{line_number} must have exactly {sorted(expected_keys)}"
            )
        rows.append(
            OpportunityAssignment(
                assignment_id=value["assignment_id"],
                arm=value["arm"],
                opportunity=value["opportunity"],
                delivered=value["delivered"],
            )
        )
    return _validate_assignments(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assignments", type=Path)
    parser.add_argument("--control-arm", default="control")
    parser.add_argument("--treatment-arm", default="d5")
    parser.add_argument("--material-threshold", type=float, default=0.2)
    parser.add_argument("--max-missing-fraction", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    """Run the strict JSONL analyzer."""
    args = _parse_args()
    result = bound_opportunity_loss(
        load_assignments(args.assignments),
        control_arm=args.control_arm,
        treatment_arm=args.treatment_arm,
        material_threshold=args.material_threshold,
        max_missing_fraction=args.max_missing_fraction,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
