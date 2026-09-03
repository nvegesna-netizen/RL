#!/usr/bin/env python3
"""Power audit for a new M4 opportunity-loss follow-up study.

This is prospective design input. It reads the immutable result inside the
preserved confirmatory artifact and cannot alter the registered conclusion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import zipfile
from pathlib import Path
from statistics import NormalDist
from typing import Any

ARTIFACT_SHA256 = "e81a29747628b6f0490323207f317755de411d0f8e7527f715368919aee12fb7"
RESULT_SHA256 = "6c3fc0cf0567d4dab63153769c075dc7d491e41cb2ceb982b569649f5b42ced3"
RESULT_MEMBER = (
    "workspace/assets/basic/m4-opportunity-loss-confirmatory-acquisition-r4/"
    "m4-opportunity-loss-result.json"
)
MATERIAL_THRESHOLD = 0.20
ALPHA = 0.05
POWER_LEVELS = (0.80, 0.90)
ALTERNATIVES = (0.21, 0.225, 0.25, 0.30, 0.40)
VARIANCE_REDUCTION_SCENARIOS = (0.0, 0.25, 0.50, 0.75)


class FollowupPowerError(ValueError):
    """Raised when the frozen result or requested design is invalid."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _load_result(artifact: Path) -> tuple[dict[str, Any], int]:
    artifact_raw = artifact.read_bytes()
    if _sha256(artifact_raw) != ARTIFACT_SHA256:
        raise FollowupPowerError("artifact SHA-256 does not match the frozen run")
    try:
        with zipfile.ZipFile(artifact) as archive:
            result_raw = archive.read(RESULT_MEMBER)
    except (OSError, KeyError, zipfile.BadZipFile) as error:
        raise FollowupPowerError(f"cannot read frozen result: {error}") from error
    if _sha256(result_raw) != RESULT_SHA256:
        raise FollowupPowerError("result SHA-256 does not match the frozen run")
    try:
        result = json.loads(result_raw)
    except json.JSONDecodeError as error:
        raise FollowupPowerError("frozen result is not JSON") from error
    if not isinstance(result, dict) or _canonical_json(result) != result_raw:
        raise FollowupPowerError("frozen result is not canonical JSON")
    return result, len(artifact_raw)


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FollowupPowerError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise FollowupPowerError(f"{name} must be finite")
    return result


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FollowupPowerError(f"{name} must be a positive integer")
    return value


def _extract_frozen_inputs(result: dict[str, Any]) -> dict[str, float | int | str]:
    try:
        causal = result["causal_inference"]
        lower = causal["lower_endpoint"]
        mechanism = result["mechanism_replication"]["arm_summaries"]
    except (KeyError, TypeError) as error:
        raise FollowupPowerError("frozen result lacks required fields") from error
    if (
        result.get("causal_conclusion") != "INCONCLUSIVE"
        or result.get("mechanism_replication_conclusion") != "REPLICATED"
        or result.get("portability_qualifier") != "SUPPORTED"
    ):
        raise FollowupPowerError("frozen scientific conclusions disagree")
    threshold = _finite_number(causal.get("material_threshold"), name="threshold")
    if threshold != MATERIAL_THRESHOLD:
        raise FollowupPowerError("material threshold disagrees")
    control = _integer(mechanism["control"]["assignment_count"], name="control count")
    treatment = _integer(mechanism["d5"]["assignment_count"], name="d5 count")
    positive_control = _integer(mechanism["d10"]["assignment_count"], name="d10 count")
    total = _integer(result.get("primary_assignment_count"), name="total count")
    if control + treatment + positive_control != total:
        raise FollowupPowerError("arm counts do not sum to the primary count")
    return {
        "estimate": _finite_number(lower.get("estimate"), name="estimate"),
        "standard_error": _finite_number(
            lower.get("standard_error"), name="standard error"
        ),
        "control_assignments": control,
        "d5_assignments": treatment,
        "d10_assignments": positive_control,
        "primary_pair_assignments": control + treatment,
        "total_primary_assignments": total,
        "causal_conclusion": str(result["causal_conclusion"]),
        "mechanism_conclusion": str(result["mechanism_replication_conclusion"]),
        "portability_qualifier": str(result["portability_qualifier"]),
    }


def _required_multiplier(
    *,
    current_standard_error: float,
    alternative: float,
    power: float,
    variance_remaining: float,
) -> float:
    margin = alternative - MATERIAL_THRESHOLD
    if margin <= 0.0:
        raise FollowupPowerError("alternative must exceed the material threshold")
    z_alpha = NormalDist().inv_cdf(1.0 - ALPHA)
    z_power = NormalDist().inv_cdf(power)
    return (
        variance_remaining
        * ((z_alpha + z_power) * current_standard_error / margin) ** 2
    )


def build_audit(artifact: Path) -> dict[str, object]:
    """Build one canonical prospective power audit."""
    result, artifact_size = _load_result(artifact)
    frozen = _extract_frozen_inputs(result)
    pair_count = int(frozen["primary_pair_assignments"])
    total_count = int(frozen["total_primary_assignments"])
    standard_error = float(frozen["standard_error"])
    estimate = float(frozen["estimate"])

    scenarios: list[dict[str, object]] = []
    for variance_reduction in VARIANCE_REDUCTION_SCENARIOS:
        variance_remaining = 1.0 - variance_reduction
        for alternative in ALTERNATIVES:
            required = {}
            for power in POWER_LEVELS:
                multiplier = _required_multiplier(
                    current_standard_error=standard_error,
                    alternative=alternative,
                    power=power,
                    variance_remaining=variance_remaining,
                )
                primary_assignments = math.ceil(pair_count * multiplier)
                required[str(power)] = {
                    "information_multiplier_vs_current_primary_pair": multiplier,
                    "required_control_plus_d5_assignments": primary_assignments,
                    "required_total_assignments_with_original_5_5_2_allocation": (
                        math.ceil(primary_assignments * 12 / 10)
                    ),
                }
            scenarios.append(
                {
                    "alternative_delta_l": alternative,
                    "margin_above_material_threshold": (
                        alternative - MATERIAL_THRESHOLD
                    ),
                    "variance_reduction": variance_reduction,
                    "required_by_power": required,
                }
            )

    primary_fraction = pair_count / total_count
    same_compute_se_without_d10 = standard_error * math.sqrt(primary_fraction)
    observed_multiplier_80 = _required_multiplier(
        current_standard_error=standard_error,
        alternative=estimate,
        power=0.80,
        variance_remaining=1.0,
    )
    return {
        "schema": "m4-opportunity-loss-prospective-followup-power-v1",
        "status": "EXPLORATORY_DESIGN_ONLY_NO_GPU_AUTHORITY",
        "source": {
            "artifact_sha256": ARTIFACT_SHA256,
            "artifact_size": artifact_size,
            "result_member": RESULT_MEMBER,
            "result_sha256": RESULT_SHA256,
        },
        "frozen_result": frozen,
        "design": {
            "alpha_one_sided": ALPHA,
            "material_threshold": MATERIAL_THRESHOLD,
            "normal_approximation": True,
            "standard_error_scaling": "inverse_square_root_of_primary_pair_count",
            "power_levels": list(POWER_LEVELS),
            "alternatives": list(ALTERNATIVES),
            "variance_reduction_scenarios": list(VARIANCE_REDUCTION_SCENARIOS),
        },
        "allocation_observation": {
            "d10_role_in_completed_study": "positive_control_support_condition",
            "primary_pair_fraction_of_total_assignments": primary_fraction,
            "same_compute_standard_error_if_all_assignments_were_control_or_d5": (
                same_compute_se_without_d10
            ),
            "variance_reduction_from_removing_d10_at_fixed_total_assignments": (
                1.0 - primary_fraction
            ),
        },
        "observed_effect_planning_warning": {
            "observed_delta_l": estimate,
            "observed_margin_above_threshold": estimate - MATERIAL_THRESHOLD,
            "information_multiplier_for_80_percent_power_at_observed_effect": (
                observed_multiplier_80
            ),
            "required_control_plus_d5_assignments": math.ceil(
                pair_count * observed_multiplier_80
            ),
            "interpretation": (
                "The observed margin is too close to the threshold for a simple "
                "same-estimator replication to be an efficient next acquisition."
            ),
        },
        "scenarios": scenarios,
        "decision": {
            "completed_study_conclusion_unchanged": "INCONCLUSIVE",
            "identical_rerun": "REJECT",
            "next_step": "OFFLINE_PRETREATMENT_COVARIATE_EFFICIENCY_AUDIT",
            "gpu_launch_authorized": False,
            "reason": (
                "Estimate prospective variance reduction before selecting a new "
                "effect margin, maximum information, and sequential stopping rule."
            ),
        },
    }


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    _write_atomic(args.output, _canonical_json(build_audit(args.artifact)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
