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

"""Detached, fail-closed opportunity-loss result pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.observer_duty_analysis import assess_observer_duty
from tools.opportunity_ledger_join import (
    LedgerJoinProtocol,
    OpportunityAuditContract,
    ReleaseArm,
    join_opportunity_ledgers,
)
from tools.opportunity_loss_inference import infer_opportunity_loss


class OpportunityLossPipelineError(ValueError):
    """Raised when detached evidence disagrees with the registered protocol."""


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise OpportunityLossPipelineError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _decode_json(data: bytes, *, name: str) -> object:
    try:
        value = json.loads(data, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OpportunityLossPipelineError(f"{name} is invalid JSON") from error
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise OpportunityLossPipelineError(f"{name} contains nonfinite JSON") from error
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise OpportunityLossPipelineError(f"{name} must be an object")
    return value


def _array(value: object, *, name: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise OpportunityLossPipelineError(f"{name} must be an array")
    return value


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OpportunityLossPipelineError(f"{name} must be an integer")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpportunityLossPipelineError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise OpportunityLossPipelineError(f"{name} must be finite")
    return result


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise OpportunityLossPipelineError(f"{name} must be a nonempty string")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise OpportunityLossPipelineError(f"{name} must be boolean")
    return value


def _sha_size(data: bytes) -> dict[str, object]:
    return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


def _parse_jsonl(
    data: bytes, *, compact: bool, name: str
) -> tuple[Mapping[str, Any], ...]:
    if not data or not data.endswith(b"\n"):
        raise OpportunityLossPipelineError(
            f"{name} must be nonempty terminal-newline JSONL"
        )
    rows: list[Mapping[str, Any]] = []
    for index, line in enumerate(data.splitlines(keepends=True)):
        if line == b"\n":
            raise OpportunityLossPipelineError(f"{name} contains a blank line")
        value = _decode_json(line, name=f"{name} line {index + 1}")
        row = _mapping(value, name=f"{name} line {index + 1}")
        separators = (",", ":") if compact else None
        expected = (
            json.dumps(
                row,
                allow_nan=False,
                ensure_ascii=False,
                separators=separators,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        if line != expected:
            raise OpportunityLossPipelineError(
                f"{name} line {index + 1} is not canonical"
            )
        rows.append(row)
    return tuple(rows)


def _parse_canonical_object(data: bytes, *, name: str) -> Mapping[str, object]:
    if not data.endswith(b"\n") or data.count(b"\n") != 1:
        raise OpportunityLossPipelineError(f"{name} must be one terminal-newline line")
    value = _mapping(_decode_json(data, name=name), name=name)
    expected = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if data != expected:
        raise OpportunityLossPipelineError(f"{name} is not canonical")
    return value


def _parse_protocol(
    data: bytes,
) -> tuple[Mapping[str, object], LedgerJoinProtocol, dict[str, object]]:
    raw = _mapping(_decode_json(data, name="protocol"), name="protocol")
    if raw.get("protocol") != "m4-opportunity-loss-common-instrumentation-v1":
        raise OpportunityLossPipelineError("protocol identity disagrees")
    assignment = _mapping(raw.get("assignment"), name="assignment")
    if assignment.get("algorithm") != "sha256-rejection-sampling" or _boolean(
        assignment.get("rerandomization"), name="rerandomization"
    ):
        raise OpportunityLossPipelineError("assignment algorithm disagrees")
    arms_raw = _array(raw.get("arms"), name="arms")
    arms = tuple(
        ReleaseArm(
            label=_string(_mapping(item, name="arm").get("label"), name="arm label"),
            delay_seconds=_number(
                _mapping(item, name="arm").get("delay_seconds"), name="delay_seconds"
            ),
            mass=_integer(_mapping(item, name="arm").get("mass"), name="arm mass"),
        )
        for item in arms_raw
    )
    windows = _mapping(raw.get("windows"), name="windows")
    primary = _array(
        windows.get("primary_start_versions"), name="primary_start_versions"
    )
    if len(primary) != 2:
        raise OpportunityLossPipelineError("primary window must have two endpoints")
    runtime = _mapping(raw.get("runtime"), name="runtime")
    instrumentation = _mapping(raw.get("instrumentation"), name="instrumentation")
    if (
        _boolean(instrumentation.get("common_in_all_arms"), name="common_in_all_arms")
        is not True
        or _boolean(instrumentation.get("off_on_abba"), name="off_on_abba") is not False
        or _number(
            instrumentation.get("opportunity_coverage_required"),
            name="opportunity_coverage_required",
        )
        != 1.0
    ):
        raise OpportunityLossPipelineError("common instrumentation contract disagrees")
    replay = _mapping(
        instrumentation.get("opportunity_replay"), name="opportunity_replay"
    )
    if (
        replay.get("comparison") != "ieee754_binary32_bit_exact"
        or replay.get("sibling_order") != "ascending_sibling_index"
    ):
        raise OpportunityLossPipelineError("opportunity replay identity disagrees")
    loss = _mapping(instrumentation.get("loss_replay"), name="loss_replay")
    audit = OpportunityAuditContract(
        estimator_name=_string(replay.get("estimator"), name="estimator"),
        baseline_algorithm=_string(
            replay.get("baseline_algorithm"), name="baseline_algorithm"
        ),
        normalize_rewards=_boolean(
            replay.get("normalize_rewards"), name="normalize_rewards"
        ),
        normalization_epsilon=_number(
            replay.get("normalization_epsilon"), name="normalization_epsilon"
        ),
        reduction_dtype=_string(replay.get("reduction_dtype"), name="reduction_dtype"),
        reward_dtype=_string(replay.get("reward_dtype"), name="reward_dtype"),
        use_leave_one_out_baseline=_boolean(
            replay.get("use_leave_one_out_baseline"), name="use_leave_one_out_baseline"
        ),
        allowed_reward_values=tuple(
            _number(item, name="allowed_reward_value")
            for item in _array(
                replay.get("allowed_reward_values"), name="allowed_reward_values"
            )
        ),
        disable_ppo_ratio=_boolean(
            loss.get("disable_ppo_ratio"), name="disable_ppo_ratio"
        ),
        positive_example_nll_weight=_number(
            loss.get("positive_example_nll_weight"), name="positive_example_nll_weight"
        ),
        sequence_level_importance_ratios=_boolean(
            loss.get("sequence_level_importance_ratios"),
            name="sequence_level_importance_ratios",
        ),
        token_level_loss=_boolean(
            loss.get("token_level_loss"), name="token_level_loss"
        ),
        use_cispo=_boolean(loss.get("use_cispo"), name="use_cispo"),
    )
    join_protocol = LedgerJoinProtocol(
        assignment_domain=_string(assignment.get("domain"), name="assignment domain"),
        assignment_seed=_integer(assignment.get("seed"), name="assignment seed"),
        arms=arms,
        primary_start_version=_integer(primary[0], name="primary start"),
        primary_end_version=_integer(primary[1], name="primary end"),
        siblings_per_group=_integer(
            runtime.get("generations_per_prompt"), name="generations_per_prompt"
        ),
        train_batch_size=_integer(
            runtime.get("train_global_batch_size"), name="train_global_batch_size"
        ),
        opportunity_audit=audit,
    )
    analysis = _mapping(raw.get("analysis"), name="analysis")
    pair = _array(analysis.get("primary_pair"), name="primary_pair")
    if pair != ["d5", "control"]:
        raise OpportunityLossPipelineError("primary pair disagrees")
    if analysis.get("primary_estimand") != (
        "(mean_d5(Q_times_D)-mean_control(Q_times_D))/mean_control(Q)"
    ):
        raise OpportunityLossPipelineError("primary estimand disagrees")
    missingness = _mapping(analysis.get("missingness"), name="missingness")
    inference = _mapping(analysis.get("inference"), name="inference")
    expected_inference_labels = {
        "sampling_unit": "assignment_clustered_by_start_weight_version",
        "endpoint_interval": "outer_envelope_of_hac_and_bootstrap",
        "material_p_value": "max_hac_normal_and_centered_bootstrap_lower_endpoint",
    }
    if any(
        inference.get(key) != value for key, value in expected_inference_labels.items()
    ):
        raise OpportunityLossPipelineError("registered inference identity disagrees")
    expected_missingness = {
        "missing_opportunity": "RED",
        "missing_terminal_lower_qd": 0,
        "missing_terminal_upper_qd": "observed_Q",
        "terminal_bound_required_for_material_claim": "lower_gt_material_threshold",
        "terminal_bound_required_for_nonmaterial_claim": "upper_lte_material_threshold",
    }
    if any(
        missingness.get(key) != value for key, value in expected_missingness.items()
    ):
        raise OpportunityLossPipelineError("registered missingness identity disagrees")
    options: dict[str, object] = {
        "treatment_arm": pair[0],
        "control_arm": pair[1],
        "material_threshold": _number(
            analysis.get("material_threshold_delta_l"), name="material threshold"
        ),
        "max_missing_fraction": _number(
            missingness.get("maximum_terminal_missing_fraction_each_primary_arm"),
            name="maximum terminal missing fraction",
        ),
        "confidence": _number(inference.get("confidence"), name="confidence"),
        "material_alpha": _number(
            inference.get("material_alpha"), name="material_alpha"
        ),
        "hac_lag": _integer(
            inference.get("hac_bartlett_lag_start_versions"), name="HAC lag"
        ),
        "block_size": _integer(
            inference.get("circular_block_size_start_versions"), name="block size"
        ),
        "bootstrap_draws": _integer(
            inference.get("bootstrap_draws"), name="bootstrap draws"
        ),
        "bootstrap_seed": _integer(
            inference.get("bootstrap_seed"), name="bootstrap seed"
        ),
        "maximum_corrected_observer_duty": _number(
            instrumentation.get("corrected_observer_duty_portability_target"),
            name="observer duty target",
        ),
    }
    return raw, join_protocol, options


def build_result(
    *,
    protocol_path: str | Path,
    lifecycle_path: str | Path,
    opportunity_path: str | Path,
    observer_duty_path: str | Path,
) -> dict[str, object]:
    """Strictly reconstruct and analyze one detached evidence bundle."""
    protocol_data = Path(protocol_path).read_bytes()
    _, protocol, options = _parse_protocol(protocol_data)
    lifecycle_data = Path(lifecycle_path).read_bytes()
    opportunity_data = Path(opportunity_path).read_bytes()
    duty_data = Path(observer_duty_path).read_bytes()
    lifecycle_rows = _parse_jsonl(
        lifecycle_data, compact=False, name="lifecycle ledger"
    )
    opportunity_rows = _parse_jsonl(
        opportunity_data, compact=True, name="opportunity ledger"
    )
    rows = join_opportunity_ledgers(
        protocol=protocol,
        lifecycle_rows=lifecycle_rows,
        opportunity_rows=opportunity_rows,
    )
    total_mass = sum(arm.mass for arm in protocol.arms)
    propensities = {arm.label: arm.mass / total_mass for arm in protocol.arms}
    causal = infer_opportunity_loss(
        rows,
        propensities=propensities,
        primary_start_version=protocol.primary_start_version,
        primary_end_version=protocol.primary_end_version,
        control_arm=str(options["control_arm"]),
        treatment_arm=str(options["treatment_arm"]),
        material_threshold=float(options["material_threshold"]),
        max_missing_fraction=float(options["max_missing_fraction"]),
        confidence=float(options["confidence"]),
        material_alpha=float(options["material_alpha"]),
        hac_lag=int(options["hac_lag"]),
        block_size=int(options["block_size"]),
        bootstrap_draws=int(options["bootstrap_draws"]),
        bootstrap_seed=int(options["bootstrap_seed"]),
    )
    group_count = sum(row.get("event_type") == "group" for row in opportunity_rows)
    duty = assess_observer_duty(
        _parse_canonical_object(duty_data, name="observer duty"),
        assignment_domain=protocol.assignment_domain,
        release_arm_labels=tuple(arm.label for arm in protocol.arms),
        expected_observation_count=group_count,
        maximum_corrected_observer_duty=float(
            options["maximum_corrected_observer_duty"]
        ),
    )
    return {
        "schema_version": 1,
        "result_scope": "primary_opportunity_loss_and_observer_duty",
        "protocol": _sha_size(protocol_data),
        "inputs": {
            "lifecycle": {
                **_sha_size(lifecycle_data),
                "row_count": len(lifecycle_rows),
            },
            "opportunity": {
                **_sha_size(opportunity_data),
                "row_count": len(opportunity_rows),
                "group_count": group_count,
            },
            "observer_duty": _sha_size(duty_data),
        },
        "primary_assignment_count": len(rows),
        "causal_conclusion": causal.conclusion,
        "causal_inference": causal.to_dict(),
        "portability_qualifier": duty.conclusion,
        "observer_duty": duty.to_dict(),
        "portability_does_not_modify_causal_conclusion": True,
    }


def write_result(result: Mapping[str, object], output_path: str | Path) -> None:
    """Atomically write one canonical result artifact."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(
            result,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """Run the detached pipeline without importing a training entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--lifecycle", required=True)
    parser.add_argument("--opportunity", required=True)
    parser.add_argument("--observer-duty", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = build_result(
        protocol_path=args.protocol,
        lifecycle_path=args.lifecycle,
        opportunity_path=args.opportunity,
        observer_duty_path=args.observer_duty,
    )
    write_result(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
