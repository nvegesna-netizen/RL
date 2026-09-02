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

"""End-to-end tests for the detached opportunity-loss result pipeline."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tests.unit.tools.test_opportunity_ledger_join import _fixture
from tools.observer_duty_analysis import ObserverDutyAnalysisError
from tools.opportunity_ledger_join import OpportunityLedgerJoinError
from tools.opportunity_loss_pipeline import (
    OpportunityLossPipelineError,
    build_result,
    main,
)


def _protocol() -> dict[str, object]:
    return {
        "protocol": "m4-opportunity-loss-common-instrumentation-v1",
        "assignment": {
            "algorithm": "sha256-rejection-sampling",
            "domain": "m4-opportunity-loss-confirmatory-v1",
            "rerandomization": False,
            "seed": 20260810,
        },
        "arms": [
            {"delay_seconds": 0, "label": "control", "mass": 5},
            {"delay_seconds": 5, "label": "d5", "mass": 5},
            {"delay_seconds": 10, "label": "d10", "mass": 2},
        ],
        "windows": {"primary_start_versions": [0, 1]},
        "runtime": {
            "generations_per_prompt": 2,
            "max_staleness_versions": 1,
            "train_global_batch_size": 2,
        },
        "mechanism_replication": {
            "accepted_predecessor_commit": "a6be6971b79ad0d6b48c0dc85bb7f09682c2f7dd",
            "role": "support_condition_not_primary_endpoint",
            "score_window": "primary_start_versions",
            "thresholds": {
                "maximum_control_direct_chain_rate": 0.02,
                "maximum_delay_overshoot_p99_seconds": 2.0,
                "minimum_d10_d5_direct_chain_contrast": 0.2,
                "minimum_d10_d5_version_advance_contrast": 0.25,
                "minimum_d10_direct_chain_rate": 0.4,
                "minimum_d5_control_direct_chain_contrast": 0.1,
                "minimum_d5_control_version_advance_contrast": 0.2,
                "minimum_d5_direct_chain_rate": 0.1,
                "minimum_nonzero_delay_compliance": 0.99,
            },
        },
        "instrumentation": {
            "common_in_all_arms": True,
            "off_on_abba": False,
            "opportunity_coverage_required": 1.0,
            "corrected_observer_duty_portability_target": 0.01,
            "opportunity_replay": {
                "allowed_reward_values": [0, 1],
                "baseline_algorithm": "nemo_rl_grpo_v1",
                "comparison": "ieee754_binary32_bit_exact",
                "estimator": "GRPOAdvantageEstimator",
                "normalization_epsilon": 1e-6,
                "normalize_rewards": False,
                "reduction_dtype": "float32",
                "reward_dtype": "float32",
                "sibling_order": "ascending_sibling_index",
                "use_leave_one_out_baseline": False,
            },
            "loss_replay": {
                "disable_ppo_ratio": False,
                "positive_example_nll_weight": 0.0,
                "sequence_level_importance_ratios": False,
                "token_level_loss": True,
                "use_cispo": False,
            },
        },
        "analysis": {
            "primary_pair": ["d5", "control"],
            "primary_estimand": "(mean_d5(Q_times_D)-mean_control(Q_times_D))/mean_control(Q)",
            "material_threshold_delta_l": 0.2,
            "missingness": {
                "maximum_terminal_missing_fraction_each_primary_arm": 0.01,
                "missing_opportunity": "RED",
                "missing_terminal_lower_qd": 0,
                "missing_terminal_upper_qd": "observed_Q",
                "terminal_bound_required_for_material_claim": "lower_gt_material_threshold",
                "terminal_bound_required_for_nonmaterial_claim": "upper_lte_material_threshold",
            },
            "inference": {
                "bootstrap_draws": 31,
                "bootstrap_seed": 17,
                "circular_block_size_start_versions": 2,
                "confidence": 0.95,
                "endpoint_interval": "outer_envelope_of_hac_and_bootstrap",
                "hac_bartlett_lag_start_versions": 1,
                "material_alpha": 0.05,
                "material_p_value": "max_hac_normal_and_centered_bootstrap_lower_endpoint",
                "sampling_unit": "assignment_clustered_by_start_weight_version",
            },
        },
    }


def _duty(count: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "measurement_scope": "pre_delay_prepare_observer_callback",
        "assignment_domain": "m4-opportunity-loss-confirmatory-v1",
        "release_arm_labels": ["control", "d5", "d10"],
        "arm_dependent_branch_forbidden": True,
        "calibration_pairs": 64,
        "paired_clock_overhead_ns": 1,
        "observation_count": count,
        "active_window_ns": 10_000,
        "raw_observer_ns": count * 10,
        "corrected_observer_ns": count * 9,
        "raw_observer_duty": count * 10 / 10_000,
        "corrected_observer_duty": count * 9 / 10_000,
    }


def _compact(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def _write_bundle(tmp_path: Path) -> dict[str, Path]:
    lifecycle, opportunity, _ = _fixture(["stale"] * 12)
    paths = {
        "protocol": tmp_path / "protocol.json",
        "lifecycle": tmp_path / "lifecycle.jsonl",
        "opportunity": tmp_path / "opportunity.jsonl",
        "duty": tmp_path / "duty.json",
        "output": tmp_path / "result.json",
    }
    paths["protocol"].write_text(json.dumps(_protocol(), indent=2) + "\n")
    paths["lifecycle"].write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in lifecycle)
    )
    paths["opportunity"].write_bytes(b"".join(_compact(row) for row in opportunity))
    paths["duty"].write_bytes(_compact(_duty(12)))
    return paths


def _build(paths: dict[str, Path]) -> dict[str, object]:
    return build_result(
        protocol_path=paths["protocol"],
        lifecycle_path=paths["lifecycle"],
        opportunity_path=paths["opportunity"],
        observer_duty_path=paths["duty"],
    )


def test_pipeline_reconstructs_causal_result_and_separate_portability(tmp_path) -> None:
    paths = _write_bundle(tmp_path)

    result = _build(paths)

    assert result["primary_assignment_count"] == 12
    assert result["causal_conclusion"] == "NOT_MATERIAL"
    assert result["portability_qualifier"] == "AMBER_OBSERVER_DUTY"
    assert result["mechanism_replication_conclusion"] == (
        "INSUFFICIENT_MECHANISM_EVIDENCE"
    )
    assert result["portability_does_not_modify_causal_conclusion"] is True
    assert result["inputs"]["opportunity"]["group_count"] == 12


def test_cli_writes_deterministic_canonical_result(tmp_path) -> None:
    paths = _write_bundle(tmp_path)
    arguments = [
        "--protocol",
        str(paths["protocol"]),
        "--lifecycle",
        str(paths["lifecycle"]),
        "--opportunity",
        str(paths["opportunity"]),
        "--observer-duty",
        str(paths["duty"]),
        "--output",
        str(paths["output"]),
    ]

    assert main(arguments) == 0
    first = paths["output"].read_bytes()
    assert main(arguments) == 0
    assert paths["output"].read_bytes() == first
    parsed = json.loads(first)
    assert first == _compact(parsed)


def test_duty_count_must_overlap_all_opportunity_groups(tmp_path) -> None:
    paths = _write_bundle(tmp_path)
    paths["duty"].write_bytes(_compact(_duty(11)))

    with pytest.raises(ObserverDutyAnalysisError, match="counters"):
        _build(paths)


def test_noncanonical_ledger_fails_before_analysis(tmp_path) -> None:
    paths = _write_bundle(tmp_path)
    rows = [json.loads(line) for line in paths["opportunity"].read_text().splitlines()]
    paths["opportunity"].write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )

    with pytest.raises(OpportunityLossPipelineError, match="not canonical"):
        _build(paths)


def test_protocol_loss_mutation_is_bound_to_raw_header(tmp_path) -> None:
    paths = _write_bundle(tmp_path)
    protocol = copy.deepcopy(_protocol())
    protocol["instrumentation"]["loss_replay"]["use_cispo"] = True
    paths["protocol"].write_text(json.dumps(protocol, indent=2) + "\n")

    with pytest.raises(OpportunityLedgerJoinError, match="loss settings"):
        _build(paths)


def test_duplicate_protocol_key_fails_closed(tmp_path) -> None:
    paths = _write_bundle(tmp_path)
    paths["protocol"].write_text('{"protocol":"a","protocol":"b"}\n')

    with pytest.raises(OpportunityLossPipelineError, match="duplicate JSON key"):
        _build(paths)


def test_each_input_is_captured_exactly_once(tmp_path, monkeypatch) -> None:
    paths = _write_bundle(tmp_path)
    original = Path.read_bytes
    counts: dict[Path, int] = {}

    def counted(path: Path) -> bytes:
        resolved = path.resolve()
        counts[resolved] = counts.get(resolved, 0) + 1
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", counted)

    _build(paths)

    assert counts == {
        paths[name].resolve(): 1
        for name in ("protocol", "lifecycle", "opportunity", "duty")
    }
