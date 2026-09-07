# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Detached analysis pipeline for prospectively registered GSM8K M4 studies."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

from tools.observer_duty_analysis import assess_observer_duty
from tools.opportunity_ledger_join import LedgerJoinProtocol, join_opportunity_ledgers
from tools.opportunity_loss_adjusted_inference import infer_adjusted_opportunity_loss
from tools.opportunity_loss_inference import infer_opportunity_loss
from tools.opportunity_loss_mechanism import assess_followup_mechanism
from tools.opportunity_loss_pipeline import (
    OpportunityLossPipelineError,
    _parse_canonical_object,
    _parse_jsonl,
    _parse_protocol,
    _sha_size,
    write_result,
)


WORKLOAD_TRANSPORT_IDENTITIES: Mapping[str, Mapping[str, object]] = {
    "m4-opportunity-loss-qwen3-1p7b-gsm8k-workload-transport-v1": {
        "assignment_domain": "m4-opportunity-loss-qwen3-1p7b-gsm8k-v1",
        "assignment_seed": 20260911,
        "bootstrap_seed": 20260912,
        "max_num_epochs": None,
        "model": "Qwen3-1.7B",
        "result_scope": "qwen3_1p7b_gsm8k_workload_transport_checks",
    },
    "m4-opportunity-loss-qwen3-1p7b-gsm8k-workload-transport-r2-v1": {
        "assignment_domain": "m4-opportunity-loss-qwen3-1p7b-gsm8k-r2-v1",
        "assignment_seed": 20260913,
        "bootstrap_seed": 20260914,
        "max_num_epochs": 2,
        "model": "Qwen3-1.7B",
        "result_scope": "qwen3_1p7b_gsm8k_workload_transport_r2_checks",
    },
    "m4-opportunity-loss-qwen3-0p6b-gsm8k-grid-completion-v1": {
        "assignment_domain": "m4-opportunity-loss-qwen3-0p6b-gsm8k-grid-v1",
        "assignment_seed": 20260915,
        "bootstrap_seed": 20260916,
        "max_num_epochs": 2,
        "model": "Qwen3-0.6B",
        "result_scope": "qwen3_0p6b_gsm8k_grid_completion_primary_checks",
    },
}

GRID_SECONDARY_CONTRACT: Mapping[str, object] = {
    "bootstrap_draws": 20000,
    "cell_bootstrap_seeds": {
        "qwen3_0p6b_openmath": 20260917,
        "qwen3_1p7b_openmath": 20260918,
        "qwen3_1p7b_gsm8k": 20260919,
        "qwen3_0p6b_gsm8k": 20260920,
    },
    "common_primary_start_versions": [8, 407],
    "complete_terminal_scoring_required": True,
    "confidence": 0.95,
    "interaction": (
        "(qwen3_0p6b_openmath-qwen3_0p6b_gsm8k)-(qwen3_1p7b_openmath-qwen3_1p7b_gsm8k)"
    ),
    "role": "secondary_grid_synthesis_cannot_override_primary_cell",
    "uncertainty": "independent_cell_version_block_bootstrap_and_hac_envelope",
}


def validate_workload_transport_contract(
    raw: Mapping[str, object],
    protocol: LedgerJoinProtocol,
    options: Mapping[str, object],
) -> None:
    """Validate the exact prospectively frozen GSM8K transport contract."""
    identity = options["protocol_identity"]
    if identity not in WORKLOAD_TRANSPORT_IDENTITIES:
        raise OpportunityLossPipelineError("workload transport protocol disagrees")
    identity_contract = WORKLOAD_TRANSPORT_IDENTITIES[identity]
    if tuple((arm.label, arm.delay_seconds, arm.mass) for arm in protocol.arms) != (
        ("control", 0.0, 1),
        ("d5", 5.0, 1),
    ):
        raise OpportunityLossPipelineError("workload transport arms disagree")
    if (
        protocol.assignment_domain != identity_contract["assignment_domain"]
        or protocol.assignment_seed != identity_contract["assignment_seed"]
        or protocol.primary_start_version != 8
        or protocol.primary_end_version != 507
    ):
        raise OpportunityLossPipelineError(
            "workload transport assignment or window disagrees"
        )
    if (
        options.get("cross_fit_folds") != 8
        or options.get("hac_lag") != 4
        or options.get("block_size") != 8
        or options.get("bootstrap_draws") != 20000
        or options.get("bootstrap_seed") != identity_contract["bootstrap_seed"]
        or options.get("max_staleness_versions") != 1
    ):
        raise OpportunityLossPipelineError(
            "workload transport inference geometry disagrees"
        )
    if raw.get("windows") != {
        "burn_in_start_versions": [0, 7],
        "primary_start_versions": [8, 507],
        "terminal_guard_start_versions": [508, 557],
    }:
        raise OpportunityLossPipelineError("workload transport windows disagree")
    expected_runtime = {
        "buffer_capacity": 64,
        "generations_per_prompt": 8,
        "inflight_prompts": 16,
        "max_staleness_versions": 1,
        "model": identity_contract["model"],
        "prompts_per_step": 4,
        "sampler": "windowed_fifo",
        "train_global_batch_size": 32,
        "trainer_steps": 558,
    }
    if identity_contract["max_num_epochs"] is not None:
        expected_runtime["max_num_epochs"] = identity_contract["max_num_epochs"]
    if raw.get("runtime") != expected_runtime:
        raise OpportunityLossPipelineError("workload transport runtime disagrees")
    if identity.endswith("-r2-v1") and raw.get("exposure") != {
        "epoch_specific_group_instance_is_assignment_unit": True,
        "max_num_epochs": 2,
        "r1_observations_enter_estimator": False,
        "repeated_prompt_exposure_declared": True,
        "rollout_pump_cancelled_at_completed_step": 558,
    }:
        raise OpportunityLossPipelineError("workload transport exposure disagrees")
    if identity.endswith("-grid-completion-v1"):
        if raw.get("exposure") != {
            "epoch_specific_group_instance_is_assignment_unit": True,
            "max_num_epochs": 2,
            "prior_study_observations_enter_estimator": False,
            "repeated_prompt_exposure_declared": True,
            "rollout_pump_cancelled_at_completed_step": 558,
        }:
            raise OpportunityLossPipelineError(
                "grid completion exposure contract disagrees"
            )
        if raw.get("grid_secondary_analysis") != GRID_SECONDARY_CONTRACT:
            raise OpportunityLossPipelineError(
                "grid completion secondary analysis disagrees"
            )
    if raw.get("resource_caps") != {
        "automatic_extension": False,
        "automatic_retry": False,
        "gpu_hour_cap": 8.0,
        "gpus": 2,
        "wall_clock_cap_hours": 4.0,
    }:
        raise OpportunityLossPipelineError("workload transport resource caps disagree")
    if raw.get("workload") != {
        "dataset_name": "gsm8k",
        "extract_answer": True,
        "huggingface_path": "openai/gsm8k",
        "processor": "math_hf_data_processor",
        "prompt_file": "examples/prompts/cot.txt",
        "reward_verifier": "hf_math_verify",
        "split": "train",
        "split_seed": None,
        "split_validation_size": 0.0,
        "subset": "main",
        "task_name": "gsm8k",
    }:
        raise OpportunityLossPipelineError("workload transport dataset disagrees")


def build_workload_transport_result(
    *,
    protocol_path: str | Path,
    lifecycle_path: str | Path,
    opportunity_path: str | Path,
    observer_duty_path: str | Path,
) -> dict[str, object]:
    """Strictly reconstruct and analyze one GSM8K workload-transport bundle."""
    protocol_data = Path(protocol_path).read_bytes()
    raw, protocol, options = _parse_protocol(protocol_data)
    validate_workload_transport_contract(raw, protocol, options)

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
    common = {
        "propensities": {"control": 0.5, "d5": 0.5},
        "primary_start_version": protocol.primary_start_version,
        "primary_end_version": protocol.primary_end_version,
        "control_arm": "control",
        "treatment_arm": "d5",
        "material_threshold": float(options["material_threshold"]),
        "max_missing_fraction": float(options["max_missing_fraction"]),
        "confidence": float(options["confidence"]),
        "material_alpha": float(options["material_alpha"]),
        "hac_lag": int(options["hac_lag"]),
        "block_size": int(options["block_size"]),
        "bootstrap_draws": int(options["bootstrap_draws"]),
        "bootstrap_seed": int(options["bootstrap_seed"]),
    }
    adjusted = infer_adjusted_opportunity_loss(
        rows,
        folds=int(options["cross_fit_folds"]),
        **common,
    )
    unadjusted = infer_opportunity_loss(rows, **common)
    mechanism = assess_followup_mechanism(
        lifecycle_rows=lifecycle_rows,
        assignments=rows,
        contract=options["mechanism_followup"],
        max_staleness_versions=int(options["max_staleness_versions"]),
    )
    group_count = sum(row.get("event_type") == "group" for row in opportunity_rows)
    duty = assess_observer_duty(
        _parse_canonical_object(duty_data, name="observer duty"),
        assignment_domain=protocol.assignment_domain,
        release_arm_labels=("control", "d5"),
        expected_observation_count=group_count,
        maximum_corrected_observer_duty=float(
            options["maximum_corrected_observer_duty"]
        ),
    )
    return {
        "schema_version": 1,
        "result_scope": WORKLOAD_TRANSPORT_IDENTITIES[
            str(options["protocol_identity"])
        ]["result_scope"],
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
        "causal_conclusion": adjusted.conclusion,
        "primary_adjusted_inference": adjusted.to_dict(),
        "supporting_unadjusted_inference": unadjusted.to_dict(),
        "supporting_unadjusted_cannot_override_primary": True,
        "mechanism_followup_conclusion": mechanism.conclusion,
        "mechanism_followup": mechanism.to_dict(),
        "mechanism_is_support_condition_not_primary_endpoint": True,
        "portability_qualifier": duty.conclusion,
        "observer_duty": duty.to_dict(),
        "portability_does_not_modify_causal_conclusion": True,
        "qualification_data_entered_estimator": False,
        "automatic_extension_or_retry_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--lifecycle", required=True)
    parser.add_argument("--opportunity", required=True)
    parser.add_argument("--observer-duty", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = build_workload_transport_result(
        protocol_path=args.protocol,
        lifecycle_path=args.lifecycle,
        opportunity_path=args.opportunity,
        observer_duty_path=args.observer_duty,
    )
    write_result(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
