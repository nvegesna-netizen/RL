"""Full synthetic 64-group traces exercise real validators, without generation."""

import json
from pathlib import Path

import pytest

from nemo_rl.algorithms.async_utils.paced_exposure import (
    PacedExposurePlan,
    compute_paced_exposure_plan_id,
)
from tests.unit.single_controller.test_paced_exposure import plan_record
from tools.analyze_paced_exposure import analyze_pilot


def write_trace(plan: PacedExposurePlan, *, arm_id: str, root: Path) -> Path:
    arm = plan.arm(arm_id)
    records: list[dict] = []
    clock_ns = 0

    def emit(kind: str, **fields: object) -> None:
        nonlocal clock_ns
        clock_ns += 1_000_000
        records.append(
            {
                "schema_version": 1,
                "trace_run_id": arm_id,
                "process_epoch": arm_id,
                "event_seq": len(records),
                "monotonic_ns": clock_ns,
                "event_type": kind,
                "trainer_version": 0,
                **fields,
            }
        )

    emit(
        "run_started",
        run_mode="scheduler_assay",
        sampler_name=arm.sampler,
        pool_id=plan.pools[0].pool_id,
        pool_manifest_sha256=plan.pools[0].manifest_sha256,
        model_revision=plan.model_revision,
        model_weights_sha256=plan.model_weights_sha256,
        scalar_summaries={
            "scheduler_assay_plan_id": plan.plan_id,
            "scheduler_assay_arm_id": arm_id,
            "generation_study_seed": arm.generation_seed,
            "fixed_pool_design_id": plan.source_design_id,
            "max_inflight_prompts": 4,
            "max_buffered_rollouts": 16,
            "num_generations_per_prompt": 16,
            "num_prompts_per_step": 4,
            "configured_max_new_tokens": 4096,
            "max_total_sequence_length": 6144,
            "generation_temperature": 1.0,
            "generation_top_p": 0.7,
            "generation_backend": "vllm",
            "generation_ignore_eos": False,
        },
    )
    for step in range(16):
        admission = f"a-{step}"
        emit(
            "admission_granted",
            admission_id=admission,
            sampler_dispatch_index=step,
            scalar_summaries={"expected_prompt_groups": 4},
        )
        identities = [
            {
                "logical_group_id": f"g-{ordinal}",
                "attempt_id": f"g-{ordinal}",
                "admission_id": admission,
                "source_pool_ordinal": ordinal,
                "source_prompt_id": plan.pools[0].source_prompt_ids[ordinal],
            }
            for ordinal in range(step * 4, step * 4 + 4)
        ]
        for identity in identities:
            emit("attempt_dispatched", **identity)
        for identity in identities:
            emit(
                "rollout_completed",
                **identity,
                scalar_summaries={
                    "completion_count": 16,
                    "reward_mean": 0.5,
                    "reward_min": 0.0,
                    "reward_max": 1.0,
                    "mean_gen_tokens_per_sample": 100.0,
                },
            )
            emit("group_ready", **identity)
        ids = [identity["logical_group_id"] for identity in identities]
        emit(
            "select_decision",
            selected_logical_group_ids=ids,
            eligible_logical_group_ids=ids,
            min_prompt_groups=4,
            max_prompt_groups=4,
            ready_prompt_groups=4,
            eligible_prompt_groups=4,
            scalar_summaries={
                "selected_prompt_groups": 4,
                "scheduler_assay_step": step,
            },
        )
        summaries = {
            "scheduler_assay_step": step,
            "physical_weight_version": 0,
            "selected_prompt_groups": 4,
            "requested_consumer_seconds": arm.consumer_duration_seconds,
        }
        emit(
            "consumer_buffer_released",
            selected_logical_group_ids=ids,
            scalar_summaries=summaries,
        )
        emit(
            "consumer_started",
            selected_logical_group_ids=ids,
            scalar_summaries=summaries,
        )
        clock_ns += round(arm.consumer_duration_seconds * 1e9)
        emit(
            "consumer_completed",
            selected_logical_group_ids=ids,
            scalar_summaries={
                **summaries,
                "actual_consumer_seconds": arm.consumer_duration_seconds,
            },
        )
    emit(
        "consumer_drained",
        scalar_summaries={
            "scheduler_assay_step": 16,
            "physical_weight_version": 0,
            "selected_prompt_groups": 64,
            "buffered_prompt_groups": 0,
        },
    )
    emit(
        "run_ended",
        terminal_reason="scheduler_assay_complete",
        scalar_summaries={
            "completed_train_steps": 0,
            "final_physical_weight_version": 0,
            "final_scheduler_assay_step": 16,
            "assay_selection_steps": 16,
            "assay_selected_prompt_groups": 64,
        },
    )
    path = root / arm_id / "scheduler_trace.v1.jsonl"
    path.parent.mkdir()
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def test_complete_and_partial_pilot_use_real_lifecycle_validators(
    tmp_path: Path,
) -> None:
    plan = PacedExposurePlan.model_validate(plan_record())
    plan = plan.model_copy(update={"plan_id": compute_paced_exposure_plan_id(plan)})
    paths = [write_trace(plan, arm_id=arm.arm_id, root=tmp_path) for arm in plan.arms]
    result = analyze_pilot(plan, arms_root=tmp_path, allow_partial=False)
    assert len(result["arms"]) == 6 and len(result["paired_contrasts"]) == 3
    for arm in result["arms"]:
        assert len(arm["cumulative_prefixes"]) == 16
        assert len(arm["rolling_16_group_windows"]) == 13
        assert len(arm["disjoint_16_group_blocks"]) == 4
        assert arm["maximum_active_generation_groups"] == 4
        assert arm["final_drain"]["lower_load_share"] == 0.5
        assert arm["final_drain"]["harder_share"] == 0.5
    text = paths[0].read_text().splitlines()
    paths[0].write_text("\n".join(text[:-1]) + "\n")
    with pytest.raises(RuntimeError):
        analyze_pilot(plan, arms_root=tmp_path, allow_partial=False)
    partial = analyze_pilot(plan, arms_root=tmp_path, allow_partial=True)
    assert len(partial["arms"]) == 5 and len(partial["paired_contrasts"]) == 2
    assert len(partial["unavailable_arms"]) == 1
    assert "g-0" not in json.dumps(partial)
