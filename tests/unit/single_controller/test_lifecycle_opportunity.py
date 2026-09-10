"""Tests for arm-blind lifecycle-derived GRPO opportunity reconstruction."""

from __future__ import annotations

import copy

import pytest
import torch

from nemo_rl.algorithms.advantage_estimator import (
    AdvEstimatorConfig,
    GRPOAdvantageEstimator,
)
from nemo_rl.algorithms.async_utils.controlled_release import (
    ControlledReleaseArmConfig,
    ControlledReleaseAssigner,
    ControlledReleaseDelayConfig,
)
from nemo_rl.algorithms.async_utils.lifecycle_opportunity import (
    LifecycleOpportunityError,
    derive_lifecycle_opportunity_rows,
)
from nemo_rl.algorithms.async_utils.gradient_opportunity import (
    GRPOOpportunityInputs,
    compute_grpo_gradient_opportunity,
)
from nemo_rl.algorithms.async_utils.rollout_lifecycle import (
    ControllerEventSequencer,
    RolloutLifecycleRecorder,
    RolloutLifecycleStage,
    RolloutRemovalReason,
)
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from tools.opportunity_ledger_join import (
    LedgerJoinProtocol,
    ReleaseArm,
    join_opportunity_ledgers,
)

_DOMAIN = "lifecycle-derived-test-v1"
_SEED = 20261007
_LOSS_SETTINGS = {
    "disable_ppo_ratio": False,
    "positive_example_nll_weight": 0.0,
    "sequence_level_importance_ratios": False,
    "token_level_loss": True,
    "use_cispo": False,
}


def _estimator() -> GRPOAdvantageEstimator:
    return GRPOAdvantageEstimator(AdvEstimatorConfig(), ClippedPGLossConfig())


def _fixture() -> tuple[list[dict[str, object]], ControlledReleaseDelayConfig]:
    sequencer = ControllerEventSequencer(clock_ns=iter(range(1000)).__next__)
    recorder = RolloutLifecycleRecorder(
        run_id="run",
        clock_domain_id="controller",
        controller_sequencer=sequencer,
    )
    release = ControlledReleaseDelayConfig(
        enabled=True,
        seed=_SEED,
        assignment_domain=_DOMAIN,
        arms=(
            ControlledReleaseArmConfig(label="control", delay_seconds=0.0, mass=1),
            ControlledReleaseArmConfig(label="d5", delay_seconds=5.0, mass=1),
        ),
    )
    assignment = ControlledReleaseAssigner(release).assign()
    group_id = "group"
    recorder.record(
        group_id=group_id,
        stage=RolloutLifecycleStage.RESERVED,
        start_weight_version=0,
    )
    release_fields = {
        "release_arm": assignment.arm_label,
        "release_delay_seconds": assignment.delay_seconds,
        "release_arm_mass": assignment.arm_mass,
        "release_total_mass": assignment.total_mass,
        "release_global_ordinal": assignment.global_ordinal,
        "release_draw": assignment.draw,
        "release_nonce": assignment.nonce,
        "generation_inflight": 1,
        "active_release_holds": 0,
        "reserved_buffer_occupancy": 1,
        "ready_buffer_depth": 0,
        "buffer_admission_stalls": 0,
    }
    recorder.record(
        group_id=group_id,
        stage=RolloutLifecycleStage.RELEASE_DELAY_ASSIGNED,
        start_weight_version=0,
        **release_fields,
    )
    for index in range(8):
        recorder.record(
            group_id=group_id,
            stage=RolloutLifecycleStage.SIBLING_DONE,
            start_weight_version=0,
            trajectory_id=f"{group_id}_g{index}",
            sibling_idx=index,
            assistant_tokens=index + 1,
            reward=float(index % 2),
            truncated=index == 7,
        )
    recorder.record(
        group_id=group_id,
        stage=RolloutLifecycleStage.RELEASE_DELAY_STARTED,
        start_weight_version=0,
        **release_fields,
    )
    recorder.record(
        group_id=group_id,
        stage=RolloutLifecycleStage.RELEASE_DELAY_COMPLETED,
        start_weight_version=0,
        **release_fields,
    )
    sample_ids = tuple(f"{group_id}_g{index}" for index in range(8))
    recorder.record(
        group_id=group_id,
        stage=RolloutLifecycleStage.GROUP_COMPLETED,
        start_weight_version=0,
        end_weight_version=0,
    )
    recorder.record(
        group_id=group_id,
        stage=RolloutLifecycleStage.GROUP_READY,
        start_weight_version=0,
        end_weight_version=0,
        sample_ids=sample_ids,
    )
    recorder.record(
        group_id=group_id,
        stage=RolloutLifecycleStage.REMOVED,
        start_weight_version=0,
        end_weight_version=0,
        learner_weight_version=0,
        sample_ids=sample_ids,
        removal_reason=RolloutRemovalReason.SELECTED,
    )
    recorder.record_learner_version_advanced(
        previous_version=0, learner_weight_version=1
    )
    return [event.to_dict() for event in recorder.snapshot()], release


def _derive(rows):
    return derive_lifecycle_opportunity_rows(
        rows,
        estimator=_estimator(),
        loss_settings=_LOSS_SETTINGS,
        siblings_per_group=8,
        train_batch_size=8,
    )


def test_derivation_reconstructs_exact_identity_q_and_completed_step() -> None:
    lifecycle, _ = _fixture()
    rows = _derive(lifecycle)
    group = next(row for row in rows if row["event_type"] == "group")
    step = next(row for row in rows if row["event_type"] == "train_step_completed")

    assert group["sample_ids"] == [f"group_g{index}" for index in range(8)]
    assert [sibling["valid_actor_tokens"] for sibling in group["siblings"]] == list(
        range(1, 9)
    )
    assert group["opportunity"] > 0
    assert step["sample_ids"] == group["sample_ids"]
    assert (
        step["source_controller_sequence_max"] > group["source_controller_sequence_max"]
    )


def test_q_derivation_is_byte_stable_under_release_field_mutation() -> None:
    lifecycle, _ = _fixture()
    mutated = copy.deepcopy(lifecycle)
    for row in mutated:
        if row["stage"].startswith("release_delay_"):
            row["release_arm"] = "arbitrary"
            row["release_delay_seconds"] = 999.0
            row["release_draw"] = 123
            row["release_nonce"] = 456
    assert _derive(lifecycle) == _derive(mutated)


def test_all_binary_reward_vectors_match_accepted_instrument_exactly() -> None:
    lifecycle, _ = _fixture()
    sibling_rows = [row for row in lifecycle if row["stage"] == "sibling_done"]
    sample_ids = tuple(f"group_g{index}" for index in range(8))
    token_counts = tuple(range(1, 9))
    actor_mask = torch.zeros((8, 9), dtype=torch.float32)
    for index, count in enumerate(token_counts):
        actor_mask[index, 1 : count + 1] = 1

    for reward_bits in range(1 << 8):
        rewards = torch.tensor(
            [(reward_bits >> index) & 1 for index in range(8)], dtype=torch.float32
        )
        for index, row in enumerate(sibling_rows):
            row["reward"] = float(rewards[index].item())
        derived = next(
            row for row in _derive(lifecycle) if row["event_type"] == "group"
        )
        inputs = GRPOOpportunityInputs(
            prompt_ids=torch.zeros(8, dtype=torch.long),
            rewards=rewards,
            actor_mask=actor_mask,
            repeated_batch={"total_reward": rewards},
            estimator_kwargs={},
        )
        accepted = compute_grpo_gradient_opportunity(
            {},
            group_id="group",
            sample_ids=sample_ids,
            start_weight_version=0,
            truncation=(False,) * 7 + (True,),
            estimator=_estimator(),
            prepare_inputs=lambda _: inputs,
        ).to_dict()

        assert {key: derived[key] for key in accepted} == accepted


def test_derived_rows_pass_strict_causal_join() -> None:
    lifecycle, release = _fixture()
    opportunity = _derive(lifecycle)
    protocol = LedgerJoinProtocol(
        assignment_domain=_DOMAIN,
        assignment_seed=_SEED,
        arms=tuple(
            ReleaseArm(arm.label, arm.delay_seconds, arm.mass) for arm in release.arms
        ),
        primary_start_version=0,
        primary_end_version=0,
        siblings_per_group=8,
        train_batch_size=8,
    )

    joined = join_opportunity_ledgers(
        protocol=protocol,
        lifecycle_rows=lifecycle,
        opportunity_rows=opportunity,
    )

    assert len(joined) == 1
    assert joined[0].delivered is True
    assert joined[0].opportunity > 0


def test_derivation_rejects_incomplete_sibling_evidence() -> None:
    lifecycle, _ = _fixture()
    lifecycle = [
        row
        for row in lifecycle
        if not (row["stage"] == "sibling_done" and row["sibling_idx"] == 7)
    ]
    for sequence, row in enumerate(lifecycle):
        row["sequence"] = sequence

    with pytest.raises(LifecycleOpportunityError, match="selected sample lacks"):
        _derive(lifecycle)
