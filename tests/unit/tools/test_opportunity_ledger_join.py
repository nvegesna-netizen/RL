"""Tests for strict lifecycle/opportunity all-assignment joins."""

from __future__ import annotations

import copy
from collections.abc import Sequence

import pytest

from nemo_rl.algorithms.async_utils.controlled_release import (
    ControlledReleaseArmConfig,
    ControlledReleaseAssigner,
    ControlledReleaseDelayConfig,
)
from nemo_rl.algorithms.async_utils.gradient_opportunity import (
    GradientOpportunityRecorder,
    GroupOpportunitySummary,
    SiblingOpportunitySummary,
)
from nemo_rl.algorithms.async_utils.rollout_lifecycle import (
    ControllerEventSequencer,
    RolloutLifecycleRecorder,
    RolloutLifecycleStage,
    RolloutRemovalReason,
)
from tools.opportunity_ledger_join import (
    LedgerJoinProtocol,
    OpportunityAuditContract,
    OpportunityLedgerJoinError,
    ReleaseArm,
    join_opportunity_ledgers,
)
from tools.opportunity_loss_analysis import bound_opportunity_loss

_ARMS = (
    ControlledReleaseArmConfig(label="control", delay_seconds=0.0, mass=5),
    ControlledReleaseArmConfig(label="d5", delay_seconds=5.0, mass=5),
    ControlledReleaseArmConfig(label="d10", delay_seconds=10.0, mass=2),
)
_DOMAIN = "m4-opportunity-loss-confirmatory-v1"
_SEED = 20260810


def _protocol(*, train_batch_size: int = 2) -> LedgerJoinProtocol:
    return LedgerJoinProtocol(
        assignment_domain=_DOMAIN,
        assignment_seed=_SEED,
        arms=tuple(ReleaseArm(arm.label, arm.delay_seconds, arm.mass) for arm in _ARMS),
        primary_start_version=0,
        primary_end_version=0,
        siblings_per_group=2,
        train_batch_size=train_batch_size,
        opportunity_audit=OpportunityAuditContract(
            normalize_rewards=False,
            use_leave_one_out_baseline=False,
        ),
    )


def _summary(group_id: str) -> GroupOpportunitySummary:
    sample_ids = (f"{group_id}_s0", f"{group_id}_s1")
    return GroupOpportunitySummary(
        group_id=group_id,
        sample_ids=sample_ids,
        start_weight_version=0,
        opportunity=2.0,
        l2_coefficient_mass=1.0,
        signed_coefficient_mass=0.0,
        positive_coefficient_mass=1.0,
        negative_coefficient_mass=1.0,
        valid_actor_tokens=4,
        nonzero_advantage_siblings=2,
        nonzero_advantage_tokens=4,
        siblings=(
            SiblingOpportunitySummary(
                sample_id=sample_ids[0],
                sibling_index=0,
                reward=1.0,
                scalar_advantage=0.5,
                valid_actor_tokens=2,
                opportunity=1.0,
                truncated=False,
            ),
            SiblingOpportunitySummary(
                sample_id=sample_ids[1],
                sibling_index=1,
                reward=0.0,
                scalar_advantage=-0.5,
                valid_actor_tokens=2,
                opportunity=1.0,
                truncated=False,
            ),
        ),
    )


def _fixture(
    states: Sequence[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]], LedgerJoinProtocol]:
    timestamps = iter(range(1, 10_000))
    sequencer = ControllerEventSequencer(clock_ns=lambda: next(timestamps))
    lifecycle = RolloutLifecycleRecorder(
        run_id="run",
        clock_domain_id="controller",
        controller_sequencer=sequencer,
    )
    opportunity = GradientOpportunityRecorder(
        run_id="run",
        clock_domain_id="controller",
        sequencer=sequencer,
    )
    opportunity.append_header(
        estimator_name="GRPOAdvantageEstimator",
        estimator_settings={
            "baseline_algorithm": "nemo_rl_grpo_v1",
            "normalization_epsilon": 1e-6,
            "normalize_rewards": False,
            "reduction_dtype": "float32",
            "reward_dtype": "float32",
            "use_leave_one_out_baseline": False,
        },
        loss_settings={
            "disable_ppo_ratio": False,
            "positive_example_nll_weight": 0.0,
            "sequence_level_importance_ratios": False,
            "token_level_loss": True,
            "use_cispo": False,
        },
    )
    release_config = ControlledReleaseDelayConfig(
        enabled=True,
        seed=_SEED,
        assignment_domain=_DOMAIN,
        arms=_ARMS,
    )
    assigner = ControlledReleaseAssigner(release_config)
    completed_ids: list[str] = []

    for index, state in enumerate(states):
        group_id = f"g{index}"
        summary = _summary(group_id)
        assignment = assigner.assign()
        lifecycle.record(
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
        lifecycle.record(
            group_id=group_id,
            stage=RolloutLifecycleStage.RELEASE_DELAY_ASSIGNED,
            start_weight_version=0,
            **release_fields,
        )
        for sibling_index, sample_id in enumerate(summary.sample_ids):
            lifecycle.record(
                group_id=group_id,
                stage=RolloutLifecycleStage.SIBLING_DONE,
                start_weight_version=0,
                trajectory_id=sample_id,
                sibling_idx=sibling_index,
                reward=float(1 - sibling_index),
                truncated=False,
            )
        opportunity.append_group(summary)
        lifecycle.record(
            group_id=group_id,
            stage=RolloutLifecycleStage.RELEASE_DELAY_STARTED,
            start_weight_version=0,
            **release_fields,
        )
        lifecycle.record(
            group_id=group_id,
            stage=RolloutLifecycleStage.RELEASE_DELAY_COMPLETED,
            start_weight_version=0,
            **release_fields,
        )
        if state in {
            "selected",
            "selected_incomplete",
            "stale",
            "oars_excess",
            "failed",
        }:
            lifecycle.record(
                group_id=group_id,
                stage=RolloutLifecycleStage.GROUP_READY,
                start_weight_version=0,
                end_weight_version=0,
                sample_ids=summary.sample_ids,
            )
        if state == "selected":
            lifecycle.record(
                group_id=group_id,
                stage=RolloutLifecycleStage.REMOVED,
                start_weight_version=0,
                sample_ids=summary.sample_ids,
                removal_reason=RolloutRemovalReason.SELECTED,
            )
            completed_ids.extend(summary.sample_ids)
        elif state == "selected_incomplete":
            lifecycle.record(
                group_id=group_id,
                stage=RolloutLifecycleStage.REMOVED,
                start_weight_version=0,
                sample_ids=summary.sample_ids,
                removal_reason=RolloutRemovalReason.SELECTED,
            )
        elif state == "stale":
            lifecycle.record(
                group_id=group_id,
                stage=RolloutLifecycleStage.REMOVED,
                start_weight_version=0,
                sample_ids=summary.sample_ids,
                removal_reason=RolloutRemovalReason.STALE_EVICTED,
            )
        elif state == "oars_excess":
            lifecycle.record(
                group_id=group_id,
                stage=RolloutLifecycleStage.REMOVED,
                start_weight_version=0,
                sample_ids=summary.sample_ids,
                removal_reason=RolloutRemovalReason.OARS_CANDIDATE_EXCESS,
            )
        elif state == "failed":
            lifecycle.record(
                group_id=group_id,
                stage=RolloutLifecycleStage.REMOVED,
                start_weight_version=0,
                sample_ids=summary.sample_ids,
                removal_reason=RolloutRemovalReason.FAILED,
            )
        elif state != "missing":
            raise AssertionError(f"unsupported fixture state {state}")

    if completed_ids:
        lifecycle.record_learner_version_advanced(
            previous_version=0, learner_weight_version=1
        )
        opportunity.append_train_step_completed(
            previous_learner_version=0,
            learner_version=1,
            sample_ids=completed_ids,
        )

    return (
        [event.to_dict() for event in lifecycle.snapshot()],
        list(opportunity.snapshot()),
        _protocol(train_batch_size=len(completed_ids) or 2),
    )


def test_join_classifies_completed_stale_failed_and_missing_terminal() -> None:
    lifecycle, opportunity, protocol = _fixture(
        [
            "selected",
            "stale",
            "oars_excess",
            "failed",
            "missing",
            "selected_incomplete",
        ]
    )

    rows = join_opportunity_ledgers(
        protocol=protocol,
        lifecycle_rows=lifecycle,
        opportunity_rows=opportunity,
    )

    assert [row.assignment_id for row in rows] == [
        "g0",
        "g1",
        "g2",
        "g3",
        "g4",
        "g5",
    ]
    assert [row.ordinal for row in rows] == [0, 1, 2, 3, 4, 5]
    assert {row.start_version for row in rows} == {0}
    assert [row.delivered for row in rows] == [
        True,
        False,
        False,
        False,
        None,
        None,
    ]
    assert [row.opportunity for row in rows] == [2.0] * 6


def test_joined_rows_feed_observed_q_missingness_bounds() -> None:
    states = ["stale"] * 24
    lifecycle, opportunity, protocol = _fixture(states)
    rows = [
        row.analysis_assignment()
        for row in join_opportunity_ledgers(
            protocol=protocol,
            lifecycle_rows=lifecycle,
            opportunity_rows=opportunity,
        )
    ]
    control_count = sum(row.arm == "control" for row in rows)
    d5_count = sum(row.arm == "d5" for row in rows)
    assert control_count > 0 and d5_count > 0

    # Turn exactly one observed disposition into terminal missingness without
    # removing its pre-delay opportunity.
    missing_index = next(index for index, row in enumerate(rows) if row.arm == "d5")
    row = rows[missing_index]
    rows[missing_index] = type(row)(row.assignment_id, row.arm, row.opportunity, None)
    result = bound_opportunity_loss(
        rows,
        max_missing_fraction=1.0 / d5_count,
    )

    assert result.treatment.missing_terminals == 1
    assert result.delta_l_upper >= result.delta_l_lower


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_opportunity", "lacks opportunity"),
        ("assignment_draw", "reconstruction"),
        ("sample_identity", "sibling identity disagrees"),
        ("ready_identity", "group_ready sample IDs disagree"),
        ("sibling_sum", "sibling opportunity recomputation failed"),
        ("estimator_setting", "estimator settings disagree"),
        ("loss_setting", "loss settings disagree"),
        ("reward", "sibling facts disagree"),
        ("advantage", "scalar advantage replay failed"),
        ("token_count", "sibling opportunity recomputation failed"),
        ("group_mass", "group l2_coefficient_mass recomputation failed"),
        ("opportunity_after_delay", "strictly pre-delay"),
        ("partial_completed_step", "completed sample lacks|partial"),
        ("unknown_terminal", "unknown terminal"),
    ],
)
def test_join_rejects_identity_order_and_terminal_contradictions(
    mutation: str, match: str
) -> None:
    lifecycle, opportunity, protocol = _fixture(["selected"])
    lifecycle = copy.deepcopy(lifecycle)
    opportunity = copy.deepcopy(opportunity)

    if mutation == "missing_opportunity":
        opportunity = [row for row in opportunity if row["event_type"] != "group"]
        opportunity = [
            row for row in opportunity if row["event_type"] != "train_step_completed"
        ]
    elif mutation == "assignment_draw":
        assigned = next(
            row for row in lifecycle if row["stage"] == "release_delay_assigned"
        )
        assigned["release_draw"] = (int(assigned["release_draw"]) + 1) % 12
    elif mutation == "sample_identity":
        sibling = next(row for row in lifecycle if row["stage"] == "sibling_done")
        sibling["trajectory_id"] = "forged"
    elif mutation == "ready_identity":
        ready = next(row for row in lifecycle if row["stage"] == "group_ready")
        ready["sample_ids"][0] = "forged"
    elif mutation == "sibling_sum":
        group = next(row for row in opportunity if row["event_type"] == "group")
        group["siblings"][0]["opportunity"] += 1.0
    elif mutation == "estimator_setting":
        opportunity[0]["estimator_settings"]["normalize_rewards"] = True
    elif mutation == "loss_setting":
        opportunity[0]["loss_settings"]["token_level_loss"] = False
    elif mutation == "reward":
        group = next(row for row in opportunity if row["event_type"] == "group")
        group["siblings"][0]["reward"] = 0.0
    elif mutation == "advantage":
        group = next(row for row in opportunity if row["event_type"] == "group")
        group["siblings"][0]["scalar_advantage"] = -0.5
    elif mutation == "token_count":
        group = next(row for row in opportunity if row["event_type"] == "group")
        group["siblings"][0]["valid_actor_tokens"] = 3
    elif mutation == "group_mass":
        group = next(row for row in opportunity if row["event_type"] == "group")
        group["l2_coefficient_mass"] += 1.0
    elif mutation == "opportunity_after_delay":
        group = next(row for row in opportunity if row["event_type"] == "group")
        delay = next(
            row for row in lifecycle if row["stage"] == "release_delay_started"
        )
        group["controller_sequence"], delay["controller_sequence"] = (
            delay["controller_sequence"],
            group["controller_sequence"],
        )
        group["timestamp_ns"], delay["timestamp_ns"] = (
            delay["timestamp_ns"],
            group["timestamp_ns"],
        )
    elif mutation == "partial_completed_step":
        completed = next(
            row for row in opportunity if row["event_type"] == "train_step_completed"
        )
        completed["sample_ids"][0] = "unknown"
    elif mutation == "unknown_terminal":
        removed = next(row for row in lifecycle if row["stage"] == "removed")
        removed["removal_reason"] = "unknown"

    with pytest.raises(OpportunityLedgerJoinError, match=match):
        join_opportunity_ledgers(
            protocol=protocol,
            lifecycle_rows=lifecycle,
            opportunity_rows=opportunity,
        )
