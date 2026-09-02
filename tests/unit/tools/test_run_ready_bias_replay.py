# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

from types import SimpleNamespace

from nemo_rl.algorithms.async_utils.scheduler_replay import (
    ReleaseSchedule,
    ReplayGroup,
    ReplayPolicy,
    ReplayTick,
    replay_schedule,
)
from tools.run_ready_bias_replay import (
    _interaction_summaries,
    _metric_or_unreached,
)


def _group() -> ReplayGroup:
    return ReplayGroup(
        logical_group_id="group",
        prompt_uid="prompt",
        source_prompt_id="source",
        task_stratum="task",
        repeated_prompt_cluster_uid="cluster",
        dispatch_cohort="0",
        decorrelation_block="block",
        slot_order=0,
        dispatch_ns=0,
        ready_ns=10,
        nominal_start_version=0,
        target_step=0,
    )


def test_counterfactual_queue_residence_uses_active_release_schedule():
    groups = (_group(),)
    release = ReleaseSchedule(
        kind="decorrelated", ready_ns_by_group={"group": 5}, seed=0
    )
    result = replay_schedule(
        groups,
        (ReplayTick(0, 10, 0, 1, 1),),
        ReplayPolicy("ready_first"),
        releases=release,
    )
    metrics = _metric_or_unreached(groups, result, release, 1)
    assert metrics["ready_to_select_mean_ns_by_task"] == {"task": 5.0}


def _metric_row(policy: str, kind: str, seed: int | None) -> dict[str, object]:
    return {
        "run_id": "run",
        "deadline_ns": 1,
        "policy": policy,
        "release_kind": kind,
        "release_seed": seed,
        "horizon": 1,
        "horizon_reached": True,
        "total_variation": 0.1 if policy == "policy" else 0.0,
        "signed_share_shift": {"task": 0.1 if policy == "policy" else 0.0},
    }


def test_interaction_requires_every_predeclared_permutation():
    plan = SimpleNamespace(
        policies=(
            SimpleNamespace(label="baseline", kernel="in_order"),
            SimpleNamespace(label="policy", kernel="ready_first"),
        ),
        source_runs=(SimpleNamespace(run_id="run"),),
        deadlines_ns=(1,),
        horizons=(1,),
        release_intervention=SimpleNamespace(seeds=range(2), seed_count=2),
    )
    rows = [
        _metric_row("baseline", "natural", None),
        _metric_row("policy", "natural", None),
        _metric_row("baseline", "decorrelated", 0),
        _metric_row("policy", "decorrelated", 0),
        _metric_row("baseline", "decorrelated", 1),
        {
            **_metric_row("policy", "decorrelated", 1),
            "horizon_reached": False,
        },
    ]
    summaries = _interaction_summaries(rows, plan)
    assert all(item["horizon_reached"] is False for item in summaries)
    assert {item["paired_permutations"] for item in summaries} == {1}
    assert {item["required_permutations"] for item in summaries} == {2}
