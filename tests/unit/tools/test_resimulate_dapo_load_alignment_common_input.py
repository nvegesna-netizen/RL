from __future__ import annotations

from tools.dapo_common_input_resimulation import (
    FixedStreamGroup,
    compare_samplers,
    shifted_ready_times,
    simulate_fixed_stream,
)


def _groups() -> tuple[FixedStreamGroup, ...]:
    values = []
    for index in range(32):
        cohort = index // 4
        values.append(
            FixedStreamGroup(
                logical_group_id=f"group-{index}",
                source_prompt_id=f"prompt-{index}",
                source_pool_ordinal=index,
                dispatch_cohort=cohort,
                dispatch_order=index,
                dispatch_ns=index,
                rollout_completed_ns=1_000 + (31 - index) * 100,
                observed_ready_ns=1_000 + (31 - index) * 100,
            )
        )
    return tuple(values)


def test_ready_first_reorders_identical_fixed_stream() -> None:
    groups = _groups()
    ready = shifted_ready_times(
        groups, delayed_prompt_ids=frozenset(), delay_seconds=0.0
    )
    in_order = simulate_fixed_stream(
        groups, sampler="in_order", ready_ns_by_group=ready
    )
    ready_first = simulate_fixed_stream(
        groups, sampler="ready_first", ready_ns_by_group=ready
    )

    assert in_order.selections[0].selected_group_ids == (
        "group-0",
        "group-1",
        "group-2",
        "group-3",
    )
    assert ready_first.selections[0].selected_group_ids == (
        "group-28",
        "group-29",
        "group-30",
        "group-31",
    )


def test_cohort_restricted_ready_first_is_exact_negative_control() -> None:
    groups = _groups()
    ready = shifted_ready_times(
        groups,
        delayed_prompt_ids=frozenset(f"prompt-{index}" for index in range(16)),
        delay_seconds=16.0,
    )
    in_order = simulate_fixed_stream(
        groups, sampler="in_order", ready_ns_by_group=ready
    )
    ready_first = simulate_fixed_stream(
        groups,
        sampler="ready_first",
        ready_ns_by_group=ready,
        cohort_restricted_ready_first=True,
    )

    assert tuple(x.selected_group_ids for x in in_order.selections) == tuple(
        x.selected_group_ids for x in ready_first.selections
    )


def test_selection_ticks_coalesce_ready_events_before_selection() -> None:
    groups = _groups()
    ready = {
        group.logical_group_id: 100 + group.source_pool_ordinal for group in groups
    }
    ready["group-0"] = 150
    immediate = simulate_fixed_stream(
        groups, sampler="ready_first", ready_ns_by_group=ready
    )
    ticked = simulate_fixed_stream(
        groups,
        sampler="ready_first",
        ready_ns_by_group=ready,
        selection_tick_ns=(1_000,),
    )

    assert immediate.selections[0].selected_group_ids == (
        "group-1",
        "group-2",
        "group-3",
        "group-4",
    )
    assert ticked.selections[0].selected_group_ids == (
        "group-0",
        "group-1",
        "group-2",
        "group-3",
    )


def test_signed_delay_comparison_uses_fixed_reference_halves() -> None:
    groups = _groups()
    lower = frozenset(f"prompt-{index}" for index in range(0, 32, 2))
    easier = frozenset(f"prompt-{index}" for index in range(16))
    higher = frozenset(f"prompt-{index}" for index in range(1, 32, 2))
    high_ready = shifted_ready_times(
        groups, delayed_prompt_ids=higher, delay_seconds=16.0
    )
    low_ready = shifted_ready_times(
        groups, delayed_prompt_ids=lower, delay_seconds=16.0
    )

    def compare(ready: dict[str, int]):
        return compare_samplers(
            simulate_fixed_stream(groups, sampler="in_order", ready_ns_by_group=ready),
            simulate_fixed_stream(
                groups, sampler="ready_first", ready_ns_by_group=ready
            ),
            lower_load_prompt_ids=lower,
            easier_prompt_ids=easier,
        )

    high = compare(high_ready)
    low = compare(low_ready)
    assert high.lower_load_mean_normalized_promotion > 0
    assert low.lower_load_mean_normalized_promotion < 0
    assert high.lower_load_net_step_displacement > 0
    assert low.lower_load_net_step_displacement < 0
