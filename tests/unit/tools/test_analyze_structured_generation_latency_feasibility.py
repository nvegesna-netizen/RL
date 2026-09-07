import pytest

from tools.analyze_structured_generation_latency_feasibility import (
    DesignItem,
    Observation,
    summarize,
)


def _observations(*, ready_long: int = 220, reward: float = 1.0):
    output = []
    for pair in range(8):
        for stratum in ("short", "long"):
            is_long = stratum == "long"
            ordinal = pair * 2 + int(is_long)
            dispatch = ordinal * 10
            ready_latency = ready_long if is_long else 100
            item = DesignItem(
                source_pool_ordinal=ordinal,
                source_prompt_id=f"prompt-{ordinal}",
                task_name=f"structured_{stratum}",
                pair_id=f"pair-{pair}",
                stratum=stratum,
                left_operand=100 + pair,
                right_operand=200 + pair,
                expected_answer=300 + 2 * pair,
                required_check_lines=16 if is_long else 2,
                rendered_prompt_tokens=80,
            )
            output.append(
                Observation(
                    item=item,
                    dispatch_ns=dispatch,
                    completed_ns=dispatch + ready_latency - 1,
                    ready_ns=dispatch + ready_latency,
                    archived_ns=dispatch + ready_latency + 1,
                    reward_mean=reward,
                    mean_generated_tokens=240 if is_long else 60,
                    min_generated_tokens=240 if is_long else 60,
                    max_generated_tokens=240 if is_long else 60,
                    length_terminations=0,
                )
            )
    return tuple(output)


def test_all_locked_feasibility_gates_pass() -> None:
    result = summarize(_observations())

    assert result["decision"] == "pass_to_fresh_replicated_scheduler_design"
    assert all(result["locked_checks"].values())
    assert result["generated_output_tokens"]["median_long_short_ratio"] == 4
    assert result["ready_latency"]["median_long_short_ratio"] == 2.2
    assert result["ready_latency"]["paired_longer_count"] == 8
    assert result["ready_latency"]["matched_pairs_rank_biserial"] == 1
    assert result["training_authorized"] is False
    assert result["scheduler_comparison_authorized"] is False


def test_failed_quality_gate_stops_scheduler_comparison() -> None:
    result = summarize(_observations(reward=0.5))

    assert result["decision"] == "stop_no_scheduler_comparison"
    assert not result["locked_checks"]["reward_mean_ge_0_75_each"]
    assert result["replay_authorized"] is False


def test_failed_latency_gate_stops_scheduler_comparison() -> None:
    result = summarize(_observations(ready_long=120))

    assert result["decision"] == "stop_no_scheduler_comparison"
    assert result["ready_latency"]["median_long_short_ratio"] == pytest.approx(1.2)
    assert not result["locked_checks"]["ready_latency_median_ratio_ge_1_5"]
