from dataclasses import replace

from tools.analyze_sliding_puzzle_latency_feasibility import (
    DesignItem,
    GroupObservation,
    analyze_observations,
)


def _observations() -> tuple[GroupObservation, ...]:
    observations = []
    for pair in range(8):
        for stratum in ("easy", "hard"):
            ordinal = pair * 2 + (stratum == "hard")
            hard = stratum == "hard"
            dispatch = (ordinal // 4) * 10_000 + (ordinal % 4) * 10
            turns = 6 if hard else 2
            tokens = 240 if hard else 100
            latency = 240 if hard else 100
            observations.append(
                GroupObservation(
                    item=DesignItem(
                        source_pool_ordinal=ordinal,
                        source_prompt_id=f"prompt-{ordinal}",
                        task_name=f"sliding_puzzle_{stratum}",
                        pair_id=f"pair-{pair}",
                        stratum=stratum,
                        optimal_distance=10 if hard else 2,
                        rendered_prompt_tokens=200,
                        board_state_sha256=f"{ordinal:064x}",
                    ),
                    dispatch_ns=dispatch,
                    completed_ns=dispatch + latency - 10,
                    ready_ns=dispatch + latency,
                    archived_ns=dispatch + latency + 5,
                    mean_turns=turns,
                    min_turns=turns,
                    max_turns=turns,
                    mean_generated_tokens=tokens,
                    min_generated_tokens=tokens,
                    max_generated_tokens=tokens,
                    solved_completions=1,
                    truncations=0,
                    max_turns_reached=0,
                    action_turns=turns * 2,
                    format_valid_actions=turns * 2,
                    legal_moves=turns * 2,
                    invalid_format_actions=0,
                    invalid_moves=0,
                    view_actions=0,
                )
            )
    return tuple(observations)


def test_passing_screen_only_authorizes_fresh_calibration() -> None:
    result = analyze_observations(_observations())
    assert result["decision"] == "pass_to_fresh_disjoint_replicated_calibration_design"
    assert result["calibration_only"] is True
    assert result["confirmatory_eligible"] is False
    assert result["replay_authorized"] is False
    assert result["training_authorized"] is False
    assert result["turns"]["median_hard_easy_ratio"] == 3
    assert result["ready_latency"]["median_hard_easy_ratio"] == 2.4


def test_invalid_action_format_fails_locked_screen() -> None:
    observations = list(_observations())
    observations[0] = replace(
        observations[0], format_valid_actions=0, invalid_format_actions=4
    )
    result = analyze_observations(observations)
    assert result["locked_checks"]["action_format_valid_rate_ge_0_9"] is False
    assert result["decision"] == "stop_no_replay"


def test_latency_gate_cannot_be_passed_by_one_outlier() -> None:
    observations = list(_observations())
    for index, observation in enumerate(observations):
        if observation.item.stratum == "hard" and observation.item.pair_id != "pair-0":
            observations[index] = replace(
                observation,
                completed_ns=observation.dispatch_ns + 90,
                ready_ns=observation.dispatch_ns + 100,
                archived_ns=observation.dispatch_ns + 105,
            )
    result = analyze_observations(observations)
    assert result["locked_checks"]["pair_ready_sign_count_ge_6"] is False
    assert result["decision"] == "stop_no_replay"
