from dataclasses import replace

from tools.analyze_sliding_puzzle_7b_competence import _summarize
from tools.analyze_sliding_puzzle_latency_feasibility import (
    DesignItem,
    GroupObservation,
)


def _observations() -> tuple[GroupObservation, ...]:
    values = []
    for pair in range(8):
        for stratum in ("easy", "hard"):
            ordinal = pair * 2 + (stratum == "hard")
            easy = stratum == "easy"
            values.append(
                GroupObservation(
                    item=DesignItem(
                        source_pool_ordinal=ordinal,
                        source_prompt_id=f"prompt-{ordinal}",
                        task_name=f"sliding_puzzle_{stratum}",
                        pair_id=f"pair-{pair}",
                        stratum=stratum,
                        optimal_distance=2 if easy else 10,
                        rendered_prompt_tokens=200,
                        board_state_sha256=f"{ordinal:064x}",
                    ),
                    dispatch_ns=ordinal * 10,
                    completed_ns=ordinal * 10 + 100,
                    ready_ns=ordinal * 10 + 110,
                    archived_ns=ordinal * 10 + 120,
                    mean_turns=2,
                    min_turns=2,
                    max_turns=2,
                    mean_generated_tokens=20,
                    min_generated_tokens=20,
                    max_generated_tokens=20,
                    solved_completions=2 if easy else 0,
                    truncations=0,
                    max_turns_reached=0,
                    action_turns=4,
                    format_valid_actions=4,
                    legal_moves=3,
                    invalid_format_actions=0,
                    invalid_moves=1,
                    view_actions=0,
                )
            )
    return tuple(values)


def test_passing_competence_only_authorizes_fresh_design() -> None:
    result = _summarize(_observations())
    assert result["decision"] == "pass_competence_to_fresh_disjoint_latency_design"
    assert result["actions"]["format_valid_rate"] == 1
    assert result["actions"]["movement_legal_rate"] == 0.75
    assert result["solve_rate_by_stratum"]["easy"] == 1
    assert result["latency_or_readiness_inference_allowed"] is False
    assert result["replay_authorized"] is False
    assert result["training_authorized"] is False
    assert "ready_latency" not in result


def test_easy_competence_failure_stops_progression() -> None:
    values = [
        replace(item, solved_completions=0) if item.item.stratum == "easy" else item
        for item in _observations()
    ]
    result = _summarize(tuple(values))
    assert result["locked_checks"]["easy_solve_rate_ge_0_5"] is False
    assert result["decision"] == "stop_model_not_competent"


def test_invalid_movements_fail_legal_action_gate() -> None:
    values = [replace(item, legal_moves=2, invalid_moves=2) for item in _observations()]
    result = _summarize(tuple(values))
    assert result["actions"]["movement_legal_rate"] == 0.5
    assert result["locked_checks"]["movement_legal_rate_ge_0_75"] is False
    assert result["decision"] == "stop_model_not_competent"
