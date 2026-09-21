import json
from pathlib import Path
from types import SimpleNamespace

from tools import analyze_dapo_load_alignment_replications as analyzer


def _result(plan_id: str, seed: int) -> dict[str, object]:
    pressure = lambda decisions: {
        "selection_pressure": {
            "decisions_with_ready_greater_than_eligible": decisions,
        }
    }
    comparison = lambda lower, easier: {
        "fixed_lower_load_mean_normalized_selection_step_promotion": lower,
        "fixed_easier_mean_normalized_selection_step_promotion": easier,
    }
    return {
        "plan_id": plan_id,
        "order_seed": seed,
        "decision": "pass_to_next_replication",
        "arms": {
            "l0_natural_in_order": pressure(0),
            "l3_high_load_delayed_in_order": pressure(6),
            "l3_low_load_delayed_in_order": pressure(6),
        },
        "comparisons": {
            "l0_natural": comparison(0.0, 0.0),
            "l3_natural": comparison(1 / 14, 1 / 14),
            "l3_high_load_delayed": comparison(1 / 14, 0.0),
            "l3_low_load_delayed": comparison(-1 / 14, 0.0),
        },
    }


def test_all_locked_gates_only_draft_a_separate_candidate(tmp_path: Path) -> None:
    plan_id = "a" * 64
    thresholds = SimpleNamespace(
        signed_pressure_median_min=0.5,
        signed_pressure_replications_exceeding_l0_min=2,
        high_load_delayed_promotion_median_min=1 / 28,
        high_load_delayed_replications_at_or_above_min=2,
        low_load_delayed_promotion_median_max=-1 / 28,
        low_load_delayed_replications_at_or_below_max=2,
        signed_separation_median_min=1 / 14,
        natural_lower_load_promotion_median_min=1 / 28,
        natural_lower_load_replications_at_or_above_min=2,
        natural_lower_load_negative_replications_max=1,
        natural_easier_promotion_median_min=1 / 28,
        natural_easier_replications_at_or_above_min=2,
        natural_easier_negative_replications_max=1,
    )
    plan = SimpleNamespace(
        plan_id=plan_id,
        replication_order=(51001, 51002, 51003),
        thresholds=thresholds,
    )
    paths = []
    for seed in plan.replication_order:
        path = tmp_path / f"{seed}.json"
        path.write_text(json.dumps(_result(plan_id, seed)))
        paths.append(path)

    result = analyzer.analyze(plan, paths)  # type: ignore[arg-type]

    assert result["decision"] == "draft_separate_fresh_counterfactual_replay_candidate"
    assert result["automatic_replay"] is False
    assert result["automatic_training"] is False
