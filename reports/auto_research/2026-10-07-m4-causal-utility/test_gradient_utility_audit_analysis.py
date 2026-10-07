import json
from pathlib import Path

import numpy as np

from analyze_gradient_utility_audit import analyze


def test_analysis_is_deterministic_and_enforces_no_update_gate(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    groups = 32
    bins = 16
    seeds = [11, 13]
    protocol = {
        "acquisition": {"prompt_groups": groups, "siblings_per_group": 8},
        "gradient_measurement": {"sketch_seeds": seeds, "sketch_bins": bins},
        "analysis": {"folds": 8, "bootstrap_resamples": 20, "bootstrap_seed": 17},
        "integrity_gate": {
            "median_sketch_norm_relative_error_max": 10.0,
            "p95_sketch_norm_relative_error_max": 10.0,
        },
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))

    events = [
        {
            "event_type": "header",
            "schema_version": 1,
            "sketch_bins": bins,
            "sketch_seeds": seeds,
        }
    ]
    for index in range(groups):
        m4 = float(index + 1)
        first = rng.normal(size=bins) + m4 / 10.0
        second = first + rng.normal(scale=0.01, size=bins)
        exact_norm = float(np.linalg.norm(first))
        events.append(
            {
                "event_type": "group_gradient",
                "audit_index": index,
                "fold": index % 8,
                "group_id": f"g{index}",
                "sample_ids": [f"g{index}-s{s}" for s in range(8)],
                "abort_acknowledged": True,
                "metadata": {
                    "gradient_opportunity_l1": m4,
                    "gradient_opportunity_valid_actor_tokens": 100 + index,
                    "gradient_opportunity_reward_mean": index % 3 / 2.0,
                    "gradient_opportunity_reward_variance": float(index % 5),
                    "gradient_opportunity_truncation_count": index % 2,
                },
                "summary": {
                    "exact_l2_norm": exact_norm,
                    "sketches": [first.tolist(), second.tolist()],
                },
            }
        )
    events.append(
        {
            "event_type": "terminal",
            "finish_train_step_calls": 0,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "learner_version": 0,
            "parameter_hash_unchanged": True,
        }
    )
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_text("".join(json.dumps(event) + "\n" for event in events))

    first = analyze(ledger_path, protocol_path, bootstrap_resamples=20)
    second = analyze(ledger_path, protocol_path, bootstrap_resamples=20)

    assert first == second
    assert first["groups"] == groups
    assert first["integrity"]["optimizer_steps_zero"] is True
    assert first["status"] == "MEASUREMENT_FAILURE"


def test_analysis_rejects_incomplete_ledger(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(
        json.dumps(
            {
                "acquisition": {"prompt_groups": 1, "siblings_per_group": 8},
                "gradient_measurement": {"sketch_seeds": [1], "sketch_bins": 2},
                "analysis": {"folds": 8, "bootstrap_resamples": 1, "bootstrap_seed": 1},
                "integrity_gate": {
                    "median_sketch_norm_relative_error_max": 1,
                    "p95_sketch_norm_relative_error_max": 1,
                },
            }
        )
    )
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_text('{"event_type":"header"}\n')

    try:
        analyze(ledger_path, protocol_path)
    except ValueError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("incomplete ledger was accepted")
