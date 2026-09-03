import copy
import hashlib
import json

import pytest

from tools.run_scheduler_positive_control import (
    EXPECTED_THRESHOLDS,
    PositiveControlError,
    _canonical_json,
    _plan_id,
    analyze_plan,
    load_plan,
)


def _plan() -> dict[str, object]:
    value = {
        "schema_version": 1,
        "analysis_status": "engineering_scheduler_latency_positive_control",
        "calibration_only": True,
        "confirmatory_eligible": False,
        "replay_authorized": False,
        "training_authorized": False,
        "analysis_code_commit": "a" * 40,
        "pool_groups": 32,
        "groups_per_stratum": 16,
        "fast_latency_ns": 100_000_000,
        "slow_latency_ns": 900_000_000,
        "tick_times_ns": [200_000_000, 400_000_000],
        "min_max_groups_per_tick": 4,
        "horizon": 8,
        "permutation_seed_start": 0,
        "permutation_seed_count": 1000,
        "thresholds": EXPECTED_THRESHOLDS,
    }
    value["plan_id"] = _plan_id(value)
    return value


def test_positive_control_detects_constructed_latency_mixture() -> None:
    result, permutations, assignments = analyze_plan(_plan())

    assert result["decision"] == "positive_control_passed"
    assert result["natural"]["fast_share"] == 1
    assert result["natural"]["total_variation"] == 0.5
    assert result["permutations"]["horizon_reached"] == 1000
    assert result["fast_share_interaction"] >= 0.4
    assert len(permutations) == 1000
    assert len(assignments) == 32_002


def test_plan_hash_and_surface_are_fail_closed(tmp_path) -> None:
    plan = _plan()
    path = tmp_path / "plan.json"
    path.write_text(_canonical_json(plan))
    assert load_plan(path) == plan

    changed = copy.deepcopy(plan)
    changed["horizon"] = 4
    path.write_text(json.dumps(changed))
    with pytest.raises(PositiveControlError, match="plan_id mismatch"):
        load_plan(path)

    changed["plan_id"] = _plan_id(changed)
    path.write_text(json.dumps(changed))
    with pytest.raises(PositiveControlError, match="invalid horizon"):
        load_plan(path)


def test_plan_id_is_canonical() -> None:
    plan = _plan()
    expected = hashlib.sha256(
        _canonical_json(
            {key: value for key, value in plan.items() if key != "plan_id"}
        ).encode()
    ).hexdigest()
    assert plan["plan_id"] == expected
