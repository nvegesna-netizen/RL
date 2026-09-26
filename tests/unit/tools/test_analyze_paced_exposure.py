from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.analyze_paced_exposure import (
    SelectedGroup,
    summarize,
    contrast_series,
    analyze_pilot,
)


def test_empty_wall_checkpoint_is_not_zero_reward() -> None:
    result = summarize([])
    assert result["selected_groups"] == 0
    assert result["mean_reward"] is None
    assert result["lower_load_share"] is None


def test_exposure_and_age_summary() -> None:
    groups = [
        SelectedGroup(10.0, True, False, 0.25, 100.0, True, 1, 2.0),
        SelectedGroup(10.0, False, True, 0.75, 300.0, False, 3, 4.0),
    ]
    result = summarize(groups)
    assert result == {
        "selected_groups": 2,
        "lower_load_share": 0.5,
        "harder_share": 0.5,
        "mean_reward": 0.5,
        "mean_tokens": 200.0,
        "nonconstant_reward_share": 0.5,
        "mean_admission_age_steps": 2.0,
        "mean_ready_wait_seconds": 3.0,
    }


def test_contrasts_keep_empty_wall_checkpoints_undefined() -> None:
    ready = [
        {"selected_groups": 4, "mean_reward": 0.5, "seconds_from_first_dispatch": 10.0}
    ]
    ordered = [
        {"selected_groups": 0, "mean_reward": None, "seconds_from_first_dispatch": 10.0}
    ]
    result = contrast_series(ready, ordered, equal_count=False)[0]
    assert result["selected_groups"] == 4
    assert result["mean_reward"] is None
    assert result["seconds_from_first_dispatch"] == 10.0
    with pytest.raises(ValueError, match="equal-count"):
        contrast_series(ready, ordered, equal_count=True)


def test_contrast_direction() -> None:
    result = contrast_series(
        [{"selected_groups": 16, "harder_share": 0.625}],
        [{"selected_groups": 16, "harder_share": 0.5}],
        equal_count=True,
    )
    assert result[0]["harder_share"] == 0.125
    assert result[0]["ready_first_selected_groups"] == 16


def test_missing_arms_are_not_zero_effects(tmp_path: Path) -> None:
    plan = SimpleNamespace(
        plan_id="test",
        arms=tuple(
            SimpleNamespace(
                arm_id=f"{sampler}-{factor}", sampler=sampler, cadence_multiplier=factor
            )
            for factor in (0.5, 1.0, 2.0)
            for sampler in ("in_order", "ready_first")
        ),
    )
    result = analyze_pilot(plan, arms_root=tmp_path, allow_partial=True)
    assert result["analysis_status"] == "exploratory_paced_exposure_partial_pilot"
    assert len(result["unavailable_arms"]) == 6
    assert result["arms"] == result["paired_contrasts"] == []
    with pytest.raises(FileNotFoundError):
        analyze_pilot(plan, arms_root=tmp_path, allow_partial=False)
