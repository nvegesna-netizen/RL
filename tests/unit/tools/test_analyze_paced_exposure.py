from tools.analyze_paced_exposure import SelectedGroup, summarize


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
