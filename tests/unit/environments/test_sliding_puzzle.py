from nemo_rl.environments.games.sliding_puzzle import SlidingPuzzleRunner


def _metadata():
    return {
        "game_state": {
            "size": 3,
            "grid": [[1, 2, 3], [4, 5, 6], [7, 0, 8]],
            "solution": [[1, 2, 3], [4, 5, 6], [7, 8, 0]],
            "empty_pos": [2, 1],
            "commands": {},
        },
        "num_moves": 0,
        "max_moves": 12,
        "optimal_distance": 1,
    }


def test_sliding_puzzle_reports_invalid_format_without_dropping_state():
    observation, reward, terminated, _, metadata, _ = (
        SlidingPuzzleRunner().process_turn(
            [{"role": "assistant", "content": "left"}], _metadata()
        )
    )
    assert observation["action_status"] == "invalid_format"
    assert reward == 0
    assert not terminated
    assert metadata is not None
    assert metadata["num_moves"] == 0


def test_sliding_puzzle_reports_legal_solving_move():
    observation, reward, terminated, _, metadata, _ = (
        SlidingPuzzleRunner().process_turn(
            [{"role": "assistant", "content": "<action>left</action>"}], _metadata()
        )
    )
    assert observation["action_status"] == "valid_move"
    assert reward == 1
    assert terminated
    assert metadata is None
