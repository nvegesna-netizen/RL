import torch

from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.processors import (
    PROCESSOR_REGISTRY,
    sliding_puzzle_fixed_pool_data_processor,
)


class _Tokenizer:
    def __call__(self, text, *, return_tensors, add_special_tokens):
        assert return_tensors == "pt" and not add_special_tokens
        return {"input_ids": torch.tensor([[1, 2, 3]])}


def test_sliding_puzzle_fixed_pool_processor_preserves_bound_state():
    task = TaskDataSpec(task_name="sliding_puzzle_easy")
    extra = {
        "game_state": {"grid": [[1, 2, 3], [4, 5, 6], [7, 0, 8]]},
        "num_moves": 0,
        "max_moves": 12,
        "optimal_distance": 1,
    }
    result = sliding_puzzle_fixed_pool_data_processor(
        {
            "messages": [{"role": "user", "content": "rendered puzzle"}],
            "extra_env_info": extra,
            "stop_strings": ["</action>"],
            "task_name": "sliding_puzzle_easy",
        },
        task,
        _Tokenizer(),
        512,
        7,
    )
    assert result["message_log"][0]["content"] == "rendered puzzle"
    assert result["extra_env_info"] == extra
    assert result["stop_strings"] == ["</action>"]
    assert result["idx"] == 7
    assert (
        PROCESSOR_REGISTRY["sliding_puzzle_fixed_pool_data_processor"]
        is sliding_puzzle_fixed_pool_data_processor
    )
