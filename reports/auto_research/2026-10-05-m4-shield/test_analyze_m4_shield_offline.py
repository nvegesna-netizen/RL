"""Unit tests for the frozen M4-Shield development selector."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "analyze_m4_shield_offline.py"
SPEC = importlib.util.spec_from_file_location("m4_shield_offline", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
PROTOCOL = json.loads((HERE / "m4_shield_protocol.json").read_text())


def candidate(
    group_id: str,
    *,
    l1: float,
    reward_variance: float,
    tokens: int = 100,
    start: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        group_id=group_id,
        start_weight_version=start,
        ready_timestamp_ns=0,
        l1=l1,
        l2=l1,
        valid_actor_tokens=tokens,
        mean_reward=0.0,
        reward_variance=reward_variance,
    )


def decision(candidates: tuple[SimpleNamespace, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        identity="test",
        index=0,
        current_learner_version=1,
        candidates=candidates,
        baseline_group_ids=("a", "b", "c", "d"),
        baseline_tokens=400,
    )


def ids(selection: tuple[SimpleNamespace, ...]) -> tuple[str, ...]:
    return tuple(item.group_id for item in selection)


def test_repairs_exact_utility_tie_with_two_replacements() -> None:
    choices = tuple(
        candidate(letter, l1=value, reward_variance=1.0)
        for letter, value in zip("abcdefgh", (1, 2, 3, 4, 5, 6, 7, 8))
    )
    case = decision(choices)
    base = MODULE.select_reward_variance_base(case, PROTOCOL)
    shield = MODULE.select_m4_shield(case, base, PROTOCOL)
    assert ids(base) == ("a", "b", "c", "d")
    assert ids(shield) == ("c", "d", "g", "h")
    assert sum(item.reward_variance for item in shield) == sum(
        item.reward_variance for item in base
    )


def test_refuses_reward_variance_utility_loss() -> None:
    choices = tuple(
        candidate(
            letter,
            l1=100 if letter >= "e" else 1,
            reward_variance=1.0 if letter < "e" else 0.0,
        )
        for letter in "abcdefgh"
    )
    case = decision(choices)
    base = MODULE.select_reward_variance_base(case, PROTOCOL)
    assert ids(MODULE.select_m4_shield(case, base, PROTOCOL)) == ids(base)


def test_enforces_minimum_two_group_proposer_overlap() -> None:
    choices = tuple(
        candidate(letter, l1=value, reward_variance=1.0)
        for letter, value in zip("abcdefgh", (4, 3, 2, 1, 10, 20, 30, 40))
    )
    case = decision(choices)
    base = MODULE.select_reward_variance_base(case, PROTOCOL)
    shield = MODULE.select_m4_shield(case, base, PROTOCOL)
    assert ids(shield) == ("a", "b", "g", "h")
    assert len(set(ids(base)) & set(ids(shield))) == 2


def test_total_l1_tie_without_imminent_gain_returns_base() -> None:
    choices = tuple(
        candidate(
            letter,
            l1=value,
            reward_variance=1.0,
            start=1,
        )
        for letter, value in zip("abcdefgh", (1, 2, 3, 4, 5, 6, 7, 8))
    )
    case = decision(choices)
    base = MODULE.select_reward_variance_base(case, PROTOCOL)
    assert ids(MODULE.select_m4_shield(case, base, PROTOCOL)) == ids(base)


def test_token_band_blocks_otherwise_better_shield_batch() -> None:
    choices = tuple(
        candidate(
            letter,
            l1=100 if letter >= "e" else 1,
            reward_variance=1.0,
            tokens=120 if letter >= "e" else 100,
        )
        for letter in "abcdefgh"
    )
    case = decision(choices)
    base = MODULE.select_reward_variance_base(case, PROTOCOL)
    assert ids(MODULE.select_m4_shield(case, base, PROTOCOL)) == ids(base)


def test_lexicographic_tie_break_is_input_order_independent() -> None:
    choices = tuple(
        candidate(letter, l1=1.0, reward_variance=1.0) for letter in "abcdefgh"
    )
    forward = decision(choices)
    reverse = decision(tuple(reversed(choices)))
    assert ids(MODULE.select_reward_variance_base(forward, PROTOCOL)) == (
        "a",
        "b",
        "c",
        "d",
    )
    assert ids(MODULE.select_reward_variance_base(reverse, PROTOCOL)) == (
        "a",
        "b",
        "c",
        "d",
    )
