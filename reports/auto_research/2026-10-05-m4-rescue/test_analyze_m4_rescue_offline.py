"""Unit tests for the frozen M4-Rescue selector."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "analyze_m4_rescue_offline.py"
SPEC = importlib.util.spec_from_file_location("m4_rescue_offline", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
PROTOCOL = json.loads((HERE / "m4_rescue_protocol.json").read_text())


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


def ids(selection: tuple[SimpleNamespace, ...]) -> set[str]:
    return {candidate.group_id for candidate in selection}


def test_enacts_one_swap_for_strict_imminent_gain() -> None:
    choices = (
        candidate("a", l1=1, reward_variance=1),
        candidate("b", l1=1, reward_variance=1),
        candidate("c", l1=1, reward_variance=1),
        candidate("d", l1=1, reward_variance=1),
        candidate("e", l1=5, reward_variance=2),
        candidate("f", l1=4, reward_variance=2),
        candidate("g", l1=3, reward_variance=2),
        candidate("h", l1=2, reward_variance=2),
    )
    selected = MODULE.select_m4_rescue(decision(choices), PROTOCOL)
    assert ids(selected) == {"a", "b", "c", "e"}
    assert len(ids(selected) & {"a", "b", "c", "d"}) == 3


def test_reward_variance_floor_blocks_higher_m4_swap() -> None:
    choices = (
        candidate("a", l1=1, reward_variance=2),
        candidate("b", l1=1, reward_variance=2),
        candidate("c", l1=1, reward_variance=2),
        candidate("d", l1=1, reward_variance=2),
        candidate("e", l1=9, reward_variance=0),
        candidate("f", l1=0, reward_variance=0),
        candidate("g", l1=0, reward_variance=0),
        candidate("h", l1=0, reward_variance=0),
    )
    assert ids(MODULE.select_m4_rescue(decision(choices), PROTOCOL)) == {
        "a",
        "b",
        "c",
        "d",
    }


def test_token_band_blocks_higher_m4_swap() -> None:
    choices = (
        candidate("a", l1=1, reward_variance=1),
        candidate("b", l1=1, reward_variance=1),
        candidate("c", l1=1, reward_variance=1),
        candidate("d", l1=1, reward_variance=1),
        candidate("e", l1=9, reward_variance=2, tokens=120),
        candidate("f", l1=0, reward_variance=0),
        candidate("g", l1=0, reward_variance=0),
        candidate("h", l1=0, reward_variance=0),
    )
    assert ids(MODULE.select_m4_rescue(decision(choices), PROTOCOL)) == {
        "a",
        "b",
        "c",
        "d",
    }


def test_total_l1_without_imminent_gain_does_not_trigger() -> None:
    choices = (
        candidate("a", l1=1, reward_variance=1, start=0),
        candidate("b", l1=1, reward_variance=1, start=0),
        candidate("c", l1=1, reward_variance=1, start=0),
        candidate("d", l1=1, reward_variance=1, start=0),
        candidate("e", l1=99, reward_variance=2, start=1),
        candidate("f", l1=0, reward_variance=0, start=1),
        candidate("g", l1=0, reward_variance=0, start=1),
        candidate("h", l1=0, reward_variance=0, start=1),
    )
    assert ids(MODULE.select_m4_rescue(decision(choices), PROTOCOL)) == {
        "a",
        "b",
        "c",
        "d",
    }


def test_lexicographic_tie_break_is_deterministic() -> None:
    choices = (
        candidate("a", l1=1, reward_variance=1),
        candidate("b", l1=1, reward_variance=1),
        candidate("c", l1=1, reward_variance=1),
        candidate("d", l1=1, reward_variance=1),
        candidate("e", l1=5, reward_variance=2),
        candidate("f", l1=5, reward_variance=2),
        candidate("g", l1=0, reward_variance=0),
        candidate("h", l1=0, reward_variance=0),
    )
    selected = MODULE.select_m4_rescue(decision(choices), PROTOCOL)
    assert tuple(candidate.group_id for candidate in selected) == ("a", "b", "c", "e")
    assert (
        MODULE.select_m4_rescue(decision(tuple(reversed(choices))), PROTOCOL)
        == selected
    )
