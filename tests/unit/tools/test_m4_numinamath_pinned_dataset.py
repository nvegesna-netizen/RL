# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import pytest

from tools import m4_numinamath_pinned_dataset as subject


class _FakeDataset:
    column_names = list(subject.EXPECTED_COLUMNS)

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, key: str) -> list[object]:
        return [row[key] for row in self.rows]

    def filter(self, function):
        return _FakeDataset([row for row in self.rows if function(row)])

    def map(self, function, *, remove_columns):
        assert set(remove_columns) == subject.EXPECTED_COLUMNS
        return [function(row) for row in self.rows]


def _row(index: int, **changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "problem": f"problem-{index}",
        "solution": f"solution-{index}",
        "answer": str(index),
        "problem_type": "Algebra",
        "question_type": "math-word-problem",
        "problem_is_valid": "Yes",
        "solution_is_valid": "Yes",
        "source": "fixture",
        "synthetic": False,
    }
    row.update(changes)
    return row


def _patch_cardinality(monkeypatch, *, source: int = 4, filtered: int = 2) -> None:
    monkeypatch.setattr(subject, "EXPECTED_SOURCE_ROWS", source)
    monkeypatch.setattr(subject, "EXPECTED_FILTERED_ROWS", filtered)


def test_pinned_numinamath_loads_exact_revision_and_filters(monkeypatch) -> None:
    _patch_cardinality(monkeypatch, source=7, filtered=2)
    rows = [
        _row(0),
        _row(1),
        _row(2, answer="proof", question_type="proof"),
        _row(3, solution_is_valid="No"),
        _row(4, problem=""),
        _row(5, problem=None),
        _row(6, problem=" \t"),
    ]
    calls = []

    def load_dataset(repository, *, revision, split):
        calls.append((repository, revision, split))
        return _FakeDataset(rows)

    monkeypatch.setattr(subject, "load_dataset", load_dataset)
    dataset = subject.M4NuminaMathPinnedDataset()
    assert calls == [(subject.DATASET_REPOSITORY, subject.DATASET_REVISION, "train")]
    assert len(dataset.dataset) == 2
    assert dataset.dataset[0]["messages"][0]["content"] == "problem-0"


@pytest.mark.parametrize(
    "mutation", ["row_count", "schema", "filtered", "duplicate", "blank_problem"]
)
def test_pinned_numinamath_rejects_identity_mutation(monkeypatch, mutation) -> None:
    _patch_cardinality(monkeypatch)
    rows = [_row(0), _row(1), _row(2, answer="proof"), _row(3, answer="notfound")]
    dataset = _FakeDataset(rows)
    if mutation == "row_count":
        dataset.rows.pop()
    elif mutation == "schema":
        dataset.column_names.remove("source")
    elif mutation == "filtered":
        dataset.rows[2]["answer"] = "2"
    elif mutation == "blank_problem":
        dataset.rows[1]["problem"] = ""
    else:
        dataset.rows[1]["problem"] = dataset.rows[0]["problem"]
    monkeypatch.setattr(subject, "load_dataset", lambda *args, **kwargs: dataset)
    with pytest.raises(ValueError):
        subject.M4NuminaMathPinnedDataset()
