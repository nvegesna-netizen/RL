# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pinned, valid, verifiable NuminaMath-1.5 workload for M4 generalization."""

from __future__ import annotations

from typing import Any

from datasets import load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


DATASET_REPOSITORY = "AI-MO/NuminaMath-1.5"
DATASET_REVISION = "1b05109f9e5c1ad06c0663519502416c30b300f8"
DATA_FILE_SHA256 = {
    "data/train-00000-of-00003.parquet": (
        "1e37cedcc5104c0d5a9afbf8b26328a59d7005cd37c2bd08352419f5f0e03214"
    ),
    "data/train-00001-of-00003.parquet": (
        "bc48fdddc1ab4e559727ad4f1f03b0263785c1c16b7964fb885fba07e18d6a39"
    ),
    "data/train-00002-of-00003.parquet": (
        "951aa899fe2a1b64f5b40c6636372d35b9db4f0b9c922e874d4bf357db64da66"
    ),
}
DATA_FILE_SIZE = {
    "data/train-00000-of-00003.parquet": 195_064_200,
    "data/train-00001-of-00003.parquet": 175_917_356,
    "data/train-00002-of-00003.parquet": 160_374_171,
}
EXPECTED_SOURCE_ROWS = 896_215
EXPECTED_FILTERED_ROWS = 680_787
EXPECTED_COLUMNS = {
    "problem",
    "solution",
    "answer",
    "problem_type",
    "question_type",
    "problem_is_valid",
    "solution_is_valid",
    "source",
    "synthetic",
}
NON_VERIFIABLE_ANSWERS = frozenset({"proof", "notfound"})


class M4NuminaMathPinnedDataset(RawDataset):
    """Load the immutable source and retain only valid, verifiable unique rows."""

    def __init__(self, **kwargs: object) -> None:
        del kwargs
        self.task_name = "NuminaMath-1.5"
        source = load_dataset(
            DATASET_REPOSITORY,
            revision=DATASET_REVISION,
            split="train",
        )
        if len(source) != EXPECTED_SOURCE_ROWS or set(source.column_names) != (
            EXPECTED_COLUMNS
        ):
            raise ValueError("pinned NuminaMath source identity disagrees")
        source = source.filter(self._is_valid_verifiable)
        if len(source) != EXPECTED_FILTERED_ROWS:
            raise ValueError("pinned NuminaMath filtered cardinality disagrees")
        problems = source["problem"]
        if len(set(problems)) != EXPECTED_FILTERED_ROWS:
            raise ValueError("pinned NuminaMath filtered problems must be unique")
        self.dataset = source.map(
            self.format_data,
            remove_columns=source.column_names,
        )

    @staticmethod
    def _is_valid_verifiable(data: dict[str, Any]) -> bool:
        answer = (data.get("answer") or "").strip()
        question_type = (data.get("question_type") or "").strip().lower()
        return (
            bool(answer)
            and answer.lower() not in NON_VERIFIABLE_ANSWERS
            and question_type != "proof"
            and data.get("problem_is_valid") == "Yes"
            and data.get("solution_is_valid") == "Yes"
        )

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Map a pinned source row to the existing math response schema."""
        problem = data["problem"]
        answer = data["answer"]
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError("NuminaMath problem must be nonempty text")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("NuminaMath answer must be nonempty text")
        return {
            "messages": [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": answer},
            ],
            "task_name": self.task_name,
        }
