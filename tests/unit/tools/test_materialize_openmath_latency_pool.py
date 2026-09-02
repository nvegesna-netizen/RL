import hashlib
import json
import stat
from dataclasses import dataclass
from pathlib import Path

import pytest

from nemo_rl.algorithms.async_utils import fixed_pool as fixed_pool_module
from nemo_rl.algorithms.async_utils.fixed_pool import (
    load_fixed_pool_manifest,
    validate_fixed_pool_manifest_design,
    validate_fixed_pool_materialization,
)
from tools import materialize_openmath_latency_pool as materializer


@dataclass(frozen=True)
class _Feature:
    dtype: str = "string"


class _Dataset:
    column_names = list(materializer.DATASET_SCHEMA)
    features = {name: _Feature() for name in materializer.DATASET_SCHEMA}
    _fingerprint = "synthetic-openmath-fixture"

    def __init__(self, rows: list[dict[str, str]]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)


class _Tokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        add_special_tokens: bool,
    ) -> str:
        assert not tokenize
        assert add_generation_prompt
        assert not add_special_tokens
        return messages[0]["content"]

    def __call__(
        self, text: str, *, return_tensors, add_special_tokens: bool
    ) -> dict[str, list[int]]:
        assert return_tensors is None
        assert not add_special_tokens
        if text.startswith("REF:") or text.startswith("ANS:"):
            length = int(text.split(":")[1])
        elif text.startswith("PROBLEM:"):
            length = int(text.rsplit(":", 1)[1])
        else:
            length = len(text.split())
        return {"input_ids": list(range(length))}


def _rows() -> list[dict[str, str]]:
    rows = []
    for index in range(materializer.PAIRS):
        source = f"source-{index % 2}"
        rows.append(
            {
                "problem": f"PROBLEM:short-{index}:{40 + index}",
                "generated_solution": f"REF:{40 + index}",
                "expected_answer": f"ANS:{2 + index % 2}",
                "problem_source": source,
            }
        )
        rows.append(
            {
                "problem": f"PROBLEM:long-{index}:{42 + index}",
                "generated_solution": f"REF:{300 + index}",
                "expected_answer": f"ANS:{3 + index % 2}",
                "problem_source": source,
            }
        )
    return rows


def _patch_external_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    dataset: _Dataset,
    prompt_file: Path,
) -> None:
    weights = b"synthetic pinned weights"
    prompt_sha = hashlib.sha256(prompt_file.read_bytes()).hexdigest()
    weights_sha = hashlib.sha256(weights).hexdigest()
    monkeypatch.setattr(materializer, "DATASET_CARDINALITY", len(dataset))
    monkeypatch.setattr(materializer, "PROMPT_FILE_SHA256", prompt_sha)
    monkeypatch.setattr(materializer, "MODEL_WEIGHTS_SHA256", weights_sha)
    monkeypatch.setattr(fixed_pool_module, "OPENMATH_PROMPT_FILE_SHA256", prompt_sha)
    monkeypatch.setattr(
        fixed_pool_module,
        "OPENMATH_MODEL_WEIGHTS_SHA256",
        weights_sha,
    )

    def fake_snapshot_download(*, repo_id: str, revision: str, local_dir: Path) -> None:
        assert repo_id == materializer.MODEL_REPO
        assert revision == materializer.MODEL_REVISION
        local_dir.mkdir()
        (local_dir / "model.safetensors").write_bytes(weights)
        (local_dir / "tokenizer.json").write_text("{}")

    def fake_load_dataset(repo_id: str, *, split: str, revision: str) -> _Dataset:
        assert repo_id == materializer.DATASET_REPO
        assert split == materializer.DATASET_SPLIT
        assert revision == materializer.DATASET_REVISION
        return dataset

    monkeypatch.setattr(materializer, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(materializer, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        materializer.AutoTokenizer,
        "from_pretrained",
        lambda model_dir: _Tokenizer(),
    )


def test_source_and_design_pins_are_exact() -> None:
    assert materializer.DATASET_REPO == "nvidia/OpenMathInstruct-2"
    assert materializer.DATASET_REVISION == "469216e3f46f4dacf476b382e192485ea51a143e"
    assert materializer.DATASET_SPLIT == "train_1M"
    assert materializer.DATASET_CARDINALITY == 1_000_000
    assert materializer.DATASET_SCHEMA == {
        "problem",
        "generated_solution",
        "expected_answer",
        "problem_source",
    }
    assert materializer.SHORT_REFERENCE_RANGE == (32, 96)
    assert materializer.LONG_REFERENCE_RANGE == (256, 384)
    assert materializer.INPUT_CALIPER_TOKENS == 8
    assert materializer.ANSWER_CALIPER_TOKENS == 4
    assert materializer.SELECTION_SEED == 20260902
    assert materializer.ORDER_SEED == 43001
    assert (
        materializer.MATCHING_ALGORITHM_VERSION
        == "bounded_reservoir_lazy_augmenting_v1"
    )


def test_dataset_contract_fails_closed_on_schema_and_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _Dataset(_rows())
    monkeypatch.setattr(materializer, "DATASET_CARDINALITY", len(dataset))
    materializer._validate_dataset_contract(dataset)

    dataset.column_names = ["problem", "generated_solution", "expected_answer"]
    with pytest.raises(RuntimeError, match="column mismatch"):
        materializer._validate_dataset_contract(dataset)

    dataset.column_names = list(materializer.DATASET_SCHEMA)
    dataset.features = {
        **_Dataset.features,
        "problem": _Feature(dtype="large_string"),
    }
    with pytest.raises(RuntimeError, match="not strings"):
        materializer._validate_dataset_contract(dataset)


def test_candidate_building_excludes_entire_repeated_problem_cluster() -> None:
    rows = _rows()
    rows.extend(
        [
            {
                "problem": "duplicate\r\nproblem",
                "generated_solution": "REF:40",
                "expected_answer": "ANS:2",
                "problem_source": "source-0",
            },
            {
                "problem": "duplicate\nproblem ",
                "generated_solution": "REF:300",
                "expected_answer": "ANS:2",
                "problem_source": "source-0",
            },
        ]
    )
    short, long, counts = materializer._build_candidates(
        _Dataset(rows), _Tokenizer(), "{}"
    )
    assert len(short) == materializer.PAIRS
    assert len(long) == materializer.PAIRS
    assert counts["rejected_repeated_problem"] == 2
    assert {candidate.normalized_problem_sha256 for candidate in short}.isdisjoint(
        candidate.normalized_problem_sha256 for candidate in long
    )


def test_pair_selection_is_deterministic_and_obeys_locked_matching() -> None:
    short, long, _ = materializer._build_candidates(
        _Dataset(_rows()), _Tokenizer(), "{}"
    )
    first, first_counts = materializer._select_pairs(short, long)
    second, second_counts = materializer._select_pairs(short, long)
    assert first == second
    assert first_counts == second_counts
    assert len(first) == materializer.PAIRS
    for short_candidate, long_candidate in first:
        assert short_candidate.problem_source == long_candidate.problem_source
        assert (
            abs(
                len(short_candidate.input_token_ids)
                - len(long_candidate.input_token_ids)
            )
            <= materializer.INPUT_CALIPER_TOKENS
        )
        assert (
            abs(
                short_candidate.reference_answer_token_count
                - long_candidate.reference_answer_token_count
            )
            <= materializer.ANSWER_CALIPER_TOKENS
        )


def test_candidate_reservoir_is_bounded_and_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(materializer, "MAX_RETAINED_CANDIDATES_PER_STRATUM", 16)
    rows = []
    for index in range(100):
        rows.append(
            {
                "problem": f"PROBLEM:short-reservoir-{index}:40",
                "generated_solution": "REF:40",
                "expected_answer": "ANS:2",
                "problem_source": "math",
            }
        )
        rows.append(
            {
                "problem": f"PROBLEM:long-reservoir-{index}:40",
                "generated_solution": "REF:300",
                "expected_answer": "ANS:2",
                "problem_source": "math",
            }
        )
    first_short, first_long, first_counts = materializer._build_candidates(
        _Dataset(rows), _Tokenizer(), "{}"
    )
    second_short, second_long, second_counts = materializer._build_candidates(
        _Dataset(rows), _Tokenizer(), "{}"
    )
    assert first_short == second_short
    assert first_long == second_long
    assert first_counts == second_counts
    assert len(first_short) == len(first_long) == 16
    assert first_counts["eligible_openmath_short"] == 100
    assert first_counts["eligible_openmath_long"] == 100
    assert first_counts["retained_openmath_short"] == 16
    assert first_counts["retained_openmath_long"] == 16


def test_pair_selection_fails_when_problem_source_does_not_match() -> None:
    rows = _rows()
    for row in rows:
        if row["generated_solution"].startswith("REF:3"):
            row["problem_source"] = "long-only"
    short, long, _ = materializer._build_candidates(_Dataset(rows), _Tokenizer(), "{}")
    with pytest.raises(RuntimeError, match="only 0 OpenMath pairs"):
        materializer._select_pairs(short, long)


def test_bucketed_matching_does_not_expand_the_full_bipartite_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_tokens = tuple(range(40))

    def candidate(index: int, *, reference_length: int) -> materializer.Candidate:
        return materializer.Candidate(
            source_index=index,
            problem=f"problem-{index}",
            answer="2",
            problem_source="math",
            normalized_problem_sha256=f"{index:064x}",
            input_token_ids=input_tokens,
            reference_solution_token_count=reference_length,
            reference_answer_token_count=2,
        )

    short = [candidate(index, reference_length=40) for index in range(8)]
    long = [candidate(10_000 + index, reference_length=300) for index in range(50_000)]
    original = materializer._compatible_long_indices
    yielded = 0

    def instrumented(candidate_value, buckets):
        nonlocal yielded
        for long_index in original(candidate_value, buckets):
            yielded += 1
            yield long_index

    monkeypatch.setattr(materializer, "_compatible_long_indices", instrumented)
    matches, matching_counts = materializer._find_bounded_matching(
        short, long, seed=materializer.SELECTION_SEED
    )
    assert len(matches) == materializer.PAIRS
    assert yielded < 1_000
    assert matching_counts["compatibility_edges_examined"] == yielded
    assert matching_counts["short_reservoir_size"] == len(short)
    assert matching_counts["long_reservoir_size"] == len(long)


def test_materialize_writes_private_bound_and_validated_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_file = tmp_path / "cot.txt"
    prompt_file.write_text("{}")
    dataset = _Dataset(_rows())
    _patch_external_dependencies(monkeypatch, dataset, prompt_file)
    output_dir = tmp_path / "private-materialization"

    materializer.materialize(output_dir, prompt_file, b"k" * 32)

    assert stat.S_IMODE(output_dir.stat().st_mode) == 0o700
    for path in output_dir.rglob("*"):
        expected = 0o700 if path.is_dir() else 0o600
        assert stat.S_IMODE(path.stat().st_mode) == expected

    manifest_path = output_dir / "fixed_pool_manifest.v1.43001.json"
    manifest = load_fixed_pool_manifest(manifest_path)
    validate_fixed_pool_materialization(manifest)
    validate_fixed_pool_manifest_design(manifest, "openmath_latency_feasibility_v1")
    assert len(manifest.items) == 16
    assert manifest.order_seed == materializer.ORDER_SEED
    assert [source.source_id for source in manifest.sources] == list(
        materializer.SOURCE_IDS
    )
    assert all(
        source.dataset_id == materializer.DATASET_REPO for source in manifest.sources
    )
    assert all(
        source.revision == materializer.DATASET_REVISION for source in manifest.sources
    )
    assert all(
        source.split == materializer.DATASET_SPLIT for source in manifest.sources
    )

    by_cohort: dict[int, list[str]] = {}
    by_pair: dict[str, list[str]] = {}
    for item in manifest.items:
        by_cohort.setdefault(item.dispatch_cohort, []).append(item.task_name)
        by_pair.setdefault(item.matching_pair_id, []).append(item.task_name)
    assert len(by_cohort) == 4
    assert all(
        sorted(tasks) == sorted([*materializer.SOURCE_IDS, *materializer.SOURCE_IDS])
        for tasks in by_cohort.values()
    )
    assert len(by_pair) == 8
    assert all(
        sorted(tasks) == sorted(materializer.SOURCE_IDS) for tasks in by_pair.values()
    )

    private_records = []
    for source_id in materializer.SOURCE_IDS:
        private_records.extend(
            json.loads(line)
            for line in (output_dir / f"{source_id}.jsonl").read_text().splitlines()
        )
    assert len(private_records) == 16
    assert {record["selection_stratum"] for record in private_records} == set(
        materializer.SOURCE_IDS
    )
    assert all(
        record["source_revision"] == materializer.DATASET_REVISION
        and record["source_split"] == materializer.DATASET_SPLIT
        and record["model_revision"] == materializer.MODEL_REVISION
        and record["selection_seed"] == materializer.SELECTION_SEED
        for record in private_records
    )

    ledger_path = output_dir / "private_exclusion_ledger.v1.json"
    ledger = json.loads(ledger_path.read_text())
    exclusions = ledger["excluded_problem_clusters"]
    assert len(exclusions) == 16
    assert len({row["normalized_problem_sha256"] for row in exclusions}) == 16
    report = json.loads((output_dir / "materialization_report.v1.json").read_text())
    assert (
        report["private_exclusion_ledger_sha256"]
        == hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    )
    assert report["selected_reference_median_ratio"] >= 3.0
    assert report["observed_max_input_pair_delta_tokens"] <= 8
    assert report["observed_max_answer_pair_delta_tokens"] <= 4
    assert (
        report["matching_algorithm_version"] == materializer.MATCHING_ALGORITHM_VERSION
    )
    assert report["max_retained_candidates_per_stratum"] == 4096
    assert report["selection_counts"]["rows_scanned_problem_multiplicity"] == 16
    assert report["selection_counts"]["rows_scanned_candidate_selection"] == 16
    assert report["selection_counts"]["rendered_prompts_tokenized"] == 16
    assert report["matching_counts"]["matches_found"] == 8

    selection_design_path = output_dir / "selection_design.v1.json"
    selection_design = json.loads(selection_design_path.read_text())
    assert selection_design["analysis_status"] == "preregistered_pre_generation"
    assert selection_design["calibration_only"] is True
    assert selection_design["confirmatory_eligible"] is False
    assert selection_design["fixed_pool_id"] == manifest.pool_id
    assert (
        selection_design["fixed_pool_manifest_sha256"]
        == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    assert selection_design["bootstrap_seed"] == materializer.BOOTSTRAP_SEED
    assert selection_design["bootstrap_reps"] == materializer.BOOTSTRAP_REPS
    assert len(selection_design["items"]) == 16
    assert {item["stratum"] for item in selection_design["items"]} == {
        "short",
        "long",
    }
    assert all(
        "reference_solution_tokens" in item
        and "reference_answer_token_count" in item
        and "rendered_prompt_tokens" in item
        for item in selection_design["items"]
    )
    assert (
        report["selection_design_sha256"]
        == hashlib.sha256(selection_design_path.read_bytes()).hexdigest()
    )

    public = manifest_path.read_text() + json.dumps(report)
    assert "PROBLEM:" not in public
    assert "normalized_problem_sha256" not in public
    assert (b"k" * 32).decode() not in public

    expected_hashes = {
        line.split("  ", 1)[1]: line.split("  ", 1)[0]
        for line in (output_dir / "SHA256SUMS").read_text().splitlines()
    }
    top_level_files = {
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    }
    assert set(expected_hashes) == top_level_files
    assert all(
        hashlib.sha256((output_dir / name).read_bytes()).hexdigest() == digest
        for name, digest in expected_hashes.items()
    )


def test_materialize_refuses_short_key_and_existing_output(
    tmp_path: Path,
) -> None:
    prompt_file = tmp_path / "cot.txt"
    prompt_file.write_text("{}")
    output_dir = tmp_path / "output"
    with pytest.raises(RuntimeError, match="shorter than 32"):
        materializer.materialize(output_dir, prompt_file, b"short")
    assert not output_dir.exists()

    output_dir.mkdir()
    with pytest.raises(FileExistsError):
        materializer.materialize(output_dir, prompt_file, b"k" * 32)
