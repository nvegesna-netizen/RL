import json
from pathlib import Path

import pytest

from nemo_rl.algorithms.async_utils import fixed_pool
from nemo_rl.algorithms.async_utils.fixed_pool import (
    load_fixed_pool_manifest,
    validate_fixed_pool_manifest_design,
)
from tools import materialize_dapo_operational_latency_discovery as materializer


class _Tokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        add_special_tokens: bool,
    ) -> str:
        assert not tokenize and add_generation_prompt and not add_special_tokens
        return f"<chat>{messages[0]['content']}</chat>"

    def __call__(self, text, *, add_special_tokens: bool):
        assert not add_special_tokens
        return {"input_ids": list(range(len(text.split())))}


def _row(prompt: str, answer: str, extra_index: str) -> dict[str, object]:
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "reward_model": {"ground_truth": answer},
        "extra_info": {"index": extra_index},
    }


def _unique_prompts(count: int = 64) -> list[materializer.UniquePrompt]:
    dataset = [
        _row(f"problem {index}", str(index), f"source-{index}")
        for index in range(count)
    ]
    prompts, conflicts, _ = materializer._deduplicate_dataset(dataset)
    assert not conflicts
    return prompts


def test_deduplication_counts_repeated_rows_and_excludes_whole_conflicts() -> None:
    rows = [_row("same", "7", "a"), _row("same", "7", "a"), _row("other", "9", "b")]

    prompts, conflicts, audit = materializer._deduplicate_dataset(rows)

    assert len(prompts) == 2
    assert not conflicts
    assert audit == {
        "published_rows": 3,
        "unique_canonical_prompts": 2,
        "unique_nonconflicting_canonical_prompts": 2,
        "duplicate_rows": 1,
        "minimum_multiplicity": 1,
        "maximum_multiplicity": 2,
        "excluded_conflicting_identity_count": 0,
        "excluded_conflicting_row_count": 0,
        "conflicting_ground_truth_count": 0,
    }

    mixed = [
        _row("same", "7", "a"),
        _row("same", "8", "b"),
        _row("same", "7", "a"),
        _row("other", "9", "c"),
        _row("same", "8", "b"),
    ]
    prompts, conflicts, audit = materializer._deduplicate_dataset(mixed)

    assert [item.prompt for item in prompts] == ["other"]
    assert len(conflicts) == 1
    assert conflicts[0].row_count == 4
    assert conflicts[0].ground_truth_counts == (("7", 2), ("8", 2))
    assert conflicts[0].extra_indices == ("a", "b")
    assert conflicts[0].answer_index_counts == (("7", "a", 2), ("8", "b", 2))
    assert audit["unique_canonical_prompts"] == 2
    assert audit["unique_nonconflicting_canonical_prompts"] == 1
    assert audit["excluded_conflicting_identity_count"] == 1
    assert audit["excluded_conflicting_row_count"] == 4


def test_three_selected_pools_are_deterministic_and_disjoint() -> None:
    prompts = _unique_prompts()

    first = materializer._select_disjoint_pools(prompts)
    second = materializer._select_disjoint_pools(prompts)

    assert first == second
    identities = [item.canonical_sha256 for pool in first.values() for item in pool]
    assert len(identities) == 48
    assert len(set(identities)) == 48


def test_source_audit_requires_exact_exclusion_counts_and_structure() -> None:
    conflicts = [
        materializer.ConflictPrompt(
            canonical_sha256=f"{index:064x}",
            row_count=200,
            ground_truth_counts=((f"a-{index}", 100), (f"b-{index}", 100)),
            extra_indices=(f"i-{index}", f"j-{index}"),
            answer_index_counts=(
                (f"a-{index}", f"i-{index}", 100),
                (f"b-{index}", f"j-{index}", 100),
            ),
        )
        for index in range(materializer.EXPECTED_CONFLICT_IDENTITIES)
    ]
    audit = {
        "published_rows": materializer.EXPECTED_TOTAL_ROWS,
        "unique_canonical_prompts": materializer.EXPECTED_UNIQUE_PROMPTS,
        "unique_nonconflicting_canonical_prompts": (
            materializer.EXPECTED_NONCONFLICTING_PROMPTS
        ),
        "duplicate_rows": (
            materializer.EXPECTED_TOTAL_ROWS - materializer.EXPECTED_UNIQUE_PROMPTS
        ),
        "minimum_multiplicity": 100,
        "maximum_multiplicity": 400,
        "excluded_conflicting_identity_count": (
            materializer.EXPECTED_CONFLICT_IDENTITIES
        ),
        "excluded_conflicting_row_count": materializer.EXPECTED_CONFLICT_ROWS,
        "conflicting_ground_truth_count": 0,
    }

    materializer._validate_source_audit(conflicts, audit)

    malformed = [*conflicts]
    malformed[0] = materializer.ConflictPrompt(
        canonical_sha256=malformed[0].canonical_sha256,
        row_count=200,
        ground_truth_counts=(("a", 99), ("b", 101)),
        extra_indices=("i", "j"),
        answer_index_counts=(("a", "i", 99), ("b", "j", 101)),
    )
    with pytest.raises(
        materializer.DapoOperationalMaterializationError,
        match="source conflict structure mismatch",
    ):
        materializer._validate_source_audit(malformed, audit)


def test_materialized_pool_passes_strict_design_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts = _unique_prompts(16)
    protocol_raw = b"candidate"
    selection_seed, generation_seed = materializer.POOL_SPECS[0]

    report = materializer._make_pool(
        root=tmp_path,
        tokenizer=_Tokenizer(),
        prompt_template="{}",
        key=b"k" * 32,
        protocol_raw=protocol_raw,
        snapshot_raw=b"[]\n",
        model_weights_sha256="a" * 64,
        selection_seed=selection_seed,
        generation_seed=generation_seed,
        prompts=prompts,
    )
    manifest = load_fixed_pool_manifest(tmp_path / str(report["manifest"]))
    monkeypatch.setattr(
        fixed_pool,
        "DAPO_OPERATIONAL_PROTOCOL_SHA256",
        materializer._sha_bytes(protocol_raw),
    )

    assert len(manifest.items) == 16
    assert len({item.source_prompt_id for item in manifest.items}) == 16
    validate_fixed_pool_manifest_design(manifest, materializer.DESIGN_ID)
    design = json.loads((tmp_path / str(report["selection_design"])).read_text())
    assert len(design["items"]) == 16
    assert design["protocol_sha256"] == materializer.PROTOCOL_SHA256
