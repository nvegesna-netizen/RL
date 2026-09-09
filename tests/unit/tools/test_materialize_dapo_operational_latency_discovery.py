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
    prompts, _ = materializer._deduplicate_dataset(dataset)
    return prompts


def test_deduplication_counts_repeated_rows_and_rejects_conflicts() -> None:
    rows = [_row("same", "7", "a"), _row("same", "7", "a"), _row("other", "9", "b")]

    prompts, audit = materializer._deduplicate_dataset(rows)

    assert len(prompts) == 2
    assert audit == {
        "published_rows": 3,
        "unique_canonical_prompts": 2,
        "duplicate_rows": 1,
        "minimum_multiplicity": 1,
        "maximum_multiplicity": 2,
        "conflicting_ground_truth_count": 0,
    }
    conflicting = [_row("same", "7", "a"), _row("same", "8", "a")]
    with pytest.raises(
        materializer.DapoOperationalMaterializationError,
        match="conflicting ground truths",
    ):
        materializer._deduplicate_dataset(conflicting)


def test_three_selected_pools_are_deterministic_and_disjoint() -> None:
    prompts = _unique_prompts()

    first = materializer._select_disjoint_pools(prompts)
    second = materializer._select_disjoint_pools(prompts)

    assert first == second
    identities = [item.canonical_sha256 for pool in first.values() for item in pool]
    assert len(identities) == 48
    assert len(set(identities)) == 48


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
