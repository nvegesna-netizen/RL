import hashlib
from pathlib import Path

import pytest

from tools import materialize_dapo_paced_exposure as materializer


def test_bound_input_rejects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    path.write_bytes(b"{}")
    digest = hashlib.sha256(b"{}").hexdigest()
    assert materializer.read_bound(path, digest) == b"{}"
    path.write_bytes(b"{ }")
    with pytest.raises(ValueError, match="hash mismatch"):
        materializer.read_bound(path, digest)


def test_selection_is_fresh_and_reproducible(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = tuple(
        materializer.base.UniquePrompt(
            canonical_sha256=f"{index:064x}",
            prompt="fixture",
            ground_truth="1",
            first_source_index=index,
            source_extra_index=str(index),
            duplicate_count=1,
        )
        for index in range(500)
    )
    earlier = {prompt.canonical_sha256 for prompt in prompts[:208]}
    all_prior = {prompt.canonical_sha256 for prompt in prompts[:400]}
    monkeypatch.setattr(
        materializer.prior,
        "_reconstructed_prior_ids",
        lambda _: (earlier, set(), set()),
    )
    first = materializer.select_fresh_pool(
        prompts, ledger_ids=all_prior, selection_seed=123
    )
    second = materializer.select_fresh_pool(
        prompts, ledger_ids=all_prior, selection_seed=123
    )
    assert first == second and len(first) == 64
    assert not all_prior.intersection(prompt.canonical_sha256 for prompt in first)
    with pytest.raises(ValueError, match="omitted"):
        materializer.select_fresh_pool(prompts, ledger_ids=set(), selection_seed=123)
    with pytest.raises(ValueError, match="insufficient"):
        materializer.select_fresh_pool(
            prompts[:450], ledger_ids=all_prior, selection_seed=123
        )


def test_ledgers_require_provenance_and_unique_identities() -> None:
    with pytest.raises(ValueError):
        materializer.IdentityLedger(
            canonical_prompt_sha256=("a" * 64, "a" * 64),
            source_artifact_sha256=("b" * 64,),
        )
    with pytest.raises(ValueError):
        materializer.SeedLedger(
            used_or_reserved_seeds=(123,), source_artifact_sha256=()
        )
