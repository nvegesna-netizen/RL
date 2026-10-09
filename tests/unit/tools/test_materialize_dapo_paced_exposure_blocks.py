import os
from pathlib import Path

import pytest

from tools import materialize_dapo_paced_exposure_blocks as materializer


def prompts(count: int):
    return tuple(
        materializer.base.UniquePrompt(
            canonical_sha256=f"{index:064x}",
            prompt="fixture",
            ground_truth="1",
            first_source_index=index,
            source_extra_index=str(index),
            duplicate_count=1,
        )
        for index in range(count)
    )


def test_disjoint_selection_is_reproducible(monkeypatch: pytest.MonkeyPatch) -> None:
    values = prompts(1200)
    prior = {prompt.canonical_sha256 for prompt in values[:500]}
    monkeypatch.setattr(
        materializer.single.prior,
        "_reconstructed_prior_ids",
        lambda _: (set(list(prior)[:100]), set(), set()),
    )
    first = materializer.select_disjoint_pools(
        values, ledger_ids=prior, selection_seeds=range(10, 16)
    )
    second = materializer.select_disjoint_pools(
        values, ledger_ids=prior, selection_seeds=range(10, 16)
    )
    assert first == second and [len(pool) for pool in first] == [64] * 6
    identities = [prompt.canonical_sha256 for pool in first for prompt in pool]
    assert len(identities) == len(set(identities)) == 384
    assert not prior.intersection(identities)


def test_disjoint_selection_rejects_bad_seed_or_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = prompts(500)
    monkeypatch.setattr(
        materializer.single.prior,
        "_reconstructed_prior_ids",
        lambda _: (set(), set(), set()),
    )
    with pytest.raises(ValueError, match="six unique"):
        materializer.select_disjoint_pools(
            values, ledger_ids=set(), selection_seeds=[1] * 6
        )
    with pytest.raises(ValueError, match="insufficient"):
        materializer.select_disjoint_pools(
            values,
            ledger_ids={prompt.canonical_sha256 for prompt in values[:200]},
            selection_seeds=range(6),
        )


def test_snapshot_is_hardlinked_and_local(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "model_snapshot/subdir").mkdir(parents=True)
    destination.mkdir()
    asset = source / "model_snapshot/subdir/weight.bin"
    asset.write_bytes(b"weights")
    raw = b"[]\n"
    materializer.hardlink_snapshot(source, destination, raw)
    target = destination / "model_snapshot/subdir/weight.bin"
    assert target.read_bytes() == b"weights"
    assert os.stat(asset).st_ino == os.stat(target).st_ino
    assert (destination / "model_snapshot_manifest.v1.json").read_bytes() == raw
