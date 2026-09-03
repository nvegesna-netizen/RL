import hashlib
import json
import stat
from pathlib import Path

import pytest

from nemo_rl.algorithms.async_utils import fixed_pool as fixed_pool_module
from nemo_rl.algorithms.async_utils.fixed_pool import (
    load_fixed_pool_manifest,
    validate_fixed_pool_manifest_design,
    validate_fixed_pool_materialization,
)
from tools import materialize_sliding_puzzle_latency_pool as materializer


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
        return messages[0]["content"] + "\nassistant:"

    def __call__(self, text, *, return_tensors, add_special_tokens):
        assert return_tensors is None and not add_special_tokens
        return {"input_ids": list(range(len(text.split())))}


def _patch_snapshot(monkeypatch: pytest.MonkeyPatch, weights: bytes) -> None:
    weights_sha = hashlib.sha256(weights).hexdigest()
    monkeypatch.setattr(materializer, "MODEL_WEIGHTS_SHA256", weights_sha)
    monkeypatch.setattr(
        fixed_pool_module, "SLIDING_PUZZLE_MODEL_WEIGHTS_SHA256", weights_sha
    )

    def snapshot_download(*, repo_id: str, revision: str, local_dir: Path) -> None:
        assert repo_id == materializer.MODEL_REPO
        assert revision == materializer.MODEL_REVISION
        local_dir.mkdir()
        (local_dir / "model.safetensors").write_bytes(weights)
        (local_dir / "tokenizer.json").write_text("{}")

    monkeypatch.setattr(materializer, "snapshot_download", snapshot_download)
    monkeypatch.setattr(
        materializer.AutoTokenizer, "from_pretrained", lambda _: _Tokenizer()
    )


def test_exact_bfs_distance_and_stable_board_rank() -> None:
    states = materializer._states_by_distance(12)
    assert [len(states[index]) for index in range(4)] == [1, 2, 4, 8]
    assert all(
        fixed_pool_module._sliding_puzzle_distance(state) == distance
        for distance, values in states.items()
        for state in values
    )
    assert len(
        {
            materializer._permutation_rank(state)
            for values in states.values()
            for state in values
        }
    ) == sum(map(len, states.values()))


def test_materializer_is_deterministic_and_strictly_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_snapshot(monkeypatch, b"synthetic pinned instruct weights")
    output = tmp_path / "private-puzzle"
    materializer.materialize(output, b"x" * 32)

    manifest_path = output / "fixed_pool_manifest.v1.44001.json"
    manifest = load_fixed_pool_manifest(manifest_path)
    validate_fixed_pool_materialization(manifest)
    validate_fixed_pool_manifest_design(
        manifest, "sliding_puzzle_latency_feasibility_v1"
    )
    assert len(manifest.items) == 16
    assert manifest.cohort_sizes == (4, 4, 4, 4)
    design = json.loads((output / "selection_design.v1.json").read_text())
    assert design["fixed_pool_id"] == manifest.pool_id
    assert design["fixed_pool_manifest_sha256"] == manifest.manifest_sha256
    assert {item["stratum"] for item in design["items"]} == {"easy", "hard"}
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE(path.stat().st_mode) == (0o700 if path.is_dir() else 0o600)
        for path in output.rglob("*")
    )


def test_strict_validator_rejects_forged_optimal_distance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_snapshot(monkeypatch, b"synthetic pinned instruct weights")
    output = tmp_path / "private-puzzle"
    materializer.materialize(output, b"y" * 32)
    source_path = output / "sliding_puzzle_easy.jsonl"
    rows = [json.loads(line) for line in source_path.read_text().splitlines()]
    rows[0]["optimal_distance"] = 12
    source_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )
    manifest = load_fixed_pool_manifest(output / "fixed_pool_manifest.v1.44001.json")
    with pytest.raises(Exception, match="SHA mismatch"):
        validate_fixed_pool_manifest_design(
            manifest, "sliding_puzzle_latency_feasibility_v1"
        )
