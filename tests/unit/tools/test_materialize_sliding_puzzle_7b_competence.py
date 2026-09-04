import hashlib
import json
from pathlib import Path

import pytest

from nemo_rl.algorithms.async_utils import fixed_pool as fixed_pool_module
from nemo_rl.algorithms.async_utils.fixed_pool import (
    load_fixed_pool_manifest,
    validate_fixed_pool_manifest_design,
    validate_fixed_pool_materialization,
)
from tools import materialize_sliding_puzzle_7b_competence as competence
from tools import materialize_sliding_puzzle_latency_pool as source_materializer


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


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def test_retokenize_preserves_pre_rendered_prompt_bytes() -> None:
    content = "<rendered> puzzle prompt </rendered>"
    record = {"messages": [{"role": "user", "content": content}]}

    updated = competence._retokenize(_Tokenizer(), record)

    expected_ids = list(range(len(content.split())))
    assert updated["messages"] == record["messages"]
    assert updated["input_token_count"] == len(expected_ids)
    assert updated["input_token_ids_sha256"] == _sha(_canonical(expected_ids))


def _make_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    weights = b"source model"
    weights_sha = _sha(weights)
    monkeypatch.setattr(source_materializer, "MODEL_WEIGHTS_SHA256", weights_sha)
    monkeypatch.setattr(
        fixed_pool_module, "SLIDING_PUZZLE_MODEL_WEIGHTS_SHA256", weights_sha
    )

    def snapshot_download(*, repo_id: str, revision: str, local_dir: Path) -> None:
        assert repo_id == source_materializer.MODEL_REPO
        assert revision == source_materializer.MODEL_REVISION
        local_dir.mkdir()
        (local_dir / "model.safetensors").write_bytes(weights)
        (local_dir / "tokenizer.json").write_text("{}")

    monkeypatch.setattr(source_materializer, "snapshot_download", snapshot_download)
    monkeypatch.setattr(
        source_materializer.AutoTokenizer,
        "from_pretrained",
        lambda _: _Tokenizer(),
    )
    output = tmp_path / "source"
    source_materializer.materialize(output, b"s" * 32)
    return output / "fixed_pool_manifest.v1.44001.json"


def test_rematerializes_exact_boards_with_sharded_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = _make_source(tmp_path, monkeypatch)
    source_manifest = load_fixed_pool_manifest(source_path)
    source_design_path = source_path.parent / "selection_design.v1.json"
    monkeypatch.setattr(
        competence, "SOURCE_MANIFEST_SHA256", _sha(source_path.read_bytes())
    )
    monkeypatch.setattr(competence, "SOURCE_POOL_ID", source_manifest.pool_id)
    monkeypatch.setattr(
        competence,
        "SOURCE_SELECTION_DESIGN_SHA256",
        _sha(source_design_path.read_bytes()),
    )

    shard_values = (b"one", b"two", b"three", b"four")
    weight_files = tuple(
        (f"model-0000{index}-of-00004.safetensors", len(value), _sha(value))
        for index, value in enumerate(shard_values, start=1)
    )
    weight_records = [
        {"path": path, "bytes": size, "sha256": sha256}
        for path, size, sha256 in weight_files
    ]
    weight_manifest_sha = _sha(_canonical(weight_records))
    generation_config = b'{"temperature":0.7,"top_p":0.8,"top_k":20}'
    monkeypatch.setattr(competence, "EXPECTED_WEIGHT_FILES", weight_files)
    monkeypatch.setattr(competence, "MODEL_WEIGHTS_SHA256", weight_manifest_sha)
    monkeypatch.setattr(competence, "GENERATION_CONFIG_SHA256", _sha(generation_config))
    monkeypatch.setattr(
        fixed_pool_module,
        "SLIDING_PUZZLE_7B_MODEL_WEIGHTS_SHA256",
        weight_manifest_sha,
    )

    def snapshot_download(*, repo_id: str, revision: str, local_dir: Path) -> None:
        assert repo_id == competence.MODEL_REPO
        assert revision == competence.MODEL_REVISION
        local_dir.mkdir()
        for (relative, _, _), value in zip(weight_files, shard_values, strict=True):
            (local_dir / relative).write_bytes(value)
        (local_dir / "generation_config.json").write_bytes(generation_config)
        (local_dir / "tokenizer.json").write_text("{}")

    monkeypatch.setattr(competence, "snapshot_download", snapshot_download)
    monkeypatch.setattr(
        competence.AutoTokenizer, "from_pretrained", lambda _: _Tokenizer()
    )
    plan = {
        "analysis_status": "calibration_only_exposed_pool_model_competence",
        "attempt_limit": 1,
        "calibration_only": True,
        "confirmatory_eligible": False,
        "expected_boards": 16,
        "hard_board_metrics_gate": False,
        "latency_or_readiness_inference_allowed": False,
        "model": {
            "generation_config_sha256": competence.GENERATION_CONFIG_SHA256,
            "sampling_source": "pinned_generation_config_supported_fields",
            "repo_id": competence.MODEL_REPO,
            "revision": competence.MODEL_REVISION,
            "weight_files": weight_records,
            "weight_manifest_sha256": weight_manifest_sha,
        },
        "replay_authorized": False,
        "runtime": {
            "generation_study_seed": 63001,
            "max_inflight_prompts": 4,
            "max_moves": 12,
            "max_new_tokens_per_turn": 128,
            "max_rollout_turns": 12,
            "max_total_sequence_length": 2048,
            "num_generations_per_prompt": 2,
            "repetition_penalty": 1.0,
            "temperature": 0.7,
            "top_k": 20,
            "top_p": 0.8,
        },
        "schema_version": 1,
        "source_pool": {
            "board_exposure_status": "previously_exposed_calibration_pool",
            "fixed_pool_manifest_sha256": competence.SOURCE_MANIFEST_SHA256,
            "pool_id": competence.SOURCE_POOL_ID,
            "selection_design_sha256": competence.SOURCE_SELECTION_DESIGN_SHA256,
        },
        "thresholds": {
            "action_format_valid_rate_min": 0.9,
            "all_expected_completions_required": True,
            "easy_max_turn_rate_max": 0.25,
            "easy_solve_rate_min": 0.5,
            "easy_truncation_rate_max": 0.25,
            "movement_legal_rate_min": 0.75,
        },
        "training_authorized": False,
    }
    plan_id = _sha(_canonical(plan))
    plan["plan_id"] = plan_id
    monkeypatch.setattr(competence, "PLAN_ID", plan_id)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr(competence, "PLAN_SHA256", _sha(plan_path.read_bytes()))

    output = tmp_path / "competence"
    competence.materialize(
        source_path=source_path, plan_path=plan_path, output_dir=output
    )
    manifest = load_fixed_pool_manifest(
        output / "fixed_pool_manifest.v1.7b_competence.json"
    )
    validate_fixed_pool_materialization(manifest)
    validate_fixed_pool_manifest_design(manifest, "sliding_puzzle_7b_competence_v1")
    lineage = json.loads((output / "source_lineage.v1.json").read_text())
    source_rows = {
        json.loads(line)["board_state_sha256"]
        for source in source_manifest.sources
        for line in (source_path.parent / source.materialized_file)
        .read_text()
        .splitlines()
    }
    assert set(lineage["board_state_sha256"]) == source_rows
    assert manifest.model_revision == competence.MODEL_REVISION
    assert manifest.model_weights_sha256 == weight_manifest_sha
