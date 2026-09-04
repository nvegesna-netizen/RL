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
from tools import materialize_sliding_puzzle_7b_compact_action as compact
from tools import materialize_sliding_puzzle_7b_competence as competence
from tests.unit.tools.test_materialize_sliding_puzzle_7b_competence import (
    _make_source,
)


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

    def __call__(self, text, *, return_tensors, add_special_tokens):
        assert return_tensors is None and not add_special_tokens
        return {"input_ids": list(range(len(text.split())))}


def _game_state() -> dict[str, object]:
    return {
        "size": 3,
        "grid": [[1, 2, 3], [4, 5, 6], [7, 0, 8]],
        "solution": [[1, 2, 3], [4, 5, 6], [7, 8, 0]],
        "empty_pos": [2, 1],
        "commands": {},
    }


def test_rewrite_prompt_is_compact_action_only() -> None:
    record = {
        "messages": [{"role": "user", "content": "legacy prompt"}],
        "extra_env_info": {"game_state": _game_state()},
        "prompt_builder_version": "legacy_sliding_puzzle_prompt_v1",
        "source_id": "sliding_puzzle_easy",
    }
    result = compact._rewrite_prompt(_Tokenizer(), record)
    content = result["messages"][0]["content"]

    assert content.startswith("<chat>Solve this 3x3 sliding puzzle")
    assert "Reply with exactly one legal move and no other text." in content
    assert "<action>up</action>" in content
    assert "<action></action>" not in content
    assert "step-by-step" not in content
    assert result["prompt_builder_version"] == "compact_action_only_v2"
    assert result["source_id"] == record["source_id"]
    expected_ids = list(range(len(content.split())))
    assert result["input_token_count"] == len(expected_ids)
    assert (
        result["input_token_ids_sha256"]
        == hashlib.sha256(
            json.dumps(expected_ids, separators=(",", ":")).encode()
        ).hexdigest()
    )


def test_load_plan_rejects_unpinned_bytes(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text("{}\n")

    with pytest.raises(
        compact.CompactActionMaterializationError,
        match="plan byte hash mismatch",
    ):
        compact._load_plan(plan)


def test_materializes_same_boards_with_compact_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = _make_source(tmp_path, monkeypatch)
    source_manifest = load_fixed_pool_manifest(source_path)
    selection_design = source_path.parent / "selection_design.v1.json"
    source_manifest_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    source_selection_sha = hashlib.sha256(selection_design.read_bytes()).hexdigest()
    monkeypatch.setattr(competence, "SOURCE_MANIFEST_SHA256", source_manifest_sha)
    monkeypatch.setattr(competence, "SOURCE_POOL_ID", source_manifest.pool_id)
    monkeypatch.setattr(
        competence, "SOURCE_SELECTION_DESIGN_SHA256", source_selection_sha
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
        "analysis_status": "calibration_only_exposed_pool_prompt_competence",
        "attempt_limit": 1,
        "calibration_only": True,
        "confirmatory_eligible": False,
        "expected_boards": 16,
        "hard_board_metrics_gate": False,
        "latency_or_readiness_inference_allowed": False,
        "model": {
            "generation_config_sha256": competence.GENERATION_CONFIG_SHA256,
            "repo_id": competence.MODEL_REPO,
            "revision": competence.MODEL_REVISION,
            "sampling_source": "pinned_generation_config_supported_fields",
            "weight_files": weight_records,
            "weight_manifest_sha256": weight_manifest_sha,
        },
        "prompt_intervention": {
            "builder_version": compact.PROMPT_BUILDER_VERSION,
            "changes_only_prompt": True,
            "legacy_builder_version": "legacy_sliding_puzzle_prompt_v1",
            "paired_prior_pipeline": 66188117,
            "response_contract": "exactly_one_action_tag_and_no_other_text",
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
        "schema_version": 2,
        "source_pool": {
            "board_exposure_status": "previously_exposed_calibration_pool",
            "fixed_pool_manifest_sha256": source_manifest_sha,
            "pool_id": source_manifest.pool_id,
            "selection_design_sha256": source_selection_sha,
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
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr(compact, "PLAN_ID", plan_id)
    monkeypatch.setattr(compact, "PLAN_SHA256", _sha(plan_path.read_bytes()))

    output = tmp_path / "compact"
    compact.materialize(source_path=source_path, plan_path=plan_path, output_dir=output)
    manifest = load_fixed_pool_manifest(output / compact.OUTPUT_MANIFEST)
    validate_fixed_pool_materialization(manifest)
    validate_fixed_pool_manifest_design(manifest, compact.DESIGN_ID)
    rows = [
        json.loads(line)
        for source in manifest.sources
        for line in (output / source.materialized_file).read_text().splitlines()
    ]
    assert {row["prompt_builder_version"] for row in rows} == {
        compact.PROMPT_BUILDER_VERSION
    }
    assert all("step-by-step" not in row["messages"][0]["content"] for row in rows)
    assert all("<action></action>" not in row["messages"][0]["content"] for row in rows)
    assert sorted(row["board_state_sha256"] for row in rows) == sorted(
        json.loads(line)["board_state_sha256"]
        for source in source_manifest.sources
        for line in (source_path.parent / source.materialized_file)
        .read_text()
        .splitlines()
    )


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
