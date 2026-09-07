import json
from pathlib import Path

import pytest

from tools import materialize_structured_generation_latency_pool as materializer


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


def test_generated_records_are_fresh_balanced_pairs() -> None:
    records = materializer._make_records(_Tokenizer(), b"k" * 32)

    assert tuple(records) == materializer.SOURCE_IDS
    assert all(len(rows) == 8 for rows in records.values())
    for pair_index in range(8):
        short = records["structured_short"][pair_index]
        long = records["structured_long"][pair_index]
        assert short["matching_pair_id"] == long["matching_pair_id"]
        assert short["repeated_prompt_cluster_id"] == long["repeated_prompt_cluster_id"]
        assert short["source_prompt_id"] != long["source_prompt_id"]
        assert short["required_check_lines"] == 2
        assert long["required_check_lines"] == 16
        assert short["expected_answer"] == long["expected_answer"]
        assert short["output"] == str(short["expected_answer"])
        assert "exactly 2 numbered check lines" in short["input"]
        assert "exactly 16 numbered check lines" in long["input"]
        assert "\\boxed{" in short["input"]


def test_plan_id_is_canonical_and_bytes_are_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = {
        "analysis_status": "preregistered_controlled_generative_demand_feasibility",
        "attempt_limit": 1,
        "calibration_only": True,
        "confirmatory_eligible": False,
        "model": {
            "generation_config_sha256": materializer.model_pin.GENERATION_CONFIG_SHA256,
            "repo_id": materializer.model_pin.MODEL_REPO,
            "revision": materializer.model_pin.MODEL_REVISION,
            "weight_files": [
                {"path": name, "bytes": size, "sha256": digest}
                for name, size, digest in materializer.model_pin.EXPECTED_WEIGHT_FILES
            ],
            "weight_manifest_sha256": materializer.model_pin.MODEL_WEIGHTS_SHA256,
        },
        "pool": {
            "cohort_size": 4,
            "generated_pair_count": 8,
            "order_seed": 45001,
            "pairing": "same arithmetic problem with short and long response contracts",
            "prompt_groups": 16,
            "selection_seed": 20260907,
            "source_exposure_status": "fresh_pre_generation",
            "task_count_per_cohort": {
                "structured_long": 2,
                "structured_short": 2,
            },
        },
        "runtime": {"generation_study_seed": 64001},
        "schema_version": 1,
        "training_authorized": False,
    }
    plan_id = materializer._sha_bytes(materializer._canonical(plan))
    plan["plan_id"] = plan_id
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr(materializer, "PLAN_ID", plan_id)
    monkeypatch.setattr(materializer, "PLAN_SHA256", materializer._sha_path(path))

    loaded, raw = materializer._load_plan(path)

    assert loaded == plan
    assert raw == path.read_bytes()
    path.write_text(path.read_text() + "\n")
    with pytest.raises(
        materializer.StructuredGenerationMaterializationError,
        match="plan byte hash mismatch",
    ):
        materializer._load_plan(path)
