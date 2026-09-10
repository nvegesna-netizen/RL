# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from pathlib import Path

import pytest

from tools import materialize_scheduler_pressure_response as materializer


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs) -> str:
        return f"<chat>{messages[0]['content']}</chat>"

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text.split())))}


def _candidate() -> dict[str, object]:
    return {
        "schema_version": 1,
        "analysis_status": "candidate_zero_update_scheduler_pressure_response_surface",
        "calibration_only": True,
        "confirmatory_eligible": False,
        "implementation_authorized": False,
        "materialization_authorized": False,
        "eos_launch_authorized": False,
        "counterfactual_replay_authorized": False,
        "training_authorized": False,
        "requires_separate_final_plan_confirmation_before_any_eos_launch": True,
        "design": {
            "fresh_pool_order_seeds": [49001, 49002, 49003],
            "selection_seeds": [2026091001, 2026091002, 2026091003],
            "generation_study_seeds": [69001, 69002, 69003],
            "prompt_groups_per_pool": 32,
            "dispatch_cohorts": 8,
        },
    }


def _authorization(protocol_sha: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "analysis_status": "confirmed_scheduler_pressure_response_validation_and_materialization",
        "confirmed_candidate_sha256": protocol_sha,
        "confirmed_authorization_candidate_sha256": "a" * 64,
        "implementation_commit": "b" * 40,
        "jet_route_commit": "c" * 40,
        "incremental_bundle_sha256": "d" * 64,
        "authorization": {
            "exact_image_validation": True,
            "materialize_three_fresh_pools": True,
            "scheduler_arm_rollout": False,
            "counterfactual_replay": False,
            "learner_training": False,
        },
    }


def test_materializes_balanced_32_group_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = tmp_path / "candidate.json"
    protocol.write_text(json.dumps(_candidate(), sort_keys=True))
    monkeypatch.setattr(
        materializer, "PROTOCOL_SHA256", materializer.base._sha_path(protocol)
    )
    authorization = tmp_path / "authorization.json"
    authorization.write_text(
        json.dumps(_authorization(materializer.PROTOCOL_SHA256), sort_keys=True)
    )

    def _snapshot(root: Path):
        snapshot = b'{"schema_version":1}\n'
        (root / "model_snapshot").mkdir()
        (root / "model_snapshot_manifest.v1.json").write_bytes(snapshot)
        return _Tokenizer(), snapshot

    monkeypatch.setattr(materializer.base.model_pin, "_snapshot_model", _snapshot)
    output = tmp_path / "pool"
    materializer.materialize(
        output_dir=output,
        protocol_path=protocol,
        authorization_path=authorization,
        order_seed=49002,
        key=b"k" * 32,
    )
    manifest = json.loads((output / "fixed_pool_manifest.v1.49002.json").read_text())
    design = json.loads((output / "selection_design.v1.json").read_text())
    report = json.loads((output / "materialization_report.v1.json").read_text())
    assert len(manifest["items"]) == len(design["items"]) == 32
    assert sorted(item["pair_id"] for item in design["items"]) == [
        f"pair-{pair_index}" for pair_index in range(16) for _ in range(2)
    ]
    assert {item["dispatch_cohort"] for item in manifest["items"]} == set(range(8))
    assert [source["split"] for source in manifest["sources"]] == [
        "scheduler_pressure_response",
        "scheduler_pressure_response",
    ]
    assert report[
        "materialization_authorization_sha256"
    ] == materializer.base._sha_path(authorization)
