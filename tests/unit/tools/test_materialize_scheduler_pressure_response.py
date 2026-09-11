# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from collections import Counter
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


def _amendment() -> dict[str, object]:
    return {
        "candidate_status": "awaiting_exact_user_confirmation",
        "requires_exact_hash_confirmation": True,
        "non_retroactivity": {"reclassify_replication_49001": False},
        "prospective_replications": {
            "replication_order": [49002, 49003, 49004],
            "new_pool_required": {
                "order_seed": 49004,
                "selection_seed": 2026091004,
                "generation_study_seed": 69004,
                "status": "not_materialized",
            },
        },
    }


def _amendment_confirmation(amendment_sha: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "analysis_status": "confirmed_prospective_scheduler_pressure_response_concurrency_amendment",
        "confirmed_amendment_candidate_sha256": amendment_sha,
        "authorization": {
            "materialize_and_validate_pool_49004_without_rollout": True,
            "scheduler_arm_49002": False,
            "scheduler_arm_49003": False,
            "scheduler_arm_49004": False,
            "counterfactual_replay": False,
            "learner_training": False,
            "population_claim": False,
        },
    }


def _replacement_amendment() -> dict[str, object]:
    return {
        "status": "candidate_awaiting_exact_confirmation",
        "analysis_status": "prospective_runtime_and_replacement_amendment",
        "integrity_decisions": {
            "replication_49004_consumed": True,
            "same_seed_retry_allowed": False,
        },
        "prospective_replacement": {
            "replacement_order_seed": 49005,
            "selection_seed": 2026091005,
            "generation_study_seed": 69005,
            "fresh_pool_required": True,
            "replacement_replication_order": [49002, 49003, 49005],
        },
        "authorization_if_exactly_confirmed": {
            "fresh_pool_49005_materialization_and_validation": True,
            "scheduler_arm_49005": False,
            "counterfactual_replay": False,
            "learner_training": False,
            "population_claim": False,
        },
    }


def _replacement_confirmation(amendment_sha: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "confirmed_for_bounded_runtime_repair_validation_and_replacement_pool_materialization",
        "confirmed_amendment_candidate_sha256": amendment_sha,
        "confirmed_on": "2026-09-11",
        "confirmation_source": "test",
        "source_plan_id": "a" * 64,
        "source_plan_file_sha256": "b" * 64,
        "triggering_failure_audit_sha256": "c" * 64,
        "authorization": {
            "bounded_runtime_repair": True,
            "local_validation": True,
            "pinned_image_no_rollout_validation": True,
            "fresh_pool_49005_materialization_and_validation_after_pinned_image_pass": True,
            "scheduler_arm_49005": False,
            "counterfactual_replay": False,
            "learner_training": False,
            "population_claim": False,
        },
        "required_sequence": ["test"],
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
        amendment_path=None,
        order_seed=49002,
        key=b"k" * 32,
    )
    manifest = json.loads((output / "fixed_pool_manifest.v1.49002.json").read_text())
    design = json.loads((output / "selection_design.v1.json").read_text())
    report = json.loads((output / "materialization_report.v1.json").read_text())
    assert len(manifest["items"]) == len(design["items"]) == 32
    assert Counter(item["pair_id"] for item in design["items"]) == {
        f"pair-{pair_index}": 2 for pair_index in range(16)
    }
    assert {item["dispatch_cohort"] for item in manifest["items"]} == set(range(8))
    assert [source["split"] for source in manifest["sources"]] == [
        "scheduler_pressure_response",
        "scheduler_pressure_response",
    ]
    assert report[
        "materialization_authorization_sha256"
    ] == materializer.base._sha_path(authorization)


def test_materializes_amendment_bound_49004_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = tmp_path / "candidate.json"
    protocol.write_text(json.dumps(_candidate(), sort_keys=True))
    monkeypatch.setattr(
        materializer, "PROTOCOL_SHA256", materializer.base._sha_path(protocol)
    )
    amendment = tmp_path / "amendment.json"
    amendment.write_text(json.dumps(_amendment(), sort_keys=True))
    monkeypatch.setattr(
        materializer, "AMENDMENT_SHA256", materializer.base._sha_path(amendment)
    )
    confirmation = tmp_path / "amendment-confirmation.json"
    confirmation.write_text(
        json.dumps(
            _amendment_confirmation(materializer.AMENDMENT_SHA256), sort_keys=True
        )
    )
    monkeypatch.setattr(
        materializer,
        "AMENDMENT_CONFIRMATION_SHA256",
        materializer.base._sha_path(confirmation),
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
        authorization_path=confirmation,
        amendment_path=amendment,
        order_seed=49004,
        key=b"k" * 32,
    )

    manifest = json.loads((output / "fixed_pool_manifest.v1.49004.json").read_text())
    report = json.loads((output / "materialization_report.v1.json").read_text())
    assert manifest["order_seed"] == 49004
    assert report["selection_seed"] == 2026091004
    assert report["concurrency_amendment_sha256"] == materializer.base._sha_path(
        amendment
    )
    assert report[
        "concurrency_amendment_confirmation_sha256"
    ] == materializer.base._sha_path(confirmation)


def test_materializes_replacement_bound_49005_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = tmp_path / "candidate.json"
    protocol.write_text(json.dumps(_candidate(), sort_keys=True))
    monkeypatch.setattr(
        materializer, "PROTOCOL_SHA256", materializer.base._sha_path(protocol)
    )
    amendment = tmp_path / "replacement-amendment.json"
    amendment.write_text(json.dumps(_replacement_amendment(), sort_keys=True))
    monkeypatch.setattr(
        materializer,
        "REPLACEMENT_AMENDMENT_SHA256",
        materializer.base._sha_path(amendment),
    )
    confirmation = tmp_path / "replacement-confirmation.json"
    confirmation.write_text(
        json.dumps(
            _replacement_confirmation(materializer.REPLACEMENT_AMENDMENT_SHA256),
            sort_keys=True,
        )
    )
    monkeypatch.setattr(
        materializer,
        "REPLACEMENT_CONFIRMATION_SHA256",
        materializer.base._sha_path(confirmation),
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
        authorization_path=confirmation,
        amendment_path=amendment,
        order_seed=49005,
        key=b"k" * 32,
    )
    manifest = json.loads((output / "fixed_pool_manifest.v1.49005.json").read_text())
    report = json.loads((output / "materialization_report.v1.json").read_text())
    assert manifest["order_seed"] == 49005
    assert report["selection_seed"] == 2026091005
    assert report["generation_seed"] == 69005
    assert report["replacement_runtime_amendment_sha256"] == (
        materializer.base._sha_path(amendment)
    )
    assert report["replacement_runtime_amendment_confirmation_sha256"] == (
        materializer.base._sha_path(confirmation)
    )
