# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for confirmed structured scheduler-crossover materialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import materialize_structured_generation_scheduler_crossover as materializer


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs) -> str:
        return f"<chat>{messages[0]['content']}</chat>"

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text.split())))}


def _candidate() -> dict[str, object]:
    return {
        "schema_version": 1,
        "analysis_status": "candidate_controlled_natural_latency_scheduler_crossover",
        "calibration_only": True,
        "candidate_status": "awaiting_explicit_user_confirmation",
        "confirmatory_eligible": False,
        "authorization": {"implementation_authorized": False},
        "replications": [
            {
                "order_seed": order_seed,
                "selection_seed": selection_seed,
                "generation_study_seed": generation_seed,
            }
            for order_seed, (
                selection_seed,
                generation_seed,
            ) in materializer.REPLICATION_SEEDS.items()
        ],
    }


def test_confirmed_candidate_bytes_and_replication_seeds_are_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(_candidate(), sort_keys=True))
    monkeypatch.setattr(
        materializer,
        "PROTOCOL_SHA256",
        materializer.base._sha_path(path),
    )

    assert materializer._load_confirmed_candidate(path) == path.read_bytes()

    candidate = _candidate()
    candidate["replications"][0]["selection_seed"] = 1  # type: ignore[index]
    path.write_text(json.dumps(candidate, sort_keys=True))
    monkeypatch.setattr(
        materializer,
        "PROTOCOL_SHA256",
        materializer.base._sha_path(path),
    )
    with pytest.raises(
        materializer.StructuredSchedulerCrossoverMaterializationError,
        match="replication seeds mismatch",
    ):
        materializer._load_confirmed_candidate(path)


def test_materialization_binds_fresh_seed_and_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = tmp_path / "candidate.json"
    protocol.write_text(json.dumps(_candidate(), sort_keys=True))
    protocol_sha = materializer.base._sha_path(protocol)
    monkeypatch.setattr(materializer, "PROTOCOL_SHA256", protocol_sha)

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
        order_seed=46002,
        key=b"k" * 32,
    )

    manifest = json.loads((output / "fixed_pool_manifest.v1.46002.json").read_text())
    design = json.loads((output / "selection_design.v1.json").read_text())
    report = json.loads((output / "materialization_report.v1.json").read_text())
    assert manifest["order_seed"] == 46002
    assert manifest["design_protocol_sha256"] == protocol_sha
    assert {source["split"] for source in manifest["sources"]} == {
        "scheduler_crossover"
    }
    assert design["protocol_id"] == protocol_sha
    assert report["selection_seed"] == 2026090802
    assert report["protocol_sha256"] == protocol_sha
    assert len(manifest["items"]) == 16
