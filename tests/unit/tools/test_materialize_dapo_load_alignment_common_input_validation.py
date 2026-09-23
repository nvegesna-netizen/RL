import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import (
    materialize_dapo_load_alignment_common_input_validation as materializer,
)


def test_protocol_confirmation_cannot_authorize_materialization(
    tmp_path: Path,
) -> None:
    path = tmp_path / "authority.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "analysis_status": (
                    "dapo_load_alignment_fresh_common_input_validation_"
                    "protocol_confirmation"
                ),
                "candidate_sha256": materializer.PROTOCOL_SHA256,
                "authorized": {
                    "eos_launch": False,
                    "fresh_pool_materialization": False,
                },
            }
        )
    )

    with pytest.raises(
        materializer.DapoFreshCommonInputMaterializationError,
        match="authority mismatch",
    ):
        materializer._load_execution_authority(path)


def test_execution_authority_is_exact_and_calibration_only(tmp_path: Path) -> None:
    path = tmp_path / "authority.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "analysis_status": (
                    "dapo_load_alignment_fresh_common_input_"
                    "materialization_calibration_confirmation"
                ),
                "confirmed": True,
                "protocol_sha256": materializer.PROTOCOL_SHA256,
                "protocol_confirmation_sha256": materializer.CONFIRMATION_SHA256,
                "authorization": {
                    "eos_launch": True,
                    "fresh_pool_materialization": True,
                    "calibration_generation": True,
                    "validation_generation": False,
                    "live_scheduler_arm": False,
                    "learner_replay": False,
                    "learner_training": False,
                    "population_claim": False,
                },
            }
        )
    )

    raw, digest = materializer._load_execution_authority(path)

    assert raw == path.read_bytes()
    assert len(digest) == 64


def test_fresh_pool_selection_excludes_complete_prior_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts = [
        SimpleNamespace(canonical_sha256=f"{index:064x}") for index in range(420)
    ]
    discovery = {item.canonical_sha256 for item in prompts[:48]}
    crossover = {item.canonical_sha256 for item in prompts[48:112]}
    mixture = {item.canonical_sha256 for item in prompts[112:208]}
    ledger = {item.canonical_sha256 for item in prompts[208:304]}
    monkeypatch.setattr(
        materializer.prior_load_alignment,
        "_reconstructed_prior_ids",
        lambda _prompts: (discovery, crossover, mixture),
    )

    pools, exclusions = materializer._select_fresh_pools(prompts, ledger)  # type: ignore[arg-type]

    selected = {item.canonical_sha256 for pool in pools.values() for item in pool}
    assert len(selected) == 96
    assert not selected.intersection(discovery | crossover | mixture | ledger)
    assert exclusions["complete_prior_study_ledger"] == ledger
