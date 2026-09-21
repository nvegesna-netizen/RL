import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import materialize_dapo_load_alignment as materializer


def test_current_implementation_scope_cannot_authorize_materialization(
    tmp_path: Path,
) -> None:
    path = tmp_path / "implementation-confirmation.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "analysis_status": "candidate_confirmation",
                "candidate_sha256": materializer.PROTOCOL_SHA256,
                "confirmed": True,
                "eos_launch_authorized": False,
                "materialization_authorized": False,
                "reference_collection_authorized": False,
                "scheduler_arm_authorized": False,
                "counterfactual_replay_authorized": False,
                "learner_training_authorized": False,
                "population_claim_authorized": False,
            }
        )
    )

    with pytest.raises(
        materializer.DapoLoadAlignmentMaterializationError,
        match="authority boundary mismatch",
    ):
        materializer._load_materialization_authority(path)


def test_fresh_pool_selection_excludes_every_prior_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts = [
        SimpleNamespace(canonical_sha256=f"{index:064x}") for index in range(320)
    ]
    discovery = {item.canonical_sha256 for item in prompts[:48]}
    crossover = {item.canonical_sha256 for item in prompts[48:112]}
    mixture = {item.canonical_sha256 for item in prompts[112:208]}
    ledger = {item.canonical_sha256 for item in prompts[208:212]}
    monkeypatch.setattr(
        materializer,
        "_reconstructed_prior_ids",
        lambda _prompts: (discovery, crossover, mixture),
    )

    pools, exclusions = materializer._select_fresh_pools(prompts, ledger)  # type: ignore[arg-type]

    selected = {item.canonical_sha256 for pool in pools.values() for item in pool}
    assert len(selected) == 96
    assert not selected.intersection(discovery | crossover | mixture | ledger)
    assert exclusions["ledger"] == ledger
