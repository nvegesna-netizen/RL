import json
from pathlib import Path

import pytest

from tools import analyze_dapo_load_alignment_common_input_calibration as analyzer


def test_balanced_validation_ordinals_are_deterministic_and_exact() -> None:
    lower = tuple(range(0, 32, 2))

    first = analyzer.balanced_validation_ordinals(lower, pool_seed=55001)
    second = analyzer.balanced_validation_ordinals(lower, pool_seed=55001)

    assert first == second
    assert set(first) == set(range(32))
    for offset in range(0, 32, 4):
        cohort = first[offset : offset + 4]
        assert sum(value in lower for value in cohort) == 2


def test_balanced_artifacts_reindex_manifest_and_design(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(analyzer, "compute_fixed_pool_id", lambda _value: "f" * 64)
    manifest_items = [
        {
            "ordinal": index,
            "source_prompt_id": f"prompt-{index}",
            "dispatch_cohort": index // 4,
            "decorrelation_block": f"cohort-{index // 4}",
        }
        for index in range(32)
    ]
    design_items = [
        {
            "source_pool_ordinal": index,
            "source_prompt_id": f"prompt-{index}",
        }
        for index in range(32)
    ]
    (tmp_path / "fixed_pool_manifest.v1.55001.json").write_text(
        json.dumps({"items": manifest_items, "pool_id": "old"})
    )
    (tmp_path / "selection_design.v1.55001.json").write_text(
        json.dumps({"items": design_items, "fixed_pool_id": "old"})
    )

    manifest_raw, design_raw, old_to_new = analyzer._balanced_artifacts(
        materialization_root=tmp_path,
        pool_seed=55001,
        lower_load_ordinals=tuple(range(16)),
    )

    manifest = json.loads(manifest_raw)
    design = json.loads(design_raw)
    assert design["fixed_pool_id"] == manifest["pool_id"]
    assert design["analysis_status"] == (
        "controlled_dapo_common_input_balanced_validation_pool"
    )
    assert {item["ordinal"] for item in manifest["items"]} == set(range(32))
    assert set(old_to_new) == set(range(32))
    for cohort in range(8):
        members = [
            old
            for old, new in old_to_new.items()
            if new // analyzer.GROUPS_PER_COHORT == cohort
        ]
        assert sum(old < 16 for old in members) == 2
