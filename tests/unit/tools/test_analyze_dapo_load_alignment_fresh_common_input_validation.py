from pathlib import Path

import pytest

from tools import analyze_dapo_load_alignment_fresh_common_input_validation as analyzer
from tools import (
    materialize_dapo_load_alignment_common_input_validation as materializer,
)


def _runs(tmp_path: Path) -> dict[tuple[int, int], Path]:
    return {
        (pool, generation): tmp_path / f"{pool}-{generation}"
        for pool in materializer.POOL_SEEDS
        for generation in materializer.VALIDATION_SEEDS[pool]
    }


def _patch_inputs(
    monkeypatch: pytest.MonkeyPatch, *, fail_last_poll: bool = False
) -> None:
    monkeypatch.setattr(
        analyzer, "_validate_execution_authority", lambda _path: "a" * 64
    )
    monkeypatch.setattr(
        analyzer,
        "_load_private_calibration",
        lambda _path, expected_sha256: {
            pool: (frozenset(f"prompt-{index}" for index in range(16)), "b" * 64)
            for pool in materializer.POOL_SEEDS
        },
    )
    monkeypatch.setattr(
        analyzer,
        "_validate_trace",
        lambda **_kwargs: {
            "generation_seed": 1,
            "prompt_groups": 32,
            "completions": 512,
        },
    )

    def effects(**kwargs: object) -> dict[str, float | bool]:
        poll = float(kwargs["poll_interval_ms"])
        low = 0.01 if fail_last_poll and poll == 20.0 else -0.05
        return {
            "natural": 0.0,
            "high_load_delayed": 0.06,
            "low_load_delayed": low,
            "signed_separation": 0.06 - low,
            "l0_negative_control_exact": True,
        }

    monkeypatch.setattr(analyzer, "_stream_effects", effects)


def test_fresh_validation_passes_only_when_every_poll_model_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_inputs(monkeypatch)

    result = analyzer.analyze(
        _runs(tmp_path),
        balanced_manifest_paths={
            pool: tmp_path / str(pool) for pool in materializer.POOL_SEEDS
        },
        private_calibration_manifest_path=tmp_path / "private.json",
        expected_private_calibration_sha256="c" * 64,
        validation_authority_path=tmp_path / "authority.json",
    )

    assert result["decision"] == (
        "close_as_replicated_common_input_selection_layer_mechanism_at_24_seconds"
    )
    assert result["locked_checks"] == {
        "all_six_validation_collections_valid": True,
        "success_gate_passed_at_every_polling_interval": True,
    }


def test_one_poll_model_failure_closes_without_replication_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_inputs(monkeypatch, fail_last_poll=True)

    result = analyzer.analyze(
        _runs(tmp_path),
        balanced_manifest_paths={
            pool: tmp_path / str(pool) for pool in materializer.POOL_SEEDS
        },
        private_calibration_manifest_path=tmp_path / "private.json",
        expected_private_calibration_sha256="c" * 64,
        validation_authority_path=tmp_path / "authority.json",
    )

    assert result["decision"] == "close_as_exploratory_regime_not_freshly_replicated"
    poll = result["poll_timing_sensitivity"]
    assert poll["20"]["passed"] is False  # type: ignore[index]
