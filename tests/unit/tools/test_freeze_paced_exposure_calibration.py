"""Freeze arithmetic/binding tests; trace validators are mocked, not certified here."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import freeze_paced_exposure_calibration as freezer


def fixture_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, tuple[Path, Path]]:
    root = tmp_path / "pool"
    root.mkdir()
    spec = {
        "schema_version": 1,
        "analysis_status": "exploratory_paced_exposure_execution_spec",
        "source_design_id": "dapo_math_paced_exposure_v1",
        "selection_seed": 100,
        "source_shuffle_seed": 106,
        "calibration_generation_seeds": [101, 102],
        "cadence_pair_generation_seeds": [103, 104, 105],
        "prior_identity_ledger_sha256": "e" * 64,
        "prior_seed_ledger_sha256": "f" * 64,
        "implementation_commit": "a" * 40,
        "runtime_validation_result_sha256": "d" * 64,
        "prompt_groups": 64,
        "completions_per_group": 16,
        "maximum_completions": 8192,
        "scientific_allocation_seconds": 7200,
        "allocated_gpus": 2,
        "training_authorized": False,
    }
    (root / "execution_spec.v1.json").write_text(json.dumps(spec))
    manifest = SimpleNamespace(
        pool_id="b" * 64,
        manifest_sha256="c" * 64,
        model_revision="b101308fe89651ea5ce025f25317fea6fc07e96e",
        model_weights_sha256="70a914a3466bf064ca88ae875cb33c366856147e475a2b8da54e7a3d94d8074a",
        items=tuple(
            SimpleNamespace(source_prompt_id=f"{index:064x}") for index in range(64)
        ),
    )
    monkeypatch.setattr(freezer, "load_fixed_pool_manifest", lambda _: manifest)
    monkeypatch.setattr(freezer, "validate_fixed_pool_materialization", lambda _: None)
    monkeypatch.setattr(
        freezer, "validate_dapo_paced_exposure_manifest_design", lambda _: None
    )
    monkeypatch.setattr(
        freezer, "validate_fixed_pool_trace", lambda *args, **kwargs: None
    )
    paths = (tmp_path / "first.jsonl", tmp_path / "second.jsonl")
    streams = {}
    for index, path in enumerate(paths):
        path.write_text(f"synthetic fixture {index}")
        runtime = {
            "fixed_pool_design_id": "dapo_math_paced_exposure_v1",
            "generation_study_seed": 101 + index,
            "generation_backend": "vllm",
            "configured_max_new_tokens": 4096,
            "max_total_sequence_length": 6144,
            "generation_temperature": 1.0,
            "generation_top_p": 0.7,
            "generation_ignore_eos": False,
            "num_generations_per_prompt": 16,
            "num_prompts_per_step": 4,
            "max_inflight_prompts": 4,
            "max_buffered_rollouts": 16,
        }
        events = [
            SimpleNamespace(
                event_type=freezer.SchedulerEventType.RUN_STARTED,
                scalar_summaries=runtime,
            )
        ]
        for ordinal in range(64):
            events.append(
                SimpleNamespace(
                    event_type=freezer.SchedulerEventType.ATTEMPT_DISPATCHED,
                    monotonic_ns=0,
                )
            )
            events.append(
                SimpleNamespace(
                    event_type=freezer.SchedulerEventType.ROLLOUT_COMPLETED,
                    monotonic_ns=(ordinal + 1) * (index + 1) * 1_000_000_000,
                    source_pool_ordinal=ordinal,
                    scalar_summaries={
                        "mean_gen_tokens_per_sample": ordinal,
                        "reward_mean": 1 - ordinal / 64,
                    },
                )
            )
        streams[path] = events
    monkeypatch.setattr(
        freezer, "iter_scheduler_trace", lambda path: iter(streams[path])
    )
    return root, paths


def test_freeze_uses_calibration_timing_and_preserves_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, paths = fixture_inputs(tmp_path, monkeypatch)
    output = tmp_path / "frozen"
    plan = freezer.freeze_calibration(
        materialization_root=root, trace_paths=paths, output_dir=output
    )
    assert plan.cadence_seconds == 6.0
    assert [arm.consumer_duration_seconds for arm in plan.arms] == [
        3.0,
        3.0,
        6.0,
        6.0,
        12.0,
        12.0,
    ]
    assert plan.pools[0].source_prompt_ids == tuple(
        f"{index:064x}" for index in range(64)
    )
    assert set(plan.pools[0].fixed_lower_load_prompt_ids) == {
        f"{index:064x}" for index in range(32)
    }
    assert set(plan.pools[0].fixed_harder_prompt_ids) == {
        f"{index:064x}" for index in range(32, 64)
    }
    assert (output / "paced_exposure_plan.v1.json").stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        freezer.freeze_calibration(
            materialization_root=root, trace_paths=paths, output_dir=output
        )


def test_reversed_seed_order_cannot_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, paths = fixture_inputs(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="runtime"):
        freezer.freeze_calibration(
            materialization_root=root,
            trace_paths=(paths[1], paths[0]),
            output_dir=tmp_path / "frozen",
        )
    assert not (tmp_path / "frozen").exists()
