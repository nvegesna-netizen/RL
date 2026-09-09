"""Tests for the outcome-free Llama observer benchmark."""

from __future__ import annotations

from tools.m4_llama_observer_benchmark import run_benchmark


def test_observer_benchmark_exercises_exact_end_to_end_paths() -> None:
    result = run_benchmark(sequence_length=2048, iterations=8, trials=3)

    assert result["schema"] == "m4-llama-observer-no-training-benchmark-v1"
    assert result["siblings"] == 8
    assert result["sequence_length"] == 2048
    assert result["exact_summary_equivalence"] is True
    assert result["candidate_trial_median_mean_ms"] > 0
    assert result["reference_trial_median_mean_ms"] > 0
