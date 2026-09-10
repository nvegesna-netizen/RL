# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import pytest

from tools.m4_llama_size_extension_analysis import LlamaSizeExtensionError, infer_size_contrast
from tools.m4_llama_v5_analysis import ReplicateEndpoint, ReplicateInference


def replicate(name: str, estimate: float, shift: float = 0.0) -> ReplicateInference:
    endpoint = ReplicateEndpoint(estimate, 0.02, (shift,) * 100)
    return ReplicateInference(name, endpoint, endpoint, 0.0, 0.0)


def test_size_contrast_retains_both_sizes_and_replicates() -> None:
    result = infer_size_contrast(
        {"r1": replicate("r1", 0.30), "r2": replicate("r2", 0.34)},
        {"r1": replicate("r1", 0.40), "r2": replicate("r2", 0.44)},
    )
    assert result.estimate == pytest.approx(0.10)
    assert result.hac_interval[0] > 0.0
    assert result.conclusion == "POSITIVE"


def test_size_contrast_requires_two_replicates_per_size() -> None:
    with pytest.raises(LlamaSizeExtensionError, match="r1 and r2"):
        infer_size_contrast(
            {"r1": replicate("r1", 0.30)},
            {"r1": replicate("r1", 0.40), "r2": replicate("r2", 0.44)},
        )


def test_size_contrast_rejects_mismatched_bootstrap_draw_counts() -> None:
    short = ReplicateInference(
        "r1",
        ReplicateEndpoint(0.30, 0.02, (0.0,) * 99),
        ReplicateEndpoint(0.30, 0.02, (0.0,) * 99),
        0.0,
        0.0,
    )
    with pytest.raises(LlamaSizeExtensionError, match="draw counts"):
        infer_size_contrast(
            {"r1": short, "r2": replicate("r2", 0.34)},
            {"r1": replicate("r1", 0.40), "r2": replicate("r2", 0.44)},
        )
