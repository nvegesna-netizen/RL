"""Hash-bound reuse of the actual accepted minimal-observer runtime."""

from __future__ import annotations

import json
from pathlib import Path

from research.m4_eligible_replay.contracts import digest_file


REFERENCE_HASH = "6616d16e0b415fc02511e3ea2cdef1832c562eaac127f7ff9fd51e01d7ab2723"
REFERENCE_FILE = "reports/auto_research/2026-10-07-m4-causal-utility/eligible_live_minimal_observer_terminal_result.json"


def verify_reference_runtime(runtime: dict) -> None:
    """Require the complete accepted actor runtime and custom worker file hash."""
    root = Path(__file__).resolve().parents[2]
    path = root / REFERENCE_FILE
    if digest_file(path) != REFERENCE_HASH:
        raise ValueError("accepted runtime evidence changed")
    reference = json.loads(path.read_bytes())
    if (
        reference["qualification_passed"] is not True
        or runtime != reference["runtime_common"]
    ):
        raise ValueError("runtime differs from accepted minimal-observer variant")
