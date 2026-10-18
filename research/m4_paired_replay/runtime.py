"""Hash-bound reuse of the actual accepted minimal-observer runtime."""

from __future__ import annotations

import json
from pathlib import Path

from research.m4_eligible_replay.contracts import digest_file


REFERENCE_HASH = "6616d16e0b415fc02511e3ea2cdef1832c562eaac127f7ff9fd51e01d7ab2723"
REFERENCE_FILE = "reports/auto_research/2026-10-07-m4-causal-utility/eligible_live_minimal_observer_terminal_result.json"
INTEGRATION_FILE = "reports/auto_research/2026-10-07-m4-causal-utility/paired_integration_protocol.json"
INTEGRATED_MODULES = (
    "research.m4_eligible_replay.worker",
    "research.m4_paired_replay.secondary",
    "research.m4_paired_replay.worker_support",
)


def verify_reference_runtime(
    runtime: dict, *, require_integrated: bool = False
) -> None:
    """Require accepted settings and only explicitly hash-bound source amendments."""
    root = Path(__file__).resolve().parents[2]
    path = root / REFERENCE_FILE
    if digest_file(path) != REFERENCE_HASH:
        raise ValueError("accepted runtime evidence changed")
    reference = json.loads(path.read_bytes())
    if reference["qualification_passed"] is not True:
        raise ValueError("accepted runtime qualification did not pass")
    expected = reference["runtime_common"]
    integrated = "research.m4_paired_replay.secondary" in runtime["module_origins"]
    if require_integrated and not integrated:
        raise ValueError("paired integration worker was not loaded")
    if integrated:
        amendment = json.loads((root / INTEGRATION_FILE).read_bytes())
        if (
            amendment["schema"] != "m4-paired-integration-protocol-v1"
            or amendment["accepted_reference_sha256"] != REFERENCE_HASH
            or set(amendment["actor_modules"]) != set(INTEGRATED_MODULES)
        ):
            raise ValueError("paired runtime amendment identity differs")
        for module_name in INTEGRATED_MODULES:
            relative = module_name.replace(".", "/") + ".py"
            pinned = amendment["actor_modules"][module_name]
            if digest_file(root / relative) != pinned:
                raise ValueError("paired runtime amendment source changed")
            expected["module_origins"][module_name] = {
                "path": "/workspace/m4-duplicate-update-repo/" + relative,
                "sha256": pinned,
            }
    if runtime != expected:
        raise ValueError("runtime differs from accepted minimal-observer variant")
