"""Persist only training-state hashes/counters, never tensors or loss outcomes."""

from pathlib import Path
import re
from typing import Any

from research.m4_eligible_replay.contracts import canonical_json


HASH_FIELDS = {"parameter_sha256", "optimizer_sha256", "scheduler_sha256", "rng_sha256"}
COUNTER_FIELDS = {"optimizer_step", "scheduler_num_steps"}
BOOL_FIELDS = {"load_optim", "load_rng", "finetune"}
RNG_FIELDS = {"python", "numpy", "torch", "cuda", "tracker"}


def validate_hash(value: str) -> None:
    """Require a lowercase SHA-256, not arbitrary source data."""
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("diagnostic requires SHA-256 fingerprints only")


def validate_state(state: dict[str, Any]) -> None:
    """Enforce the exact outcome-free snapshot schema."""
    if set(state) != HASH_FIELDS | COUNTER_FIELDS | BOOL_FIELDS | {
        "rng_component_sha256"
    }:
        raise ValueError("unexpected diagnostic state fields")
    for field in HASH_FIELDS:
        validate_hash(state[field])
    for field in COUNTER_FIELDS:
        if type(state[field]) is not int or state[field] < 0:
            raise ValueError("invalid diagnostic state counter")
    for field in BOOL_FIELDS:
        if type(state[field]) is not bool:
            raise ValueError("invalid diagnostic state flag")
    components = state["rng_component_sha256"]
    if not isinstance(components, dict) or set(components) != RNG_FIELDS:
        raise ValueError("invalid RNG component schema")
    for value in components.values():
        validate_hash(value)


def changed_fields(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """List aggregate and RNG-subcomponent differences without guessing a cause."""
    validate_state(before)
    validate_state(after)
    result = [
        key
        for key in before
        if before[key] != after[key] and key != "rng_component_sha256"
    ]
    result += [
        "rng_component_sha256." + key
        for key in RNG_FIELDS
        if before["rng_component_sha256"][key] != after["rng_component_sha256"][key]
    ]
    return sorted(result)


def write_diagnostic(
    directory: Path,
    *,
    replicate: str,
    stage: str,
    states: dict[str, dict[str, Any]],
    protocol_hash: str,
    manifest_hash: str,
) -> Path:
    """Save an exclusive compact record before a fail-closed gate can raise."""
    stages = {
        "before": {"before"},
        "idle": {"idle"},
        "after-eval": {"after_eval"},
        "comparison": {"before", "idle", "after_eval"},
    }
    if (
        replicate not in ("a", "b")
        or stage not in stages
        or set(states) != stages[stage]
    ):
        raise ValueError("invalid diagnostic identity/stage")
    validate_hash(protocol_hash)
    validate_hash(manifest_hash)
    for state in states.values():
        validate_state(state)
    record = {
        "schema": "m4-eligible-live-evaluation-state-diagnostic-v1",
        "replicate": replicate,
        "stage": stage,
        "protocol_sha256": protocol_hash,
        "capsule_manifest_sha256": manifest_hash,
        "states": states,
        "comparator_update_started": False,
        "post_update_heldout_loss_computed": False,
        "causal_contrast_computed": False,
    }
    if stage == "comparison":
        record["idle_changed_fields"] = changed_fields(states["before"], states["idle"])
        record["evaluation_changed_fields"] = changed_fields(
            states["before"], states["after_eval"]
        )
        record["post_idle_evaluation_changed_fields"] = changed_fields(
            states["idle"], states["after_eval"]
        )
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"replay-{replicate}-state-{stage}.json"
    with output.open("xb") as stream:
        stream.write(canonical_json(record))
        stream.flush()
    return output
