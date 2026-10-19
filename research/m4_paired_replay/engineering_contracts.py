"""Synthetic-only engineering identities and outcome-free comparison gates."""

from __future__ import annotations

from research.m4_paired_replay.contracts import validate_transition
from research.m4_paired_replay.execution import strict_runtime
from research.m4_paired_replay.runtime import verify_reference_runtime


IDENTITY = "engineering-synthetic-v0-20261018"
DOMAIN = "m4-paired-synthetic-engineering-v1"
SEED = 2026101807
PRIMARIES = {
    "base-control": ("reward_variance_base", False),
    "base-a": ("reward_variance_base", True),
    "base-b": ("reward_variance_base", True),
    "shield-a": ("m4_shield", True),
    "shield-b": ("m4_shield", True),
}
PROBES = {
    "base-train": ("reward_variance_base", "selected"),
    "base-heldout": ("reward_variance_base", "heldout"),
    "shield-train": ("m4_shield", "selected"),
    "shield-heldout": ("m4_shield", "heldout"),
}
STAGES = ("bootstrap", *PROBES, *PRIMARIES)


def synthetic_decision() -> dict:
    """Manufacture routing metadata; this is not a measured M4 eligibility event."""
    base = [f"engineering-g{i}" for i in (0, 1, 2, 3)]
    shield = [f"engineering-g{i}" for i in (0, 1, 4, 5)]
    common = {
        "status": "proposed",
        "search_strategy": "exact",
        "proposed_reward_variance_sum": 4.0,
        "proposed_imminent_reward_variance_sum": 4.0,
        "proposed_valid_actor_tokens": 1024,
    }
    return {
        "skip_reason": None,
        "candidate_group_count": 8,
        "actuation_scorer": "m4_shield",
        "proposal_matches_actual": True,
        "baseline_valid_actor_tokens": 1024,
        "actual_selected_group_ids": shield,
        "proposals": {
            "reward_variance_risk": {
                **common,
                "proposed_group_ids": base,
                "proposed_imminent_l1": 1.0,
            },
            "m4_shield": {
                **common,
                "proposed_group_ids": shield,
                "proposed_imminent_l1": 2.0,
            },
        },
    }


def validate_synthetic_manifest(manifest: dict) -> None:
    if (
        manifest["trajectory_identity"] != IDENTITY
        or manifest["trajectory_seed"] != SEED
        or manifest["assignment_domain"] != DOMAIN
        or manifest["learner_version"] != 0
        or manifest["engineering_synthetic_only"] is not True
        or manifest["decision"] != synthetic_decision()
    ):
        raise ValueError(
            "synthetic qualification cannot use a scientific or historical capsule"
        )


def compare_primaries(records: dict[str, dict], manifest: dict) -> dict:
    """Require both action duplicates and observer noninterference, never loss signs."""
    validate_synthetic_manifest(manifest)
    if set(records) != set(PRIMARIES):
        raise ValueError("synthetic primary process set incomplete")
    initial = records["base-control"]["state_before"]
    runtime = records["base-control"]["runtime"]
    for name, (arm, observed) in PRIMARIES.items():
        record = records[name]
        if (
            record["stage"] != name
            or record["arm"] != arm
            or record["observed"] is not observed
            or record["status"] != "COMPLETE_ENGINEERING_PRIMARY"
            or record["state_before"] != initial
            or record["runtime"] != runtime
            or record["secondaries_complete"] is not observed
        ):
            raise ValueError(
                "synthetic process identity, initial state, or measurement mode differs"
            )
        strict_runtime(record["runtime"])
        verify_reference_runtime(record["runtime"], require_integrated=True)
        validate_transition(
            initial, initial, record["state_after"], record["state_after"], manifest
        )
    if initial["optimizer_step"] != 0 or initial["scheduler_num_steps"] != 0:
        raise ValueError("native checkpoint is not a genuine version-zero restore")
    if (
        not records["base-control"]["state_after"]
        == records["base-a"]["state_after"]
        == records["base-b"]["state_after"]
    ):
        raise ValueError("base duplicate or observer noninterference gate failed")
    if records["shield-a"]["state_after"] != records["shield-b"]["state_after"]:
        raise ValueError("alternate action duplicate gate failed")
    return {
        "status": "PASS_RUNTIME_ENGINEERING_COLLECTOR_CHECK_PENDING",
        "synthetic_only": True,
        "version_zero_restored": True,
        "both_action_duplicates_equal": True,
        "observer_post_state_equal": True,
        "registered_secondaries_complete": True,
        "scientific_outcomes_opened": False,
        "collector_plaintext_exclusion_verified": False,
        "fresh_acquisition_unlocked": False,
    }
