"""Fresh-cohort gates and fixed arm membership, independent of CUDA."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from research.m4_eligible_replay.contracts import canonical_json, safe_member
from research.m4_eligible_replay.state_diagnostics import validate_hash, validate_state


ARMS = ("reward_variance_base", "m4_shield")
DOMAIN = "m4-eligible-live-causal-acquisition-v2"
PLAN_HASH = "f90956f368753960473c980f2faaccd33375f80736fd36e9afda9049404b4209"
TERMINAL = {"success", "failed", "canceled", "skipped"}


def object_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def validate_plan(plan: dict) -> None:
    """Bind the untouched twenty-seed prospective plan, not caller-selected seeds."""
    if object_hash(plan) != PLAN_HASH:
        raise ValueError("frozen acquisition plan changed")


def capture_gate(plan: dict, captures: list[dict]) -> dict:
    """Require every predeclared trajectory terminal before admitting any replay.

    Inputs are compact authenticated capture classifications. Network/source
    authentication must precede this logical gate; it does not authenticate
    GitLab itself and is not a signed authorization.
    """
    validate_plan(plan)
    if len(captures) != 20:
        raise ValueError("all twenty capture classifications required")
    admitted = []
    jobs = set()
    for expected, row in zip(plan["trajectories"], captures):
        if set(row) != {
            "identity",
            "seed",
            "assignment_domain",
            "terminal_status",
            "eos_job_id",
            "classification",
            "manifest_sha256",
            "evidence_sha256",
        }:
            raise ValueError("unexpected capture classification fields")
        if (row["identity"], row["seed"], row["assignment_domain"]) != (
            expected["identity"],
            expected["seed"],
            DOMAIN,
        ):
            raise ValueError("historical, reordered or substituted assignment")
        if row["terminal_status"] not in TERMINAL:
            raise ValueError("capture chain not terminal")
        if row["eos_job_id"] is None:
            if row["classification"] != "FAILED" or row["terminal_status"] not in {
                "failed",
                "canceled",
                "skipped",
            }:
                raise ValueError(
                    "only authenticated pre-workload failure may lack EOS job"
                )
        else:
            if (
                type(row["eos_job_id"]) is not int
                or row["eos_job_id"] <= 0
                or row["eos_job_id"] in jobs
            ):
                raise ValueError("missing or reused EOS identity")
            jobs.add(row["eos_job_id"])
        validate_hash(row["evidence_sha256"])
        if row["classification"] == "CAPSULE":
            validate_hash(row["manifest_sha256"])
            if row["terminal_status"] != "success":
                raise ValueError("capsule workload did not finish successfully")
            admitted.append(dict(row))
        elif row["classification"] in ("NO_ELIGIBLE", "FAILED"):
            if row["manifest_sha256"] is not None:
                raise ValueError("excluded classification carries a capsule")
            if (
                row["classification"] == "NO_ELIGIBLE"
                and row["terminal_status"] != "success"
            ):
                raise ValueError("no-eligibility record did not finish successfully")
        else:
            raise ValueError("unclassified capture identity")
    if len(admitted) < 16:
        raise ValueError("fewer than sixteen valid capsules; no replay permitted")
    if len({x["manifest_sha256"] for x in admitted}) != len(admitted):
        raise ValueError("capsule reused across trajectories")
    return {
        "schema": "m4-paired-capture-gate-v1",
        "plan_sha256": PLAN_HASH,
        "captures_sha256": object_hash(captures),
        "admitted": admitted,
        "valid_capsules": len(admitted),
        "primary_outcomes_opened": False,
    }


def admit(
    plan: dict, captures: list[dict], identity: str, manifest_bytes: bytes
) -> tuple[dict, dict]:
    """Authenticate fresh manifest identity before checkpoint I/O."""
    gate = capture_gate(plan, captures)
    rows = [r for r in gate["admitted"] if r["identity"] == identity]
    if len(rows) != 1:
        raise ValueError("identity not admitted for replay")
    source = rows[0]
    if hashlib.sha256(manifest_bytes).hexdigest() != source["manifest_sha256"]:
        raise ValueError("capsule bytes do not match admitted hash")
    manifest = json.loads(manifest_bytes)
    if (
        manifest["schema"] != "m4-shield-eligible-live-capsule-v1"
        or manifest["status"] != "QUALIFIED_OUTCOME_EXCLUDED_ELIGIBLE_CAPTURE"
        or (
            manifest["trajectory_identity"],
            manifest["trajectory_seed"],
            manifest["assignment_domain"],
        )
        != (source["identity"], source["seed"], DOMAIN)
        or manifest["captured_update_executed"] is not False
        or manifest["post_update_outcomes_opened"] is not False
        or manifest["parameter_hash_unchanged"] is not True
        or manifest["parameter_sha256_before"] != manifest["parameter_sha256_after"]
        or type(manifest["learner_version"]) is not int
        or not 0 <= manifest["learner_version"] < 64
        or manifest["learner_version"] != manifest["train_steps_before_capture"]
    ):
        raise ValueError("fresh capsule boundary differs")
    validate_hash(manifest["parameter_sha256_before"])
    return gate, manifest


def arm_groups(manifest: dict, arm: str) -> list[dict]:
    """Use recorded membership in serialized frontier order, as in qualification."""
    if arm not in ARMS:
        raise ValueError("unknown replay arm")
    groups = manifest["groups"]
    if len(groups) != 12 or len({r["group_id"] for r in groups}) != 12:
        raise ValueError("exact unique frontier and held-out groups required")
    base, shield = manifest["base_group_ids"], manifest["shield_group_ids"]
    if (
        len(base) != 4
        or len(set(base)) != 4
        or len(shield) != 4
        or len(set(shield)) != 4
        or set(base) == set(shield)
        or len(set(base) & set(shield)) < 2
        or not (set(base) | set(shield)) <= {r["group_id"] for r in groups[:8]}
    ):
        raise ValueError("frozen action membership differs")
    samples = [s for row in groups for s in row["sample_ids"]]
    if len(samples) != 96 or len(set(samples)) != 96:
        raise ValueError("group sample counts or uniqueness differ")
    for i, row in enumerate(groups):
        expected_roles = []
        if row["group_id"] in base:
            expected_roles.append("base")
        if row["group_id"] in shield:
            expected_roles.append("shield")
        if i >= 8:
            expected_roles.append("heldout")
        if (
            row["index"] != i
            or row["raw_file"] != f"groups/group-{i:02d}.pt"
            or row["roles"] != expected_roles
            or len(row["sample_ids"]) != 8
        ):
            raise ValueError("group ordering, roles or canonical path differs")
        validate_hash(row["raw_file_sha256"])
        validate_hash(row["tensor_content_sha256"])
    selected_ids = set(base if arm == ARMS[0] else shield)
    return [row for row in groups[:8] if row["group_id"] in selected_ids]


def members(manifest: dict, arm: str) -> list[dict]:
    """Permit only shared checkpoint, fixed selected action and held-out members."""
    selected = arm_groups(manifest, arm)
    if manifest["checkpoint_root"] != "checkpoint" or not manifest["checkpoint_files"]:
        raise ValueError("checkpoint boundary differs")
    result = []
    for row in manifest["checkpoint_files"]:
        name = str(safe_member(row["path"]))
        if type(row["bytes"]) is not int or row["bytes"] <= 0:
            raise ValueError("checkpoint byte count invalid")
        validate_hash(row["sha256"])
        result.append(
            {
                "path": "checkpoint/" + name,
                "bytes": row["bytes"],
                "sha256": row["sha256"],
            }
        )
    for row in selected + manifest["groups"][8:]:
        result.append(
            {
                "path": str(safe_member(row["raw_file"])),
                "bytes": None,
                "sha256": row["raw_file_sha256"],
            }
        )
    if len({r["path"] for r in result}) != len(result):
        raise ValueError("duplicate capsule path")
    paths = {r["path"] for r in result}
    if any(
        any(str(parent) in paths for parent in safe_member(name).parents)
        for name in paths
    ):
        raise ValueError("file/parent path collision")
    return result


def validate_transition(
    before: dict, before_update: dict, after: dict, after_eval: dict, manifest: dict
) -> None:
    """Enforce full restoration, exactly one real update and both evaluation invariants."""
    for state in (before, before_update, after, after_eval):
        validate_state(state)
    version = manifest["learner_version"]
    if (
        before != before_update
        or after != after_eval
        or before["parameter_sha256"] != manifest["parameter_sha256_before"]
        or before["optimizer_step"] != version
        or before["scheduler_num_steps"] != 32 * version
        or before["load_optim"] is not True
        or before["load_rng"] is not True
        or before["finetune"] is not False
        or after["optimizer_step"] != version + 1
        or after["scheduler_num_steps"] != 32 * (version + 1)
        or after["parameter_sha256"] == before["parameter_sha256"]
    ):
        raise ValueError("restore, one-update or evaluation-state gate failed")
