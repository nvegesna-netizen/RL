#!/usr/bin/env python3
"""Validate the fresh causal-utility design without submitting or opening outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPORT_DIR = Path(__file__).resolve().parent
REPO = REPORT_DIR.parents[2]
PINNED_PREDECESSORS = {
    "causal_protocol": (
        "eligible_live_decision_causal_protocol.json",
        "716ba0eb8f980ca5af0ef3d57e2415befc1fb1a315afeb01b9bc326a60341b48",
    ),
    "v1_terminal": (
        "eligible_live_acquisition_v1_terminal_authentication.json",
        "cebc287b0edd71c313bf1b70a21e70ff9f8c7f844880c4ab0cf459d19f02b456",
    ),
    "restore_gate": (
        "eligible_live_restore_qualification_terminal_result.json",
        "dbe064e4c294694daccc9be6c86622bf033d81ac915724b4034acff5385103aa",
    ),
    "liveness_recovery": (
        "eligible_live_v2_liveness_qualification_offline_recovery.json",
        "4cbf20424f34a804df98a8ebb8798d2fdb3693539e0813a2a7e4b31b894c8f2f",
    ),
    "qualified_package": (
        "eligible_live_v2_liveness_qualification_package_receipt.json",
        "7b564f9496898c46883b174c27fd2edd3c828aaf83d379345d987e7697439b75",
    ),
}


def require(condition: bool, message: str) -> None:
    """Reject any changed design boundary explicitly."""
    if not condition:
        raise ValueError(message)


def file_hash(path: Path) -> str:
    """Return a small design or source file's SHA-256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_design(plan: dict[str, Any]) -> dict[str, Any]:
    """Check frozen assignments, predecessor evidence, and launch prerequisites.

    This validates design preparation, not the execution package or raw artifacts.
    The raw-artifact recovery must be replayed again before locking execution.
    """
    require(
        plan["schema"] == "m4-shield-eligible-live-acquisition-plan-v2",
        "wrong design schema",
    )
    require(
        plan["status"] == "FROZEN_BEFORE_V2_PACKAGING_AND_EXECUTION",
        "wrong design stage",
    )
    require(
        plan["assignment_domain"] == "m4-eligible-live-causal-acquisition-v2",
        "assignment domain changed",
    )
    expected_rows = [
        {"identity": f"v2a{index:02d}", "seed": 20261200 + index}
        for index in range(1, 21)
    ]
    require(plan["trajectories"] == expected_rows, "fresh fixed seed cohort changed")
    require(
        plan["independence_unit"] == "preselected_training_seed_trajectory",
        "independence unit changed",
    )
    require(
        plan["estimand"]
        == "mean paired one-update contrast conditional on an authenticated first eligible decision within the fixed 64-update window",
        "estimand changed",
    )
    predecessors = {}
    for key, (name, expected_hash) in PINNED_PREDECESSORS.items():
        require(
            plan["predecessors"][key] == {"path": name, "sha256": expected_hash},
            f"{key} binding changed",
        )
        path = REPORT_DIR / name
        require(file_hash(path) == expected_hash, f"{key} evidence changed")
        predecessors[key] = json.loads(path.read_bytes())

    capture = plan["capture_phase"]
    require(
        capture["selection_rule"]
        == "first eligible decision under the unchanged causal protocol",
        "eligibility selection changed",
    )
    required_capture = {
        "trajectory_count": 20,
        "maximum_completed_updates_per_trajectory": 64,
        "maximum_capsules_per_trajectory": 1,
        "parallel_execution_permitted": True,
        "suppress_capture_for_liveness_qualification": False,
        "captured_update_executed": False,
        "post_update_outcomes_opened": False,
        "all_20_terminal_and_authenticated_before_gate": True,
        "minimum_authenticated_capsules_to_open_replay": 16,
        "replacement_seeds": 0,
        "pool_v1_or_qualification_capsules": False,
    }
    for key, expected in required_capture.items():
        require(capture[key] == expected, f"capture boundary changed: {key}")
    require(
        plan["execution"]
        == {
            "launcher": "runllm.py --no_wait",
            "submission_attempt_limit_per_identity": 1,
            "workload_time_limit_seconds": 14400,
            "client_queue_deadline": None,
            "progress_watchdog_seconds": 1200,
            "automatic_retry": False,
            "automatic_replacement": False,
            "automatic_extension": False,
        },
        "execution boundary changed",
    )
    transport = plan["artifact_transport"]
    require(
        transport["checkpoint_source"]
        == "successful EOS workload artifact, accessed directly",
        "checkpoint transport source changed",
    )
    for key, expected in {
        "large_archive_download_in_logs_after": False,
        "compact_terminal_record_first": True,
        "recursive_capsule_sha256_inventory_required": True,
        "mutable_jet_wrapper_log_in_workload_checksum_inventory": False,
        "transport_clean_room_test_required_before_launch": True,
    }.items():
        require(transport[key] == expected, f"transport boundary changed: {key}")

    replay = plan["replay_phase"]
    for key, expected in {
        "locked_until_capture_completion_gate": True,
        "arms": ["reward_variance_base", "m4_shield"],
        "updates_per_arm": 1,
        "identical_pre_update_state_required": True,
        "optimizer_scheduler_and_rng_restore_evidence_required": True,
        "one_update_replay_qualification_required_before_acquisition_launch": True,
        "qualification_uses_only_excluded_existing_capsule": True,
        "regeneration_forbidden": True,
        "action_reselection_forbidden": True,
        "outcome_access_before_all_valid_replays_authenticate": False,
    }.items():
        require(replay[key] == expected, f"replay boundary changed: {key}")
    require(
        plan["primary_analysis"]
        == {
            "endpoint": "shield_post_minus_pre_heldout_clipped_pg_loss_minus_base_post_minus_pre",
            "favorable_direction": "negative",
            "unit": "one paired contrast per valid trajectory",
            "summary": "paired mean with 95 percent studentized bootstrap interval",
            "robustness": ["paired t interval", "exact sign flip test"],
            "report_every_contrast": True,
            "categorical_decision_labels": False,
            "terminal_accuracy_endpoint": False,
        },
        "primary analysis changed",
    )
    require(
        plan["launch_readiness"]
        == {
            "design_only": True,
            "execution_lock_and_package_receipt_required": True,
            "fresh_unconsumed_submission_guards_required": True,
            "source_rotated_setup_jet_before_submission": True,
            "v2_acquisition_started": False,
            "paired_replay_started": False,
        },
        "launch readiness changed",
    )
    require(
        plan["stopping_rules"]
        == {
            "no_eligible_decision_by_update_64": "valid no-capsule record; do not replace",
            "runtime_or_integrity_failure": "identity fails closed; preserve evidence; no automatic retry",
            "unauthenticated_or_unclassified_identity": "completion gate remains closed",
            "fewer_than_16_authenticated_capsules": "stop without paired replay",
            "paired_replay_missing_or_invalid_arm": "do not open any primary outcomes",
            "outcome_guided_tuning": "forbidden",
        },
        "stopping rules changed",
    )

    recovery = predecessors["liveness_recovery"]
    require(
        recovery["status"]
        == "PASS_ALL_SIX_ORIGINAL_LIVENESS_REQUIREMENTS_OFFLINE_RECOVERED",
        "liveness gate not recovered",
    )
    require(
        recovery["repair"]["all_six_original_failures_reproduced"]
        and recovery["repair"]["all_six_repaired_classifiers_pass"],
        "classifier reproduction incomplete",
    )
    require(
        not recovery["repair"]["training_or_scheduler_code_changed"]
        and not recovery["repair"]["scientific_or_engineering_thresholds_changed"],
        "recovery changed requirements",
    )
    chains = recovery["chains"]
    expected_qualification = [
        ("lq01", 20261102),
        ("lq02", 20261103),
        ("lq03", 20261109),
        ("lq04", 20261110),
        ("lq05", 20261111),
        ("lq06", 20261119),
    ]
    require(
        [(row["identity"], row["seed"]) for row in chains] == expected_qualification,
        "qualification identities differ",
    )
    for row in chains:
        result = row["recovered_qualification"]
        require(
            result["status"] == "PASS_FULL_64_UPDATE_LIVENESS",
            "qualification identity failed",
        )
        for key in (
            "train_steps",
            "trainer_version",
            "lifecycle_learner_advances",
            "oars_decisions",
            "oars_actions_matching_m4_shield",
        ):
            require(result[key] == 64, f"qualification incomplete: {key}")
        require(result["run_exit_code"] == 0, "qualification training failed")
        for key in (
            "watchdog_timeout_record_present",
            "capsule_written",
            "checkpoint_written",
            "paired_replay_started",
            "post_update_heldout_outcomes_opened",
        ):
            require(result[key] is False, f"qualification exclusion changed: {key}")
        core = [
            record
            for name, record in row["artifact_records"].items()
            if not name.startswith("jet_assets/")
        ]
        require(
            len(core) == 6 and all(record["manifest_hash_matches"] for record in core),
            "core evidence not authenticated",
        )

    require(
        predecessors["restore_gate"]["status"] == "PASS_RESTORE_AND_EVALUATION_GATE",
        "restore gate failed",
    )
    old_seeds = {
        item["seed"]
        for item in json.loads(
            (REPORT_DIR / "eligible_live_acquisition_plan.json").read_bytes()
        )["trajectories"]
    }
    old_seeds.update(row["seed"] for row in chains)
    old_seeds.add(20261031)
    require(
        not old_seeds.intersection(row["seed"] for row in expected_rows),
        "historical seed reused",
    )

    package = predecessors["qualified_package"]
    source = plan["source"]
    require(
        source["qualified_runtime_commit"] == package["source_commit"],
        "qualified commit changed",
    )
    require(
        source["qualified_runtime_archive_sha256"] == package["source_archive_sha256"],
        "qualified archive changed",
    )
    require(
        source["new_execution_archive_requires_separate_receipt"]
        and not source["training_or_scheduler_change_permitted"]
        and not source["liveness_stall_root_cause_proven"],
        "source or liveness claim boundary changed",
    )
    runtime_hashes = {
        "nemo_rl/algorithms/async_utils/opportunity_at_risk_v2.py": "d6221b7f9766d07f9e1a4f1a0886af9c5625f3e4a6eaf0ddc283078c6d982c3b",
        "nemo_rl/algorithms/single_controller.py": "97164bd7649830c9a5499f0f6825e257c3314a34e5b27488955213de3d02047f",
        "nemo_rl/algorithms/single_controller_utils/config.py": "e7f86e102ebb5094e65cd377ac531cf9c7152b6e43362065021721d16327dab4",
        "tests/unit/single_controller/test_opportunity_at_risk_v2.py": "e5de5f257460c04de4d157ab755c2407d6ab93e04f91f9caf34407b7ef541515",
        "tests/unit/single_controller/test_single_controller.py": "ca1879a47ee7d4a8aa2322d59fd137fbbd7fb1ff3fc94012639730b70a7e15ff",
        "uv.lock": "bc69ceddc9f35d2c120235c96afac2e6c7b76827e044efda3e61f908be72c4cc",
        "examples/configs/grpo_math_1B_megatron_single_controller_m4_eligible_live_capture.yaml": "94b6bb744beca304be20371135b9c0251d9e2b3d4b870b4be86d9c1c0be112f2",
        "nemo_rl/algorithms/async_utils/eligible_live_decision_capsule.py": "d647cf0113cc4caefa244e61f26622f1453ee569b60a30105b17ac486c0d8913",
        "reports/auto_research/2026-10-07-m4-causal-utility/analyze_eligible_live_decision_capture.py": "bd6734b03f9be962377373ce2f019e5243e07796ae5297af0fe233b3e16bd6ee",
    }
    for name, expected_hash in runtime_hashes.items():
        require(
            file_hash(REPO / name) == expected_hash,
            f"qualified runtime changed: {name}",
        )

    return {
        "schema": "m4-eligible-live-v2-design-validation-v1",
        "status": "PASS_FROZEN_DESIGN_ONLY",
        "trajectory_count": 20,
        "fresh_seeds": True,
        "qualified_runtime_files_checked": len(runtime_hashes),
        "liveness_passes": 6,
        "learner_advances": 384,
        "predecessor_hashes": {
            key: value[1] for key, value in PINNED_PREDECESSORS.items()
        },
        "launch_ready": False,
        "pending_prelaunch_gates": [
            "excluded-capsule one-update replay qualification",
            "clean-room direct-EOS checkpoint transport",
            "new source package receipt and execution lock",
            "new unconsumed submission guards",
        ],
        "new_eos_launches": 0,
        "paired_outcomes_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=REPORT_DIR / "eligible_live_acquisition_v2_plan.json",
    )
    args = parser.parse_args()
    result = validate_design(json.loads(args.plan.read_bytes()))
    result["plan_sha256"] = file_hash(args.plan)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
