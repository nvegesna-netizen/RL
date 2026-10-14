"""CPU-only negative controls for the prospective acquisition design."""

import copy
import json
import unittest

from validate_eligible_live_acquisition_v2_design import REPORT_DIR, validate_design


class DesignGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = json.loads(
            (REPORT_DIR / "eligible_live_acquisition_v2_plan.json").read_bytes()
        )

    def test_frozen_design_passes_but_does_not_authorize_execution(self) -> None:
        result = validate_design(self.plan)
        self.assertEqual(result["status"], "PASS_FROZEN_DESIGN_ONLY")
        self.assertFalse(result["launch_ready"])
        self.assertEqual(result["learner_advances"], 384)

    def test_changed_boundaries_fail_closed(self) -> None:
        mutations = [
            ("capture_phase", "minimum_authenticated_capsules_to_open_replay", 13),
            ("capture_phase", "suppress_capture_for_liveness_qualification", True),
            ("capture_phase", "pool_v1_or_qualification_capsules", True),
            ("capture_phase", "all_20_terminal_and_authenticated_before_gate", False),
            ("execution", "client_queue_deadline", 14400),
            ("execution", "workload_time_limit_seconds", None),
            ("execution", "automatic_retry", True),
            ("artifact_transport", "large_archive_download_in_logs_after", True),
            (
                "artifact_transport",
                "mutable_jet_wrapper_log_in_workload_checksum_inventory",
                True,
            ),
            (
                "replay_phase",
                "one_update_replay_qualification_required_before_acquisition_launch",
                False,
            ),
            (
                "replay_phase",
                "optimizer_scheduler_and_rng_restore_evidence_required",
                False,
            ),
            (
                "replay_phase",
                "outcome_access_before_all_valid_replays_authenticate",
                True,
            ),
            ("primary_analysis", "favorable_direction", "positive"),
            ("primary_analysis", "terminal_accuracy_endpoint", True),
            ("launch_readiness", "v2_acquisition_started", True),
            ("source", "liveness_stall_root_cause_proven", True),
            ("stopping_rules", "outcome_guided_tuning", "allowed"),
        ]
        for section, key, value in mutations:
            with self.subTest(section=section, key=key):
                mutated = copy.deepcopy(self.plan)
                mutated[section][key] = value
                with self.assertRaises(ValueError):
                    validate_design(mutated)

    def test_historical_or_duplicate_seed_rejected(self) -> None:
        for seed in (20261102, 20261202):
            with self.subTest(seed=seed):
                mutated = copy.deepcopy(self.plan)
                mutated["trajectories"][0]["seed"] = seed
                with self.assertRaisesRegex(ValueError, "cohort changed"):
                    validate_design(mutated)

    def test_predecessor_substitution_rejected(self) -> None:
        mutated = copy.deepcopy(self.plan)
        mutated["predecessors"]["liveness_recovery"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "binding changed"):
            validate_design(mutated)


if __name__ == "__main__":
    unittest.main()
