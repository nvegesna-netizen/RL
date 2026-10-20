"""Prospective pretraining retry authority must fail closed on changed scope."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/auto_research/2026-10-07-m4-causal-utility"


class RetryAmendmentTests(unittest.TestCase):
    def test_accepted_preflight_and_original_guards_authenticate(self):
        spec = importlib.util.spec_from_file_location(
            "retry_builder", REPORT / "build_eligible_live_v2_capture_package.py"
        )
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        path = REPORT / "eligible_live_v2_capture_retry_r1_amendment.json"
        self.assertEqual(
            builder.validate_retry_amendment(path)["identity"], "capture-retry-r1"
        )

    def test_changed_authority_and_predecessor_hashes_rejected(self):
        spec = importlib.util.spec_from_file_location(
            "retry_builder", REPORT / "build_eligible_live_v2_capture_package.py"
        )
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        original = json.loads(
            (REPORT / "eligible_live_v2_capture_retry_r1_amendment.json").read_bytes()
        )
        mutations = {
            "maximum_new_launcher_invocations": 21,
            "new_attempts_per_identity": 2,
            "attempt_namespace": "m4-v2-capture-fixed-cohort-guards",
            "changed_scientific_fields": ["seed"],
            "replacement_seeds": 1,
            "automatic_retry": True,
            "automatic_replacement": True,
            "automatic_extension": True,
            "paired_replay_authorized": True,
            "post_update_outcomes_opened": True,
            "allocated_seconds": 14401,
            "client_queue_deadline": 1200,
            "progress_watchdog_seconds": 0,
            "minimum_authenticated_capsules": 15,
            "preserve_original_attempts_and_guards": False,
            "all_twenty_terminal_and_authenticated_before_gate": False,
            "preflight_result_sha256": "0" * 64,
            "original_terminal_result_sha256": "0" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "amendment.json"
            for field, value in mutations.items():
                with self.subTest(field=field):
                    changed = copy.deepcopy(original)
                    changed[field] = value
                    path.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):
                        builder.validate_retry_amendment(path)


if __name__ == "__main__":
    unittest.main()
