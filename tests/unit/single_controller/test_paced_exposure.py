"""Isolated protocol tests; no model, dataset, or study seeds are consumed."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "nemo_rl/algorithms/async_utils/paced_exposure.py"
)
SPEC = importlib.util.spec_from_file_location("paced_exposure_under_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def plan_record() -> dict:
    """Synthetic identifiers and seeds are fixtures, not a proposed execution."""
    inventory = [f"{index:064x}" for index in range(64)]
    return {
        "schema_version": 1,
        "analysis_status": "exploratory_paced_zero_update_exposure",
        "confirmatory_eligible": False,
        "population_claim_authorized": False,
        "training_authorized": False,
        "source_design_id": "dapo_math_paced_exposure_v1",
        "execution_spec_sha256": "9" * 64,
        "analysis_code_commit": "a" * 40,
        "expected_base_commit": "ae07eafe8035b5b2e84efa7234e70e7fd7e493c1",
        "expected_image_sha256": "3df8114a0b3e60ef95ce13f8b982c7cc45d164aca71c63a87388b1e8434ee470",
        "model_revision": "b101308fe89651ea5ce025f25317fea6fc07e96e",
        "model_weights_sha256": "70a914a3466bf064ca88ae875cb33c366856147e475a2b8da54e7a3d94d8074a",
        "dataset_revision": "65877096c24ffa7abc4e4fa5edb95cf3413a5674",
        "pools": [
            {
                "order_seed": 100,
                "pool_id": "b" * 64,
                "manifest_sha256": "c" * 64,
                "private_calibration_manifest_sha256": "d" * 64,
                "prior_identity_exclusions_sha256": "e" * 64,
                "prior_seed_ledger_sha256": "f" * 64,
                "source_prompt_ids": inventory,
                "fixed_lower_load_prompt_ids": inventory[:32],
                "fixed_harder_prompt_ids": inventory[32:],
            }
        ],
        "calibration_draws": [
            {
                "generation_seed": 101,
                "trace_sha256": "1" * 64,
                "completion_16_seconds": 10,
                "completion_48_seconds": 90,
            },
            {
                "generation_seed": 102,
                "trace_sha256": "2" * 64,
                "completion_16_seconds": 20,
                "completion_48_seconds": 180,
            },
        ],
        "cadence_seconds": 15,
        "arms": [
            {
                "arm_id": name,
                "sampler": sampler,
                "cadence_multiplier": multiplier,
                "consumer_duration_seconds": 15 * multiplier,
                "generation_seed": seed,
                "sampler_lookahead_versions": 3,
                "max_buffered_rollouts": 16,
            }
            for name, sampler, multiplier, seed in (
                ("in_order_half", "in_order", 0.5, 103),
                ("ready_first_half", "ready_first", 0.5, 103),
                ("ready_first_one", "ready_first", 1, 104),
                ("in_order_one", "in_order", 1, 104),
                ("in_order_two", "in_order", 2, 105),
                ("ready_first_two", "ready_first", 2, 105),
            )
        ],
        "prompt_groups": 64,
        "completions_per_group": 16,
        "selection_groups_per_step": 4,
        "selection_steps": 16,
        "max_inflight_prompts": 4,
        "max_total_sequence_length": 6144,
        "data_max_input_seq_length": 2048,
        "hf_config_override_max_position_embeddings": 6144,
        "max_new_tokens": 4096,
        "temperature": 1.0,
        "top_p": 0.7,
        "maximum_completions": 8192,
        "scientific_allocation_seconds": 7200,
        "allocated_gpus": 2,
        "plan_id": "0" * 64,
    }


class PacedExposureTests(unittest.TestCase):
    def test_pre_generation_spec_has_no_calibration_outcomes(self) -> None:
        record = {
            "schema_version": 1,
            "analysis_status": "exploratory_paced_exposure_execution_spec",
            "source_design_id": "dapo_math_paced_exposure_v1",
            "selection_seed": 100,
            "source_shuffle_seed": 101,
            "calibration_generation_seeds": [102, 103],
            "cadence_pair_generation_seeds": [104, 105, 106],
            "prior_identity_ledger_sha256": "a" * 64,
            "prior_seed_ledger_sha256": "b" * 64,
            "implementation_commit": "c" * 40,
            "runtime_validation_result_sha256": "d" * 64,
            "prompt_groups": 64,
            "completions_per_group": 16,
            "maximum_completions": 8192,
            "scientific_allocation_seconds": 7200,
            "allocated_gpus": 2,
            "training_authorized": False,
        }
        MODULE.PacedExposureExecutionSpec.model_validate(record)
        for key, value in (
            ("source_shuffle_seed", 100),
            ("cadence_seconds", 10),
            ("training_authorized", True),
            ("selection_seed", True),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                MODULE.PacedExposureExecutionSpec.model_validate({**record, key: value})

    def test_round_trip_and_tamper_detection(self) -> None:
        record = plan_record()
        plan = MODULE.PacedExposurePlan.model_validate(record)
        record["plan_id"] = MODULE.compute_paced_exposure_plan_id(plan)
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "plan.json"
            filename.write_text(json.dumps(record))
            loaded = MODULE.load_paced_exposure_plan(filename)
            self.assertEqual(loaded.arm("in_order_half").consumer_duration_seconds, 7.5)
            record["analysis_code_commit"] = "b" * 40
            filename.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "ID mismatch"):
                MODULE.load_paced_exposure_plan(filename)

    def test_live_plan_cannot_change_frozen_execution_seeds(self) -> None:
        plan = MODULE.PacedExposurePlan.model_validate(plan_record())
        spec = MODULE.PacedExposureExecutionSpec.model_validate(
            {
                "schema_version": 1,
                "analysis_status": "exploratory_paced_exposure_execution_spec",
                "source_design_id": "dapo_math_paced_exposure_v1",
                "selection_seed": 100,
                "source_shuffle_seed": 106,
                "calibration_generation_seeds": [101, 102],
                "cadence_pair_generation_seeds": [103, 104, 105],
                "prior_identity_ledger_sha256": "e" * 64,
                "prior_seed_ledger_sha256": "f" * 64,
                "implementation_commit": "a" * 40,
                "runtime_validation_result_sha256": "d" * 64,
                "prompt_groups": 64,
                "completions_per_group": 16,
                "maximum_completions": 8192,
                "scientific_allocation_seconds": 7200,
                "allocated_gpus": 2,
                "training_authorized": False,
            }
        )
        MODULE.validate_paced_plan_execution_spec(plan, spec)
        altered = spec.model_dump()
        altered["cadence_pair_generation_seeds"] = (107, 108, 109)
        with self.assertRaisesRegex(ValueError, "changed"):
            MODULE.validate_paced_plan_execution_spec(
                plan, MODULE.PacedExposureExecutionSpec.model_validate(altered)
            )

    def test_geometry_budget_and_claims_are_frozen(self) -> None:
        for key, value in (
            ("prompt_groups", 32),
            ("training_authorized", True),
            ("scientific_allocation_seconds", 7201),
            ("maximum_completions", 8193),
            ("selection_steps", 8),
        ):
            record = plan_record()
            record[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                MODULE.PacedExposurePlan.model_validate(record)

    def test_calibration_and_arm_cadences_bound(self) -> None:
        for value in (0, -1, float("nan"), float("inf"), 16):
            record = plan_record()
            record["cadence_seconds"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                MODULE.PacedExposurePlan.model_validate(record)
        record = plan_record()
        record["arms"][0]["consumer_duration_seconds"] = 8
        with self.assertRaises(ValueError):
            MODULE.PacedExposurePlan.model_validate(record)

    def test_order_and_seed_pairing_bound(self) -> None:
        for change in ("order", "seed", "calibration_seed"):
            record = plan_record()
            if change == "order":
                record["arms"].reverse()
            elif change == "seed":
                record["arms"][0]["generation_seed"] = 106
            else:
                record["calibration_draws"][0]["generation_seed"] = 103
            with self.subTest(change=change), self.assertRaises(ValueError):
                MODULE.PacedExposurePlan.model_validate(record)

    def test_halves_and_inventory_bound(self) -> None:
        for key in (
            "source_prompt_ids",
            "fixed_lower_load_prompt_ids",
            "fixed_harder_prompt_ids",
        ):
            record = plan_record()
            record["pools"][0][key][1] = record["pools"][0][key][0]
            with self.subTest(key=key), self.assertRaises(ValueError):
                MODULE.PacedExposurePlan.model_validate(record)


if __name__ == "__main__":
    unittest.main()
