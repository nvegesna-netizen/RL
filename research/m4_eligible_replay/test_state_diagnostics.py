"""Outcome-free, durable state diagnostics and unchanged fail-closed ordering."""

import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest

from research.m4_eligible_replay.state_diagnostics import (
    RNG_FIELDS,
    changed_fields,
    validate_state,
    write_diagnostic,
)


def fixture() -> dict:
    """Return fingerprints/counters only, without runtime dependencies."""
    return {
        "parameter_sha256": "a" * 64,
        "optimizer_sha256": "b" * 64,
        "scheduler_sha256": "c" * 64,
        "rng_sha256": "d" * 64,
        "optimizer_step": 7,
        "scheduler_num_steps": 224,
        "load_optim": True,
        "load_rng": True,
        "finetune": False,
        "rng_component_sha256": {name: "e" * 64 for name in RNG_FIELDS},
    }


class StateDiagnosticTests(unittest.TestCase):
    def test_unchanged_snapshot(self) -> None:
        self.assertEqual(changed_fields(fixture(), fixture()), [])

    def test_component_changes_are_named(self) -> None:
        before, after = fixture(), fixture()
        after["rng_sha256"] = "f" * 64
        after["rng_component_sha256"]["python"] = "f" * 64
        self.assertEqual(
            changed_fields(before, after), ["rng_component_sha256.python", "rng_sha256"]
        )

    def test_parameters_optimizer_and_counters_are_not_ignored(self) -> None:
        for name, value in (
            ("parameter_sha256", "f" * 64),
            ("optimizer_sha256", "f" * 64),
            ("optimizer_step", 8),
            ("scheduler_num_steps", 256),
        ):
            after = fixture()
            after[name] = value
            self.assertIn(name, changed_fields(fixture(), after))

    def test_raw_outcomes_rejected(self) -> None:
        for key in ("loss", "accuracy", "reward", "raw_rng", "tensor"):
            state = fixture()
            state[key] = 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_state(state)

    def test_invalid_hash_and_counter_rejected(self) -> None:
        for field, value in (
            ("parameter_sha256", "raw-data"),
            ("optimizer_step", True),
            ("load_rng", 1),
        ):
            state = fixture()
            state[field] = value
            with self.assertRaises(ValueError):
                validate_state(state)

    def test_all_rng_components_required(self) -> None:
        state = fixture()
        state["rng_component_sha256"].pop("tracker")
        with self.assertRaises(ValueError):
            validate_state(state)

    def test_record_persists_before_guard_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            before, idle, after = fixture(), fixture(), fixture()
            after["rng_sha256"] = "f" * 64
            states = {"before": before, "idle": idle, "after_eval": after}
            path = write_diagnostic(
                Path(directory),
                replicate="a",
                stage="comparison",
                states=states,
                protocol_hash="1" * 64,
                manifest_hash="2" * 64,
            )
            with self.assertRaisesRegex(ValueError, "mutated"):
                if before != after:
                    raise ValueError("held-out evaluation mutated training state")
            record = json.loads(path.read_bytes())
            self.assertEqual(record["idle_changed_fields"], [])
            self.assertEqual(record["evaluation_changed_fields"], ["rng_sha256"])
            self.assertFalse(record["comparator_update_started"])
            self.assertNotIn("loss", record)
            self.assertEqual(
                states, {"before": fixture(), "idle": fixture(), "after_eval": after}
            )

    def test_no_overwrite_or_identity_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            kwargs = dict(
                directory=Path(directory),
                replicate="b",
                stage="before",
                states={"before": fixture()},
                protocol_hash="1" * 64,
                manifest_hash="2" * 64,
            )
            write_diagnostic(**kwargs)
            with self.assertRaises(FileExistsError):
                write_diagnostic(**kwargs)
            kwargs["replicate"] = "../escape"
            with self.assertRaises(ValueError):
                write_diagnostic(**kwargs)

    def test_idle_drift_distinguished_from_eval_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            before, idle = fixture(), fixture()
            idle["rng_component_sha256"]["python"] = "f" * 64
            after = copy.deepcopy(idle)
            path = write_diagnostic(
                Path(directory),
                replicate="a",
                stage="comparison",
                states={"before": before, "idle": idle, "after_eval": after},
                protocol_hash="1" * 64,
                manifest_hash="2" * 64,
            )
            record = json.loads(path.read_bytes())
            self.assertTrue(record["idle_changed_fields"])
            self.assertEqual(record["post_idle_evaluation_changed_fields"], [])

    def test_actual_runner_writes_before_original_gate_and_update(self) -> None:
        tree = ast.parse(Path(__file__).with_name("run_qualification.py").read_text())
        function = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_run"
        )
        gate = next(
            n
            for n in ast.walk(function)
            if isinstance(n, ast.If)
            and ast.unparse(n.test) == "before != before_update"
        )
        calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)]
        comparison = next(
            n
            for n in calls
            if isinstance(n.func, ast.Name)
            and n.func.id == "record_state"
            and n.args
            and isinstance(n.args[0], ast.Constant)
            and n.args[0].value == "comparison"
        )
        update = next(
            n
            for n in calls
            if isinstance(n.func, ast.Attribute)
            and n.func.attr == "train_from_meta"
            and any(
                k.arg == "eval_mode"
                and isinstance(k.value, ast.Constant)
                and k.value.value is False
                for k in n.keywords
            )
        )
        self.assertLess(comparison.lineno, gate.lineno)
        self.assertLess(gate.lineno, update.lineno)
        self.assertIsInstance(gate.body[0], ast.Raise)


if __name__ == "__main__":
    unittest.main()
