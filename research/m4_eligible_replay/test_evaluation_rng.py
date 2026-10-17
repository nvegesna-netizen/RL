"""Exercise actual worker dispatch without importing the optional CUDA backend."""

import ast
from collections.abc import Iterable
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock

from research.m4_eligible_replay.evaluation_rng import isolate_evaluation_tracker


class TensorState:
    """Mutable stand-in; real tensor checks also run in EOS preflight."""

    dtype = "uint8"
    ndim = 1

    def __init__(self, value: int):
        self.value = value

    def clone(self) -> "TensorState":
        return TensorState(self.value)


class TestTracker:
    def __init__(self):
        self.states = {"model-parallel-rng": TensorState(7)}
        self.initialized = True

    def is_initialized(self) -> bool:
        return self.initialized

    def get_states(self) -> dict:
        return dict(self.states)

    def set_states(self, states: dict) -> None:
        self.states = states


def digest(states: dict) -> str:
    return repr(sorted((name, value.value) for name, value in states.items()))


def clone_state(value: Any) -> TensorState:
    if not isinstance(value, TensorState):
        raise TypeError("unsupported RNG state")
    return value.clone()


class EvaluationRngTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = TestTracker()
        self.call = Mock(return_value={"metrics": "unchanged"})
        tracker, call = self.tracker, self.call

        class Base:
            def __init__(self, config: Any, *args: Any, **kwargs: Any) -> None:
                self.cfg = config

            def train(self, data: Any, **kwargs: Any) -> dict:
                tracker.states["model-parallel-rng"].value += 1
                return call(data, **kwargs)

        source = Path(__file__).with_name("worker.py").read_text()
        worker = next(
            node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)
        )
        worker.decorator_list = []
        worker.bases = [ast.Name(id="Base", ctx=ast.Load())]
        namespace = {
            "Base": Base,
            "Any": Any,
            "Iterable": Iterable,
            "json": json,
            "torch": SimpleNamespace(Tensor=TensorState, uint8="uint8"),
            "tensor_parallel": SimpleNamespace(get_cuda_rng_tracker=lambda: tracker),
            "isolate_evaluation_tracker": isolate_evaluation_tracker,
            "state_digest": digest,
        }
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[worker], type_ignores=[])),
                "worker-under-test",
                "exec",
            ),
            namespace,
        )
        self.worker = namespace["EligibleLiveReplayWorker"]({})
        self.worker.cfg = {}

    def evaluate(self, **kwargs: Any) -> dict:
        with redirect_stdout(io.StringIO()):
            return self.worker.train("batch", "loss", eval_mode=True, **kwargs)

    def test_actual_override_restores_in_place_mutation(self) -> None:
        original = self.tracker.states["model-parallel-rng"]
        self.evaluate()
        self.assertEqual(
            digest(self.tracker.get_states()), "[('model-parallel-rng', 7)]"
        )
        self.assertIsNot(self.tracker.states["model-parallel-rng"], original)

    def test_actual_override_forwards_arguments_and_return(self) -> None:
        result = self.evaluate(gbs=32, mbs=1, check_dim_skip_keys=["x"])
        self.assertIs(result, self.call.return_value)
        self.call.assert_called_once_with(
            "batch",
            loss_fn="loss",
            eval_mode=True,
            gbs=32,
            mbs=1,
            check_dim_skip_keys=["x"],
        )

    def test_actual_override_restores_on_exception(self) -> None:
        failure = RuntimeError("evaluation failed")
        self.call.side_effect = failure
        with self.assertRaises(RuntimeError) as caught:
            self.evaluate()
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.tracker.states["model-parallel-rng"].value, 7)

    def test_training_advances_and_bypasses_isolation(self) -> None:
        self.tracker.is_initialized = Mock(
            side_effect=AssertionError("must not inspect")
        )
        result = self.worker.train("batch", "loss", gbs=32, mbs=1)
        self.assertIs(result, self.call.return_value)
        self.assertEqual(self.tracker.states["model-parallel-rng"].value, 8)
        self.call.assert_called_once_with(
            "batch",
            loss_fn="loss",
            eval_mode=False,
            gbs=32,
            mbs=1,
            check_dim_skip_keys=None,
        )

    def test_repeated_evaluation_sees_same_tracker_state(self) -> None:
        seen = []
        self.call.side_effect = lambda *a, **kw: seen.append(
            digest(self.tracker.get_states())
        )
        self.evaluate()
        self.evaluate()
        self.assertEqual(seen[0], seen[1])

    def test_uninitialized_tracker_rejected_before_forward(self) -> None:
        self.tracker.initialized = False
        with self.assertRaisesRegex(ValueError, "initialized"):
            self.evaluate()
        self.call.assert_not_called()

    def test_unsupported_generator_state_rejected(self) -> None:
        self.tracker.states = {"graph": object()}
        with self.assertRaisesRegex(TypeError, "byte-tensor"):
            self.evaluate()
        self.call.assert_not_called()

    def test_wrong_tensor_dtype_rejected(self) -> None:
        self.tracker.states["model-parallel-rng"].dtype = "float32"
        with self.assertRaisesRegex(TypeError, "byte-tensor"):
            self.evaluate()
        self.call.assert_not_called()

    def test_empty_tracker_rejected(self) -> None:
        self.tracker.states = {}
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.evaluate()

    def test_changed_stream_names_restore_then_fail(self) -> None:
        self.call.side_effect = lambda *a, **kw: self.tracker.states.update(
            extra=TensorState(9)
        )
        with self.assertRaisesRegex(ValueError, "stream names"):
            self.evaluate()
        self.assertEqual(set(self.tracker.states), {"model-parallel-rng"})

    def test_failed_restore_is_fatal(self) -> None:
        original = self.tracker.set_states
        self.tracker.set_states = Mock(
            side_effect=[original(self.tracker.states), OSError("restore failed")]
        )
        with self.assertRaisesRegex(OSError, "restore failed"):
            self.evaluate()

    def test_failed_restore_preserves_primary_error(self) -> None:
        primary = RuntimeError("primary evaluation failure")
        self.call.side_effect = primary
        self.tracker.set_states = Mock(side_effect=[None, OSError("restore failed")])
        with self.assertRaises(RuntimeError) as caught:
            self.evaluate()
        self.assertIs(caught.exception, primary)
        self.assertIn("restore failed", primary.__notes__[0])

    def test_round_trip_mismatch_rejected_before_forward(self) -> None:
        self.tracker.set_states = lambda states: self.tracker.states.update(
            extra=TensorState(0)
        )
        with self.assertRaisesRegex(ValueError, "round trip"):
            self.evaluate()
        self.call.assert_not_called()

    def test_actual_tq_shim_dispatches_to_override(self) -> None:
        # Compile the packaged base shim, not a hand-written approximation.
        root = Path(__file__).resolve().parents[2]
        source = (root / "nemo_rl/data_plane/worker_mixin.py").read_text()
        shim = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef) and node.name == "train_presharded"
        )
        shim.returns = None
        shim.decorator_list = []
        for argument in shim.args.args + shim.args.kwonlyargs:
            argument.annotation = None
        namespace = {}
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[shim], type_ignores=[])),
                "pinned-tq-shim",
                "exec",
            ),
            namespace,
        )
        self.worker._fetch = lambda meta: "batch"
        self.worker._attach_or_repack_pack_metadata = lambda data, meta: data
        with redirect_stdout(io.StringIO()):
            result = namespace["train_presharded"](
                self.worker, "meta", "loss", eval_mode=True, gbs=32, mbs=1
            )
        self.assertIs(result, self.call.return_value)
        self.assertEqual(self.tracker.states["model-parallel-rng"].value, 7)


if __name__ == "__main__":
    unittest.main()
