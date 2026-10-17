"""Behavioral tests of observation, restoration and first-divergence analysis."""

import ast
from collections import UserDict
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

from research.m4_eligible_replay.analyze_training_trace import (
    compare_traces,
    load_trace,
    RUNTIME_KEYS,
    require_identical,
)
from research.m4_eligible_replay import test_evaluation_rng as rng_tests
from research.m4_eligible_replay.training_trace import (
    ENVIRONMENT_FIELDS,
    MODEL_MODE_FIELDS,
    normalize_payload,
    observer_cleanup,
    observe_training,
    source_identity,
    trace_training,
)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class Handle:
    def __init__(self, hooks: list, hook: Any):
        self.hooks, self.hook = hooks, hook

    def remove(self) -> None:
        self.hooks.remove(self.hook)


class Model:
    def __init__(self):
        self.pre, self.post = [], []
        self.gradient = 0
        self.value = 2
        self.failure = None

    def register_forward_pre_hook(self, hook: Any, *, with_kwargs: bool) -> Handle:
        assert with_kwargs
        self.pre.append(hook)
        return Handle(self.pre, hook)

    def register_forward_hook(self, hook: Any, *, with_kwargs: bool) -> Handle:
        assert with_kwargs
        self.post.append(hook)
        return Handle(self.post, hook)

    def __call__(self, **kwargs: Any) -> int:
        for hook in self.pre:
            assert hook(self, (), kwargs) is None
        if self.failure:
            raise self.failure
        output = kwargs["input_ids"] * self.value
        for hook in self.post:
            assert hook(self, (), kwargs, output) is None
        return output


class Optimizer:
    def __init__(self, model: Model):
        self.model = model
        self.result = object()

    def step(self) -> object:
        self.model.value -= self.model.gradient
        return self.result


def simulate() -> tuple:
    model = Model()
    optimizer = Optimizer(model)

    def get_iterator(batch: list[int], *args: Any, **kwargs: Any) -> tuple:
        return iter(batch), len(batch), 1, 8, 8

    def forward_backward(**kwargs: Any) -> object:
        result = []
        for item in kwargs["data_iterator"]:
            result.append(model(input_ids=item))
            model.gradient += item
        return result

    backend = SimpleNamespace(
        get_microbatch_iterator=get_iterator, megatron_forward_backward=forward_backward
    )
    return model, optimizer, backend


class TrainingTraceTests(unittest.TestCase):
    def exercise(
        self, *, failure: BaseException | None = None, local_step: bool = False
    ) -> tuple:
        model, optimizer, backend = simulate()
        model.failure = failure
        if local_step:
            optimizer.step = optimizer.step
        old_step = optimizer.step
        old_iterator, old_forward = (
            backend.get_microbatch_iterator,
            backend.megatron_forward_backward,
        )
        events = []
        emit = lambda phase, payload: events.append((phase, payload))
        snapshot = lambda: {
            "inner_optimizer_sha256": fingerprint(0),
            "master_parameters_sha256": fingerprint(model.value),
            "parameter_sha256": fingerprint(model.value),
        }
        context = observe_training(
            backend=backend,
            model=model,
            optimizer=optimizer,
            data=[1, 2],
            fingerprint=fingerprint,
            gradients=lambda: fingerprint(model.gradient),
            optimizer_state=snapshot,
            emit=emit,
        )
        try:
            with context:
                iterator, size, mbs, seq, padded = backend.get_microbatch_iterator(
                    [1, 2]
                )
                result = backend.megatron_forward_backward(
                    model=model,
                    data_iterator=iterator,
                    num_microbatches=size,
                    seq_length=padded,
                    mbs=mbs,
                    forward_only=False,
                    global_valid_seqs=2,
                    global_valid_toks=16,
                )
                self.assertEqual(result, [2, 4])
                self.assertIs(optimizer.step(), optimizer.result)
        except BaseException as error:
            if error is not failure:
                raise
        self.assertIs(backend.get_microbatch_iterator, old_iterator)
        self.assertIs(backend.megatron_forward_backward, old_forward)
        self.assertEqual(optimizer.step, old_step)
        self.assertEqual("step" in vars(optimizer), local_step)
        self.assertEqual(model.pre, [])
        self.assertEqual(model.post, [])
        return events, model

    def test_normal_observation_preserves_math_returns_and_order(self) -> None:
        events, model = self.exercise()
        self.assertEqual(model.value, -1)
        self.assertEqual(model.gradient, 3)
        self.assertEqual(
            [phase for phase, _ in events],
            [
                "worker_input",
                "forward_backward_start",
                "microbatch",
                "model_input",
                "model_output",
                "microbatch",
                "model_input",
                "model_output",
                "forward_backward_end",
                "optimizer_before",
                "optimizer_after",
                "complete",
            ],
        )

    def test_original_bindings_and_hooks_restored_on_exception(self) -> None:
        events, _ = self.exercise(failure=RuntimeError("native forward failed"))
        self.assertNotIn("complete", [phase for phase, _ in events])

    def test_cleanup_failure_preserves_training_exception(self) -> None:
        primary = RuntimeError("primary training failure")
        with self.assertRaises(RuntimeError) as caught:
            with observer_cleanup() as cleanup:
                cleanup.callback(
                    lambda: (_ for _ in ()).throw(OSError("cleanup failed"))
                )
                raise primary
        self.assertIs(caught.exception, primary)
        self.assertIn("cleanup failed", primary.__notes__[0])

    def test_cleanup_failure_without_training_error_is_fatal(self) -> None:
        with self.assertRaisesRegex(OSError, "cleanup failed"):
            with observer_cleanup() as cleanup:
                cleanup.callback(
                    lambda: (_ for _ in ()).throw(OSError("cleanup failed"))
                )

    def test_cancellation_also_restores_bindings(self) -> None:
        self.exercise(failure=KeyboardInterrupt())

    def test_existing_instance_step_binding_restored(self) -> None:
        self.exercise(local_step=True)

    def test_incomplete_execution_fails_closed(self) -> None:
        model, optimizer, backend = simulate()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            with observe_training(
                backend=backend,
                model=model,
                optimizer=optimizer,
                data=[],
                fingerprint=fingerprint,
                gradients=lambda: fingerprint(1),
                optimizer_state=lambda: {},
                emit=lambda *args: None,
            ):
                pass
        self.assertNotIn("step", vars(optimizer))
        self.assertEqual(model.pre + model.post, [])

    def test_iterator_observation_is_lazy_and_preserves_objects(self) -> None:
        model, optimizer, backend = simulate()
        values = [{"x": 1}, {"x": 2}]
        emitted = []
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with observe_training(
                backend=backend,
                model=model,
                optimizer=optimizer,
                data=[],
                fingerprint=fingerprint,
                gradients=lambda: fingerprint(1),
                optimizer_state=lambda: {},
                emit=lambda p, v: emitted.append(p),
            ):
                iterator = backend.get_microbatch_iterator(values)[0]
                self.assertEqual(emitted, ["worker_input"])
                self.assertIs(next(iterator), values[0])
                self.assertEqual(emitted, ["worker_input", "microbatch"])
                raise RuntimeError("intentional")

    def test_unsupported_payload_fails_instead_of_using_repr(self) -> None:
        with self.assertRaisesRegex(TypeError, "unsupported"):
            normalize_payload(
                object(),
                tensor_type=bytes,
                process_group_type=memoryview,
                describe_group=lambda g: {},
            )

    def test_dataclass_and_userdict_contents_preserved(self) -> None:
        @dataclass
        class Payload:
            data: Any

        value = Payload(UserDict({"tokens": b"tensor", "order": [3, 1]}))
        normalized = normalize_payload(
            value,
            tensor_type=bytes,
            process_group_type=memoryview,
            describe_group=lambda g: {},
        )
        self.assertIs(normalized["fields"]["data"]["tokens"], value.data["tokens"])
        self.assertEqual(normalized["fields"]["data"]["order"], [3, 1])

    def test_process_group_uses_rank_size_not_process_address(self) -> None:
        group = memoryview(b"x")
        self.assertEqual(
            normalize_payload(
                group,
                tensor_type=bytes,
                process_group_type=memoryview,
                describe_group=lambda g: {"rank": 0, "size": 1},
            ),
            {"process_group": {"rank": 0, "size": 1}},
        )

    def test_missing_loaded_module_recorded_without_import(self) -> None:
        self.assertEqual(
            source_identity("nonexistent.diagnostic.module"),
            {"path": None, "sha256": None},
        )

    def test_pinned_native_iterator_returns_five_values(self) -> None:
        path = Path(__file__).resolve().parents[2] / "nemo_rl/models/megatron/data.py"
        tree = ast.parse(path.read_text())
        factory = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "get_microbatch_iterator"
        )
        returns = [node for node in ast.walk(factory) if isinstance(node, ast.Return)]
        self.assertEqual(len(returns), 1)
        self.assertEqual(len(returns[0].value.elts), 5)

    def test_actual_worker_opt_in_and_passthrough(self) -> None:
        # Reuse the actual-class AST fixture, not a second handwritten override.
        fixture = rng_tests.EvaluationRngTests()
        fixture.setUp()
        namespace = fixture.worker.train.__globals__
        calls = []

        @contextmanager
        def tracer(worker: Any, data: Any, **kwargs: Any):
            calls.append((worker, data, kwargs))
            yield

        namespace["trace_training"] = tracer
        fixture.worker.cfg = {"eligible_replay_training_trace": {"identity": "test"}}
        result = fixture.worker.train("batch", "loss", gbs=32, mbs=1)
        self.assertIs(result, fixture.call.return_value)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], fixture.worker)
        self.assertEqual(fixture.tracker.states["model-parallel-rng"].value, 8)
        fixture.tracker.states["model-parallel-rng"].value = 7
        fixture.evaluate()
        self.assertEqual(
            len(calls), 1
        )  # Held-out evaluation never enters training trace.

    def test_full_adapter_writes_valid_exclusive_trace(self) -> None:
        model, optimizer, pipeline = simulate()
        model.named_parameters = lambda: [
            ("weight", SimpleNamespace(grad=3, main_grad=None))
        ]
        optimizer.optimizer = SimpleNamespace(state_dict=lambda: {"step": 7})
        optimizer.get_parameters = lambda: [model.value]
        backend = ModuleType("nemo_rl.models.policy.workers.megatron_policy_worker")
        backend.get_microbatch_iterator = pipeline.get_microbatch_iterator
        backend.megatron_forward_backward = pipeline.megatron_forward_backward
        worker = SimpleNamespace(
            model=model,
            optimizer=optimizer,
            _get_model_config=lambda: SimpleNamespace(),
            model_parameter_sha256=lambda: fingerprint(model.value),
        )
        torch = ModuleType("torch")
        torch.__version__ = "test-torch"
        torch.Tensor = bytes
        torch.are_deterministic_algorithms_enabled = lambda: False
        torch.is_deterministic_algorithms_warn_only_enabled = lambda: False
        torch.distributed = SimpleNamespace(
            get_world_size=lambda *args: 1,
            get_rank=lambda *args: 0,
            ProcessGroup=memoryview,
        )
        torch.cuda = SimpleNamespace(synchronize=lambda: None)
        torch.backends = SimpleNamespace(
            cudnn=SimpleNamespace(
                deterministic=False, benchmark=False, allow_tf32=False
            ),
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
        )
        megatron_optimizer = ModuleType("megatron.core.optimizer")
        megatron_optimizer.ChainedOptimizer = type("UnusedChainedOptimizer", (), {})
        workers = ModuleType("nemo_rl.models.policy.workers")
        workers.megatron_policy_worker = backend
        modules = {
            "torch": torch,
            "megatron": ModuleType("megatron"),
            "megatron.core": ModuleType("megatron.core"),
            "megatron.core.optimizer": megatron_optimizer,
            "nemo_rl": ModuleType("nemo_rl"),
            "nemo_rl.models": ModuleType("nemo_rl.models"),
            "nemo_rl.models.policy": ModuleType("nemo_rl.models.policy"),
            "nemo_rl.models.policy.workers": workers,
            "nemo_rl.models.policy.workers.megatron_policy_worker": backend,
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("sys.modules", modules),
        ):
            path = Path(directory) / "replay-a-training-trace.jsonl"
            config = {
                "output": str(path),
                "replicate": "a",
                "protocol_sha256": "a" * 64,
                "capsule_manifest_sha256": "b" * 64,
            }
            with trace_training(worker, [1, 2], config=config, digest=fingerprint):
                iterator, size, mbs, _, padded = backend.get_microbatch_iterator([1, 2])
                backend.megatron_forward_backward(
                    data_iterator=iterator,
                    num_microbatches=size,
                    seq_length=padded,
                    mbs=mbs,
                    forward_only=False,
                    global_valid_seqs=2,
                    global_valid_toks=16,
                )
                optimizer.step()
            rows = load_trace(
                path, replicate="a", protocol_hash="a" * 64, capsule_hash="b" * 64
            )
            self.assertEqual(rows[-1]["phase"], "complete")
            self.assertEqual(model.value, -1)
            self.assertNotIn("step", vars(optimizer))
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                with trace_training(worker, [1, 2], config=config, digest=fingerprint):
                    self.fail("existing trace must not be overwritten")
            self.assertEqual(path.read_bytes(), original)


class TraceAnalysisTests(unittest.TestCase):
    def test_strict_trace_gate_rejects_runtime_or_gradient_divergence(self) -> None:
        for runtime_change in (False, True):
            left, right = self.trace(), self.trace("b")
            if runtime_change:
                right[0]["payload"]["deterministic_algorithms"] = True
            else:
                right[-4]["payload"]["gradient_sha256"] = "c" * 64
            with self.assertRaisesRegex(ValueError, "traces differ"):
                require_identical(compare_traces(left, right))

    def test_strict_trace_match_does_not_claim_full_qualification(self) -> None:
        result = compare_traces(self.trace(), self.trace("b"))
        require_identical(result)
        self.assertFalse(result["qualification_passed"])

    def trace(self, replicate: str = "a") -> list[dict]:
        events, _ = TrainingTraceTests().exercise()
        runtime = {key: False for key in RUNTIME_KEYS}
        runtime.update(
            {
                "torch_version": "test-torch",
                "model_modes": {name: "None" for name in MODEL_MODE_FIELDS},
                "environment": {name: None for name in ENVIRONMENT_FIELDS},
                "module_origins": {},
                "callable_origins": {
                    "microbatch_iterator": "/test/iterator.py",
                    "forward_backward": "/test/train.py",
                },
            }
        )
        rows = [("runtime", runtime), *events]
        return [
            {
                "schema": "m4-eligible-live-training-trace-v1",
                "sequence": index,
                "replicate": replicate,
                "protocol_sha256": "a" * 64,
                "capsule_manifest_sha256": "b" * 64,
                "phase": phase,
                "payload": payload,
            }
            for index, (phase, payload) in enumerate(rows)
        ]

    def load(self, rows: list[dict]) -> list[dict]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            return load_trace(
                path, replicate="a", protocol_hash="a" * 64, capsule_hash="b" * 64
            )

    def test_valid_trace_and_identical_execution(self) -> None:
        left = self.load(self.trace())
        result = compare_traces(left, self.trace("b"))
        self.assertIsNone(result["earliest_observed_divergence"])
        self.assertFalse(result["qualification_passed"])

    def test_earliest_forward_divergence_not_last_parameter_difference(self) -> None:
        left, right = self.trace(), self.trace("b")
        right[5]["payload"]["sha256"] = "c" * 64
        right[-2]["payload"]["parameter_sha256"] = "d" * 64
        result = compare_traces(left, right)
        self.assertEqual(
            result["earliest_observed_divergence"]["a_phase"], "model_output"
        )
        self.assertFalse(result["exact_root_cause_established"])

    def test_runtime_difference_separated_from_execution_difference(self) -> None:
        left, right = self.trace(), self.trace("b")
        right[0]["payload"]["torch_version"] = "different"
        result = compare_traces(left, right)
        self.assertEqual(result["runtime_changed_fields"], ["torch_version"])
        self.assertIsNone(result["earliest_observed_divergence"])

    def test_truncated_trace_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "completion"):
            self.load(self.trace()[:-1])

    def test_wrong_identity_rejected(self) -> None:
        rows = self.trace()
        rows[3]["replicate"] = "b"
        with self.assertRaisesRegex(ValueError, "identity"):
            self.load(rows)

    def test_raw_outcomes_rejected(self) -> None:
        rows = self.trace()
        rows[5]["payload"]["loss"] = 0.5
        with self.assertRaisesRegex(ValueError, "phase fields"):
            self.load(rows)

    def test_raw_outcomes_in_runtime_metadata_rejected(self) -> None:
        rows = self.trace()
        rows[0]["payload"]["model_modes"]["loss"] = "0.5"
        with self.assertRaisesRegex(ValueError, "mode/environment fields"):
            self.load(rows)

    def test_reordered_events_rejected(self) -> None:
        rows = self.trace()
        rows[3]["phase"], rows[4]["phase"] = rows[4]["phase"], rows[3]["phase"]
        with self.assertRaisesRegex(ValueError, "execution ordering"):
            self.load(rows)

    def test_duplicate_microbatch_index_rejected(self) -> None:
        rows = self.trace()
        rows[6]["payload"]["index"] = 0
        with self.assertRaisesRegex(ValueError, "microbatch ordering"):
            self.load(rows)


if __name__ == "__main__":
    unittest.main()
