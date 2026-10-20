"""CPU execution of native buffer and watchdog definitions, not GPU imports.

The complete real TQReplayBuffer class and unmodified controller diagnostic
methods are compiled from source. Only optional GPU module imports are omitted.
The native full-import lifecycle tests additionally run on EOS before training.
"""

from __future__ import annotations

import ast
import asyncio
import copy
from enum import Enum
import json
import os
from pathlib import Path
import tempfile
import time
import traceback
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]


class DiagnosticSampler:
    def liveness_diagnostic_state(self) -> dict:
        return {"selection_candidate_watermark": 8}


def native_definitions() -> dict:
    """Compile original definitions without importing Torch or Ray on the host."""
    lifecycle_tree = ast.parse(
        (ROOT / "nemo_rl/algorithms/async_utils/rollout_lifecycle.py").read_bytes()
    )
    reason = next(
        n
        for n in lifecycle_tree.body
        if isinstance(n, ast.ClassDef) and n.name == "RolloutRemovalReason"
    )
    buffer_tree = ast.parse(
        (ROOT / "nemo_rl/algorithms/async_utils/replay_buffer.py").read_bytes()
    )
    buffer = next(
        n
        for n in buffer_tree.body
        if isinstance(n, ast.ClassDef) and n.name == "TQReplayBuffer"
    )
    controller_tree = ast.parse(
        (ROOT / "nemo_rl/algorithms/single_controller.py").read_bytes()
    )
    controller = next(
        n
        for n in controller_tree.body
        if isinstance(n, ast.ClassDef) and n.name == "SingleControllerActor"
    )
    names = {
        "_eligible_live_liveness_state",
        "_task_liveness_state",
        "_eligible_live_progress_watchdog",
    }
    methods = [
        n
        for n in controller.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names
    ]
    assert len(methods) == 3
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            reason,
            buffer,
            ast.ClassDef(
                name="DiagnosticController",
                bases=[],
                keywords=[],
                body=methods,
                decorator_list=[],
            ),
        ],
        type_ignores=[],
    )
    namespace = dict(
        asyncio=asyncio,
        json=json,
        os=os,
        Path=Path,
        time=time,
        traceback=traceback,
        OpportunityAtRiskV2ShadowSampler=DiagnosticSampler,
        Enum=Enum,
    )
    exec(
        compile(
            ast.fix_missing_locations(module),
            "actual-native-watchdog-definitions",
            "exec",
        ),
        namespace,
    )
    return namespace


def fixture(directory: Path, size: int) -> object:
    namespace = native_definitions()
    ctrl = namespace["DiagnosticController"]()
    ctrl._buffer = namespace["TQReplayBuffer"](
        dp_client=None, partition_id="offline-watchdog", pad_value_dict={}
    )
    for index in range(size):
        ctrl._buffer.reserve(weight_version=index % 2, group_id=f"g{index}")
    ctrl._sampler = DiagnosticSampler()
    ctrl._train_steps = ctrl._trainer_version = 3
    ctrl._eligible_live_progress_sequence = 1
    ctrl._eligible_live_progress_phase = "pumps_started"
    ctrl._eligible_live_suppressed_decisions = 0
    ctrl._current_epoch = 1
    ctrl._inflight_rollouts = 16
    ctrl._dispatched_rollouts = set()
    ctrl._rollout_permitted = asyncio.Event()
    ctrl._rollout_exhausted = asyncio.Event()
    ctrl._buffer_admission_stalls = 0
    ctrl._async_cfg = SimpleNamespace(
        eligible_live_decision_capture=SimpleNamespace(
            progress_timeout_seconds=0.01,
            liveness_diagnostic_path=str(directory / "watchdog.json"),
            trajectory_identity="excluded-test",
            trajectory_seed=20261209,
        )
    )
    return ctrl


class WatchdogMetadataTests(unittest.TestCase):
    def test_real_buffer_forced_timeout_writes_json_without_metadata_mutation(self):
        for size in (0, 12):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                ctrl = fixture(root, size)
                before = copy.deepcopy(
                    (
                        ctrl._buffer.start_weight_list,
                        ctrl._buffer.end_weight_list,
                        ctrl._buffer.ready_list,
                    )
                )

                async def exercise():
                    event = asyncio.Event()
                    tasks = [
                        asyncio.create_task(event.wait(), name=name)
                        for name in ("rollout", "train")
                    ]
                    try:
                        with self.assertRaisesRegex(
                            RuntimeError, "made no learner progress"
                        ):
                            await ctrl._eligible_live_progress_watchdog(
                                rollout_task=tasks[0], train_task=tasks[1]
                            )
                    finally:
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)

                asyncio.run(exercise())
                record = json.loads((root / "watchdog.json").read_bytes())
                self.assertEqual(record["controller_state"]["buffer_size"], size)
                self.assertEqual(
                    record["controller_state"]["buffer_unready_size"], size
                )
                self.assertEqual(
                    record["controller_state"]["buffer_end_weight_version_counts"],
                    {"-1": size} if size else {},
                )
                self.assertTrue(record["tasks"]["rollout_pump"]["stack"])
                self.assertTrue(record["tasks"]["train_pump"]["stack"])
                self.assertEqual(
                    before,
                    (
                        ctrl._buffer.start_weight_list,
                        ctrl._buffer.end_weight_list,
                        ctrl._buffer.ready_list,
                    ),
                )
                self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_old_field_does_not_exist_on_real_buffer(self):
        with tempfile.TemporaryDirectory() as temp:
            ctrl = fixture(Path(temp), 1)
            with self.assertRaises(AttributeError):
                _ = ctrl._buffer.weight_version_list


if __name__ == "__main__":
    unittest.main()
