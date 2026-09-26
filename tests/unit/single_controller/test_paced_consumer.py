"""Dependency-light pacing tests; production integration is tested separately."""

import asyncio
import importlib.util
from pathlib import Path
import sys
import unittest

# Load only this dependency-free source: the package initializer imports Ray.
# These are helper tests, not evidence of production package integration.
SOURCE = (
    Path(__file__).resolve().parents[3]
    / "nemo_rl/algorithms/async_utils/paced_consumer.py"
)
SPEC = importlib.util.spec_from_file_location("paced_consumer_under_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
wait_for_consumer = MODULE.wait_for_consumer


class PacedConsumerTests(unittest.IsolatedAsyncioTestCase):
    async def test_measures_actual_duration(self) -> None:
        now = 10.0

        async def sleep(delay: float) -> None:
            nonlocal now
            now += delay + 0.25

        result = await wait_for_consumer(2.0, clock=lambda: now, sleep=sleep)
        self.assertEqual(result.requested_seconds, 2.0)
        self.assertEqual(result.elapsed_seconds, 2.25)

    async def test_early_wakeup_does_not_finish_step(self) -> None:
        now = 0.0
        requests: list[float] = []

        async def sleep(delay: float) -> None:
            nonlocal now
            requests.append(delay)
            now += min(delay, 1.0)

        result = await wait_for_consumer(3.0, clock=lambda: now, sleep=sleep)
        self.assertEqual(requests, [3.0, 2.0, 1.0])
        self.assertEqual(result.elapsed_seconds, 3.0)

    async def test_zero_duration_yields(self) -> None:
        requests: list[float] = []

        async def sleep(delay: float) -> None:
            requests.append(delay)

        result = await wait_for_consumer(0.0, clock=lambda: 1.0, sleep=sleep)
        self.assertEqual(requests, [0.0])
        self.assertEqual(result.elapsed_seconds, 0.0)

    async def test_cancellation_propagates(self) -> None:
        async def sleep(delay: float) -> None:
            raise asyncio.CancelledError

        with self.assertRaises(asyncio.CancelledError):
            await wait_for_consumer(1.0, sleep=sleep)

    async def test_invalid_durations_rejected(self) -> None:
        for duration in (-1.0, float("nan"), float("inf")):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                await wait_for_consumer(duration)

    async def test_backward_clock_rejected(self) -> None:
        timestamps = iter((2.0, 1.0))

        async def sleep(delay: float) -> None:
            pass

        with self.assertRaises(RuntimeError):
            await wait_for_consumer(1.0, clock=lambda: next(timestamps), sleep=sleep)


if __name__ == "__main__":
    unittest.main()
