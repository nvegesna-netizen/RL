# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Measured zero-update consumer delay; contains no trainer or admission state."""

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ConsumerTiming:
    """Monotonic timestamps local to the controller process, in seconds."""

    requested_seconds: float
    started_at: float
    finished_at: float

    @property
    def elapsed_seconds(self) -> float:
        return self.finished_at - self.started_at


async def wait_for_consumer(
    duration_seconds: float,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> ConsumerTiming:
    """Wait at least the requested duration, propagating cancellation.

    The caller releases selected buffer capacity before entering this function
    and advances its logical admission clock only after successful return.
    Cancellation therefore never reports a completed consumer step. This helper
    does not clear samples, mutate weights, or advance any controller clock.
    """
    if not math.isfinite(duration_seconds) or duration_seconds < 0:
        raise ValueError("consumer duration must be finite and nonnegative")
    started_at = clock()
    if not math.isfinite(started_at):
        raise ValueError("consumer clock must be finite")
    deadline = started_at + duration_seconds
    if not math.isfinite(deadline):
        raise ValueError("consumer deadline must be finite")
    previous = started_at
    await sleep(duration_seconds)
    finished_at = clock()
    while finished_at < deadline:
        if not math.isfinite(finished_at) or finished_at < previous:
            raise RuntimeError("consumer clock must remain finite and monotonic")
        previous = finished_at
        await sleep(deadline - previous)
        finished_at = clock()
    if not math.isfinite(finished_at) or finished_at < previous:
        raise RuntimeError("consumer clock must remain finite and monotonic")
    return ConsumerTiming(duration_seconds, started_at, finished_at)
