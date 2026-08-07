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

"""Runtime-only math verifier integration with its platform dependencies."""

import itertools
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import ray

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed import ray_actor_environment_registry
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments import utils as environment_utils
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.environments.math_environment import HFVerifyWorker
from nemo_rl.environments.utils import chunk_list_to_workers
from turn_level_credit.math_repair import (
    TERMINAL_SUCCESS_KEY,
    VERIFIER_SCORE_KEY,
    MathRepairConfig,
    MathRepairMetadata,
    _latest_assistant_responses,
    _validate_metadata,
    advance_math_repair_turns,
)

MATH_REPAIR_ENVIRONMENT_FQN = (
    "turn_level_credit.math_repair_runtime.FixedHorizonMathRepairEnvironment"
)


@ray.remote
class FixedHorizonMathRepairEnvironment(  # pragma: no cover
    EnvironmentInterface[MathRepairMetadata]
):
    """Batched math verifier that always consumes the configured turn budget."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._repair_config = MathRepairConfig.model_validate(cfg)
        self.num_workers = self._repair_config.num_workers
        self._worker_counter = itertools.count()
        self.workers = [
            HFVerifyWorker.options(  # type: ignore # decorated with @ray.remote
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote()
            for _ in range(self.num_workers)
        ]

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[MathRepairMetadata],
    ) -> EnvironmentReturn[MathRepairMetadata]:
        """Verify only the newest policy answer and advance fixed-horizon state."""
        if len(message_log_batch) != len(metadata):
            raise ValueError("math repair message and metadata batches must align")
        if not metadata:
            raise ValueError("math repair cannot step an empty batch")
        for row, row_metadata in enumerate(metadata):
            _validate_metadata(
                row_metadata,
                config=self._repair_config,
                row=row,
            )
        responses = _latest_assistant_responses(message_log_batch)
        ground_truths = [row["ground_truth"] for row in metadata]
        response_chunks = chunk_list_to_workers(responses, self.num_workers)
        ground_truth_chunks = chunk_list_to_workers(ground_truths, self.num_workers)
        worker_index = next(self._worker_counter) % self.num_workers
        futures = [
            self.workers[(worker_index + index) % self.num_workers].verify.remote(
                response_chunk,
                ground_truth_chunk,
                False,
                math_verify_impl=self._repair_config.math_verify_impl,
            )
            for index, (response_chunk, ground_truth_chunk) in enumerate(
                zip(response_chunks, ground_truth_chunks, strict=True)
            )
        ]
        scores = [
            float(score)
            for worker_scores in ray.get(futures)
            for score in worker_scores
        ]
        result = advance_math_repair_turns(
            scores,
            metadata,
            config=self._repair_config,
        )
        return EnvironmentReturn(
            observations=result.observations,
            metadata=result.metadata,
            next_stop_strings=[None] * len(scores),
            rewards={
                VERIFIER_SCORE_KEY: result.verifier_scores,
                TERMINAL_SUCCESS_KEY: result.terminal_success,
            },
            terminateds=result.terminateds,
            answers=None,
        )

    def global_post_process_and_metrics(
        self,
        batch: BatchedDataDict[Any],
    ) -> tuple[BatchedDataDict[Any], dict[str, float]]:
        """Report terminal success separately from accumulated verifier scores."""
        if VERIFIER_SCORE_KEY not in batch or TERMINAL_SUCCESS_KEY not in batch:
            raise ValueError("math repair batch is missing named reward components")
        verifier_score_sum = batch[VERIFIER_SCORE_KEY].float()
        terminal_success = batch[TERMINAL_SUCCESS_KEY].float()
        return batch, {
            "accuracy": float(terminal_success.mean().item()),
            "math_repair/verifier_score_sum_mean": float(
                verifier_score_sum.mean().item()
            ),
        }

    def shutdown(self) -> None:
        """Release verifier workers owned by this environment."""
        for worker in self.workers:
            ray.kill(worker)


@contextmanager
def install_math_repair_environment() -> Iterator[None]:
    """Temporarily route the standard math task to the research environment."""
    original_entry = environment_utils.ENV_REGISTRY.get("math")
    if original_entry is None:
        raise RuntimeError("the standard math environment is not registered")
    original_fqn = original_entry.get("actor_class_fqn")
    if original_fqn is None:
        raise RuntimeError("the standard math environment has no actor class")
    actor_environments = ray_actor_environment_registry.ACTOR_ENVIRONMENT_REGISTRY
    if original_fqn not in actor_environments:
        raise RuntimeError("the standard math actor has no Python environment")
    previous_actor_environment = actor_environments.get(MATH_REPAIR_ENVIRONMENT_FQN)
    environment_utils.ENV_REGISTRY["math"] = {
        "actor_class_fqn": MATH_REPAIR_ENVIRONMENT_FQN
    }
    actor_environments[MATH_REPAIR_ENVIRONMENT_FQN] = actor_environments[original_fqn]
    try:
        yield
    finally:
        environment_utils.ENV_REGISTRY["math"] = original_entry
        if previous_actor_environment is None:
            actor_environments.pop(MATH_REPAIR_ENVIRONMENT_FQN, None)
        else:
            actor_environments[MATH_REPAIR_ENVIRONMENT_FQN] = previous_actor_environment
