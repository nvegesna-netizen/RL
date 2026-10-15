"""Explicitly selected Megatron extension for excluded-capsule qualification."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable
from typing import Any

import numpy as np
import ray
import torch

from research.m4_eligible_replay.bootstrap import initialize_bundled_megatron
from research.m4_eligible_replay.evaluation_rng import isolate_evaluation_tracker
from research.m4_eligible_replay.training_trace import trace_training

# Standalone smoke checks have not imported nemo_rl.__init__ yet. Bootstrap
# the packaged sources before importing either Megatron Core or Bridge.
initialize_bundled_megatron(require_bridge=True)

from megatron.core import tensor_parallel  # noqa: E402
from megatron.core.optimizer import ChainedOptimizer  # noqa: E402
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker  # noqa: E402
from nemo_rl.models.policy.workers.megatron_policy_worker import (  # noqa: E402
    MegatronPolicyWorkerImpl,
)


def state_digest(value: Any) -> str:
    """Hash nested optimizer/RNG state using typed, length-delimited records."""
    digest = hashlib.sha256()

    def add(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            add(("tensor", str(tensor.dtype), tuple(tensor.shape)))
            payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        elif isinstance(item, np.ndarray):
            add(("ndarray", str(item.dtype), tuple(item.shape)))
            payload = item.tobytes()
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        elif isinstance(item, dict):
            digest.update(b"dict")
            add(len(item))
            for key in sorted(
                item, key=lambda entry: (type(entry).__name__, repr(entry))
            ):
                add(key)
                add(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            add(len(item))
            for child in item:
                add(child)
        elif item is None or isinstance(
            item, (bool, str, int, float, np.integer, np.floating)
        ):
            payload = f"{type(item).__name__}:{item!r}".encode()
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        else:
            raise TypeError(f"unsupported training-state value: {type(item).__name__}")

    add(value)
    return digest.hexdigest()


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker")
)  # pragma: no cover
class EligibleLiveReplayWorker(MegatronPolicyWorkerImpl):
    """Restore RNG through Bridge's existing pre-load hook; leave acquisition alone."""

    def train(
        self,
        data: Any,
        loss_fn: Any,
        eval_mode: bool = False,
        gbs: int | None = None,
        mbs: int | None = None,
        check_dim_skip_keys: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Isolate held-out forwards; delegate actual training without restoration."""
        if not eval_mode:
            trace_config = self.cfg.get("eligible_replay_training_trace")
            if trace_config is not None:
                with trace_training(
                    self, data, config=trace_config, digest=state_digest
                ):
                    return super().train(
                        data,
                        loss_fn=loss_fn,
                        eval_mode=eval_mode,
                        gbs=gbs,
                        mbs=mbs,
                        check_dim_skip_keys=check_dim_skip_keys,
                    )
            return super().train(
                data,
                loss_fn=loss_fn,
                eval_mode=eval_mode,
                gbs=gbs,
                mbs=mbs,
                check_dim_skip_keys=check_dim_skip_keys,
            )
        tracker = tensor_parallel.get_cuda_rng_tracker()

        def clone_state(value: Any) -> Any:
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.uint8
                or value.ndim != 1
            ):
                raise TypeError(
                    "qualification supports only byte-tensor RNG tracker states"
                )
            return value.clone()

        print(
            json.dumps(
                {
                    "stage": "EVALUATION_RNG_ISOLATION",
                    "tracker_class": f"{type(tracker).__module__}.{type(tracker).__qualname__}",
                    "state_types": sorted(
                        {
                            type(value).__name__
                            for value in tracker.get_states().values()
                        }
                    ),
                    "supported_mode": "named-byte-tensor-states",
                }
            ),
            flush=True,
        )
        with isolate_evaluation_tracker(
            tracker, clone_state=clone_state, digest=state_digest
        ):
            return super().train(
                data,
                loss_fn=loss_fn,
                eval_mode=eval_mode,
                gbs=gbs,
                mbs=mbs,
                check_dim_skip_keys=check_dim_skip_keys,
            )

    def _pre_load_checkpoint_hook(self, state: Any, model: Any) -> None:
        checkpoint = state.cfg.checkpoint
        if checkpoint.load is None or not checkpoint.load_optim or checkpoint.finetune:
            raise ValueError("qualification requires a real optimizer resume")
        checkpoint.load_rng = True

    def replay_state(self) -> dict[str, Any]:
        """Return hashes and observed counters rather than parameter/state content."""
        if (
            torch.distributed.get_world_size() != 1
            or self.optimizer is None
            or self.scheduler is None
        ):
            raise ValueError("qualification requires one initialized training worker")
        checkpoint = self.mcore_state.cfg.checkpoint
        if checkpoint.finetune or not checkpoint.load_optim or not checkpoint.load_rng:
            raise ValueError("resume was silently downgraded")
        rng = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
            "tracker": tensor_parallel.get_cuda_rng_tracker().get_states(),
        }
        optimizers = (
            self.optimizer.chained_optimizers
            if isinstance(self.optimizer, ChainedOptimizer)
            else [self.optimizer]
        )
        inner_states = [optimizer.optimizer.state_dict() for optimizer in optimizers]
        steps = []

        def collect_steps(value: Any) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    if key == "step":
                        steps.append(
                            int(child.item())
                            if isinstance(child, torch.Tensor)
                            else int(child)
                        )
                    else:
                        collect_steps(child)
            elif isinstance(value, (tuple, list)):
                for child in value:
                    collect_steps(child)

        collect_steps(inner_states)
        if not steps or len(set(steps)) != 1:
            raise ValueError("optimizer step counter is missing or inconsistent")
        return {
            "parameter_sha256": self.model_parameter_sha256(),
            "optimizer_sha256": state_digest(
                {
                    "inner_states": inner_states,
                    "master_parameters": self.optimizer.get_parameters(),
                }
            ),
            "optimizer_step": steps[0],
            "scheduler_sha256": state_digest(self.scheduler.state_dict()),
            "rng_sha256": state_digest(rng),
            "rng_component_sha256": {
                key: state_digest(value) for key, value in rng.items()
            },
            "scheduler_num_steps": int(self.scheduler.num_steps),
            "load_optim": bool(checkpoint.load_optim),
            "load_rng": bool(checkpoint.load_rng),
            "finetune": bool(checkpoint.finetune),
        }
