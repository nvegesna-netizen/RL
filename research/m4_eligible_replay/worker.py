"""Explicitly selected Megatron extension for excluded-capsule qualification."""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Iterable
from typing import Any

import numpy as np
import ray
import torch

from research.m4_eligible_replay.bootstrap import initialize_bundled_megatron
from research.m4_eligible_replay.evaluation_rng import isolate_evaluation_tracker
from research.m4_eligible_replay.deterministic_backward import (
    MARKER,
    REQUIRED_ENV,
    initialize_worker,
    validate_environment,
    verify_worker,
)
from research.m4_eligible_replay.training_trace import trace_training
from research.m4_paired_replay.secondary import zero_step_if_uninitialized
from research.m4_paired_replay.worker_support import (
    gradient_probe,
    heldout_logprobs,
    parameter_begin,
    parameter_end,
)

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

    @staticmethod
    def configure_worker(
        num_gpus: int | float,
        bundle_indices: tuple[int, list[int]] | None = None,
        num_gpus_per_node: int | None = None,
    ) -> tuple[dict, dict, dict, dict]:
        """Preserve native resources and explicitly propagate early actor controls."""
        resources, environment, kwargs, runtime = (
            MegatronPolicyWorkerImpl.configure_worker(
                num_gpus,
                bundle_indices=bundle_indices,
                num_gpus_per_node=num_gpus_per_node,
            )
        )
        if os.environ.get(MARKER) == "1":
            validate_environment(os.environ)
            environment.update({MARKER: "1", **REQUIRED_ENV})
        if os.environ.get("M4_PAIRED_SYNTHETIC_ENGINEERING") == "1":
            environment["M4_PAIRED_SYNTHETIC_ENGINEERING"] = "1"
        return resources, environment, kwargs, runtime

    def __init__(self, config: Any, *args: Any, **kwargs: Any) -> None:
        """Apply deterministic config before the inherited model construction."""
        self._paired_parameter_snapshot = None
        if config.get("eligible_replay_deterministic") is True:
            with initialize_worker(config):
                super().__init__(config, *args, **kwargs)
            verify_worker(self, torch)
        else:
            super().__init__(config, *args, **kwargs)

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
        if self.cfg.get("eligible_replay_deterministic") is True:
            verify_worker(self, torch)
        if self.cfg.get("eligible_replay_minimal_observer") is True:
            if (
                self.cfg.get("eligible_replay_deterministic") is not True
                or self.cfg.get("eligible_replay_training_trace") is not None
            ):
                raise ValueError(
                    "minimal observer requires strict mode and no training trace"
                )
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

    def replay_execution_boundary(self) -> dict[str, Any]:
        """Read strict runtime modes outside training, without tensor observation."""
        if self.cfg.get("eligible_replay_minimal_observer") is not True:
            raise ValueError("minimal observer boundary was not selected")
        if self.cfg.get("eligible_replay_training_trace") is not None:
            raise ValueError("minimal observer rejects training trace")
        verify_worker(self, torch)
        # Actual backend metadata is optional and needed only in this actor RPC.
        from nemo_rl.models.policy.workers import megatron_policy_worker as backend
        from research.m4_eligible_replay.training_trace import (
            runtime_metadata,
            source_identity,
        )

        runtime = runtime_metadata(self, backend, torch)
        if self.cfg.get("eligible_replay_paired_integration") is True:
            for module_name in (
                "research.m4_paired_replay.secondary",
                "research.m4_paired_replay.worker_support",
            ):
                runtime["module_origins"][module_name] = source_identity(module_name)

        return {
            "training_trace_enabled": False,
            "runtime": runtime,
        }

    def _pre_load_checkpoint_hook(self, state: Any, model: Any) -> None:
        checkpoint = state.cfg.checkpoint
        if self.cfg.get("eligible_replay_engineering_cold_start") is True:
            if (
                os.environ.get("M4_PAIRED_SYNTHETIC_ENGINEERING") != "1"
                or self.cfg.get("eligible_replay_paired_integration") is not True
                or checkpoint.load is not None
                or checkpoint.load_optim
                or checkpoint.pretrained_checkpoint is None
            ):
                raise ValueError(
                    "cold start is restricted to synthetic engineering bootstrap"
                )
            # Preserve native pretrained initialization; this is not a resume proof.
            return
        if checkpoint.load is None or not checkpoint.load_optim or checkpoint.finetune:
            raise ValueError("qualification requires a real optimizer resume")
        checkpoint.load_rng = True

    def paired_engineering_canary(self, canary: str) -> None:
        """Exercise actor stdout/stderr collection using a synthetic private marker."""
        subject = self.cfg.get("eligible_replay_probe_subject", {})
        if (
            os.environ.get("M4_PAIRED_SYNTHETIC_ENGINEERING") != "1"
            or not subject.get("identity", "").startswith("engineering-")
            or len(canary) != 64
            or any(c not in "0123456789abcdef" for c in canary)
        ):
            raise ValueError("invalid synthetic collector canary")
        print("M4_PRIVATE_CANARY=" + canary, flush=True)
        os.write(2, ("M4_PRIVATE_CANARY=" + canary + "\n").encode())

    def paired_parameter_begin(self) -> None:
        parameter_begin(self, torch)

    def paired_parameter_end(self) -> float:
        return parameter_end(self, torch)

    def paired_logprobs_meta(self, meta: Any) -> dict:
        return heldout_logprobs(
            self,
            meta,
            torch=torch,
            tracker=tensor_parallel.get_cuda_rng_tracker(),
            digest=state_digest,
        )

    def paired_probe_meta(
        self, meta: Any, loss_fn: Any, output: str, subject: dict
    ) -> dict:
        return gradient_probe(
            self,
            meta,
            loss_fn,
            torch=torch,
            native_train=super().train,
            output=output,
            subject=subject,
        )

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
        if not steps:
            steps = [
                zero_step_if_uninitialized(
                    inner_states,
                    expected_step=self.cfg.get(
                        "eligible_replay_expected_optimizer_step"
                    ),
                    scheduler_samples=int(self.scheduler.num_steps),
                )
            ]
        if len(set(steps)) != 1:
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
