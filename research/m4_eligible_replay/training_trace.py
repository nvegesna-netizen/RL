"""Outcome-free, opt-in observation of one single-worker comparator update."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import fields, is_dataclass
import hashlib
import inspect
import os
from pathlib import Path
import sys
from typing import Any, TypedDict

from research.m4_eligible_replay.contracts import canonical_json
from research.m4_eligible_replay.state_diagnostics import validate_hash


class TraceConfig(TypedDict):
    """Explicit driver-to-worker diagnostic identity; never part of the protocol."""

    output: str
    replicate: str
    protocol_sha256: str
    capsule_manifest_sha256: str


MODEL_MODE_FIELDS = (
    "deterministic_mode",
    "attention_backend",
    "attention_dropout",
    "hidden_dropout",
    "cross_entropy_loss_fusion",
    "bf16",
    "fp16",
    "fp8",
    "cuda_graph_impl",
    "cuda_graph_scope",
    "recompute_granularity",
)
ENVIRONMENT_FIELDS = (
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_LAUNCH_BLOCKING",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
    "NVTE_FUSED_ATTN",
    "NVTE_FLASH_ATTN",
    "NVTE_UNFUSED_ATTN",
    "NCCL_ALGO",
    "OMP_NUM_THREADS",
    "PYTHONHASHSEED",
)


@contextmanager
def observer_cleanup() -> Iterator[ExitStack]:
    """Attempt every restoration and retain a primary training exception."""
    cleanup = ExitStack()
    primary = None
    try:
        yield cleanup
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            cleanup.close()
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note(
                f"training observer cleanup failed: {type(error).__name__}: {error}"
            )


def normalize_payload(
    value: Any,
    *,
    tensor_type: type,
    process_group_type: type,
    describe_group: Callable[[Any], dict[str, int]],
) -> Any:
    """Preserve ordered contents without copying tensors or stringifying objects."""

    def normalize(child: Any) -> Any:
        return normalize_payload(
            child,
            tensor_type=tensor_type,
            process_group_type=process_group_type,
            describe_group=describe_group,
        )

    if (
        isinstance(value, tensor_type)
        or value is None
        or type(value) in (bool, str, int, float)
    ):
        return value
    if isinstance(value, process_group_type):
        return {"process_group": describe_group(value)}
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("training data keys must be strings")
        return {key: normalize(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        children = [normalize(child) for child in value]
        return tuple(children) if isinstance(value, tuple) else children
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                field.name: normalize(getattr(value, field.name))
                for field in fields(value)
            },
        }
    raise TypeError(f"unsupported training diagnostic payload: {type(value).__name__}")


def source_identity(module_name: str) -> dict[str, str | None]:
    """Describe an already loaded module; do not import a different implementation."""
    module = sys.modules.get(module_name)
    path = getattr(module, "__file__", None)
    if path is None:
        return {"path": None, "sha256": None}
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def runtime_metadata(worker: Any, backend: Any, torch: Any) -> dict[str, Any]:
    """Read actual actor modes/origins without enabling deterministic algorithms."""
    config = worker._get_model_config()
    return {
        "torch_version": torch.__version__,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "model_modes": {
            name: str(getattr(config, name, None)) for name in MODEL_MODE_FIELDS
        },
        "environment": {name: os.environ.get(name) for name in ENVIRONMENT_FIELDS},
        "module_origins": {
            name: source_identity(name)
            for name in (
                type(worker).__module__,
                type(worker.optimizer).__module__,
                "nemo_rl.models.policy.workers.megatron_policy_worker",
                "nemo_rl.models.megatron.train",
                "nemo_rl.models.megatron.data",
                "nemo_rl.models.megatron.setup",
                "megatron.bridge.training.checkpointing",
                "megatron.bridge.models.conversion.auto_bridge",
                "megatron.core.optimizer",
                "megatron.core.tensor_parallel.random",
                "transformer_engine.pytorch",
            )
        },
        "callable_origins": {
            "microbatch_iterator": inspect.unwrap(
                backend.get_microbatch_iterator
            ).__code__.co_filename,
            "forward_backward": inspect.unwrap(
                backend.megatron_forward_backward
            ).__code__.co_filename,
        },
    }


@contextmanager
def observe_training(
    *,
    backend: Any,
    model: Any,
    optimizer: Any,
    data: Any,
    fingerprint: Callable[[Any], str],
    gradients: Callable[[], str],
    optimizer_state: Callable[[], dict[str, str]],
    emit: Callable[[str, dict[str, Any]], None],
) -> Iterator[None]:
    """Temporarily observe native calls; restore bindings/hooks on every exit.

    Only the single-worker qualification may use this context. No argument,
    return value, gradient or numerical setting is replaced. CPU hashing
    synchronizes tensors and is an engineering observer, not a performance run.
    """
    original_iterator = backend.get_microbatch_iterator
    original_forward_backward = backend.megatron_forward_backward
    original_step = optimizer.step
    local_step = "step" in vars(optimizer)
    counts = {
        "microbatches": 0,
        "model_inputs": 0,
        "model_outputs": 0,
        "forward_backward": 0,
        "optimizer_steps": 0,
    }
    expected_microbatches = 0

    def trace_iterator(*args: Any, **kwargs: Any) -> Any:
        nonlocal expected_microbatches
        result = original_iterator(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 5:
            raise ValueError("pinned microbatch iterator contract changed")
        expected_microbatches += int(result[1])

        def observed() -> Iterator[Any]:
            for microbatch in result[0]:
                emit(
                    "microbatch",
                    {
                        "index": counts["microbatches"],
                        "sha256": fingerprint(microbatch),
                    },
                )
                counts["microbatches"] += 1
                yield microbatch

        return (observed(), *result[1:])

    def before_forward(module: Any, args: tuple, kwargs: dict) -> None:
        emit(
            "model_input",
            {"index": counts["model_inputs"], "sha256": fingerprint((args, kwargs))},
        )
        counts["model_inputs"] += 1

    def after_forward(module: Any, args: tuple, kwargs: dict, output: Any) -> None:
        emit(
            "model_output",
            {"index": counts["model_outputs"], "sha256": fingerprint(output)},
        )
        counts["model_outputs"] += 1

    def trace_forward_backward(*args: Any, **kwargs: Any) -> Any:
        counts["forward_backward"] += 1
        if counts["forward_backward"] != 1:
            raise ValueError("diagnostic permits exactly one forward/backward call")
        emit(
            "forward_backward_start",
            {
                "sha256": fingerprint(
                    {
                        key: kwargs[key]
                        for key in (
                            "num_microbatches",
                            "seq_length",
                            "mbs",
                            "forward_only",
                            "global_valid_seqs",
                            "global_valid_toks",
                        )
                    }
                )
            },
        )
        result = original_forward_backward(*args, **kwargs)
        emit("forward_backward_end", {"gradient_sha256": gradients()})
        return result

    def trace_step(*args: Any, **kwargs: Any) -> Any:
        if counts["optimizer_steps"] != 0 or counts["forward_backward"] != 1:
            raise ValueError("unexpected optimizer call ordering")
        emit("optimizer_before", {**optimizer_state(), "gradient_sha256": gradients()})
        result = original_step(*args, **kwargs)
        counts["optimizer_steps"] += 1
        emit("optimizer_after", optimizer_state())
        return result

    emit("worker_input", {"sha256": fingerprint(data)})
    with observer_cleanup() as cleanup:
        cleanup.callback(
            backend.__setattr__, "get_microbatch_iterator", original_iterator
        )
        cleanup.callback(
            backend.__setattr__, "megatron_forward_backward", original_forward_backward
        )
        if local_step:
            cleanup.callback(optimizer.__setattr__, "step", original_step)
        else:
            cleanup.callback(optimizer.__delattr__, "step")
        backend.get_microbatch_iterator = trace_iterator
        backend.megatron_forward_backward = trace_forward_backward
        optimizer.step = trace_step
        cleanup.callback(
            model.register_forward_pre_hook(before_forward, with_kwargs=True).remove
        )
        cleanup.callback(
            model.register_forward_hook(after_forward, with_kwargs=True).remove
        )
        yield
        if (
            counts["forward_backward"] != 1
            or counts["optimizer_steps"] != 1
            or expected_microbatches < 1
            or any(
                counts[key] != expected_microbatches
                for key in ("microbatches", "model_inputs", "model_outputs")
            )
        ):
            raise ValueError("incomplete diagnostic training execution")
        emit("complete", {**counts, "expected_microbatches": expected_microbatches})


@contextmanager
def trace_training(
    worker: Any, data: Any, *, config: TraceConfig, digest: Callable[[Any], str]
) -> Iterator[None]:
    """Bind an exclusive trace to one capsule/protocol and the actual worker."""
    # Optional CUDA dependencies are needed only for the opt-in actor path.
    import torch
    from megatron.core.optimizer import ChainedOptimizer
    from nemo_rl.models.policy.workers import megatron_policy_worker as backend

    if set(config) != {
        "output",
        "replicate",
        "protocol_sha256",
        "capsule_manifest_sha256",
    }:
        raise ValueError("unexpected training trace configuration")
    replicate = config["replicate"]
    if replicate not in ("a", "b") or torch.distributed.get_world_size() != 1:
        raise ValueError(
            "training trace requires the single-worker duplicate qualification"
        )
    validate_hash(config["protocol_sha256"])
    validate_hash(config["capsule_manifest_sha256"])
    output = Path(config["output"])
    if (
        output.name != f"replay-{replicate}-training-trace.jsonl"
        or not output.is_absolute()
    ):
        raise ValueError("unexpected training trace destination")

    def fingerprint(value: Any) -> str:
        return digest(
            normalize_payload(
                value,
                tensor_type=torch.Tensor,
                process_group_type=torch.distributed.ProcessGroup,
                describe_group=lambda group: {
                    "size": torch.distributed.get_world_size(group),
                    "rank": torch.distributed.get_rank(group),
                },
            )
        )

    def gradients() -> str:
        torch.cuda.synchronize()
        entries = [
            (name, parameter.grad, getattr(parameter, "main_grad", None))
            for name, parameter in worker.model.named_parameters()
        ]
        if not entries or not any(
            grad is not None or main is not None for _, grad, main in entries
        ):
            raise ValueError("no training gradients found at diagnostic boundary")
        return digest(entries)

    def optimizer_state() -> dict[str, str]:
        torch.cuda.synchronize()
        optimizers = (
            worker.optimizer.chained_optimizers
            if isinstance(worker.optimizer, ChainedOptimizer)
            else [worker.optimizer]
        )
        return {
            "inner_optimizer_sha256": digest(
                [optimizer.optimizer.state_dict() for optimizer in optimizers]
            ),
            "master_parameters_sha256": digest(worker.optimizer.get_parameters()),
            "parameter_sha256": worker.model_parameter_sha256(),
        }

    sequence = 0
    with output.open("xb") as stream:

        def emit(phase: str, payload: dict[str, Any]) -> None:
            nonlocal sequence
            record = {
                "schema": "m4-eligible-live-training-trace-v1",
                "sequence": sequence,
                "replicate": replicate,
                "protocol_sha256": config["protocol_sha256"],
                "capsule_manifest_sha256": config["capsule_manifest_sha256"],
                "phase": phase,
                "payload": payload,
            }
            stream.write(canonical_json(record))
            stream.flush()
            sequence += 1

        emit("runtime", runtime_metadata(worker, backend, torch))
        with observe_training(
            backend=backend,
            model=worker.model,
            optimizer=worker.optimizer,
            data=data,
            fingerprint=fingerprint,
            gradients=gradients,
            optimizer_state=optimizer_state,
            emit=emit,
        ):
            yield
