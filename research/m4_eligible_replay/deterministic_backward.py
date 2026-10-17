"""Fail-closed deterministic execution for excluded-capsule qualification only."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import os
from typing import Any

from research.m4_eligible_replay.training_trace import observer_cleanup


MARKER = "M4_ELIGIBLE_REPLAY_DETERMINISTIC"
REQUIRED_ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    "NCCL_ALGO": "Ring",
}


def validate_environment(environment: Mapping[str, str]) -> None:
    """Reject missing or conflicting controls rather than repairing them late."""
    if environment.get(MARKER) != "1":
        raise ValueError("deterministic qualification marker missing")
    for name, expected in REQUIRED_ENV.items():
        if environment.get(name) != expected:
            raise ValueError(f"strict deterministic environment mismatch: {name}")


def apply_torch_controls(torch: Any, environment: Mapping[str, str]) -> None:
    """Enable strict Torch/cuDNN behavior without selecting an attention backend."""
    validate_environment(environment)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    verify_torch_controls(torch, environment)


def verify_torch_controls(torch: Any, environment: Mapping[str, str]) -> None:
    """Check actual process flags; setting a config value is insufficient."""
    validate_environment(environment)
    if (
        not torch.are_deterministic_algorithms_enabled()
        or torch.is_deterministic_algorithms_warn_only_enabled()
        or not torch.backends.cudnn.deterministic
        or torch.backends.cudnn.benchmark
        or torch.backends.cuda.matmul.allow_tf32
    ):
        raise ValueError("strict deterministic Torch controls not active")


@contextmanager
def model_setup_controls(*, setup: Any, config: dict, torch: Any) -> Iterator[None]:
    """Set model modes before native config validation and model construction.

    The pinned setup_model_config calls _apply_performance_config before its
    model __post_init__ and Bridge ConfigContainer validation. Only this
    single actor's initialization is observed; the native binding is restored.
    """
    apply_torch_controls(torch, os.environ)
    if config.get("eligible_replay_deterministic") is not True:
        raise ValueError("explicit deterministic model configuration required")
    original = setup._apply_performance_config
    applied = 0

    def configure(model: Any, policy_config: dict) -> None:
        nonlocal applied
        original(model, policy_config)
        if policy_config.get("eligible_replay_deterministic") is not True:
            raise ValueError("deterministic actor setup received another configuration")
        model.deterministic_mode = True
        # Required by the pinned Bridge deterministic validator. This does not
        # assert that fused cross entropy caused the previous gradient mismatch.
        model.cross_entropy_loss_fusion = False
        applied += 1

    with observer_cleanup() as cleanup:
        cleanup.callback(setup.__setattr__, "_apply_performance_config", original)
        setup._apply_performance_config = configure
        yield
        if applied == 0:
            raise ValueError("native deterministic model setup hook was not reached")
        verify_torch_controls(torch, os.environ)


def verify_worker(worker: Any, torch: Any) -> None:
    """Reject checkpoint/config downgrades in the actual model actor."""
    verify_torch_controls(torch, os.environ)
    config = worker._get_model_config()
    if (
        config.deterministic_mode is not True
        or config.cross_entropy_loss_fusion is not False
    ):
        raise ValueError("actual model is not in strict deterministic mode")


@contextmanager
def initialize_worker(config: dict) -> Iterator[None]:
    """Initialize with strict controls before any model modules are constructed."""
    # Optional CUDA/model dependencies are needed only in the opted-in actor.
    import torch
    from nemo_rl.models.megatron import setup

    with model_setup_controls(setup=setup, config=config, torch=torch):
        yield
