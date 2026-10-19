"""Engineering-only native Adam state allocation without an optimizer update."""

from __future__ import annotations

import inspect
import hashlib
import os
from pathlib import Path
from typing import Any, Callable


def require_engineering_zero(worker: Any, torch: Any) -> None:
    """Reject a zero-counter interpretation outside the synthetic v0 boundary."""
    if (
        os.environ.get("M4_PAIRED_SYNTHETIC_ENGINEERING") != "1"
        or worker.cfg.get("eligible_replay_paired_integration") is not True
        or worker.cfg.get("eligible_replay_expected_optimizer_step") != 0
        or type(worker.cfg.get("eligible_replay_expected_optimizer_step")) is not int
        or worker.mcore_state.train_state.step != 0
        or worker.scheduler.num_steps != 0
        or torch.distributed.get_world_size() != 1
    ):
        raise ValueError(
            "native zero state requires synthetic engineering version zero"
        )


def validate_native_zero(optimizers: list, *, torch: Any, adam_class: type) -> int:
    """Validate every native shard, including master weights and absent counters.

    TE creates its group counter only on the first step. Absence is acceptable
    here only with complete zero moments, unchanged master weights, and the
    caller's independently checked native version-zero metadata. Nothing is
    assigned to a counter or scheduler.
    """
    if not optimizers:
        raise ValueError("no optimizer shards")
    count = 0
    for optimizer in optimizers:
        raw = optimizer.optimizer
        if (
            not isinstance(raw, adam_class)
            or not optimizer.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
            or not raw.master_weights
            or not raw.param_groups
        ):
            raise ValueError("unsupported native zero-state optimizer")
        parameters = [p for g in raw.param_groups for p in g["params"]]
        if any("step" in g for g in raw.param_groups) or not parameters:
            raise ValueError("zero-state counter present or parameters absent")
        if set(raw.state) != set(parameters) or len(set(parameters)) != len(parameters):
            raise ValueError("incomplete or duplicated native optimizer state")
        for parameter in parameters:
            state = raw.state[parameter]
            if set(state) != {"exp_avg", "exp_avg_sq", "master_param"}:
                raise ValueError("native zero-state keys differ")
            for name in ("exp_avg", "exp_avg_sq"):
                value = raw.get_unscaled_state(parameter, name)
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != parameter.shape
                    or not bool(torch.isfinite(value).all().item())
                    or bool(torch.count_nonzero(value).item())
                ):
                    raise ValueError("native zero moments differ")
            master = raw.get_unscaled_state(parameter, "master_param")
            remainders = (
                raw.store_param_remainders and parameter.dtype == torch.bfloat16
            )
            if master.shape != parameter.shape or (
                (
                    master.dtype != torch.int16
                    or bool(torch.count_nonzero(master).item())
                )
                if remainders
                else not torch.equal(master, parameter.detach().float())
            ):
                raise ValueError("native initial master weights differ")
            count += 1
    return count


def materialize_native_zero(
    optimizers: list,
    *,
    torch: Any,
    adam_class: type,
    snapshot: Callable[[], dict],
) -> dict:
    """Call TE's allocation API, never step, and prove non-optimizer invariance.

    Core's pinned Adam init_state_fn calls initialize_state(p), whereas the
    pinned TE API requires store_param_remainders too. Support only these two
    explicit native signatures; never catch a failed call and retry it.
    """
    before = snapshot()
    signatures = []
    for optimizer in optimizers:
        raw = optimizer.optimizer
        if not isinstance(raw, adam_class):
            raise ValueError("not native TE Adam")
        signature = inspect.signature(raw.initialize_state)
        names = tuple(signature.parameters)
        if names not in (("param",), ("param", "store_param_remainders")):
            raise ValueError("unsupported TE initialization signature")
        signatures.append(str(signature))
        if (
            not optimizer.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
            or not raw.master_weights
            or any("step" in g for g in raw.param_groups)
            or any(raw.state.values())
        ):
            raise ValueError(
                "bootstrap optimizer is not untouched precision-aware Adam"
            )
        for group in raw.param_groups:
            for parameter in group["params"]:
                if len(names) == 2:
                    raw.initialize_state(
                        parameter,
                        store_param_remainders=(
                            raw.store_param_remainders
                            and parameter.dtype == torch.bfloat16
                        ),
                    )
                else:
                    raw.initialize_state(parameter)
    count = validate_native_zero(optimizers, torch=torch, adam_class=adam_class)
    # Exercise the exact extractor that failed before the expensive full save.
    for optimizer in optimizers:
        for parameter in optimizer.model_param_group_index_map:
            tensors = optimizer._get_main_param_and_optimizer_states(parameter)
            if set(tensors) != {"param", "exp_avg", "exp_avg_sq"}:
                raise ValueError("native checkpoint extraction incomplete")
    after = snapshot()
    if before != after:
        raise ValueError(
            "zero-state allocation changed weights, RNG, groups or counters"
        )
    return {
        "status": "NATIVE_ZERO_STATE_MATERIALIZED",
        "parameter_shards": count,
        "native_initialization_signatures": signatures,
        "native_adam_source_sha256": hashlib.sha256(
            Path(inspect.getfile(adam_class)).read_bytes()
        ).hexdigest(),
        "before": before,
        "after": after,
        "optimizer_updates": 0,
        "native_checkpoint_extractor_passed": True,
    }
