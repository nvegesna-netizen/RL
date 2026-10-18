"""Exact secondary definitions and boundary-only numerical helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def zero_step_if_uninitialized(
    inner_states: list[dict], *, expected_step: Any, scheduler_samples: int
) -> int:
    """Missing Adam step means zero only for an explicitly expected empty v0 state."""
    if type(expected_step) is not int or expected_step != 0 or scheduler_samples != 0:
        raise ValueError(
            "missing optimizer steps outside an explicit version-zero restore"
        )
    if not inner_states or any(
        set(s) != {"state", "param_groups"}
        or not isinstance(s["state"], dict)
        or s["state"]
        for s in inner_states
    ):
        raise ValueError("version-zero optimizer is not genuinely uninitialized")
    if any(not s["param_groups"] for s in inner_states):
        raise ValueError("version-zero optimizer has no parameter groups")
    return 0


def clipping_record(norm: float, threshold: float) -> dict:
    """Record the scalar min(1, threshold/(norm+epsilon)) clipping coefficient."""
    if (
        not math.isfinite(norm)
        or norm < 0
        or not math.isfinite(threshold)
        or threshold < 0
    ):
        raise ValueError("invalid native gradient norm or clip threshold")
    coefficient = min(1.0, threshold / (norm + 1e-6)) if threshold > 0 else 1.0
    return {
        "threshold": threshold,
        "coefficient": coefficient,
        "applied": coefficient < 1.0,
    }


def tensor_squared_difference(before: Any, after: Any, torch: Any) -> float:
    """Accumulate exact elementwise FP64 differences with bounded scratch memory."""
    if before.shape != after.shape or before.dtype != after.dtype:
        raise ValueError("parameter shape or dtype changed")
    left, right = before.reshape(-1), after.detach().cpu().contiguous().reshape(-1)
    sums = []
    for start in range(0, left.numel(), 1 << 20):
        delta = (
            right[start : start + (1 << 20)].double()
            - left[start : start + (1 << 20)].double()
        )
        sums.append(float(torch.sum(delta * delta).item()))
    result = math.fsum(sums)
    if not math.isfinite(result):
        raise ValueError("nonfinite parameter delta")
    return result


def gradient_rows(model: Any, torch: Any) -> list[tuple[str, Any]]:
    """Read the same main_grad used by the optimizer, once before clipping."""
    result = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        value = getattr(parameter, "main_grad", None)
        if value is None:
            value = parameter.grad
        if value is None:
            raise ValueError("trainable parameter lacks a gradient")
        result.append((name, value.detach().cpu().contiguous().clone()))
    if not result:
        raise ValueError("empty gradient diagnostic")
    return result


def exact_gradient_dot(
    left_path: Path, right_path: Path, torch: Any, *, expected_subject: dict
) -> dict:
    """Compare separately restored, pre-clipping gradients with no sketch estimator."""
    left = torch.load(left_path, weights_only=True, map_location="cpu")
    right = torch.load(right_path, weights_only=True, map_location="cpu")
    if left["subject"] != {**expected_subject, "role": "selected"} or right[
        "subject"
    ] != {**expected_subject, "role": "heldout"}:
        raise ValueError("gradient diagnostic assignment, arm, or role differs")
    if (
        left["state_before"] != right["state_before"]
        or left["parameter_sha256"] != right["parameter_sha256"]
    ):
        raise ValueError(
            "gradient diagnostics do not share the complete pre-update state"
        )
    a, b = left["gradients"], right["gradients"]
    if (
        not a
        or len({n for n, _ in a}) != len(a)
        or len(a) != len(b)
        or [n for n, _ in a] != [n for n, _ in b]
    ):
        raise ValueError("gradient parameter order differs")
    dot, square_a, square_b = [], [], []
    for (name, ga), (_, gb) in zip(a, b):
        if ga.shape != gb.shape or ga.dtype != gb.dtype:
            raise ValueError("gradient shape or dtype differs")
        aa, bb = ga.reshape(-1), gb.reshape(-1)
        for start in range(0, aa.numel(), 1 << 20):
            x, y = (
                aa[start : start + (1 << 20)].double(),
                bb[start : start + (1 << 20)].double(),
            )
            dot.append(float(torch.sum(x * y).item()))
            square_a.append(float(torch.sum(x * x).item()))
            square_b.append(float(torch.sum(y * y).item()))
    values = [
        math.fsum(dot),
        math.sqrt(math.fsum(square_a)),
        math.sqrt(math.fsum(square_b)),
    ]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("nonfinite gradient diagnostic")
    return {
        "heldout_gradient_dot_train_gradient": values[0],
        "train_batch_gradient_l2_norm_probe": values[1],
        "heldout_gradient_l2_norm": values[2],
        "state_before": left["state_before"],
        "exact": True,
    }


def update_kl(before: Any, after: Any, mask: Any, torch: Any) -> float:
    """K3 pre/post drift on frozen tokens, not an unbiased off-policy KL estimate."""
    if before.shape != after.shape or before.shape != mask.shape:
        raise ValueError("held-out logprob or mask shape differs")
    valid = mask.double()
    denominator = float(valid.sum().item())
    if denominator <= 0 or bool((valid < 0).any().item()):
        raise ValueError("invalid held-out token weights")
    ratio = after.double() - before.double()
    # No clamp or renormalization; nonfinite diagnostics fail rather than change the definition.
    value = float(((torch.expm1(ratio) - ratio) * valid).sum().item() / denominator)
    if not math.isfinite(value) or value < -1e-12:
        raise ValueError("nonfinite or invalid sampled update KL")
    return value
