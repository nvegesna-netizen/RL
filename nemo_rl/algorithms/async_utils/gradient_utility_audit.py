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

"""No-update gradient summaries for the M4 construct-validity audit."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_HASH_MULTIPLIER = 6364136223846793005
_HASH_INCREMENT = 1442695040888963407


@dataclass(frozen=True)
class GradientUtilitySummary:
    """Compact summary of one token-normalized group gradient."""

    exact_l2_norm: float
    coordinate_count: int
    nonzero_coordinate_count: int
    normalization_tokens: int
    sketches: tuple[tuple[float, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "exact_l2_norm": self.exact_l2_norm,
            "coordinate_count": self.coordinate_count,
            "nonzero_coordinate_count": self.nonzero_coordinate_count,
            "normalization_tokens": self.normalization_tokens,
            "sketches": [list(sketch) for sketch in self.sketches],
        }


def _parameter_gradient(parameter: torch.Tensor) -> torch.Tensor | None:
    main_grad = getattr(parameter, "main_grad", None)
    if isinstance(main_grad, torch.Tensor):
        return main_grad
    if isinstance(parameter.grad, torch.Tensor):
        return parameter.grad
    return None


def _countsketch_chunk(
    values: torch.Tensor,
    *,
    coordinate_offset: int,
    bins: int,
    seed: int,
) -> torch.Tensor:
    """Return one deterministic CountSketch contribution on ``values.device``."""
    indices = torch.arange(
        coordinate_offset,
        coordinate_offset + values.numel(),
        dtype=torch.int64,
        device=values.device,
    )
    seed_term = (seed * _HASH_INCREMENT) & ((1 << 63) - 1)
    mixed = indices * _HASH_MULTIPLIER + seed_term
    buckets = torch.remainder(mixed, bins)
    signs = torch.where(
        torch.bitwise_and(torch.bitwise_right_shift(mixed, 32), 1).bool(),
        -torch.ones((), dtype=torch.float64, device=values.device),
        torch.ones((), dtype=torch.float64, device=values.device),
    )
    result = torch.zeros(bins, dtype=torch.float64, device=values.device)
    result.scatter_add_(0, buckets, values.to(torch.float64) * signs)
    return result


@torch.no_grad()
def summarize_open_step_gradients(
    named_parameters: Iterable[tuple[str, torch.Tensor]],
    *,
    normalization_tokens: int,
    sketch_bins: int,
    sketch_seeds: Sequence[int],
    chunk_elements: int = 1 << 20,
) -> GradientUtilitySummary:
    """Summarize gradients without reducing, clipping, or changing them.

    The frozen acquisition uses one policy worker, so gradients are already the
    complete group gradient. Multi-worker callers must not use this helper until
    a source-bound distributed reduction contract exists.
    """
    if normalization_tokens < 1:
        raise ValueError("normalization_tokens must be positive")
    if sketch_bins < 2:
        raise ValueError("sketch_bins must be at least two")
    if not sketch_seeds or len(set(sketch_seeds)) != len(sketch_seeds):
        raise ValueError("sketch_seeds must be nonempty and unique")
    if chunk_elements < 1:
        raise ValueError("chunk_elements must be positive")

    sketches: list[torch.Tensor] | None = None
    exact_squared_norm = 0.0
    coordinate_count = 0
    nonzero_count = 0
    saw_gradient = False

    for name, parameter in named_parameters:
        if not name:
            raise ValueError("parameter names must be nonempty")
        gradient = _parameter_gradient(parameter)
        if gradient is None:
            continue
        if not bool(torch.isfinite(gradient).all().item()):
            raise ValueError(f"nonfinite gradient in parameter {name!r}")
        saw_gradient = True
        flat = gradient.detach().reshape(-1)
        if sketches is None:
            sketches = [
                torch.zeros(sketch_bins, dtype=torch.float64, device=flat.device)
                for _ in sketch_seeds
            ]
        elif any(sketch.device != flat.device for sketch in sketches):
            raise ValueError("all gradients must be resident on one device")

        for start in range(0, flat.numel(), chunk_elements):
            raw_chunk = flat[start : start + chunk_elements]
            chunk = raw_chunk.to(torch.float64) / float(normalization_tokens)
            exact_squared_norm += float(torch.sum(chunk * chunk).item())
            nonzero_count += int(torch.count_nonzero(raw_chunk).item())
            for sketch, seed in zip(sketches, sketch_seeds, strict=True):
                sketch.add_(
                    _countsketch_chunk(
                        chunk,
                        coordinate_offset=coordinate_count + start,
                        bins=sketch_bins,
                        seed=int(seed),
                    )
                )
        coordinate_count += flat.numel()

    if not saw_gradient or sketches is None:
        raise RuntimeError("the open train step produced no parameter gradients")
    exact_l2_norm = math.sqrt(exact_squared_norm)
    if not math.isfinite(exact_l2_norm):
        raise ValueError("exact gradient norm is nonfinite")

    cpu_sketches = tuple(
        tuple(float(value) for value in sketch.cpu().tolist()) for sketch in sketches
    )
    return GradientUtilitySummary(
        exact_l2_norm=exact_l2_norm,
        coordinate_count=coordinate_count,
        nonzero_coordinate_count=nonzero_count,
        normalization_tokens=normalization_tokens,
        sketches=cpu_sketches,
    )


@torch.no_grad()
def hash_model_parameters(
    named_parameters: Iterable[tuple[str, torch.Tensor]],
    *,
    chunk_bytes: int = 16 << 20,
) -> str:
    """Return a canonical SHA-256 over names, metadata, and parameter bytes."""
    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    count = 0
    for name, parameter in named_parameters:
        if not name:
            raise ValueError("parameter names must be nonempty")
        tensor = parameter.detach().contiguous()
        header = json.dumps(
            {"dtype": str(tensor.dtype), "name": name, "shape": list(tensor.shape)},
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        byte_view = tensor.view(torch.uint8).reshape(-1)
        for start in range(0, byte_view.numel(), chunk_bytes):
            payload = byte_view[start : start + chunk_bytes].cpu().numpy().tobytes()
            digest.update(payload)
        count += 1
    if count == 0:
        raise RuntimeError("model has no named parameters")
    return digest.hexdigest()


class GradientUtilityAuditRecorder:
    """Canonical in-memory JSONL recorder for a bounded no-update audit."""

    SCHEMA_VERSION = 1

    def __init__(self, *, sketch_bins: int, sketch_seeds: Sequence[int]) -> None:
        self._events: list[dict[str, Any]] = [
            {
                "event_type": "header",
                "schema_version": self.SCHEMA_VERSION,
                "sketch_bins": sketch_bins,
                "sketch_seeds": list(sketch_seeds),
                "training_updates_authorized": False,
            }
        ]

    @property
    def events(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._events)

    def append_group(
        self,
        *,
        audit_index: int,
        group_id: str,
        sample_ids: Sequence[str],
        metadata: Mapping[str, Any],
        summary: Mapping[str, Any],
    ) -> None:
        if audit_index != len(self._events) - 1:
            raise ValueError("audit indices must be contiguous and zero-based")
        if not group_id or not sample_ids or len(sample_ids) != len(set(sample_ids)):
            raise ValueError("group and sample identities must be nonempty and unique")
        self._events.append(
            {
                "event_type": "group_gradient",
                "audit_index": audit_index,
                "fold": audit_index % 8,
                "group_id": group_id,
                "sample_ids": list(sample_ids),
                "metadata": dict(metadata),
                "summary": dict(summary),
                "optimizer_step": False,
                "scheduler_step": False,
                "abort_acknowledged": True,
            }
        )

    def append_terminal(
        self,
        *,
        parameter_sha256_before: str,
        parameter_sha256_after: str,
        groups: int,
        finish_train_step_calls: int,
    ) -> None:
        if groups != len(self._events) - 1:
            raise ValueError("terminal group count does not match recorded groups")
        self._events.append(
            {
                "event_type": "terminal",
                "groups": groups,
                "finish_train_step_calls": finish_train_step_calls,
                "learner_version": 0,
                "optimizer_steps": 0,
                "scheduler_steps": 0,
                "parameter_sha256_before": parameter_sha256_before,
                "parameter_sha256_after": parameter_sha256_after,
                "parameter_hash_unchanged": (
                    parameter_sha256_before == parameter_sha256_after
                ),
            }
        )

    def flush_jsonl(self, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(
            json.dumps(
                event,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
            for event in self._events
        )
        path.write_text(payload, encoding="utf-8")
