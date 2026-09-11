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
"""Atomic, non-resumable terminal policy export support."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def export_terminal_policy(
    *,
    trainer: Any,
    master_config: Any,
    output_dir: str,
    train_steps: int,
    trainer_version: int,
) -> dict[str, Any]:
    """Export weights and resolved config at an exact terminal boundary.

    The destination and its ``.incomplete`` sibling must both be absent.  The
    incomplete directory is retained on failure for diagnosis; successful
    exports are atomically renamed into place after asynchronous saves finish.
    """
    expected_steps = master_config.grpo.max_num_steps
    if train_steps != expected_steps or trainer_version != expected_steps:
        raise RuntimeError(
            "terminal policy export requires the configured train-step/version "
            f"boundary: steps={train_steps} version={trainer_version} "
            f"expected={expected_steps}"
        )

    destination = Path(output_dir)
    incomplete = destination.with_name(f".{destination.name}.incomplete")
    if destination.exists() or incomplete.exists():
        raise FileExistsError(
            "terminal policy export refuses to overwrite an existing path: "
            f"output={destination} temporary={incomplete}"
        )
    incomplete.parent.mkdir(parents=True, exist_ok=True)
    incomplete.mkdir()

    weights_path = incomplete / "policy" / "weights"
    trainer.save_checkpoint(
        weights_path=os.fspath(weights_path),
        optimizer_path=None,
        tokenizer_path=None,
    )
    trainer.finalize_async_save()

    resolved_config_path = incomplete / "resolved_config.json"
    resolved_config_path.write_text(
        json.dumps(
            master_config.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema": "single-controller-terminal-policy-export-v1",
        "train_steps": train_steps,
        "trainer_version": trainer_version,
        "weights_path": "policy/weights",
        "resolved_config_path": "resolved_config.json",
        "optimizer_exported": False,
        "resumable_training_checkpoint": False,
    }
    (incomplete / "terminal_policy_export.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    incomplete.rename(destination)
    return manifest
