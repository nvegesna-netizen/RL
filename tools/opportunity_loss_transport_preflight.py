# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Build a no-training lock for the M4 Qwen3-1.7B neutral qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    validate_single_controller_config,
)
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

CONFIG_PATH = (
    "examples/configs/"
    "grpo_math_1B_megatron_single_controller_m4_qwen3_1p7b_neutral_qualification.yaml"
)
AUDIT_PATH = (
    "reports/auto_research/2026-09-03-m4-opportunity-loss-transport/"
    "qwen3-1p7b-compatibility-audit/audit.json"
)
PINNED_FILES = (
    CONFIG_PATH,
    AUDIT_PATH,
    "nemo_rl/algorithms/async_utils/controlled_release.py",
    "nemo_rl/algorithms/single_controller.py",
    "nemo_rl/algorithms/single_controller_utils/config.py",
    "nemo_rl/experience/rollout_manager.py",
    "tools/opportunity_loss_transport_preflight.py",
    "tests/unit/tools/test_opportunity_loss_transport_preflight.py",
)


class TransportPreflightError(ValueError):
    """Raised when the neutral qualification configuration disagrees."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _at(value: Mapping[str, object], *path: str) -> object:
    current: object = value
    for component in path:
        if not isinstance(current, Mapping) or component not in current:
            raise TransportPreflightError(f"config lacks {'.'.join(path)}")
        current = current[component]
    return current


def _lower_hex(value: str, *, length: int, name: str) -> str:
    if len(value) != length or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise TransportPreflightError(f"{name} must be lowercase hexadecimal")
    return value


def validate_neutral_qualification_config(config: Mapping[str, object]) -> None:
    expected = {
        ("grpo", "max_num_steps"): 32,
        ("grpo", "num_prompts_per_step"): 4,
        ("grpo", "num_generations_per_prompt"): 8,
        ("grpo", "normalize_rewards"): True,
        ("grpo", "use_leave_one_out_baseline"): True,
        ("policy", "model_name"): "Qwen/Qwen3-1.7B",
        ("policy", "tokenizer", "name"): "Qwen/Qwen3-1.7B",
        ("policy", "train_global_batch_size"): 32,
        ("policy", "train_micro_batch_size"): 1,
        ("policy", "max_total_sequence_length"): 2048,
        ("policy", "generation", "max_new_tokens"): 1792,
        ("async_rl", "sampler", "name"): "windowed",
        ("async_rl", "sampler", "max_staleness_versions"): 1,
        ("async_rl", "sampler", "sample_freshest_first"): False,
        ("async_rl", "min_groups_for_streaming_train"): 4,
        ("async_rl", "max_inflight_prompts"): 16,
        ("async_rl", "max_buffered_rollouts"): 64,
        ("async_rl", "controlled_release_delay", "enabled"): True,
        ("async_rl", "controlled_release_delay", "seed"): 20260906,
        ("async_rl", "controlled_release_delay", "assignment_domain"): (
            "m4-opportunity-loss-qwen3-1p7b-neutral-qualification-v1"
        ),
        ("async_rl", "gradient_opportunity_audit", "enabled"): True,
        ("loss_fn", "disable_ppo_ratio"): False,
        ("loss_fn", "positive_example_nll_weight"): 0.0,
        ("loss_fn", "sequence_level_importance_ratios"): False,
        ("loss_fn", "token_level_loss"): True,
        ("loss_fn", "use_cispo"): False,
        ("checkpointing", "enabled"): False,
        ("logger", "wandb_enabled"): False,
        ("logger", "tensorboard_enabled"): False,
        ("logger", "mlflow_enabled"): False,
        ("logger", "swanlab_enabled"): False,
        ("logger", "monitor_gpus"): False,
        ("data_plane", "enabled"): True,
        ("data_plane", "impl"): "transfer_queue",
        ("data_plane", "backend"): "simple",
        ("data", "train", "dataset_name"): "OpenMathInstruct-2",
        ("data", "default", "env_name"): "math",
        ("cluster", "gpus_per_node"): 2,
        ("cluster", "num_nodes"): 1,
    }
    for path, expected_value in expected.items():
        if _at(config, *path) != expected_value:
            raise TransportPreflightError(f"config field {'.'.join(path)} disagrees")
    arms = _at(config, "async_rl", "controlled_release_delay", "arms")
    if arms != [{"label": "neutral", "delay_seconds": 0.0, "mass": 1}]:
        raise TransportPreflightError("qualification must have one zero-dose arm")
    paths = (
        _at(config, "async_rl", "lifecycle_audit_path"),
        _at(config, "async_rl", "gradient_opportunity_audit", "output_path"),
        _at(config, "async_rl", "gradient_opportunity_audit", "observer_duty_path"),
    )
    if (
        paths
        != (
            "results/m4-qwen3-1p7b-neutral-qualification/lifecycle.jsonl",
            "results/m4-qwen3-1p7b-neutral-qualification/opportunity.jsonl",
            "results/m4-qwen3-1p7b-neutral-qualification/observer-duty.json",
        )
        or len(set(paths)) != 3
    ):
        raise TransportPreflightError("qualification output paths disagree")
    validate_single_controller_config(MasterConfig(**config))


def build_transport_preflight_lock(
    *, repo: Path, source_commit: str, source_archive_sha256: str
) -> dict[str, object]:
    source_commit = _lower_hex(source_commit, length=40, name="source commit")
    source_archive_sha256 = _lower_hex(
        source_archive_sha256, length=64, name="source archive SHA256"
    )
    register_omegaconf_resolvers()
    resolved = OmegaConf.to_container(load_config(repo / CONFIG_PATH), resolve=True)
    if not isinstance(resolved, Mapping):
        raise TransportPreflightError("resolved config must be an object")
    validate_neutral_qualification_config(resolved)
    audit_raw = (repo / AUDIT_PATH).read_bytes()
    audit = json.loads(audit_raw)
    if audit.get("schema") != "m4-opportunity-loss-qwen3-1p7b-compatibility-audit-v1":
        raise TransportPreflightError("compatibility audit schema disagrees")
    decision = audit.get("decision")
    if not isinstance(decision, Mapping) or decision.get("status") != "CONDITIONAL_GO":
        raise TransportPreflightError("compatibility audit decision disagrees")
    if decision.get("acquisition_authorized_by_this_audit") is not False:
        raise TransportPreflightError("compatibility audit grants acquisition")
    if decision.get("gpu_qualification_authorized_by_this_audit") is not False:
        raise TransportPreflightError("compatibility audit grants qualification")
    files = {}
    for name in PINNED_FILES:
        raw = (repo / name).read_bytes()
        files[name] = {"sha256": _sha(raw), "size": len(raw)}
    return {
        "schema": "m4-opportunity-loss-qwen3-1p7b-neutral-preflight-lock-v1",
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "files": files,
        "compatibility_audit": {"sha256": _sha(audit_raw), "size": len(audit_raw)},
        "qualification": {
            "model": "Qwen/Qwen3-1.7B",
            "dataset": "OpenMathInstruct-2",
            "arms": [{"label": "neutral", "delay_seconds": 0.0, "mass": 1}],
            "trainer_steps": 32,
            "candidate_gpus_per_node": 2,
            "scientific_acquisition": False,
        },
        "training_allowed": False,
        "eos_submission_allowed": False,
        "qualification_submission_allowed": False,
        "scientific_acquisition_allowed": False,
        "automatic_retry": False,
    }


def _write_canonical(path: Path, value: Mapping[str, object]) -> None:
    raw = (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_transport_preflight_lock(
        repo=args.repo.resolve(),
        source_commit=args.source_commit,
        source_archive_sha256=args.source_archive_sha256,
    )
    _write_canonical(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
