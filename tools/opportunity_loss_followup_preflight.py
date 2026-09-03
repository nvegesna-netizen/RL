# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Build a no-training lock for the prospective M4 opportunity-loss follow-up."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from tools.opportunity_loss_followup_pipeline import _validate_followup_contract
from tools.opportunity_loss_pipeline import (
    OpportunityLossPipelineError,
    _parse_protocol,
)

IMAGE_PATH = (
    "/lustre/fsw/coreai_dlalgo_ci/nvegesna/nemo_rl_images/nemo-rl-nightly-5802754.sqsh"
)
IMAGE_SHA256 = "3df8114a0b3e60ef95ce13f8b982c7cc45d164aca71c63a87388b1e8434ee470"
IMAGE_COMMIT = "ae07eafe8035b5b2e84efa7234e70e7fd7e493c1"
CONFIG_PATH = (
    "examples/configs/"
    "grpo_math_1B_megatron_single_controller_m4_opportunity_loss_followup.yaml"
)
PROTOCOL_PATH = (
    "reports/auto_research/2026-09-02-m4-opportunity-loss-followup/protocol_config.json"
)
DESIGN_ARTIFACT_SHA256 = (
    "e81a29747628b6f0490323207f317755de411d0f8e7527f715368919aee12fb7"
)
EMPIRICAL_POWER_SHA256 = (
    "96901dac5392923d36b03d7a27f74c466c6904b4c5325031e7ab018a38f354c9"
)
PINNED_FILES = (
    CONFIG_PATH,
    PROTOCOL_PATH,
    "tools/opportunity_loss_adjusted_inference.py",
    "tools/opportunity_loss_followup_pipeline.py",
    "tools/opportunity_loss_inference.py",
    "tools/opportunity_loss_mechanism.py",
    "tools/opportunity_loss_pipeline.py",
    "tests/unit/tools/test_opportunity_loss_adjusted_inference.py",
    "tests/unit/tools/test_opportunity_loss_followup_protocol.py",
    "tests/unit/tools/test_opportunity_loss_inference.py",
    "tests/unit/tools/test_opportunity_loss_mechanism.py",
)


class FollowupPreflightError(ValueError):
    """Raised when the frozen follow-up configuration disagrees."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _at(value: Mapping[str, object], *path: str) -> object:
    current: object = value
    for component in path:
        if not isinstance(current, Mapping) or component not in current:
            raise FollowupPreflightError(f"config lacks {'.'.join(path)}")
        current = current[component]
    return current


def _lower_hex(value: str, *, length: int, name: str) -> str:
    if len(value) != length or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise FollowupPreflightError(f"{name} must be lowercase hexadecimal")
    return value


def validate_followup_config(config: Mapping[str, object]) -> None:
    expected = {
        ("grpo", "max_num_steps"): 448,
        ("grpo", "num_prompts_per_step"): 4,
        ("grpo", "num_generations_per_prompt"): 8,
        ("grpo", "normalize_rewards"): True,
        ("grpo", "use_leave_one_out_baseline"): True,
        ("policy", "model_name"): "Qwen/Qwen3-0.6B",
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
        ("async_rl", "controlled_release_delay", "seed"): 20260905,
        ("async_rl", "controlled_release_delay", "assignment_domain"): (
            "m4-opportunity-loss-followup-v1"
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
            raise FollowupPreflightError(f"config field {'.'.join(path)} disagrees")
    if _at(config, "async_rl", "controlled_release_delay", "arms") != [
        {"label": "control", "delay_seconds": 0.0, "mass": 1},
        {"label": "d5", "delay_seconds": 5.0, "mass": 1},
    ]:
        raise FollowupPreflightError("follow-up arms disagree")
    paths = (
        _at(config, "async_rl", "lifecycle_audit_path"),
        _at(config, "async_rl", "gradient_opportunity_audit", "output_path"),
        _at(config, "async_rl", "gradient_opportunity_audit", "observer_duty_path"),
    )
    if (
        paths
        != (
            "results/m4-opportunity-loss-followup/lifecycle.jsonl",
            "results/m4-opportunity-loss-followup/opportunity.jsonl",
            "results/m4-opportunity-loss-followup/observer-duty.json",
        )
        or len(set(paths)) != 3
    ):
        raise FollowupPreflightError("follow-up output paths disagree")
    validate_single_controller_config(MasterConfig(**config))


def build_followup_preflight_lock(
    *, repo: Path, source_commit: str, source_archive_sha256: str
) -> dict[str, object]:
    source_commit = _lower_hex(source_commit, length=40, name="source commit")
    source_archive_sha256 = _lower_hex(
        source_archive_sha256, length=64, name="source archive SHA256"
    )
    register_omegaconf_resolvers()
    resolved = OmegaConf.to_container(load_config(repo / CONFIG_PATH), resolve=True)
    if not isinstance(resolved, Mapping):
        raise FollowupPreflightError("resolved config must be an object")
    validate_followup_config(resolved)
    protocol_raw = (repo / PROTOCOL_PATH).read_bytes()
    try:
        protocol_value, protocol, options = _parse_protocol(protocol_raw)
        _validate_followup_contract(protocol, options)
    except OpportunityLossPipelineError as error:
        raise FollowupPreflightError("follow-up protocol is invalid") from error
    evidence = protocol_value.get("design_evidence")
    if not isinstance(evidence, Mapping) or evidence != {
        "adjusted_inference_commit": "1bf0eb89666128139863ef09acd14dd8313d9b48",
        "empirical_power_draws": 20000,
        "empirical_power_sha256": EMPIRICAL_POWER_SHA256,
        "power_at_delta_l_0_25": 0.81555,
        "planning_variance_reduction": 0.5,
        "source_artifact_sha256": DESIGN_ARTIFACT_SHA256,
    }:
        raise FollowupPreflightError("design evidence disagrees")
    if not math.isclose(float(evidence["power_at_delta_l_0_25"]), 0.81555):
        raise FollowupPreflightError("power evidence disagrees")
    files = {}
    for name in PINNED_FILES:
        raw = (repo / name).read_bytes()
        files[name] = {"sha256": _sha(raw), "size": len(raw)}
    return {
        "schema": "m4-opportunity-loss-followup-no-training-lock-v1",
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "files": files,
        "protocol": {"sha256": _sha(protocol_raw), "size": len(protocol_raw)},
        "image": {
            "path": IMAGE_PATH,
            "sha256": IMAGE_SHA256,
            "embedded_commit": IMAGE_COMMIT,
        },
        "design": {
            "arms": ["control", "d5"],
            "primary_versions": 400,
            "trainer_steps": 448,
            "empirical_power_at_0_25": 0.81555,
            "empirical_power_sha256": EMPIRICAL_POWER_SHA256,
        },
        "training_allowed": False,
        "eos_submission_allowed": False,
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
    result = build_followup_preflight_lock(
        repo=args.repo.resolve(),
        source_commit=args.source_commit,
        source_archive_sha256=args.source_archive_sha256,
    )
    _write_canonical(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
