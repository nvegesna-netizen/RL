# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Build a local, no-training lock for the M4 opportunity-loss acquisition."""

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
from tools.opportunity_loss_pipeline import (
    OpportunityLossPipelineError,
    _parse_protocol,
)

IMAGE_PATH = (
    "/lustre/fsw/coreai_dlalgo_ci/nvegesna/nemo_rl_images/nemo-rl-nightly-5802754.sqsh"
)
IMAGE_SHA256 = "3df8114a0b3e60ef95ce13f8b982c7cc45d164aca71c63a87388b1e8434ee470"
IMAGE_COMMIT = "ae07eafe8035b5b2e84efa7234e70e7fd7e493c1"
M4_EVIDENCE_COMMIT = "a6be6971b79ad0d6b48c0dc85bb7f09682c2f7dd"
CONFIG_PATH = (
    "examples/configs/grpo_math_1B_megatron_single_controller_m4_opportunity_loss.yaml"
)
PIPELINE_PATH = "tools/opportunity_loss_pipeline.py"
PIPELINE_TEST_PATH = "tests/unit/tools/test_opportunity_loss_pipeline.py"
MECHANISM_PATH = "tools/opportunity_loss_mechanism.py"
MECHANISM_TEST_PATH = "tests/unit/tools/test_opportunity_loss_mechanism.py"


class OpportunityLossPreflightError(ValueError):
    """Raised when the frozen acquisition or preflight lock disagrees."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _lower_hex(value: str, *, length: int, name: str) -> str:
    if len(value) != length or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise OpportunityLossPreflightError(f"{name} must be lowercase hexadecimal")
    return value


def _at(value: Mapping[str, object], *path: str) -> object:
    current: object = value
    for component in path:
        if not isinstance(current, Mapping) or component not in current:
            raise OpportunityLossPreflightError(f"config lacks {'.'.join(path)}")
        current = current[component]
    return current


def validate_acquisition_config(config: Mapping[str, object]) -> None:
    """Require the exact registered acquisition geometry and output separation."""
    expected = {
        ("grpo", "max_num_steps"): 224,
        ("grpo", "num_prompts_per_step"): 4,
        ("grpo", "num_generations_per_prompt"): 8,
        ("grpo", "normalize_rewards"): True,
        ("grpo", "use_leave_one_out_baseline"): True,
        ("grpo", "adv_estimator", "name"): "grpo",
        ("grpo", "adv_estimator", "normalize_rewards"): True,
        ("grpo", "adv_estimator", "use_leave_one_out_baseline"): True,
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
        ("async_rl", "controlled_release_delay", "seed"): 20260810,
        (
            "async_rl",
            "controlled_release_delay",
            "assignment_domain",
        ): "m4-opportunity-loss-confirmatory-v1",
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
            raise OpportunityLossPreflightError(
                f"config field {'.'.join(path)} disagrees"
            )
    arms = _at(config, "async_rl", "controlled_release_delay", "arms")
    if arms != [
        {"label": "control", "delay_seconds": 0.0, "mass": 5},
        {"label": "d5", "delay_seconds": 5.0, "mass": 5},
        {"label": "d10", "delay_seconds": 10.0, "mass": 2},
    ]:
        raise OpportunityLossPreflightError("registered release arms disagree")
    paths = (
        _at(config, "async_rl", "lifecycle_audit_path"),
        _at(config, "async_rl", "gradient_opportunity_audit", "output_path"),
        _at(config, "async_rl", "gradient_opportunity_audit", "observer_duty_path"),
    )
    expected_paths = (
        "results/m4-opportunity-loss/lifecycle.jsonl",
        "results/m4-opportunity-loss/opportunity.jsonl",
        "results/m4-opportunity-loss/observer-duty.json",
    )
    if paths != expected_paths:
        raise OpportunityLossPreflightError("registered audit output paths disagree")
    master = MasterConfig(**config)
    validate_single_controller_config(master)


def build_preflight_lock(
    *,
    repo: Path,
    protocol: bytes,
    source_commit: str,
    source_archive_sha256: str,
) -> dict[str, object]:
    """Build value-only evidence; never call a trainer or submission client."""
    source_commit = _lower_hex(source_commit, length=40, name="source commit")
    source_archive_sha256 = _lower_hex(
        source_archive_sha256, length=64, name="source archive SHA256"
    )
    config_path = repo / CONFIG_PATH
    register_omegaconf_resolvers()
    resolved = OmegaConf.to_container(load_config(config_path), resolve=True)
    if not isinstance(resolved, Mapping):
        raise OpportunityLossPreflightError("resolved config must be an object")
    validate_acquisition_config(resolved)
    try:
        protocol_value, _, _ = _parse_protocol(protocol)
    except OpportunityLossPipelineError as error:
        raise OpportunityLossPreflightError("protocol is not strictly valid") from error
    if protocol_value.get("m4_evidence_commit") != M4_EVIDENCE_COMMIT:
        raise OpportunityLossPreflightError("protocol M4 evidence binding disagrees")
    files = {}
    for name in (
        CONFIG_PATH,
        PIPELINE_PATH,
        PIPELINE_TEST_PATH,
        MECHANISM_PATH,
        MECHANISM_TEST_PATH,
    ):
        raw = (repo / name).read_bytes()
        files[name] = {"sha256": _sha(raw), "size": len(raw)}
    return {
        "schema_version": 1,
        "record_type": "m4_opportunity_loss_no_training_preflight_lock",
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "files": files,
        "protocol": {"sha256": _sha(protocol), "size": len(protocol)},
        "image": {
            "path": IMAGE_PATH,
            "sha256": IMAGE_SHA256,
            "embedded_commit": IMAGE_COMMIT,
        },
        "training_allowed": False,
        "eos_submission_allowed": False,
        "m4_mechanism_replication": {
            "required": True,
            "accepted_predecessor_commit": M4_EVIDENCE_COMMIT,
            "current_acquisition_status": "PENDING",
        },
        "preflight_checks": [
            "validate_config",
            "verify_source_archive",
            "verify_image_identity",
            "run_selected_local_tests",
            "run_ruff_format_and_compile",
        ],
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
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_preflight_lock(
        repo=args.repo.resolve(),
        protocol=args.protocol.read_bytes(),
        source_commit=args.source_commit,
        source_archive_sha256=args.source_archive_sha256,
    )
    _write_canonical(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
