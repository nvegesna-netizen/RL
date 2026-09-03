# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Build a no-training lock for the Qwen3-1.7B M4 transport acquisition."""

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
from tools.opportunity_loss_transport_pipeline import _validate_transport_contract

IMAGE_PATH = (
    "/lustre/fsw/coreai_dlalgo_ci/nvegesna/nemo_rl_images/nemo-rl-nightly-5802754.sqsh"
)
IMAGE_SHA256 = "3df8114a0b3e60ef95ce13f8b982c7cc45d164aca71c63a87388b1e8434ee470"
IMAGE_COMMIT = "ae07eafe8035b5b2e84efa7234e70e7fd7e493c1"
CONFIG_PATH = (
    "examples/configs/"
    "grpo_math_1B_megatron_single_controller_m4_qwen3_1p7b_transport.yaml"
)
PROTOCOL_PATH = (
    "reports/auto_research/2026-09-03-m4-opportunity-loss-transport/"
    "qwen3-1p7b-confirmatory-design/protocol_config.json"
)
QUALIFICATION_RESULT_PATH = (
    "reports/auto_research/2026-09-03-m4-opportunity-loss-transport/"
    "qwen3-1p7b-compatibility-audit/qualification_result.json"
)
QUALIFICATION_RESULT_SHA256 = (
    "fbcef5983af19e5149e88834cfe0ebee101affbc92aaccfe987e86e081c91dee"
)
QUALIFICATION_ARTIFACT_SHA256 = (
    "5cf1acb1729abb104497fa78740b9cc329e4532dedb7e8ecb6237b099004ee51"
)
QUALIFICATION_SUMMARY_SHA256 = (
    "f356d8bc327e83b4818f04244d57bc2f261aef87dc5439e4f83ea391cbbed344"
)
ACCEPTED_FOLLOWUP_ARTIFACT_SHA256 = (
    "07cddc489ea20055161f9c435d44d0a52b91113fda9eac16472bdb531dc6ffb6"
)
ACCEPTED_FOLLOWUP_POWER_SHA256 = (
    "96901dac5392923d36b03d7a27f74c466c6904b4c5325031e7ab018a38f354c9"
)
PINNED_FILES = (
    CONFIG_PATH,
    PROTOCOL_PATH,
    QUALIFICATION_RESULT_PATH,
    "tools/opportunity_loss_adjusted_inference.py",
    "tools/opportunity_loss_inference.py",
    "tools/opportunity_loss_mechanism.py",
    "tools/opportunity_loss_pipeline.py",
    "tools/opportunity_loss_transport_pipeline.py",
    "tools/opportunity_loss_transport_confirmatory_preflight.py",
    "tests/unit/tools/test_opportunity_loss_adjusted_inference.py",
    "tests/unit/tools/test_opportunity_loss_inference.py",
    "tests/unit/tools/test_opportunity_loss_mechanism.py",
    "tests/unit/tools/test_opportunity_loss_transport_protocol.py",
    "tests/unit/tools/test_opportunity_loss_transport_confirmatory_preflight.py",
)


class TransportConfirmatoryPreflightError(ValueError):
    """Raised when the frozen transport design disagrees."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _at(value: Mapping[str, object], *path: str) -> object:
    current: object = value
    for component in path:
        if not isinstance(current, Mapping) or component not in current:
            raise TransportConfirmatoryPreflightError(f"config lacks {'.'.join(path)}")
        current = current[component]
    return current


def _lower_hex(value: str, *, length: int, name: str) -> str:
    if len(value) != length or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise TransportConfirmatoryPreflightError(
            f"{name} must be lowercase hexadecimal"
        )
    return value


def validate_transport_config(config: Mapping[str, object]) -> None:
    expected = {
        ("grpo", "max_num_steps"): 558,
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
        ("policy", "generation", "colocated", "enabled"): False,
        ("policy", "generation", "colocated", "resources", "gpus_per_node"): 1,
        ("policy", "generation", "colocated", "resources", "num_nodes"): 1,
        ("async_rl", "sampler", "name"): "windowed",
        ("async_rl", "sampler", "max_staleness_versions"): 1,
        ("async_rl", "sampler", "sample_freshest_first"): False,
        ("async_rl", "min_groups_for_streaming_train"): 4,
        ("async_rl", "max_inflight_prompts"): 16,
        ("async_rl", "max_buffered_rollouts"): 64,
        ("async_rl", "controlled_release_delay", "enabled"): True,
        ("async_rl", "controlled_release_delay", "seed"): 20260907,
        ("async_rl", "controlled_release_delay", "assignment_domain"): (
            "m4-opportunity-loss-qwen3-1p7b-transport-v1"
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
            raise TransportConfirmatoryPreflightError(
                f"config field {'.'.join(path)} disagrees"
            )
    if _at(config, "async_rl", "controlled_release_delay", "arms") != [
        {"label": "control", "delay_seconds": 0.0, "mass": 1},
        {"label": "d5", "delay_seconds": 5.0, "mass": 1},
    ]:
        raise TransportConfirmatoryPreflightError("transport arms disagree")
    paths = (
        _at(config, "async_rl", "lifecycle_audit_path"),
        _at(config, "async_rl", "gradient_opportunity_audit", "output_path"),
        _at(config, "async_rl", "gradient_opportunity_audit", "observer_duty_path"),
    )
    if (
        paths
        != (
            "results/m4-qwen3-1p7b-transport/lifecycle.jsonl",
            "results/m4-qwen3-1p7b-transport/opportunity.jsonl",
            "results/m4-qwen3-1p7b-transport/observer-duty.json",
        )
        or len(set(paths)) != 3
    ):
        raise TransportConfirmatoryPreflightError("transport output paths disagree")
    validate_single_controller_config(MasterConfig(**config))


def _validate_design_evidence(raw: Mapping[str, object]) -> None:
    expected = {
        "accepted_followup_artifact_sha256": ACCEPTED_FOLLOWUP_ARTIFACT_SHA256,
        "accepted_followup_power_at_delta_l_0_25": 0.81555,
        "accepted_followup_power_sha256": ACCEPTED_FOLLOWUP_POWER_SHA256,
        "conservative_assignments_per_primary_version": 15.0,
        "minimum_primary_assignments": 7395,
        "preferred_primary_assignments": 7500,
        "qualification_artifact_sha256": QUALIFICATION_ARTIFACT_SHA256,
        "qualification_result_sha256": QUALIFICATION_RESULT_SHA256,
        "qualification_summary_sha256": QUALIFICATION_SUMMARY_SHA256,
        "transport_power_assumed_equal_to_followup": False,
    }
    if raw.get("design_evidence") != expected:
        raise TransportConfirmatoryPreflightError("design evidence disagrees")


def build_transport_confirmatory_preflight_lock(
    *, repo: Path, source_commit: str, source_archive_sha256: str
) -> dict[str, object]:
    source_commit = _lower_hex(source_commit, length=40, name="source commit")
    source_archive_sha256 = _lower_hex(
        source_archive_sha256, length=64, name="source archive SHA256"
    )
    register_omegaconf_resolvers()
    resolved = OmegaConf.to_container(load_config(repo / CONFIG_PATH), resolve=True)
    if not isinstance(resolved, Mapping):
        raise TransportConfirmatoryPreflightError("resolved config must be an object")
    validate_transport_config(resolved)
    protocol_raw = (repo / PROTOCOL_PATH).read_bytes()
    try:
        protocol_value, protocol, options = _parse_protocol(protocol_raw)
        _validate_transport_contract(protocol_value, protocol, options)
    except OpportunityLossPipelineError as error:
        raise TransportConfirmatoryPreflightError(
            "transport protocol is invalid"
        ) from error
    _validate_design_evidence(protocol_value)
    if protocol_value.get("status") != (
        "FROZEN_LOCAL_PROTOCOL_PENDING_NO_TRAINING_PREFLIGHT"
    ):
        raise TransportConfirmatoryPreflightError("protocol status disagrees")
    qualification_raw = (repo / QUALIFICATION_RESULT_PATH).read_bytes()
    if _sha(qualification_raw) != QUALIFICATION_RESULT_SHA256:
        raise TransportConfirmatoryPreflightError("qualification result moved")
    qualification = json.loads(qualification_raw)
    if (
        qualification.get("gate", {}).get("decision")
        != "GO_TO_PROSPECTIVE_CONFIRMATORY_DESIGN"
        or qualification.get("gate", {}).get("acquisition_authorized") is not False
        or qualification.get("artifact", {}).get("sha256")
        != QUALIFICATION_ARTIFACT_SHA256
        or qualification.get("qualification", {}).get("qualification_complete")
        is not True
        or qualification.get("qualification", {}).get("resource_fit") is not True
    ):
        raise TransportConfirmatoryPreflightError("qualification gate disagrees")
    files = {}
    for name in PINNED_FILES:
        raw = (repo / name).read_bytes()
        files[name] = {"sha256": _sha(raw), "size": len(raw)}
    return {
        "schema": "m4-qwen3-1p7b-transport-no-training-lock-v1",
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
            "gpus": 2,
            "primary_versions": 500,
            "preferred_primary_assignments": 7500,
            "trainer_steps": 558,
            "wall_clock_cap_hours": 6.0,
            "gpu_hour_cap": 12.0,
            "qualification_artifact_sha256": QUALIFICATION_ARTIFACT_SHA256,
        },
        "training_allowed": False,
        "scientific_acquisition_allowed": False,
        "eos_submission_allowed": False,
        "automatic_retry": False,
        "automatic_extension": False,
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
    result = build_transport_confirmatory_preflight_lock(
        repo=args.repo.resolve(),
        source_commit=args.source_commit,
        source_archive_sha256=args.source_archive_sha256,
    )
    _write_canonical(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
