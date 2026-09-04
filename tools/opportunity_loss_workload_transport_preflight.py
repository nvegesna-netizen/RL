# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Build a no-training lock for the Qwen3-1.7B GSM8K M4 transport study."""

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
from tools.opportunity_loss_workload_transport_pipeline import (
    validate_workload_transport_contract,
)


IMAGE_PATH = (
    "/lustre/fsw/coreai_dlalgo_ci/nvegesna/nemo_rl_images/nemo-rl-nightly-5802754.sqsh"
)
IMAGE_SHA256 = "3df8114a0b3e60ef95ce13f8b982c7cc45d164aca71c63a87388b1e8434ee470"
IMAGE_COMMIT = "ae07eafe8035b5b2e84efa7234e70e7fd7e493c1"
CONFIG_PATH = (
    "examples/configs/"
    "grpo_math_1B_megatron_single_controller_m4_qwen3_1p7b_gsm8k_transport.yaml"
)
PROTOCOL_PATH = (
    "reports/auto_research/2026-09-04-m4-opportunity-loss-workload-transport/"
    "qwen3-1p7b-gsm8k-confirmatory-design/protocol_config.json"
)
AUDIT_PATH = (
    "reports/auto_research/2026-09-04-m4-opportunity-loss-workload-transport/"
    "qwen3-1p7b-gsm8k-compatibility-audit/audit.json"
)
ACCEPTED_RESULT_PATH = (
    "reports/auto_research/2026-09-03-m4-opportunity-loss-transport/"
    "qwen3-1p7b-confirmatory-design/acquisition_result.json"
)
AUDIT_SHA256 = "6a334580a3771c571bc360e804929a3d61351e8b092747566729d6465d71ffc7"
ACCEPTED_RESULT_SHA256 = (
    "7eec5871da27004eb665c7662abe813ff990c1a8cdad4a45773af24d0b8f7ade"
)
PINNED_FILES = (
    CONFIG_PATH,
    PROTOCOL_PATH,
    AUDIT_PATH,
    ACCEPTED_RESULT_PATH,
    (
        "reports/auto_research/2026-09-04-m4-opportunity-loss-workload-transport/"
        "qwen3-1p7b-gsm8k-confirmatory-design/design.md"
    ),
    (
        "reports/auto_research/2026-09-04-m4-opportunity-loss-workload-transport/"
        "qwen3-1p7b-gsm8k-confirmatory-design/preflight_plan.md"
    ),
    "nemo_rl/algorithms/async_utils/controlled_release.py",
    "nemo_rl/algorithms/async_utils/gradient_opportunity.py",
    "nemo_rl/algorithms/async_utils/observer_duty.py",
    "nemo_rl/algorithms/async_utils/rollout_lifecycle.py",
    "nemo_rl/algorithms/single_controller.py",
    "nemo_rl/algorithms/single_controller_utils/config.py",
    "nemo_rl/data/datasets/response_datasets/gsm8k.py",
    "nemo_rl/data/processors.py",
    "nemo_rl/environments/math_environment.py",
    "nemo_rl/experience/rollout_manager.py",
    "tools/opportunity_loss_adjusted_inference.py",
    "tools/opportunity_loss_inference.py",
    "tools/opportunity_loss_mechanism.py",
    "tools/opportunity_loss_pipeline.py",
    "tools/opportunity_loss_workload_transport_pipeline.py",
    "tools/opportunity_loss_workload_transport_preflight.py",
    "tests/unit/data/datasets/test_response_dataset.py",
    "tests/unit/single_controller/test_controlled_release.py",
    "tests/unit/single_controller/test_gradient_opportunity.py",
    "tests/unit/single_controller/test_observer_duty.py",
    "tests/unit/tools/test_observer_duty_analysis.py",
    "tests/unit/tools/test_opportunity_loss_adjusted_inference.py",
    "tests/unit/tools/test_opportunity_loss_inference.py",
    "tests/unit/tools/test_opportunity_loss_mechanism.py",
    "tests/unit/tools/test_opportunity_loss_pipeline.py",
    "tests/unit/tools/test_opportunity_loss_gsm8k_transport_protocol.py",
    "tests/unit/tools/test_opportunity_loss_workload_transport_preflight.py",
)


class WorkloadTransportPreflightError(ValueError):
    """Raised when the frozen GSM8K workload-transport design disagrees."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _at(value: Mapping[str, object], *path: str) -> object:
    current: object = value
    for component in path:
        if not isinstance(current, Mapping) or component not in current:
            raise WorkloadTransportPreflightError(f"config lacks {'.'.join(path)}")
        current = current[component]
    return current


def _lower_hex(value: str, *, length: int, name: str) -> str:
    if len(value) != length or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise WorkloadTransportPreflightError(f"{name} must be lowercase hexadecimal")
    return value


def validate_workload_transport_config(config: Mapping[str, object]) -> None:
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
        (
            "policy",
            "generation",
            "colocated",
            "resources",
            "gpus_per_node",
        ): 1,
        ("policy", "generation", "colocated", "resources", "num_nodes"): 1,
        ("async_rl", "sampler", "name"): "windowed",
        ("async_rl", "sampler", "max_staleness_versions"): 1,
        ("async_rl", "sampler", "sample_freshest_first"): False,
        ("async_rl", "min_groups_for_streaming_train"): 4,
        ("async_rl", "max_inflight_prompts"): 16,
        ("async_rl", "max_buffered_rollouts"): 64,
        ("async_rl", "controlled_release_delay", "enabled"): True,
        ("async_rl", "controlled_release_delay", "seed"): 20260911,
        ("async_rl", "controlled_release_delay", "assignment_domain"): (
            "m4-opportunity-loss-qwen3-1p7b-gsm8k-v1"
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
        ("data", "train", "dataset_name"): "gsm8k",
        ("data", "train", "subset"): "main",
        ("data", "train", "split"): "train",
        ("data", "train", "split_validation_size"): 0.0,
        ("data", "train", "seed"): None,
        ("data", "train", "extract_answer"): True,
        ("data", "default", "prompt_file"): "examples/prompts/cot.txt",
        ("data", "default", "processor"): "math_hf_data_processor",
        ("data", "default", "env_name"): "math",
        ("env", "math", "math_verify_impl"): "hf_math_verify",
        ("cluster", "gpus_per_node"): 2,
        ("cluster", "num_nodes"): 1,
    }
    for path, expected_value in expected.items():
        if _at(config, *path) != expected_value:
            raise WorkloadTransportPreflightError(
                f"config field {'.'.join(path)} disagrees"
            )
    if _at(config, "async_rl", "controlled_release_delay", "arms") != [
        {"label": "control", "delay_seconds": 0.0, "mass": 1},
        {"label": "d5", "delay_seconds": 5.0, "mass": 1},
    ]:
        raise WorkloadTransportPreflightError("workload transport arms disagree")
    paths = (
        _at(config, "async_rl", "lifecycle_audit_path"),
        _at(config, "async_rl", "gradient_opportunity_audit", "output_path"),
        _at(
            config,
            "async_rl",
            "gradient_opportunity_audit",
            "observer_duty_path",
        ),
    )
    if (
        paths
        != (
            "results/m4-qwen3-1p7b-gsm8k-transport/lifecycle.jsonl",
            "results/m4-qwen3-1p7b-gsm8k-transport/opportunity.jsonl",
            "results/m4-qwen3-1p7b-gsm8k-transport/observer-duty.json",
        )
        or len(set(paths)) != 3
    ):
        raise WorkloadTransportPreflightError(
            "workload transport output paths disagree"
        )
    validate_single_controller_config(MasterConfig(**config))


def build_workload_transport_preflight_lock(
    *, repo: Path, source_commit: str, source_archive_sha256: str
) -> dict[str, object]:
    source_commit = _lower_hex(source_commit, length=40, name="source commit")
    source_archive_sha256 = _lower_hex(
        source_archive_sha256, length=64, name="source archive SHA256"
    )
    register_omegaconf_resolvers()
    resolved = OmegaConf.to_container(load_config(repo / CONFIG_PATH), resolve=True)
    if not isinstance(resolved, Mapping):
        raise WorkloadTransportPreflightError("resolved config must be an object")
    validate_workload_transport_config(resolved)

    protocol_raw = (repo / PROTOCOL_PATH).read_bytes()
    try:
        protocol_value, protocol, options = _parse_protocol(protocol_raw)
        validate_workload_transport_contract(protocol_value, protocol, options)
    except OpportunityLossPipelineError as error:
        raise WorkloadTransportPreflightError(
            "workload transport protocol is invalid"
        ) from error
    if protocol_value.get("status") != (
        "FROZEN_LOCAL_PROTOCOL_PENDING_NO_TRAINING_PREFLIGHT"
    ):
        raise WorkloadTransportPreflightError("protocol status disagrees")
    evidence = protocol_value.get("design_evidence")
    if not isinstance(evidence, Mapping) or evidence != {
        "accepted_qwen3_1p7b_openmath_artifact_sha256": (
            "022e4d18f2eedd2339c3c7eb7decf6e21f929761de46bcd0f85a73f0a2700ac7"
        ),
        "accepted_qwen3_1p7b_openmath_result_sha256": (
            "74c3a836fcc27dd567f3c19ec8e2b8438a915eac53ecc9aa7ca6bb7b7769a3c7"
        ),
        "fixed_window_selected_before_gsm8k_qualification": True,
        "minimum_primary_assignments": 7395,
        "preferred_primary_assignments": 7500,
        "qualification_data_may_enter_estimator": False,
        "workload_compatibility_audit_sha256": AUDIT_SHA256,
    }:
        raise WorkloadTransportPreflightError("design evidence disagrees")

    audit_raw = (repo / AUDIT_PATH).read_bytes()
    if _sha(audit_raw) != AUDIT_SHA256:
        raise WorkloadTransportPreflightError("compatibility audit moved")
    audit = json.loads(audit_raw)
    if (
        audit.get("decision") != "CONDITIONAL_GO_TO_LOCAL_PROTOCOL"
        or audit.get("scope") != "local_static_audit_only_no_gpu_authority"
        or audit.get("qualification", {}).get("scientific_acquisition_allowed")
        is not False
    ):
        raise WorkloadTransportPreflightError("compatibility audit disagrees")
    accepted_raw = (repo / ACCEPTED_RESULT_PATH).read_bytes()
    if _sha(accepted_raw) != ACCEPTED_RESULT_SHA256:
        raise WorkloadTransportPreflightError("accepted result record moved")
    accepted = json.loads(accepted_raw)
    if accepted.get("status") != "MATERIAL":
        raise WorkloadTransportPreflightError("accepted result status disagrees")

    files = {}
    for name in PINNED_FILES:
        raw = (repo / name).read_bytes()
        files[name] = {"sha256": _sha(raw), "size": len(raw)}
    return {
        "schema": "m4-qwen3-1p7b-gsm8k-no-training-preflight-lock-v1",
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "files": files,
        "protocol": {"sha256": _sha(protocol_raw), "size": len(protocol_raw)},
        "compatibility_audit": {"sha256": _sha(audit_raw), "size": len(audit_raw)},
        "accepted_result_record": {
            "sha256": _sha(accepted_raw),
            "size": len(accepted_raw),
        },
        "image": {
            "path": IMAGE_PATH,
            "sha256": IMAGE_SHA256,
            "embedded_commit": IMAGE_COMMIT,
        },
        "design": {
            "arms": ["control", "d5"],
            "dataset": "openai/gsm8k:main:train",
            "gpus": 2,
            "minimum_primary_assignments": 7395,
            "primary_versions": 500,
            "trainer_steps": 558,
            "wall_clock_cap_hours": 4.0,
            "gpu_hour_cap": 8.0,
        },
        "training_allowed": False,
        "qualification_submission_allowed": False,
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
    result = build_workload_transport_preflight_lock(
        repo=args.repo.resolve(),
        source_commit=args.source_commit,
        source_archive_sha256=args.source_archive_sha256,
    )
    _write_canonical(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
