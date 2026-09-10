# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fail-closed no-training lock and config checks for the Llama M4 successor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

REPORT_ROOT = "reports/auto_research/2026-09-09-m4-llama-lifecycle-derived-transport"
PROTOCOL_PATH = f"{REPORT_ROOT}/protocol_config.json"
DESIGN_PATH = f"{REPORT_ROOT}/design.md"
BRIDGE_PATH = f"{REPORT_ROOT}/instrument_bridge.md"
PREFLIGHT_PATH = f"{REPORT_ROOT}/preflight_plan.md"
QUALIFICATION_PATH = f"{REPORT_ROOT}/qualification_plan.md"
IMAGE_PATH = (
    "/lustre/fsw/coreai_dlalgo_ci/nvegesna/nemo_rl_images/nemo-rl-nightly-5802754.sqsh"
)
IMAGE_SHA256 = "3df8114a0b3e60ef95ce13f8b982c7cc45d164aca71c63a87388b1e8434ee470"
IMAGE_COMMIT = "ae07eafe8035b5b2e84efa7234e70e7fd7e493c1"

CONFIG_PATHS = {
    "llama3p2_1b_openmath": (
        "examples/configs/grpo_math_1B_megatron_single_controller_m4_"
        "llama3p2_1b_openmath_lifecycle_derived_transport.yaml"
    ),
    "llama3p2_1b_gsm8k": (
        "examples/configs/grpo_math_1B_megatron_single_controller_m4_"
        "llama3p2_1b_gsm8k_lifecycle_derived_transport.yaml"
    ),
}
QUALIFICATION_CONFIG_PATHS = {
    "llama3p2_1b_openmath": (
        "examples/configs/grpo_math_1B_megatron_single_controller_m4_"
        "llama3p2_1b_openmath_lifecycle_derived_neutral_qualification.yaml"
    ),
    "llama3p2_1b_gsm8k": (
        "examples/configs/grpo_math_1B_megatron_single_controller_m4_"
        "llama3p2_1b_gsm8k_lifecycle_derived_neutral_qualification.yaml"
    ),
}
PINNED_FILES = (
    DESIGN_PATH,
    BRIDGE_PATH,
    PREFLIGHT_PATH,
    PROTOCOL_PATH,
    QUALIFICATION_PATH,
    *CONFIG_PATHS.values(),
    *QUALIFICATION_CONFIG_PATHS.values(),
    "examples/configs/grpo_math_1B.yaml",
    "examples/configs/grpo_math_1B_megatron_single_controller.yaml",
    "examples/configs/grpo_math_1B_megatron_single_controller_m4_opportunity_loss_followup.yaml",
    "examples/configs/grpo_math_1B_megatron_single_controller_m4_llama3p2_1b_openmath_family_transport.yaml",
    "examples/configs/grpo_math_1B_megatron_single_controller_m4_llama3p2_1b_gsm8k_family_transport.yaml",
    "nemo_rl/algorithms/advantage_estimator.py",
    "nemo_rl/algorithms/async_utils/controlled_release.py",
    "nemo_rl/algorithms/async_utils/gradient_opportunity.py",
    "nemo_rl/algorithms/async_utils/lifecycle_opportunity.py",
    "nemo_rl/algorithms/async_utils/observer_duty.py",
    "nemo_rl/algorithms/async_utils/replay_buffer.py",
    "nemo_rl/algorithms/async_utils/rollout_lifecycle.py",
    "nemo_rl/algorithms/async_utils/staleness_sampler.py",
    "nemo_rl/algorithms/single_controller.py",
    "nemo_rl/algorithms/single_controller_utils/config.py",
    "nemo_rl/algorithms/single_controller_utils/utils.py",
    "nemo_rl/experience/payload.py",
    "nemo_rl/experience/rollout_manager.py",
    "tools/m4_llama_lifecycle_derived_preflight.py",
    "tools/opportunity_ledger_join.py",
    "tests/unit/single_controller/test_gradient_opportunity.py",
    "tests/unit/single_controller/test_lifecycle_opportunity.py",
    "tests/unit/single_controller/test_observer_duty.py",
    "tests/unit/single_controller/test_rollout_lifecycle.py",
    "tests/unit/tools/test_m4_llama_lifecycle_derived_preflight.py",
    "tests/unit/tools/test_opportunity_ledger_join.py",
)


class LifecycleDerivedPreflightError(ValueError):
    """Raised when a successor protocol or config invariant differs."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _at(value: Mapping[str, object], *path: str) -> object:
    current: object = value
    for component in path:
        if not isinstance(current, Mapping) or component not in current:
            raise LifecycleDerivedPreflightError(f"config lacks {'.'.join(path)}")
        current = current[component]
    return current


def _resolve(repo: Path, path: str) -> dict[str, object]:
    from omegaconf import OmegaConf

    from nemo_rl.utils.config import load_config

    value = OmegaConf.to_container(load_config(repo / path), resolve=True)
    if not isinstance(value, dict):
        raise LifecycleDerivedPreflightError(f"{path} did not resolve to an object")
    return value


def _validate_runtime_schema(config: Mapping[str, object]) -> None:
    from nemo_rl.algorithms.single_controller_utils.config import (
        MasterConfig,
        validate_single_controller_config,
    )

    validate_single_controller_config(MasterConfig(**config))


def validate_protocol(protocol: Mapping[str, object]) -> None:
    if protocol.get("protocol") != "m4-llama3p2-1b-lifecycle-derived-transport-v1":
        raise LifecycleDerivedPreflightError("protocol identity differs")
    if protocol.get("status") != "FROZEN_IMPLEMENTED_PENDING_NO_TRAINING_PREFLIGHT":
        raise LifecycleDerivedPreflightError("protocol freeze status differs")
    predecessor = protocol.get("predecessor")
    instrument = protocol.get("instrument")
    execution = protocol.get("execution")
    if not all(
        isinstance(value, Mapping) for value in (predecessor, instrument, execution)
    ):
        raise LifecycleDerivedPreflightError("protocol sections are malformed")
    assert isinstance(predecessor, Mapping)
    assert isinstance(instrument, Mapping)
    assert isinstance(execution, Mapping)
    if (
        predecessor.get("terminal_status")
        != "TERMINAL_CLOSED_QUALIFICATION_MISS_NO_ACQUISITION"
        or instrument.get("synchronous_gradient_opportunity_callback_enabled")
        is not False
        or instrument.get("derived_after_active_window") is not True
        or instrument.get("total_lifecycle_recorder_corrected_duty_maximum") != 0.01
        or execution.get("preflight_attempt_limit") != 1
        or execution.get("qualification_attempt_limit_each_cell") != 1
        or execution.get("acquisition_attempt_limit_each_cell") != 1
        or execution.get("analysis_attempt_limit") != 1
        or execution.get("automatic_retry") is not False
        or execution.get("automatic_extension") is not False
        or execution.get("required_launcher") != "runllm.py --no_wait"
    ):
        raise LifecycleDerivedPreflightError("protocol execution lock differs")


def validate_config(
    cell: str, config: Mapping[str, object], *, qualification: bool
) -> None:
    roots = {
        ("llama3p2_1b_openmath", False): (
            "results/m4-llama3p2-1b-openmath-lifecycle-derived-transport",
            20261009,
            "m4-llama3p2-1b-openmath-lifecycle-derived-transport-v1",
            448,
            "OpenMathInstruct-2",
            1,
        ),
        ("llama3p2_1b_gsm8k", False): (
            "results/m4-llama3p2-1b-gsm8k-lifecycle-derived-transport",
            20261010,
            "m4-llama3p2-1b-gsm8k-lifecycle-derived-transport-v1",
            448,
            "gsm8k",
            2,
        ),
        ("llama3p2_1b_openmath", True): (
            "results/m4-llama3p2-1b-openmath-lifecycle-derived-neutral",
            20261007,
            "m4-llama3p2-1b-openmath-lifecycle-derived-neutral-v1",
            32,
            "OpenMathInstruct-2",
            1,
        ),
        ("llama3p2_1b_gsm8k", True): (
            "results/m4-llama3p2-1b-gsm8k-lifecycle-derived-neutral",
            20261008,
            "m4-llama3p2-1b-gsm8k-lifecycle-derived-neutral-v1",
            32,
            "gsm8k",
            2,
        ),
    }
    try:
        root, seed, domain, steps, dataset, epochs = roots[(cell, qualification)]
    except KeyError as error:
        raise LifecycleDerivedPreflightError("unknown successor cell") from error
    expected = {
        ("grpo", "max_num_steps"): steps,
        ("grpo", "max_num_epochs"): epochs,
        ("grpo", "num_prompts_per_step"): 4,
        ("grpo", "num_generations_per_prompt"): 8,
        ("policy", "model_name"): "meta-llama/Llama-3.2-1B-Instruct",
        ("policy", "train_global_batch_size"): 32,
        ("policy", "train_micro_batch_size"): 1,
        ("cluster", "gpus_per_node"): 2,
        ("cluster", "num_nodes"): 1,
        ("data", "train", "dataset_name"): dataset,
        ("async_rl", "sampler", "name"): "windowed",
        ("async_rl", "sampler", "max_staleness_versions"): 1,
        ("async_rl", "max_inflight_prompts"): 16,
        ("async_rl", "max_buffered_rollouts"): 64,
        ("async_rl", "controlled_release_delay", "enabled"): True,
        ("async_rl", "controlled_release_delay", "seed"): seed,
        ("async_rl", "controlled_release_delay", "assignment_domain"): domain,
        ("async_rl", "gradient_opportunity_audit", "enabled"): False,
        ("async_rl", "lifecycle_derived_opportunity_audit", "enabled"): True,
        ("async_rl", "lifecycle_audit_path"): f"{root}/lifecycle.jsonl",
        (
            "async_rl",
            "lifecycle_derived_opportunity_audit",
            "output_path",
        ): f"{root}/opportunity.jsonl",
        (
            "async_rl",
            "lifecycle_derived_opportunity_audit",
            "lifecycle_duty_path",
        ): f"{root}/lifecycle-duty.json",
        (
            "async_rl",
            "lifecycle_derived_opportunity_audit",
            "derivation_summary_path",
        ): f"{root}/derivation-summary.json",
        ("checkpointing", "enabled"): False,
    }
    for path, expected_value in expected.items():
        if _at(config, *path) != expected_value:
            raise LifecycleDerivedPreflightError(
                f"{cell} field {'.'.join(path)} differs"
            )
    arms = _at(config, "async_rl", "controlled_release_delay", "arms")
    expected_arms = (
        [{"label": "neutral", "delay_seconds": 0.0, "mass": 1}]
        if qualification
        else [
            {"label": "control", "delay_seconds": 0.0, "mass": 1},
            {"label": "d5", "delay_seconds": 5.0, "mass": 1},
        ]
    )
    if arms != expected_arms:
        raise LifecycleDerivedPreflightError(f"{cell} release arms differ")
    _validate_runtime_schema(config)


def build_file_lock(
    *, repo: Path, source_commit: str, source_archive_sha256: str
) -> dict[str, object]:
    if len(source_commit) != 40 or any(
        c not in "0123456789abcdef" for c in source_commit
    ):
        raise LifecycleDerivedPreflightError("source commit must be lowercase hex")
    if len(source_archive_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in source_archive_sha256
    ):
        raise LifecycleDerivedPreflightError(
            "source archive hash must be lowercase hex"
        )
    protocol = json.loads((repo / PROTOCOL_PATH).read_bytes())
    validate_protocol(protocol)
    files: dict[str, object] = {}
    for name in sorted(PINNED_FILES):
        raw = (repo / name).read_bytes()
        files[name] = {"sha256": _sha(raw), "size": len(raw)}
    return {
        "schema": "m4-llama-lifecycle-derived-no-training-lock-v1",
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "image": {
            "path": IMAGE_PATH,
            "sha256": IMAGE_SHA256,
            "embedded_commit": IMAGE_COMMIT,
        },
        "protocol_sha256": _sha((repo / PROTOCOL_PATH).read_bytes()),
        "files": files,
        "training_allowed": False,
        "qualification_submission_allowed": False,
        "scientific_acquisition_allowed": False,
        "automatic_retry": False,
        "automatic_extension": False,
    }


def build_lock(
    *, repo: Path, source_commit: str, source_archive_sha256: str
) -> dict[str, object]:
    """Validate runtime configs and return the deterministic pinned-file lock."""
    from nemo_rl.utils.config import register_omegaconf_resolvers

    register_omegaconf_resolvers()
    for cell in CONFIG_PATHS:
        validate_config(cell, _resolve(repo, CONFIG_PATHS[cell]), qualification=False)
        validate_config(
            cell,
            _resolve(repo, QUALIFICATION_CONFIG_PATHS[cell]),
            qualification=True,
        )
    return build_file_lock(
        repo=repo,
        source_commit=source_commit,
        source_archive_sha256=source_archive_sha256,
    )


def _write(path: Path, value: Mapping[str, object]) -> None:
    raw = (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash-only", action="store_true")
    args = parser.parse_args()
    builder = build_file_lock if args.hash_only else build_lock
    lock = builder(
        repo=args.repo,
        source_commit=args.source_commit,
        source_archive_sha256=args.source_archive_sha256,
    )
    _write(args.output, lock)
    print(f"lock_sha256={_sha(args.output.read_bytes())} files={len(lock['files'])}")


if __name__ == "__main__":
    main()
