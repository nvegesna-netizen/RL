# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Build the no-training lock for the capacity-corrected GSM8K M4 R2 study."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
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
from tools.opportunity_loss_workload_transport_preflight import (
    ACCEPTED_RESULT_PATH,
    ACCEPTED_RESULT_SHA256,
    AUDIT_PATH,
    AUDIT_SHA256,
    IMAGE_COMMIT,
    IMAGE_PATH,
    IMAGE_SHA256,
    PINNED_FILES as R1_PINNED_FILES,
    WorkloadTransportPreflightError,
    _lower_hex,
    _sha,
    _write_canonical,
    validate_workload_transport_config,
)


CONFIG_PATH = (
    "examples/configs/"
    "grpo_math_1B_megatron_single_controller_m4_qwen3_1p7b_gsm8k_transport_r2.yaml"
)
PROTOCOL_PATH = (
    "reports/auto_research/2026-09-04-m4-opportunity-loss-workload-transport/"
    "qwen3-1p7b-gsm8k-confirmatory-design/protocol_config_r2.json"
)
R1_FAILURE_PATH = (
    "reports/auto_research/2026-09-04-m4-opportunity-loss-workload-transport/"
    "qwen3-1p7b-gsm8k-confirmatory-design/acquisition_r1_terminal_failure.json"
)
PROTOCOL_SHA256 = "3413bde3718563157cee5d504f174710f65406dd7f1effd8ad2755a03d73b69e"
R1_FAILURE_SHA256 = "070e288335786fbd047ace303fb9046fc0bcf8e400a60228f08e1c4fd6518369"
R1_TERMINAL_ARTIFACT_SHA256 = (
    "dae0b7ee0f13df156c205a336f2c5ff14cfff7093a92e337d8a88ecb25b14785"
)
PINNED_FILES = tuple(
    dict.fromkeys(
        (*R1_PINNED_FILES, CONFIG_PATH, PROTOCOL_PATH, R1_FAILURE_PATH, __file__)
    )
)
# __file__ is replaced below with its repository-relative spelling.
PINNED_FILES = tuple(
    "tools/opportunity_loss_workload_transport_r2_preflight.py"
    if name == __file__
    else name
    for name in PINNED_FILES
)


def validate_r2_workload_transport_config(config: Mapping[str, object]) -> None:
    """Require the R2 config to differ from the frozen R1 config only as declared."""
    candidate = copy.deepcopy(dict(config))
    try:
        grpo = candidate["grpo"]
        async_rl = candidate["async_rl"]
        logger = candidate["logger"]
        if not isinstance(grpo, dict) or not isinstance(async_rl, dict):
            raise KeyError
        release = async_rl["controlled_release_delay"]
        audit = async_rl["gradient_opportunity_audit"]
        if not isinstance(release, dict) or not isinstance(audit, dict):
            raise KeyError
        if grpo.get("max_num_epochs") != 2:
            raise WorkloadTransportPreflightError("R2 max_num_epochs disagrees")
        if release.get("seed") != 20260913:
            raise WorkloadTransportPreflightError("R2 assignment seed disagrees")
        if release.get("assignment_domain") != (
            "m4-opportunity-loss-qwen3-1p7b-gsm8k-r2-v1"
        ):
            raise WorkloadTransportPreflightError("R2 assignment domain disagrees")
        r2_paths = (
            async_rl.get("lifecycle_audit_path"),
            audit.get("output_path"),
            audit.get("observer_duty_path"),
            logger.get("log_dir") if isinstance(logger, dict) else None,
        )
        if r2_paths != (
            "results/m4-qwen3-1p7b-gsm8k-transport-r2/lifecycle.jsonl",
            "results/m4-qwen3-1p7b-gsm8k-transport-r2/opportunity.jsonl",
            "results/m4-qwen3-1p7b-gsm8k-transport-r2/observer-duty.json",
            "results/m4-qwen3-1p7b-gsm8k-transport-r2/metrics",
        ):
            raise WorkloadTransportPreflightError("R2 output paths disagree")

        # Normalize only the seven preregistered R2 deltas, then apply the complete
        # frozen R1 validator to every other scientific and runtime field.
        grpo["max_num_epochs"] = 1
        release["seed"] = 20260911
        release["assignment_domain"] = "m4-opportunity-loss-qwen3-1p7b-gsm8k-v1"
        async_rl["lifecycle_audit_path"] = (
            "results/m4-qwen3-1p7b-gsm8k-transport/lifecycle.jsonl"
        )
        audit["output_path"] = "results/m4-qwen3-1p7b-gsm8k-transport/opportunity.jsonl"
        audit["observer_duty_path"] = (
            "results/m4-qwen3-1p7b-gsm8k-transport/observer-duty.json"
        )
        if isinstance(logger, dict):
            logger["log_dir"] = "results/m4-qwen3-1p7b-gsm8k-transport/metrics"
    except KeyError as error:
        raise WorkloadTransportPreflightError(
            "R2 config structure disagrees"
        ) from error
    validate_workload_transport_config(candidate)
    validate_single_controller_config(MasterConfig(**config))


def build_r2_workload_transport_preflight_lock(
    *, repo: Path, source_commit: str, source_archive_sha256: str
) -> dict[str, object]:
    source_commit = _lower_hex(source_commit, length=40, name="source commit")
    source_archive_sha256 = _lower_hex(
        source_archive_sha256, length=64, name="source archive SHA256"
    )
    register_omegaconf_resolvers()
    resolved = OmegaConf.to_container(load_config(repo / CONFIG_PATH), resolve=True)
    if not isinstance(resolved, Mapping):
        raise WorkloadTransportPreflightError("resolved R2 config must be an object")
    validate_r2_workload_transport_config(resolved)

    protocol_raw = (repo / PROTOCOL_PATH).read_bytes()
    if _sha(protocol_raw) != PROTOCOL_SHA256:
        raise WorkloadTransportPreflightError("R2 protocol moved")
    try:
        protocol_value, protocol, options = _parse_protocol(protocol_raw)
        validate_workload_transport_contract(protocol_value, protocol, options)
    except OpportunityLossPipelineError as error:
        raise WorkloadTransportPreflightError("R2 protocol is invalid") from error
    if protocol_value.get("status") != (
        "FROZEN_LOCAL_R2_PROTOCOL_PENDING_NO_TRAINING_PREFLIGHT"
    ):
        raise WorkloadTransportPreflightError("R2 protocol status disagrees")
    evidence = protocol_value.get("design_evidence")
    if not isinstance(evidence, Mapping):
        raise WorkloadTransportPreflightError("R2 design evidence is missing")
    if evidence.get("r1_observations_may_enter_estimator") is not False:
        raise WorkloadTransportPreflightError("R1 estimator exclusion disagrees")
    if evidence.get("r1_terminal_artifact_sha256") != R1_TERMINAL_ARTIFACT_SHA256:
        raise WorkloadTransportPreflightError("R1 terminal artifact binding disagrees")
    if evidence.get("r1_terminal_failure_record_sha256") != R1_FAILURE_SHA256:
        raise WorkloadTransportPreflightError("R1 failure binding disagrees")

    r1_raw = (repo / R1_FAILURE_PATH).read_bytes()
    if _sha(r1_raw) != R1_FAILURE_SHA256:
        raise WorkloadTransportPreflightError("R1 failure record moved")
    r1 = json.loads(r1_raw)
    if (
        r1.get("status") != "TERMINAL_INCOMPLETE_FINITE_EPOCH_EXHAUSTION"
        or r1.get("scientific_interpretation") != "INCOMPLETE_NO_MATERIALITY_CONCLUSION"
        or r1.get("terminal_artifact", {}).get("sha256") != R1_TERMINAL_ARTIFACT_SHA256
        or r1.get("failure", {}).get("configured_max_num_epochs") != 1
        or r1.get("analysis", {}).get("primary_assignment_shortfall") != 105
    ):
        raise WorkloadTransportPreflightError("R1 terminal failure record disagrees")

    audit_raw = (repo / AUDIT_PATH).read_bytes()
    accepted_raw = (repo / ACCEPTED_RESULT_PATH).read_bytes()
    if _sha(audit_raw) != AUDIT_SHA256:
        raise WorkloadTransportPreflightError("compatibility audit moved")
    if _sha(accepted_raw) != ACCEPTED_RESULT_SHA256:
        raise WorkloadTransportPreflightError("accepted result record moved")
    if json.loads(accepted_raw).get("status") != "MATERIAL":
        raise WorkloadTransportPreflightError("accepted result status disagrees")

    files: dict[str, dict[str, object]] = {}
    for name in PINNED_FILES:
        raw = (repo / name).read_bytes()
        files[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
    return {
        "schema": "m4-qwen3-1p7b-gsm8k-r2-no-training-preflight-lock-v1",
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "files": files,
        "protocol": {"sha256": _sha(protocol_raw), "size": len(protocol_raw)},
        "r1_terminal_failure_record": {
            "sha256": _sha(r1_raw),
            "size": len(r1_raw),
        },
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
            "epoch_specific_group_instance_is_assignment_unit": True,
            "gpus": 2,
            "max_num_epochs": 2,
            "minimum_primary_assignments": 7395,
            "primary_versions": 500,
            "r1_observations_enter_estimator": False,
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_r2_workload_transport_preflight_lock(
        repo=args.repo.resolve(),
        source_commit=args.source_commit,
        source_archive_sha256=args.source_archive_sha256,
    )
    _write_canonical(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
