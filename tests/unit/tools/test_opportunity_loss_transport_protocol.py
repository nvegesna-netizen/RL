# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Static tests for the Qwen3-1.7B M4 transport protocol."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from tools.opportunity_loss_pipeline import (
    OpportunityLossPipelineError,
    _parse_protocol,
)
from tools.opportunity_loss_transport_pipeline import _validate_transport_contract

_REPO = Path(__file__).resolve().parents[3]
_PROTOCOL = (
    _REPO / "reports/auto_research/2026-09-03-m4-opportunity-loss-transport/"
    "qwen3-1p7b-confirmatory-design/protocol_config.json"
)
_FOLLOWUP_PROTOCOL = (
    _REPO / "reports/auto_research/2026-09-02-m4-opportunity-loss-followup/"
    "protocol_config.json"
)
_CONFIG = (
    _REPO / "examples/configs/"
    "grpo_math_1B_megatron_single_controller_m4_qwen3_1p7b_transport.yaml"
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def test_transport_protocol_and_overlay_have_exact_registered_geometry() -> None:
    raw, protocol, options = _parse_protocol(_PROTOCOL.read_bytes())
    _validate_transport_contract(raw, protocol, options)
    overlay = yaml.safe_load(_CONFIG.read_bytes())

    assert options["protocol_identity"] == (
        "m4-opportunity-loss-qwen3-1p7b-transport-v1"
    )
    assert options["cross_fit_folds"] == 8
    assert protocol.primary_start_version == 8
    assert protocol.primary_end_version == 507
    assert [(arm.label, arm.delay_seconds, arm.mass) for arm in protocol.arms] == [
        ("control", 0.0, 1),
        ("d5", 5.0, 1),
    ]
    assert raw["runtime"]["trainer_steps"] == overlay["grpo"]["max_num_steps"] == 558
    assert raw["arms"] == overlay["async_rl"]["controlled_release_delay"]["arms"]
    assert (
        raw["assignment"]["domain"]
        == (overlay["async_rl"]["controlled_release_delay"]["assignment_domain"])
    )
    assert (
        raw["assignment"]["seed"]
        == (overlay["async_rl"]["controlled_release_delay"]["seed"])
    )
    assert overlay["policy"]["model_name"] == "Qwen/Qwen3-1.7B"


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("windows", "primary_start_versions"), [8, 506], "window"),
        (("runtime", "trainer_steps"), 557, "runtime"),
        (("resource_caps", "automatic_retry"), True, "resource caps"),
        (("assignment", "seed"), 1, "assignment"),
    ],
)
def test_transport_contract_mutations_fail_closed(
    path: tuple[str, ...], value: object, match: str
) -> None:
    mutated = copy.deepcopy(json.loads(_PROTOCOL.read_bytes()))
    target = mutated
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    raw, protocol, options = _parse_protocol(_canonical(mutated))
    with pytest.raises(OpportunityLossPipelineError, match=match):
        _validate_transport_contract(raw, protocol, options)


def test_transport_and_followup_pipelines_reject_each_others_protocols_first(
    tmp_path: Path,
) -> None:
    from tools.opportunity_loss_followup_pipeline import build_followup_result
    from tools.opportunity_loss_transport_pipeline import build_transport_result

    missing = tmp_path / "missing"
    with pytest.raises(OpportunityLossPipelineError, match="follow-up"):
        build_followup_result(
            protocol_path=_PROTOCOL,
            lifecycle_path=missing,
            opportunity_path=missing,
            observer_duty_path=missing,
        )
    with pytest.raises(OpportunityLossPipelineError, match="Qwen3-1.7B"):
        build_transport_result(
            protocol_path=_FOLLOWUP_PROTOCOL,
            lifecycle_path=missing,
            opportunity_path=missing,
            observer_duty_path=missing,
        )
