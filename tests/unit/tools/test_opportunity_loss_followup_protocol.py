# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Static tests for the prospective M4 follow-up protocol and config overlay."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tools.opportunity_loss_pipeline import (
    OpportunityLossPipelineError,
    _parse_protocol,
)

_REPO = Path(__file__).resolve().parents[3]
_PROTOCOL = (
    _REPO
    / "reports/auto_research/2026-09-02-m4-opportunity-loss-followup/protocol_config.json"
)
_CONFIG = (
    _REPO
    / "examples/configs/grpo_math_1B_megatron_single_controller_m4_opportunity_loss_followup.yaml"
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def test_followup_protocol_and_overlay_have_exact_registered_geometry() -> None:
    raw, protocol, options = _parse_protocol(_PROTOCOL.read_bytes())
    overlay = yaml.safe_load(_CONFIG.read_bytes())

    assert options["protocol_identity"] == "m4-opportunity-loss-adjusted-followup-v1"
    assert options["cross_fit_folds"] == 8
    assert protocol.primary_start_version == 8
    assert protocol.primary_end_version == 407
    assert [(arm.label, arm.delay_seconds, arm.mass) for arm in protocol.arms] == [
        ("control", 0.0, 1),
        ("d5", 5.0, 1),
    ]
    assert raw["runtime"]["trainer_steps"] == overlay["grpo"]["max_num_steps"] == 448
    assert raw["arms"] == overlay["async_rl"]["controlled_release_delay"]["arms"]
    assert (
        raw["assignment"]["domain"]
        == (overlay["async_rl"]["controlled_release_delay"]["assignment_domain"])
    )
    assert (
        raw["assignment"]["seed"]
        == (overlay["async_rl"]["controlled_release_delay"]["seed"])
    )
    assert (
        len(
            {
                overlay["async_rl"]["lifecycle_audit_path"],
                overlay["async_rl"]["gradient_opportunity_audit"]["output_path"],
                overlay["async_rl"]["gradient_opportunity_audit"]["observer_duty_path"],
            }
        )
        == 3
    )


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("protocol",), "m4-opportunity-loss-common-instrumentation-v1", "estimand"),
        (("analysis", "primary_estimator"), "unadjusted", "estimator"),
        (("analysis", "inference", "cross_fit_folds"), True, "cross-fit"),
        (("arms", 1, "mass"), 2, "follow-up arms"),
    ],
)
def test_followup_protocol_mutations_fail_closed(
    path: tuple[str | int, ...], value: object, match: str
) -> None:
    protocol = json.loads(_PROTOCOL.read_bytes())
    target: object = protocol
    for component in path[:-1]:
        target = target[component]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]

    if path[0] == "arms":
        from tools.opportunity_loss_followup_pipeline import _validate_followup_contract

        _, parsed, options = _parse_protocol(_canonical(protocol))
        with pytest.raises(OpportunityLossPipelineError, match=match):
            _validate_followup_contract(parsed, options)
        return
    with pytest.raises(OpportunityLossPipelineError, match=match):
        _parse_protocol(_canonical(protocol))


def test_registered_pipeline_rejects_followup_protocol_before_evidence_reads(
    tmp_path: Path,
) -> None:
    from tools.opportunity_loss_pipeline import build_result

    missing = tmp_path / "missing"
    with pytest.raises(OpportunityLossPipelineError, match="requires v1 protocol"):
        build_result(
            protocol_path=_PROTOCOL,
            lifecycle_path=missing,
            opportunity_path=missing,
            observer_duty_path=missing,
        )
