"""Static fail-closed checks for the lifecycle-derived Llama successor."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from nemo_rl.utils.config import register_omegaconf_resolvers
from tools import m4_llama_lifecycle_derived_preflight as subject

_REPO = Path(__file__).resolve().parents[3]


def test_protocol_preserves_closed_predecessor_and_one_shot_sequence() -> None:
    protocol = json.loads((_REPO / subject.PROTOCOL_PATH).read_bytes())
    subject.validate_protocol(protocol)
    assert protocol["predecessor"]["terminal_commit"] == (
        "59a0f88c8af66aa7804ad30219eccbd2caf7f2be"
    )


@pytest.mark.parametrize("qualification", [False, True])
def test_configs_resolve_and_disable_synchronous_observer(qualification: bool) -> None:
    register_omegaconf_resolvers()
    paths = (
        subject.QUALIFICATION_CONFIG_PATHS if qualification else subject.CONFIG_PATHS
    )
    for cell, path in paths.items():
        config = subject._resolve(_REPO, path)
        subject.validate_config(cell, config, qualification=qualification)
        assert config["async_rl"]["gradient_opportunity_audit"]["enabled"] is False
        assert (
            config["async_rl"]["lifecycle_derived_opportunity_audit"]["enabled"] is True
        )


def test_config_mutation_fails_closed() -> None:
    register_omegaconf_resolvers()
    config = subject._resolve(
        _REPO, subject.QUALIFICATION_CONFIG_PATHS["llama3p2_1b_gsm8k"]
    )
    mutated = copy.deepcopy(config)
    mutated["async_rl"]["gradient_opportunity_audit"]["enabled"] = True
    with pytest.raises(subject.LifecycleDerivedPreflightError, match="differs"):
        subject.validate_config("llama3p2_1b_gsm8k", mutated, qualification=True)


def test_lock_forbids_compute_and_pins_instrument() -> None:
    lock = subject.build_lock(
        repo=_REPO,
        source_commit="7" * 40,
        source_archive_sha256="8" * 64,
    )
    assert lock["training_allowed"] is False
    assert lock["qualification_submission_allowed"] is False
    assert lock["scientific_acquisition_allowed"] is False
    assert lock["automatic_retry"] is False
    assert "nemo_rl/algorithms/async_utils/lifecycle_opportunity.py" in lock["files"]
