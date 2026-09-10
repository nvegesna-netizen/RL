# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Static protocol checks for the two V4 Llama qualification configs."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).parents[3]
CONFIGS = {
    "openmath": {
        "path": ROOT
        / "examples/configs/grpo_math_1B_megatron_single_controller_m4_llama3p2_1b_openmath_lifecycle_derived_neutral_qualification_v4.yaml",
        "base": "grpo_math_1B_megatron_single_controller_m4_llama3p2_1b_openmath_lifecycle_derived_transport.yaml",
        "domain": "m4-llama3p2-1b-openmath-lifecycle-derived-neutral-v2",
        "seed": 20261013,
    },
    "gsm8k": {
        "path": ROOT
        / "examples/configs/grpo_math_1B_megatron_single_controller_m4_llama3p2_1b_gsm8k_lifecycle_derived_neutral_qualification_v4.yaml",
        "base": "grpo_math_1B_megatron_single_controller_m4_llama3p2_1b_gsm8k_lifecycle_derived_transport.yaml",
        "domain": "m4-llama3p2-1b-gsm8k-lifecycle-derived-neutral-v2",
        "seed": 20261014,
    },
}


def test_v4_configs_freeze_only_neutral_qualification_overrides() -> None:
    for cell, expected in CONFIGS.items():
        config = yaml.safe_load(expected["path"].read_bytes())
        assert set(config) == {"defaults", "grpo", "async_rl", "logger"}
        assert config["defaults"] == expected["base"]
        assert config["grpo"] == {"max_num_steps": 64}
        async_rl = config["async_rl"]
        assert set(async_rl) == {
            "lifecycle_audit_path",
            "controlled_release_delay",
            "lifecycle_derived_opportunity_audit",
        }
        release = async_rl["controlled_release_delay"]
        assert release == {
            "seed": expected["seed"],
            "assignment_domain": expected["domain"],
            "arms": [{"label": "neutral", "delay_seconds": 0.0, "mass": 1}],
        }
        rendered = expected["path"].read_text()
        assert (
            rendered.count(f"m4-llama3p2-1b-{cell}-lifecycle-derived-neutral-v4") == 5
        )
        assert "acquisition" not in rendered


def test_v4_domains_and_seeds_are_fresh_and_distinct() -> None:
    domains = {value["domain"] for value in CONFIGS.values()}
    seeds = {value["seed"] for value in CONFIGS.values()}
    assert len(domains) == len(seeds) == 2
    assert all(domain.endswith("neutral-v2") for domain in domains)
    assert seeds == {20261013, 20261014}
