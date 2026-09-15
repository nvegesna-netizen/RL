#!/usr/bin/env python3
"""Fail-closed validation for the authorized OARS shadow preflight design."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    validate_single_controller_config,
)
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PROTOCOL = HERE / "shadow_preflight_protocol.json"
AUTHORIZATION = HERE / "shadow_preflight_authorization.json"
CONFIG = (
    REPO
    / "examples/configs/grpo_math_1B_megatron_single_controller_m4_oars_shadow_preflight.yaml"
)
ANALYZER = REPO / "tools/m4_oars_shadow_preflight.py"
EXPECTED = {
    PROTOCOL: "1482d985a17197a58af018bd4e043a1008b78b6dbe4d99671954dc8452705513",
    AUTHORIZATION: "855b5e8fe61b338aedf2e79eef8f098ff00417d6540033e21fc918de0742471e",
    CONFIG: "200164a4f7beab123d687a4674ac6ba4d3a1f84299cb676e672b8df282b8acb4",
    ANALYZER: "151a9b7e726934647bd3f40ac38723c433622003cc3af772ce21c5f5ec23ae85",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for path, expected in EXPECTED.items():
        assert sha256(path) == expected, f"frozen input moved: {path}"
    protocol = json.loads(PROTOCOL.read_bytes())
    authorization = json.loads(AUTHORIZATION.read_bytes())
    assert protocol["status"] == "FROZEN_AUTHORIZED_ONE_SHOT_NO_ACTUATION"
    assert authorization["protocol_sha256"] == EXPECTED[PROTOCOL]
    assert authorization["required_launcher"] == "runllm.py --no_wait"
    assert authorization["submission_attempt_limit"] == 1
    assert authorization["eos_submission_authorized"] is True
    assert authorization["fifo_controlled_training_authorized"] is True
    assert authorization["oars_shadow_observation_authorized"] is True
    assert authorization["oars_actuation_authorized"] is False
    assert authorization["scientific_outcome_acquisition_authorized"] is False
    assert authorization["automatic_retry"] is False
    assert authorization["automatic_extension"] is False

    register_omegaconf_resolvers()
    config = OmegaConf.to_container(load_config(str(CONFIG)), resolve=True)
    assert isinstance(config, dict)
    master = MasterConfig(**config)
    validate_single_controller_config(master)
    async_config = master.async_rl
    assert master.grpo.max_num_steps == 64
    assert master.grpo.num_prompts_per_step == 4
    assert master.policy["model_name"] == "meta-llama/Llama-3.2-1B-Instruct"
    assert master.cluster["num_nodes"] == 1 and master.cluster["gpus_per_node"] == 2
    assert async_config.sampler.name == "weight_fifo"
    assert async_config.sampler.max_staleness_versions == 1
    release = async_config.controlled_release_delay
    assert release.enabled is True
    assert [(arm.label, arm.delay_seconds, arm.mass) for arm in release.arms] == [
        ("neutral", 0.0, 1)
    ]
    assert async_config.gradient_opportunity_audit.enabled is True
    assert async_config.lifecycle_derived_opportunity_audit.enabled is False
    shadow = async_config.opportunity_at_risk_shadow
    assert shadow.enabled is True
    assert shadow.service_budget_multiplier == 1.02
    assert shadow.max_candidate_groups == 25

    surface = CONFIG.read_text() + PROTOCOL.read_text() + AUTHORIZATION.read_text()
    assert "sbatch " not in surface and "srun " not in surface
    assert surface.count("runllm.py --no_wait") == 2
    print(
        json.dumps(
            {
                "status": "PASS_AUTHORIZED_LOCAL_DESIGN_NO_SUBMISSION_YET",
                "protocol_sha256": EXPECTED[PROTOCOL],
                "authorization_sha256": EXPECTED[AUTHORIZATION],
                "config_sha256": EXPECTED[CONFIG],
                "analyzer_sha256": EXPECTED[ANALYZER],
                "acting_sampler": "weight_fifo",
                "oars_actuated": False,
                "trainer_steps": 64,
                "gpus": 2,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
