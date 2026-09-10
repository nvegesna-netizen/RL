#!/usr/bin/env python3
"""Deterministically validate the frozen local Llama 3.2 3B design."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from tools.m4_llama3b_qualification import assess_llama3b_qualification
from tools.m4_llama_size_extension_analysis import infer_size_contrast
from tools.m4_llama_v5_analysis import ReplicateEndpoint, ReplicateInference


CONFIGS = {
    "openmath_qualification": "examples/configs/grpo_math_3B_megatron_single_controller_m4_llama3p2_3b_openmath_neutral_qualification_v1.yaml",
    "gsm8k_qualification": "examples/configs/grpo_math_3B_megatron_single_controller_m4_llama3p2_3b_gsm8k_neutral_qualification_v1.yaml",
    "openmath_r1": "examples/configs/grpo_math_3B_megatron_single_controller_m4_llama3p2_3b_openmath_size_extension_v1_r1.yaml",
    "openmath_r2": "examples/configs/grpo_math_3B_megatron_single_controller_m4_llama3p2_3b_openmath_size_extension_v1_r2.yaml",
    "gsm8k_r1": "examples/configs/grpo_math_3B_megatron_single_controller_m4_llama3p2_3b_gsm8k_size_extension_v1_r1.yaml",
    "gsm8k_r2": "examples/configs/grpo_math_3B_megatron_single_controller_m4_llama3p2_3b_gsm8k_size_extension_v1_r2.yaml",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replicate(name: str, estimate: float) -> ReplicateInference:
    endpoint = ReplicateEndpoint(estimate, 0.02, (0.0,) * 100)
    return ReplicateInference(name, endpoint, endpoint, 0.0, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = HERE / "prospective_protocol.json"
    protocol = json.loads(protocol_path.read_bytes())
    assert protocol["status"] == "FROZEN_LOCAL_DESIGN_NO_LAUNCH_AUTHORITY"
    assert protocol["no_training_preflight"]["authorized"] is False
    assert protocol["qualification"]["authorized"] is False
    assert protocol["acquisition"]["authorized"] is False
    assert protocol["qualification"]["qualification_data_allowed_in_causal_estimator"] is False
    assert protocol["acquisition"]["automatic_retry"] is False
    assert protocol["acquisition"]["automatic_extension"] is False
    seeds = [value["seed"] for value in protocol["cells"].values()]
    domains = [value["domain"] for value in protocol["cells"].values()]
    assert len(seeds) == len(set(seeds)) == 6
    assert len(domains) == len(set(domains)) == 6

    config_hashes = {}
    for name, relative in CONFIGS.items():
        path = ROOT / relative
        digest = sha256(path)
        assert digest == protocol["cells"][name]["config_sha256"]
        text = path.read_text()
        assert text.count("meta-llama/Llama-3.2-3B-Instruct") == 2
        assert f"seed: {protocol['cells'][name]['seed']}" in text
        assert f"assignment_domain: {protocol['cells'][name]['domain']}" in text
        if "qualification" in name:
            assert "neutral_qualification_v4.yaml" in text
            assert "size_extension" not in protocol["cells"][name]["domain"]
        else:
            assert "transport_v5_r" in text
            assert "size-extension" in protocol["cells"][name]["domain"]
        config_hashes[name] = digest

    counts_pass = {version: 16 for version in range(8, 56)}
    counts_fail = {version: 12 for version in range(8, 56)}
    pass_result = assess_llama3b_qualification(
        counts_pass,
        [20.0] * 48,
        total_qualification_runtime_seconds=1080.0,
        assignment_bootstrap_seed=20261111,
        timing_bootstrap_seed=20261113,
    )
    assignment_fail = assess_llama3b_qualification(
        counts_fail,
        [20.0] * 48,
        total_qualification_runtime_seconds=1080.0,
        assignment_bootstrap_seed=20261112,
        timing_bootstrap_seed=20261114,
    )
    timing_fail = assess_llama3b_qualification(
        counts_pass,
        [32.0] * 48,
        total_qualification_runtime_seconds=1650.0,
        assignment_bootstrap_seed=20261111,
        timing_bootstrap_seed=20261113,
    )
    assert pass_result.qualified
    assert not assignment_fail.qualified and not assignment_fail.assignment_support_passed
    assert not timing_fail.qualified and not timing_fail.timing_support_passed

    contrast = infer_size_contrast(
        {"r1": replicate("r1", 0.30), "r2": replicate("r2", 0.34)},
        {"r1": replicate("r1", 0.40), "r2": replicate("r2", 0.44)},
    )
    assert abs(contrast.estimate - 0.10) < 1e-12 and contrast.conclusion == "POSITIVE"
    result = {
        "schema": "m4-llama3p2-3b-local-design-validation-v1",
        "status": "PASS",
        "protocol_sha256": sha256(protocol_path),
        "config_sha256": config_hashes,
        "checks": {
            "six_unique_domains_and_seeds": True,
            "3b_model_and_tokenizer_exact": True,
            "qualification_inherits_neutral_64_step_contract": True,
            "acquisitions_inherit_v5_448_step_contract": True,
            "assignment_and_timing_negative_controls": True,
            "size_contrast_retains_both_sizes_and_replicates": True,
            "launch_authority_absent": True,
        },
        "launch_attempted": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
