"""Fail closed on the generated M4-Shield JET package surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--custom-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    custom_config_raw = args.custom_config.read_bytes()
    custom_config = json.loads(custom_config_raw)
    script = manifest["spec"]["script"]
    substitution_error = None
    substituted_script = None
    try:
        substitution_code = compile("f" + repr(script), "<jet-substitution>", "eval")
        substituted_script = eval(
            substitution_code,
            {"__builtins__": {}},
            {"assets_dir": "/tmp/m4-shield-assets"},
        )
    except (NameError, SyntaxError, ValueError) as error:
        substitution_error = f"{type(error).__name__}: {error}"
    expected_script = script.replace("{assets_dir}", "/tmp/m4-shield-assets")
    expected_script = expected_script.replace("{{", "{").replace("}}", "}")
    forbidden = re.compile(
        r"(?i)(private[-_ ]?token|access[-_ ]?token|authorization:\s*bearer|"
        r"glpat-|github_pat_|BEGIN (RSA |OPENSSH )?PRIVATE KEY)"
    )
    checks = {
        "manifest_shape": manifest.get("type") == "basic"
        and manifest.get("format_version") == 1,
        "one_node": manifest["spec"].get("nodes") == 1
        and manifest["launchers"]["type:slurm"].get("nodes") == 1,
        "no_manifest_scheduler_time_limit": "time_limit" in manifest["spec"]
        and manifest["spec"]["time_limit"] is None,
        "no_manifest_queue_deadline": "queue_deadline" not in json.dumps(manifest),
        "deadline_suppressed_in_custom_config": custom_config
        == {
            "launchers": {
                "dgxh100_eos": {"sbatch_additional_flags": {"deadline": False}}
            }
        },
        "custom_config_sha256_bound": hashlib.sha256(custom_config_raw).hexdigest()
        == "b5db8882c7baccd3bf8c0369d7264e80f7ea08c17482559f82875abc5207f42e",
        "credential_free_text": forbidden.search(raw.decode("utf-8")) is None,
        "outcome_excluded": "terminal_policy_export" not in script
        and "evaluation_data" not in script
        and "run_env_eval" not in script,
        "one_training_invocation": script.count(
            "examples/run_grpo_single_controller.py"
        )
        == 1,
        "m4_shield_bound": 'actuation_scorer=="m4_shield"' in script
        and "m4_shield_qualification.py" in script,
        "qualification_analyzer_source_bound": (
            "6fe084407c2ba24113bd33143e1905368085d3c2dd3ebdda17479683bd90cbb8" in script
            and "m4_shield_qualification.py" in script
        ),
        "full_preflight_tests": script.count('"$PYTHON" -m pytest -q') == 3,
        "artifact_hashes": "artifacts.sha256" in script and "sha256sum" in script,
        "jet_substitution_syntax": substitution_error is None,
        "jet_substitution_roundtrip": substituted_script == expected_script,
        "one_jet_placeholder": script.count("{assets_dir}") == 1,
    }
    result = {
        "schema": "m4-shield-qualification-package-validation-v1",
        "status": "PASS_CREDENTIAL_FREE_PACKAGE" if all(checks.values()) else "FAIL",
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "custom_config_sha256": hashlib.sha256(custom_config_raw).hexdigest(),
        "checks": checks,
        "jet_substitution_error": substitution_error,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["status"] != "PASS_CREDENTIAL_FREE_PACKAGE":
        raise SystemExit(f"package validation failed: {checks}")
    print(result["status"])


if __name__ == "__main__":
    main()
