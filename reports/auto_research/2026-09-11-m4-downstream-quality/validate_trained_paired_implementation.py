#!/usr/bin/env python3
"""Validate the local-only M4 downstream-quality implementation package."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PROTOCOL = ROOT / "trained_paired_acquisition_protocol.json"
AUTHORIZATION = ROOT / "trained_paired_local_implementation_authorization.json"
EMBARGO = ROOT / "trained_paired_outcome_embargo.json"
MANIFEST = ROOT / "trained_paired_run_manifest.json"
BUILDER = ROOT / "build_trained_paired_run_configs.py"
ANALYZER = REPO / "tools/m4_downstream_quality_analysis.py"
TEST = REPO / "tests/unit/tools/test_m4_downstream_quality_analysis.py"
EXTRACTOR = ROOT / "extract_v8_base_prompt_covariate.py"
TEMPLATES = {
    "immediate": ROOT / "templates/trained_paired_immediate.yaml.tmpl",
    "mixed_d5": ROOT / "templates/trained_paired_mixed_d5.yaml.tmpl",
}
EXPECTED_SHA256 = {
    "protocol": "dbe7a4f7d1938ef43d24e536ccf9dd57110a01bab5e6cb511980218e472ef2c0",
    "authorization": "fdb66f09b411976b5d4cc9b54ec68cef97e2d533c33fd8150d2112d32f046165",
    "embargo": "3d292020ff6d9a42a9209de3c65010603a24a5c9ffabbeaf77521be99ac84997",
    "manifest": "a461e83c7dbccdefa1c5c779062e709f34bfc502639a3e7de663a717fefa93bf",
    "builder": "cd95b6be5f0262dd896cca73704982ade84fc31fc8ffb60da6c89857ba06bda7",
    "analyzer": "6fc29e32c949276bcda33923d6294dd6e75f76ca94fecc426bfe9115cf367a09",
    "test": "e11eadfb1268d2df8898e827ebf1e059cce920e1a7f9f3c2454f1f1e7a4324a7",
    "extractor": "dbb384a4f4dea4c3c78dc18ae3b2cb86b4d7ff95867f982f3b7c821cb258bf8b",
    "immediate_template": "418fb6dccb9872ec1acaf9955566e011604a9cc6092be1ffb400b9845432f050",
    "mixed_d5_template": "5c5c61bacbc06616885317cff023983ae1405a2bd2cae0ba634d1b55c60ca3d4",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-config-dir", type=Path, required=True)
    parser.add_argument("--v8-archive", type=Path, required=True)
    parser.add_argument("--base-covariate", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "protocol": PROTOCOL,
        "authorization": AUTHORIZATION,
        "embargo": EMBARGO,
        "manifest": MANIFEST,
        "builder": BUILDER,
        "analyzer": ANALYZER,
        "test": TEST,
        "extractor": EXTRACTOR,
        "immediate_template": TEMPLATES["immediate"],
        "mixed_d5_template": TEMPLATES["mixed_d5"],
    }
    for name, path in paths.items():
        actual = sha256(path)
        if actual != EXPECTED_SHA256[name]:
            raise RuntimeError(f"{name} moved: {actual}")

    authorization = json.loads(AUTHORIZATION.read_bytes())
    forbidden = (
        "source_archive_build_authorized",
        "credential_free_eos_package_build_authorized",
        "model_weight_access_authorized",
        "optimizer_initialization_authorized",
        "training_authorized",
        "eos_submission_authorized",
        "qualification_authorized",
        "scientific_acquisition_authorized",
        "automatic_retry",
        "automatic_replacement",
        "automatic_extension",
    )
    if any(authorization[key] for key in forbidden):
        raise RuntimeError("local authority grants execution")
    embargo = json.loads(EMBARGO.read_bytes())
    if (
        embargo["status"] != "LOCKED_NO_TRAINED_OUTCOMES"
        or embargo["partial_result_analysis_allowed"] is not False
        or embargo["outcome_release_authorized"] is not False
    ):
        raise RuntimeError("outcome embargo is open")

    manifest = json.loads(MANIFEST.read_bytes())
    if (
        manifest["status"] != "PASS_LOCAL_IDENTITIES_UNLAUNCHABLE"
        or manifest["run_count"] != 32
        or manifest["block_count"] != 16
        or manifest["launch_authorized"] is not False
        or len({run["config_sha256"] for run in manifest["runs"]}) != 32
        or len(manifest["pair_invariant_sha256"]) != 16
    ):
        raise RuntimeError("run manifest boundary differs")

    builder = load_module("m4_downstream_builder", BUILDER)
    extractor = load_module("m4_downstream_extractor", EXTRACTOR)
    with tempfile.TemporaryDirectory(prefix="m4-downstream-implementation-") as raw:
        temp = Path(raw)
        rebuilt_manifest = temp / "run-manifest.json"
        builder.build(
            protocol_path=PROTOCOL,
            authorization_path=AUTHORIZATION,
            immediate_template=TEMPLATES["immediate"],
            mixed_template=TEMPLATES["mixed_d5"],
            output_dir=temp / "configs",
            manifest_path=rebuilt_manifest,
        )
        if rebuilt_manifest.read_bytes() != MANIFEST.read_bytes():
            raise RuntimeError("run manifest deterministic rebuild differs")
        for run in manifest["runs"]:
            config = temp / "configs" / run["config_filename"]
            preserved_config = args.generated_config_dir / run["config_filename"]
            text = config.read_text(encoding="utf-8")
            if (
                sha256(config) != run["config_sha256"]
                or preserved_config.read_bytes() != config.read_bytes()
                or "__" in text
                or "max_num_steps: 448" not in text
                or "seed: 20260911" not in text
                or f"seed: {run['training_seed']}" not in text
                or run["assignment_domain"] not in text
                or "terminal_policy_export:\n    enabled: true" not in text
                or "checkpointing:\n  enabled: false" not in text
            ):
                raise RuntimeError(
                    f"generated config differs: {run['config_filename']}"
                )

        tampered = json.loads(AUTHORIZATION.read_bytes())
        tampered["training_authorized"] = True
        tampered_path = temp / "tampered-authorization.json"
        tampered_path.write_text(
            json.dumps(tampered, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        try:
            builder.build(
                protocol_path=PROTOCOL,
                authorization_path=tampered_path,
                immediate_template=TEMPLATES["immediate"],
                mixed_template=TEMPLATES["mixed_d5"],
                output_dir=temp / "tampered-configs",
                manifest_path=temp / "tampered-manifest.json",
            )
        except RuntimeError as error:
            if "authority moved" not in str(error):
                raise
        else:
            raise RuntimeError("training-authorized mutation did not fail closed")

        rebuilt_covariate = temp / "v8-base-prompt-covariate.json"
        extractor.extract(archive=args.v8_archive, output=rebuilt_covariate)
        if (
            sha256(args.base_covariate)
            != "86ba0cd85d88e1bac90c27f972effa5038cc08efeae6d3a210922a2bad16cc21"
            or rebuilt_covariate.read_bytes() != args.base_covariate.read_bytes()
        ):
            raise RuntimeError("V8 base-prompt covariate deterministic rebuild differs")

    completed = subprocess.run(
        [sys.executable, str(TEST), "-q"],
        cwd=REPO,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0 or "OK" not in completed.stdout:
        raise RuntimeError(
            f"dependency-light analysis tests failed:\n{completed.stdout}"
        )

    print("M4_DOWNSTREAM_QUALITY_LOCAL_IMPLEMENTATION_PASS")
    print("run_count=32")
    print("matched_blocks=16")
    print("training_started=false")
    print("eos_submission_started=false")


if __name__ == "__main__":
    main()
