#!/usr/bin/env python3
"""Materialize the 32 frozen downstream-quality run configurations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROTOCOL_SHA256 = "dbe7a4f7d1938ef43d24e536ccf9dd57110a01bab5e6cb511980218e472ef2c0"
AUTHORIZATION_SHA256 = (
    "fdb66f09b411976b5d4cc9b54ec68cef97e2d533c33fd8150d2112d32f046165"
)
TEMPLATE_SHA256 = {
    "immediate": "418fb6dccb9872ec1acaf9955566e011604a9cc6092be1ffb400b9845432f050",
    "mixed_d5": "5c5c61bacbc06616885317cff023983ae1405a2bd2cae0ba634d1b55c60ca3d4",
}
MARKER_COUNTS = {
    "__TRAINING_SEED__": 1,
    "__ASSIGNMENT_SEED__": 1,
    "__ASSIGNMENT_DOMAIN__": 1,
    "__RUN_ROOT__": 5,
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def _load_frozen_inputs(
    protocol_path: Path, authorization_path: Path, templates: dict[str, Path]
) -> tuple[dict[str, object], dict[str, str]]:
    if sha256(protocol_path) != PROTOCOL_SHA256:
        raise RuntimeError("frozen trained-paired protocol moved")
    if sha256(authorization_path) != AUTHORIZATION_SHA256:
        raise RuntimeError("local implementation authority moved")
    protocol = json.loads(protocol_path.read_bytes())
    authorization = json.loads(authorization_path.read_bytes())
    if (
        protocol["status"]
        != "FROZEN_DESIGN_PENDING_IMPLEMENTATION_AND_EXPLICIT_TRAINING_AUTHORIZATION"
        or authorization["protocol_sha256"] != PROTOCOL_SHA256
        or not authorization["config_template_implementation_authorized"]
        or not authorization["run_identity_materialization_authorized"]
    ):
        raise RuntimeError("local implementation boundary differs")
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
        raise RuntimeError("local implementation authority grants execution")
    texts = {}
    for regime, path in templates.items():
        if sha256(path) != TEMPLATE_SHA256[regime]:
            raise RuntimeError(f"frozen {regime} template moved")
        text = path.read_text(encoding="utf-8")
        if any(text.count(marker) != count for marker, count in MARKER_COUNTS.items()):
            raise RuntimeError(f"{regime} template marker cardinality differs")
        texts[regime] = text
    return protocol, texts


def _render(
    template: str,
    *,
    block_id: str,
    regime: str,
    training_seed: int,
    assignment_seed: int,
) -> tuple[str, str, str]:
    domain = f"m4-downstream-quality-paired-v1-{block_id}-{regime.replace('_', '-')}"
    run_root = f"results/m4-downstream-quality-trained-paired/{block_id}/{regime}"
    values = {
        "__TRAINING_SEED__": str(training_seed),
        "__ASSIGNMENT_SEED__": str(assignment_seed),
        "__ASSIGNMENT_DOMAIN__": domain,
        "__RUN_ROOT__": run_root,
    }
    rendered = template
    for marker, value in values.items():
        if rendered.count(marker) != MARKER_COUNTS[marker]:
            raise RuntimeError(f"marker {marker} cardinality differs")
        rendered = rendered.replace(marker, value)
    if any(marker in rendered for marker in MARKER_COUNTS):
        raise RuntimeError("unresolved run-config marker")
    return rendered, domain, run_root


def _pair_normalized(text: str, *, block_id: str, regime: str, domain: str) -> str:
    normalized = text.replace(
        f"prospective downstream-quality {'immediate' if regime == 'immediate' else 'mixed-d5'} cell",
        "prospective downstream-quality __REGIME__ cell",
    )
    normalized = normalized.replace(
        f"results/m4-downstream-quality-trained-paired/{block_id}/{regime}",
        "results/m4-downstream-quality-trained-paired/__BLOCK__/__REGIME__",
    )
    normalized = normalized.replace(domain, "__ASSIGNMENT_DOMAIN__")
    start = normalized.index("  controlled_release_delay:\n")
    end = normalized.index("  gradient_opportunity_audit:\n", start)
    normalized = (
        normalized[:start]
        + "  controlled_release_delay: __FROZEN_REGIME_POLICY__\n"
        + normalized[end:]
    )
    return normalized


def build(
    *,
    protocol_path: Path,
    authorization_path: Path,
    immediate_template: Path,
    mixed_template: Path,
    output_dir: Path,
    manifest_path: Path,
) -> dict[str, object]:
    templates = {"immediate": immediate_template, "mixed_d5": mixed_template}
    protocol, template_text = _load_frozen_inputs(
        protocol_path, authorization_path, templates
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        raise FileExistsError(f"run manifest already exists: {manifest_path}")

    runs: list[dict[str, object]] = []
    pair_hashes: dict[str, str] = {}
    for block in protocol["blocks"]:
        block_id = block["id"]
        normalized = []
        for regime in ("immediate", "mixed_d5"):
            text, domain, run_root = _render(
                template_text[regime],
                block_id=block_id,
                regime=regime,
                training_seed=block["training_seed"],
                assignment_seed=block["mixed_assignment_seed"],
            )
            filename = f"m4-downstream-quality-{block_id}-{regime}.yaml"
            path = output_dir / filename
            if path.exists():
                raise FileExistsError(f"run config already exists: {path}")
            path.write_text(text, encoding="utf-8")
            config_sha = sha256(path)
            normalized.append(
                _pair_normalized(
                    text,
                    block_id=block_id,
                    regime=regime,
                    domain=domain,
                )
            )
            runs.append(
                {
                    "block_id": block_id,
                    "regime": regime,
                    "training_seed": block["training_seed"],
                    "assignment_seed": block["mixed_assignment_seed"],
                    "assignment_domain": domain,
                    "submission_order": block["submission_order"],
                    "config_filename": filename,
                    "config_sha256": config_sha,
                    "run_root": run_root,
                    "expected_result_path": f"{block_id}/{regime}/trained-run-result.json",
                }
            )
        if normalized[0] != normalized[1]:
            raise RuntimeError(f"{block_id} configs differ outside allowed fields")
        pair_hashes[block_id] = sha256_bytes(normalized[0].encode("utf-8"))

    if len(runs) != 32 or len({run["config_sha256"] for run in runs}) != 32:
        raise RuntimeError("run config count or identity uniqueness differs")
    manifest: dict[str, object] = {
        "schema": "m4-downstream-quality-trained-paired-run-manifest-v1",
        "status": "PASS_LOCAL_IDENTITIES_UNLAUNCHABLE",
        "protocol_sha256": PROTOCOL_SHA256,
        "authorization_sha256": AUTHORIZATION_SHA256,
        "template_sha256": TEMPLATE_SHA256,
        "block_count": 16,
        "run_count": 32,
        "pair_invariant_sha256": pair_hashes,
        "runs": runs,
        "generated_configs_are_external_to_git": True,
        "training_started": False,
        "eos_submission_started": False,
        "launch_authorized": False,
        "automatic_retry": False,
        "automatic_replacement": False,
        "automatic_extension": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(canonical(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--immediate-template", type=Path, required=True)
    parser.add_argument("--mixed-template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        protocol_path=args.protocol,
        authorization_path=args.authorization,
        immediate_template=args.immediate_template,
        mixed_template=args.mixed_template,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
    )
    print("M4_DOWNSTREAM_QUALITY_32_RUN_CONFIGS_PASS")
    print(f"runs={result['run_count']}")
    print(f"manifest_sha256={sha256(args.manifest)}")


if __name__ == "__main__":
    main()
