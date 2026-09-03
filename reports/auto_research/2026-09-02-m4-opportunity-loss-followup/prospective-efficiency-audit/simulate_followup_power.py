#!/usr/bin/env python3
"""Empirical version-block power simulation for the M4 follow-up design."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
import zipfile
from pathlib import Path

from tools.opportunity_ledger_join import join_opportunity_ledgers
from tools.opportunity_loss_adjusted_inference import _endpoint, _type7
from tools.opportunity_loss_pipeline import _parse_jsonl, _parse_protocol

ARTIFACT_SHA256 = "e81a29747628b6f0490323207f317755de411d0f8e7527f715368919aee12fb7"
RESULT_SHA256 = "6c3fc0cf0567d4dab63153769c075dc7d491e41cb2ceb982b569649f5b42ced3"
PREFIX = "workspace/assets/basic/m4-opportunity-loss-confirmatory-acquisition-r4/"
MEMBERS = {
    "lifecycle": PREFIX + "m4-opportunity-loss-lifecycle.jsonl",
    "opportunity": PREFIX + "m4-opportunity-loss-opportunity.jsonl",
    "result": PREFIX + "m4-opportunity-loss-result.json",
}
CONTROL_ARM = "control"
TREATMENT_ARM = "d5"
TARGET_PRIMARY_VERSIONS = 400
BLOCK_SIZE = 8
FOLDS = 8
HAC_LAG = 4
DRAWS = 20_000
SEED = 20260904
MATERIAL_THRESHOLD = 0.20
ALPHA = 0.05
PLANNING_VARIANCE_RATIO = 0.50
ALTERNATIVES = (0.225, 0.25, 0.30, 0.40)


class FollowupSimulationError(ValueError):
    """Raised when frozen inputs or the prospective simulation are invalid."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sample_standard_deviation(values: list[float]) -> float:
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def build_simulation(artifact: Path, protocol_path: Path) -> dict[str, object]:
    artifact_raw = artifact.read_bytes()
    if _sha256(artifact_raw) != ARTIFACT_SHA256:
        raise FollowupSimulationError("artifact SHA-256 disagrees")
    protocol_raw = protocol_path.read_bytes()
    _protocol_object, protocol, _options = _parse_protocol(protocol_raw)
    try:
        with zipfile.ZipFile(artifact) as archive:
            raw = {name: archive.read(member) for name, member in MEMBERS.items()}
    except (OSError, KeyError, zipfile.BadZipFile) as error:
        raise FollowupSimulationError(f"cannot read artifact: {error}") from error
    if _sha256(raw["result"]) != RESULT_SHA256:
        raise FollowupSimulationError("result SHA-256 disagrees")
    result = json.loads(raw["result"])
    if _canonical_json(result) != raw["result"]:
        raise FollowupSimulationError("result is not canonical JSON")
    if result["protocol"] != {
        "sha256": _sha256(protocol_raw),
        "size": len(protocol_raw),
    }:
        raise FollowupSimulationError("protocol does not match result")
    rows = join_opportunity_ledgers(
        protocol=protocol,
        lifecycle_rows=_parse_jsonl(raw["lifecycle"], compact=False, name="lifecycle"),
        opportunity_rows=_parse_jsonl(
            raw["opportunity"], compact=True, name="opportunity"
        ),
    )
    primary_rows = [row for row in rows if row.arm in (CONTROL_ARM, TREATMENT_ARM)]
    versions = tuple(
        range(protocol.primary_start_version, protocol.primary_end_version + 1)
    )
    opportunity_scale = math.sqrt(
        math.fsum(row.opportunity**2 for row in primary_rows) / len(primary_rows)
    )
    adjusted = _endpoint(
        primary_rows,
        endpoint="lower",
        control_arm=CONTROL_ARM,
        treatment_arm=TREATMENT_ARM,
        versions=versions,
        folds=FOLDS,
        hac_lag=HAC_LAG,
        opportunity_scale=opportunity_scale,
    )
    cluster_scores = {version: [] for version in versions}
    for row, score in zip(primary_rows, adjusted.scores, strict=True):
        cluster_scores[row.start_version].append(score)

    rng = random.Random(SEED)
    block_count = math.ceil(TARGET_PRIMARY_VERSIONS / BLOCK_SIZE)
    raw_noise = []
    for _draw in range(DRAWS):
        sampled_versions = []
        for _block in range(block_count):
            start = rng.randrange(len(versions))
            sampled_versions.extend(
                versions[(start + offset) % len(versions)]
                for offset in range(BLOCK_SIZE)
            )
        sampled_scores = [
            score
            for version in sampled_versions[:TARGET_PRIMARY_VERSIONS]
            for score in cluster_scores[version]
        ]
        raw_noise.append(math.fsum(sampled_scores) / len(sampled_scores))

    simulated_raw_standard_error = _sample_standard_deviation(raw_noise)
    registered_standard_error = float(
        result["causal_inference"]["lower_endpoint"]["standard_error"]
    )
    primary_fraction = len(primary_rows) / len(rows)
    conservative_target_standard_error = registered_standard_error * math.sqrt(
        len(versions)
        / TARGET_PRIMARY_VERSIONS
        * primary_fraction
        * PLANNING_VARIANCE_RATIO
    )
    inflation = conservative_target_standard_error / simulated_raw_standard_error
    conservative_noise = [value * inflation for value in raw_noise]
    critical_value = _type7(conservative_noise, 1.0 - ALPHA)
    powers = {
        str(alternative): sum(
            alternative - MATERIAL_THRESHOLD + noise > critical_value
            for noise in conservative_noise
        )
        / DRAWS
        for alternative in ALTERNATIVES
    }
    return {
        "schema": "m4-opportunity-loss-empirical-followup-power-v1",
        "status": "EXPLORATORY_DESIGN_ONLY_NO_GPU_AUTHORITY",
        "source": {
            "artifact_sha256": ARTIFACT_SHA256,
            "result_sha256": RESULT_SHA256,
            "protocol_sha256": _sha256(protocol_raw),
        },
        "simulation": {
            "draws": DRAWS,
            "seed": SEED,
            "source_primary_versions": len(versions),
            "target_primary_versions": TARGET_PRIMARY_VERSIONS,
            "circular_block_size": BLOCK_SIZE,
            "cross_fit_folds": FOLDS,
            "hac_lag": HAC_LAG,
            "primary_pair_fraction_in_source": primary_fraction,
            "planning_variance_ratio": PLANNING_VARIANCE_RATIO,
            "raw_empirical_adjusted_standard_error_at_target": (
                simulated_raw_standard_error
            ),
            "conservative_target_standard_error": conservative_target_standard_error,
            "noise_inflation_to_conservative_target": inflation,
            "empirical_one_sided_critical_value": critical_value,
        },
        "power": powers,
        "decision": {
            "target_alternative": 0.25,
            "minimum_power": 0.80,
            "target_power": powers["0.25"],
            "power_gate_passed": powers["0.25"] >= 0.80,
            "gpu_launch_authorized_by_this_artifact": False,
        },
    }


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(_canonical_json(value))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    _write_atomic(args.output, build_simulation(args.artifact, args.protocol))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
