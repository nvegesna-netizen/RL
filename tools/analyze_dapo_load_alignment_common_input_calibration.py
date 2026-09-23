# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate fresh calibration streams and freeze balanced validation cohorts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nemo_rl.algorithms.async_utils.fixed_pool import (
    FIXED_POOL_RUN_MODE,
    compute_fixed_pool_id,
    load_fixed_pool_manifest,
    validate_fixed_pool_manifest_design,
    validate_fixed_pool_materialization,
    validate_fixed_pool_trace,
)
from nemo_rl.algorithms.async_utils.scheduler_trace import iter_scheduler_trace
from tools import analyze_dapo_load_alignment_references as prior_analyzer
from tools import analyze_dapo_operational_latency_discovery as base_analyzer
from tools import (
    materialize_dapo_load_alignment_common_input_validation as materializer,
)


EXPECTED_GROUPS = 32
EXPECTED_COMPLETIONS = 16
HALF_SIZE = 16
GROUPS_PER_COHORT = 4


class DapoFreshCommonInputCalibrationError(ValueError):
    """A fresh calibration artifact or balanced cohort is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DapoFreshCommonInputCalibrationError(message)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise DapoFreshCommonInputCalibrationError(f"cannot load {label}") from error
    if not isinstance(value, dict):
        raise DapoFreshCommonInputCalibrationError(f"{label} must be an object")
    return value


def _load_design(
    path: Path, pool_seed: int
) -> tuple[str, tuple[base_analyzer.DesignItem, ...]]:
    value = _load_object(path, "selection design")
    required = {
        "schema_version",
        "analysis_status",
        "calibration_only",
        "confirmatory_eligible",
        "protocol_sha256",
        "fixed_pool_id",
        "selection_seed",
        "generation_seed",
        "items",
    }
    _require(
        set(value) == required
        and value.get("schema_version") == 1
        and value.get("analysis_status")
        == "controlled_dapo_common_input_validation_pool"
        and value.get("calibration_only") is True
        and value.get("confirmatory_eligible") is False
        and value.get("protocol_sha256") == materializer.PROTOCOL_SHA256
        and value.get("selection_seed") == pool_seed
        and value.get("generation_seed") == pool_seed,
        "selection design contract mismatch",
    )
    raw_items = value.get("items")
    fields = set(base_analyzer.DesignItem.__dataclass_fields__)
    _require(isinstance(raw_items, list), "selection design items missing")
    items: list[base_analyzer.DesignItem] = []
    for item in raw_items:
        _require(isinstance(item, dict) and set(item) == fields, "design item mismatch")
        items.append(base_analyzer.DesignItem(**item))
    _require(
        len(items) == EXPECTED_GROUPS
        and [item.source_pool_ordinal for item in items] == list(range(EXPECTED_GROUPS))
        and len({item.source_prompt_id for item in items}) == EXPECTED_GROUPS,
        "selection design geometry mismatch",
    )
    return str(value["fixed_pool_id"]), tuple(items)


def _analyze_collection(
    *,
    pool_seed: int,
    generation_seed: int,
    run_dir: Path,
    materialization_root: Path,
) -> tuple[tuple[base_analyzer.Observation, ...], dict[str, object]]:
    manifest_path = materialization_root / f"fixed_pool_manifest.v1.{pool_seed}.json"
    design_path = materialization_root / f"selection_design.v1.{pool_seed}.json"
    trace_path = run_dir / "scheduler_trace.v1.jsonl"
    pool_id, items = _load_design(design_path, pool_seed)
    manifest = load_fixed_pool_manifest(manifest_path)
    validate_fixed_pool_materialization(manifest)
    validate_fixed_pool_manifest_design(manifest, materializer.DESIGN_ID)
    _require(
        manifest.pool_id == pool_id and manifest.order_seed == pool_seed,
        "manifest/design binding mismatch",
    )
    trace_report = validate_fixed_pool_trace(
        trace_path, manifest, expected_completions_per_group=EXPECTED_COMPLETIONS
    )
    _require(trace_report.physical_weight_version == 0, "calibration changed weights")
    events = tuple(iter_scheduler_trace(trace_path))
    starts = [event for event in events if event.event_type.value == "run_started"]
    _require(
        len(starts) == 1
        and starts[0].run_mode == FIXED_POOL_RUN_MODE
        and starts[0].scalar_summaries.get("generation_study_seed") == generation_seed,
        "calibration run identity or generation seed mismatch",
    )
    observations = base_analyzer._observations_from_events(events, items, pool_seed)
    length_rate = sum(item.length_terminations for item in observations) / (
        EXPECTED_GROUPS * EXPECTED_COMPLETIONS
    )
    checks = {
        "complete_32_groups_512_completions": len(observations) == EXPECTED_GROUPS,
        "zero_learner_steps_and_physical_weight_version_zero": True,
        "zero_administrative_censoring": True,
        "backend_length_termination_rate_le_0_2": length_rate <= 0.2,
    }
    return observations, {
        "generation_seed": generation_seed,
        "trace_sha256": _sha(trace_path),
        "manifest_sha256": _sha(manifest_path),
        "selection_design_sha256": _sha(design_path),
        "prompt_groups": len(observations),
        "completions": len(observations) * EXPECTED_COMPLETIONS,
        "backend_length_termination_rate": length_rate,
        "validity_checks": checks,
        "all_validity_gates_passed": all(checks.values()),
    }


def balanced_validation_ordinals(
    lower_load_ordinals: Sequence[int], *, pool_seed: int
) -> tuple[int, ...]:
    """Return eight deterministic cohorts containing two prompts from each half."""
    lower = list(lower_load_ordinals)
    if (
        len(lower) != HALF_SIZE
        or len(set(lower)) != HALF_SIZE
        or any(value < 0 or value >= EXPECTED_GROUPS for value in lower)
    ):
        raise DapoFreshCommonInputCalibrationError("lower-load half is invalid")
    higher = [value for value in range(EXPECTED_GROUPS) if value not in set(lower)]
    rng = random.Random(pool_seed)
    rng.shuffle(lower)
    rng.shuffle(higher)
    order: list[int] = []
    for offset in range(0, HALF_SIZE, 2):
        order.extend(lower[offset : offset + 2])
        order.extend(higher[offset : offset + 2])
    if len(order) != EXPECTED_GROUPS or len(set(order)) != EXPECTED_GROUPS:
        raise DapoFreshCommonInputCalibrationError("balanced order is not a partition")
    return tuple(order)


def _balanced_artifacts(
    *,
    materialization_root: Path,
    pool_seed: int,
    lower_load_ordinals: Sequence[int],
) -> tuple[bytes, bytes, dict[int, int]]:
    source_manifest = _load_object(
        materialization_root / f"fixed_pool_manifest.v1.{pool_seed}.json",
        "source manifest",
    )
    source_design = _load_object(
        materialization_root / f"selection_design.v1.{pool_seed}.json",
        "source design",
    )
    manifest_items = source_manifest.get("items")
    design_items = source_design.get("items")
    _require(
        isinstance(manifest_items, list) and isinstance(design_items, list),
        "source item inventory missing",
    )
    manifests_by_ordinal = {
        int(item["ordinal"]): item
        for item in manifest_items
        if isinstance(item, dict) and isinstance(item.get("ordinal"), int)
    }
    designs_by_ordinal = {
        int(item["source_pool_ordinal"]): item
        for item in design_items
        if isinstance(item, dict) and isinstance(item.get("source_pool_ordinal"), int)
    }
    _require(
        set(manifests_by_ordinal) == set(designs_by_ordinal) == set(range(32)),
        "source ordinal inventory mismatch",
    )
    order = balanced_validation_ordinals(lower_load_ordinals, pool_seed=pool_seed)
    old_to_new = {old: new for new, old in enumerate(order)}
    balanced_manifest = copy.deepcopy(source_manifest)
    balanced_manifest_items = []
    for new_ordinal, old_ordinal in enumerate(order):
        item = copy.deepcopy(manifests_by_ordinal[old_ordinal])
        item["ordinal"] = new_ordinal
        item["dispatch_cohort"] = new_ordinal // GROUPS_PER_COHORT
        item["decorrelation_block"] = f"cohort-{new_ordinal // GROUPS_PER_COHORT}"
        balanced_manifest_items.append(item)
    balanced_manifest["items"] = balanced_manifest_items
    balanced_manifest.pop("pool_id", None)
    balanced_manifest["pool_id"] = compute_fixed_pool_id(balanced_manifest)

    balanced_design = copy.deepcopy(source_design)
    balanced_design_items = []
    for new_ordinal, old_ordinal in enumerate(order):
        item = copy.deepcopy(designs_by_ordinal[old_ordinal])
        item["source_pool_ordinal"] = new_ordinal
        balanced_design_items.append(item)
    balanced_design["fixed_pool_id"] = balanced_manifest["pool_id"]
    balanced_design["analysis_status"] = (
        "controlled_dapo_common_input_balanced_validation_pool"
    )
    balanced_design["items"] = balanced_design_items
    return _canonical(balanced_manifest), _canonical(balanced_design), old_to_new


def analyze(
    runs: Mapping[tuple[int, int], Path], *, materialization_root: Path
) -> tuple[dict[str, object], dict[str, object], dict[int, tuple[bytes, bytes]]]:
    """Return private calibration, public gates, and balanced pool artifacts."""
    expected = tuple(
        (pool_seed, generation_seed)
        for pool_seed in materializer.POOL_SEEDS
        for generation_seed in materializer.CALIBRATION_SEEDS[pool_seed]
    )
    _require(tuple(runs) == expected, "all six calibration runs are required in order")
    private_pools: list[dict[str, object]] = []
    public_pools: list[dict[str, object]] = []
    balanced: dict[int, tuple[bytes, bytes]] = {}
    pooled_loads: tuple[list[float], list[float]] = ([], [])
    all_collection_valid = True
    for pool_seed in materializer.POOL_SEEDS:
        generation_seeds = materializer.CALIBRATION_SEEDS[pool_seed]
        observations_by_draw: list[tuple[base_analyzer.Observation, ...]] = []
        collections: list[dict[str, object]] = []
        for generation_seed in generation_seeds:
            observations, record = _analyze_collection(
                pool_seed=pool_seed,
                generation_seed=generation_seed,
                run_dir=runs[(pool_seed, generation_seed)],
                materialization_root=materialization_root,
            )
            observations_by_draw.append(observations)
            collections.append(record)
            all_collection_valid &= bool(record["all_validity_gates_passed"])
        by_ordinal = [
            {item.item.source_pool_ordinal: item for item in observations}
            for observations in observations_by_draw
        ]
        _require(
            set(by_ordinal[0]) == set(by_ordinal[1]) == set(range(EXPECTED_GROUPS))
            and all(
                by_ordinal[0][index].item.source_prompt_id
                == by_ordinal[1][index].item.source_prompt_id
                for index in range(EXPECTED_GROUPS)
            ),
            "calibration prompt identity parity mismatch",
        )
        loads = [
            [
                by_ordinal[side][index].mean_generated_tokens
                for index in range(EXPECTED_GROUPS)
            ]
            for side in (0, 1)
        ]
        for side in (0, 1):
            pooled_loads[side].extend(loads[side])
        load_spearman = prior_analyzer._spearman(*loads)
        lower_overlap = len(
            set(prior_analyzer._lower_ordinals(loads[0]))
            & set(prior_analyzer._lower_ordinals(loads[1]))
        )
        frozen_loads = [statistics.fmean(pair) for pair in zip(*loads, strict=True)]
        lower_ordinals = prior_analyzer._lower_ordinals(frozen_loads)
        manifest_raw, design_raw, old_to_new = _balanced_artifacts(
            materialization_root=materialization_root,
            pool_seed=pool_seed,
            lower_load_ordinals=lower_ordinals,
        )
        balanced[pool_seed] = (manifest_raw, design_raw)
        prompts = [
            {
                "source_pool_ordinal": index,
                "validation_source_pool_ordinal": old_to_new[index],
                "source_prompt_id": by_ordinal[0][index].item.source_prompt_id,
                "canonical_prompt_sha256": by_ordinal[0][
                    index
                ].item.canonical_prompt_sha256,
                "generated_load_score": frozen_loads[index],
                "fixed_lower_load": index in set(lower_ordinals),
            }
            for index in range(EXPECTED_GROUPS)
        ]
        private_pools.append(
            {
                "pool_seed": pool_seed,
                "calibration_generation_seeds": list(generation_seeds),
                "validation_generation_seeds": list(
                    materializer.VALIDATION_SEEDS[pool_seed]
                ),
                "balanced_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
                "balanced_design_sha256": hashlib.sha256(design_raw).hexdigest(),
                "prompts": prompts,
            }
        )
        checks = {
            "generated_load_spearman_ge_0_6": load_spearman >= 0.6,
            "lower_load_half_overlap_ge_10": lower_overlap >= 10,
            "balanced_cohorts_exact_2_lower_2_higher": all(
                sum(
                    old_to_new[index] // GROUPS_PER_COHORT == cohort
                    for index in lower_ordinals
                )
                == 2
                for cohort in range(8)
            ),
        }
        public_pools.append(
            {
                "pool_seed": pool_seed,
                "collections": collections,
                "generated_load_spearman": load_spearman,
                "lower_load_half_overlap": lower_overlap,
                "balanced_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
                "balanced_design_sha256": hashlib.sha256(design_raw).hexdigest(),
                "stability_and_geometry_checks": checks,
            }
        )
    pooled_load = prior_analyzer._spearman(*pooled_loads)
    all_pool_gates = all(
        all(pool["stability_and_geometry_checks"].values())  # type: ignore[union-attr]
        for pool in public_pools
    )
    private: dict[str, object] = {
        "schema_version": 1,
        "analysis_status": "frozen_private_dapo_common_input_calibration",
        "protocol_sha256": materializer.PROTOCOL_SHA256,
        "contains_prompt_identities": True,
        "pools": private_pools,
    }
    private_sha = hashlib.sha256(_canonical(private)).hexdigest()
    checks = {
        "all_six_calibration_collections_valid": all_collection_valid,
        "all_three_pool_stability_and_geometry_gates_passed": all_pool_gates,
        "pooled_generated_load_spearman_ge_0_7": pooled_load >= 0.7,
    }
    decision = (
        "calibration_valid_freeze_balanced_validation_pools"
        if all(checks.values())
        else "stop_before_validation_generation"
    )
    public: dict[str, object] = {
        "schema_version": 1,
        "analysis_status": "dapo_common_input_calibration_result",
        "calibration_only": True,
        "confirmatory_eligible": False,
        "validation_generation_authorized": False,
        "live_scheduler_arm_authorized": False,
        "learner_training_authorized": False,
        "protocol_sha256": materializer.PROTOCOL_SHA256,
        "private_calibration_manifest_sha256": private_sha,
        "pools": public_pools,
        "pooled_generated_load_spearman": pooled_load,
        "locked_checks": checks,
        "decision": decision,
    }
    return private, public, balanced if all(checks.values()) else {}


def _parse_run(value: str) -> tuple[tuple[int, int], Path]:
    identity, separator, path = value.partition("=")
    pool, colon, generation = identity.partition(":")
    if (
        not separator
        or not colon
        or not pool.isdigit()
        or not generation.isdigit()
        or not path
    ):
        raise argparse.ArgumentTypeError("run must be POOL_SEED:GENERATION_SEED=PATH")
    return (int(pool), int(generation)), Path(path)


def _write_new(path: Path, raw: bytes, *, private: bool = False) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    if private:
        path.chmod(0o600)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--materialization-root", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--balanced-output-dir", type=Path, required=True)
    args = parser.parse_args()
    parsed: list[tuple[tuple[int, int], Path]] = args.run
    runs: dict[tuple[int, int], Path] = dict(parsed)
    _require(len(runs) == len(parsed), "duplicate calibration run")
    private, public, balanced = analyze(
        runs, materialization_root=args.materialization_root
    )
    _write_new(args.private_output, _canonical(private), private=True)
    _write_new(args.public_output, _canonical(public))
    if balanced:
        _require(
            args.balanced_output_dir.resolve() == args.materialization_root.resolve(),
            "balanced manifests must remain beside their bound sources and model",
        )
        for pool_seed, (manifest_raw, design_raw) in balanced.items():
            _write_new(
                args.balanced_output_dir
                / f"fixed_pool_manifest.validation.v1.{pool_seed}.json",
                manifest_raw,
                private=True,
            )
            _write_new(
                args.balanced_output_dir
                / f"selection_design.validation.v1.{pool_seed}.json",
                design_raw,
                private=True,
            )
    print(json.dumps(public, sort_keys=True))


if __name__ == "__main__":
    main()
