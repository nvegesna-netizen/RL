# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Validate dual neutral references and freeze DAPO operational difficulty."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from nemo_rl.algorithms.async_utils.fixed_pool import (
    FIXED_POOL_RUN_MODE,
    load_fixed_pool_manifest,
    validate_fixed_pool_manifest_design,
    validate_fixed_pool_materialization,
    validate_fixed_pool_trace,
)
from nemo_rl.algorithms.async_utils.scheduler_trace import iter_scheduler_trace
from tools import analyze_dapo_operational_latency_discovery as base_analyzer
from tools import materialize_dapo_operational_mixture as materializer


EXPECTED_GROUPS = 32
EXPECTED_COMPLETIONS = 16
HARDER_HALF_SIZE = 16


class DapoOperationalMixtureReferenceError(ValueError):
    """A neutral reference artifact or frozen gate is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DapoOperationalMixtureReferenceError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _pool_spec(seed: int) -> tuple[int, tuple[int, int], int]:
    for (
        pool_seed,
        dispatch_seed,
        reference_seeds,
        scheduler_seed,
    ) in materializer.POOL_SPECS:
        if pool_seed == seed:
            return dispatch_seed, reference_seeds, scheduler_seed
    raise DapoOperationalMixtureReferenceError(f"unknown pool seed {seed}")


def _load_design(
    path: Path, seed: int
) -> tuple[str, tuple[base_analyzer.DesignItem, ...]]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise DapoOperationalMixtureReferenceError(
            "invalid selection design"
        ) from error
    dispatch_seed, _, _ = _pool_spec(seed)
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
        isinstance(value, dict)
        and set(value) == required
        and value.get("schema_version") == 1
        and value.get("analysis_status") == "controlled_dapo_operational_mixture_pool"
        and value.get("calibration_only") is True
        and value.get("confirmatory_eligible") is False
        and value.get("protocol_sha256") == materializer.PROTOCOL_SHA256
        and value.get("selection_seed") == seed
        and value.get("generation_seed") == dispatch_seed,
        "selection design contract mismatch",
    )
    fields = set(base_analyzer.DesignItem.__dataclass_fields__)
    raw_items = value.get("items")
    _require(isinstance(raw_items, list), "selection design items missing")
    items = []
    for item in raw_items:
        _require(isinstance(item, dict) and set(item) == fields, "design item mismatch")
        items.append(base_analyzer.DesignItem(**item))
    _require(
        len(items) == EXPECTED_GROUPS
        and [item.source_pool_ordinal for item in items] == list(range(EXPECTED_GROUPS))
        and len({item.canonical_prompt_sha256 for item in items}) == EXPECTED_GROUPS,
        "selection design geometry mismatch",
    )
    return str(value["fixed_pool_id"]), tuple(items)


def _analyze_collection(
    *, seed: int, generation_seed: int, run_dir: Path, materialization_root: Path
) -> tuple[tuple[base_analyzer.Observation, ...], dict[str, object]]:
    manifest_path = materialization_root / f"fixed_pool_manifest.v1.{seed}.json"
    design_path = materialization_root / f"selection_design.v1.{seed}.json"
    trace_path = run_dir / "scheduler_trace.v1.jsonl"
    pool_id, items = _load_design(design_path, seed)
    manifest = load_fixed_pool_manifest(manifest_path)
    validate_fixed_pool_materialization(manifest)
    validate_fixed_pool_manifest_design(manifest, materializer.DESIGN_ID)
    _require(
        manifest.pool_id == pool_id and manifest.order_seed == seed,
        "manifest/design binding mismatch",
    )
    trace_report = validate_fixed_pool_trace(
        trace_path, manifest, expected_completions_per_group=EXPECTED_COMPLETIONS
    )
    _require(trace_report.physical_weight_version == 0, "reference changed weights")
    events = tuple(iter_scheduler_trace(trace_path))
    starts = [event for event in events if event.event_type.value == "run_started"]
    _require(
        len(starts) == 1
        and starts[0].run_mode == FIXED_POOL_RUN_MODE
        and starts[0].scalar_summaries.get("generation_study_seed") == generation_seed,
        "reference run identity or generation seed mismatch",
    )
    observations = base_analyzer._observations_from_events(events, items, seed)
    length_rate = sum(item.length_terminations for item in observations) / (
        EXPECTED_GROUPS * EXPECTED_COMPLETIONS
    )
    variable_fraction = (
        sum(item.reward_min < item.reward_max for item in observations)
        / EXPECTED_GROUPS
    )
    checks = {
        "complete_32_groups_512_completions": len(observations) == EXPECTED_GROUPS,
        "zero_learner_steps_and_physical_weight_version_zero": True,
        "zero_administrative_censoring": True,
        "backend_length_termination_rate_le_0_2": length_rate <= 0.2,
        "within_group_reward_variance_fraction_ge_0_25": variable_fraction >= 0.25,
    }
    return observations, {
        "generation_seed": generation_seed,
        "trace_sha256": _sha(trace_path),
        "manifest_sha256": _sha(manifest_path),
        "selection_design_sha256": _sha(design_path),
        "prompt_groups": len(observations),
        "completions": len(observations) * EXPECTED_COMPLETIONS,
        "backend_length_termination_rate": length_rate,
        "within_group_reward_variance_fraction": variable_fraction,
        "validity_checks": checks,
        "all_validity_gates_passed": all(checks.values()),
    }


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    _require(len(left) == len(right) and len(left) > 1, "Pearson input mismatch")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    return (
        sum(a * b for a, b in zip(left_centered, right_centered, strict=True))
        / denominator
        if denominator
        else 0.0
    )


def _harder_ordinals(rewards: Sequence[float]) -> tuple[int, ...]:
    return tuple(
        sorted(range(len(rewards)), key=lambda ordinal: (rewards[ordinal], ordinal))[
            :HARDER_HALF_SIZE
        ]
    )


def analyze(
    runs: Mapping[tuple[int, int], Path], *, materialization_root: Path
) -> tuple[dict[str, object], dict[str, object]]:
    """Return private frozen identities and a privacy-safe public result."""
    expected = tuple(
        (pool_seed, reference_seed)
        for pool_seed, _, reference_seeds, _ in materializer.POOL_SPECS
        for reference_seed in reference_seeds
    )
    _require(tuple(runs) == expected, "all six reference runs are required in order")
    private_pools = []
    public_pools = []
    pooled_left: list[float] = []
    pooled_right: list[float] = []
    all_collection_valid = True
    for pool_seed, _, reference_seeds, scheduler_seed in materializer.POOL_SPECS:
        collections = []
        observation_sets = []
        for generation_seed in reference_seeds:
            observations, record = _analyze_collection(
                seed=pool_seed,
                generation_seed=generation_seed,
                run_dir=runs[(pool_seed, generation_seed)],
                materialization_root=materialization_root,
            )
            observation_sets.append(observations)
            collections.append(record)
            all_collection_valid &= bool(record["all_validity_gates_passed"])
        left_by_ordinal = {
            item.item.source_pool_ordinal: item for item in observation_sets[0]
        }
        right_by_ordinal = {
            item.item.source_pool_ordinal: item for item in observation_sets[1]
        }
        _require(
            set(left_by_ordinal) == set(right_by_ordinal) == set(range(EXPECTED_GROUPS))
            and all(
                left_by_ordinal[index].item.source_prompt_id
                == right_by_ordinal[index].item.source_prompt_id
                for index in range(EXPECTED_GROUPS)
            ),
            "reference prompt identity parity mismatch",
        )
        left = [left_by_ordinal[index].reward_mean for index in range(EXPECTED_GROUPS)]
        right = [
            right_by_ordinal[index].reward_mean for index in range(EXPECTED_GROUPS)
        ]
        pooled_left.extend(left)
        pooled_right.extend(right)
        correlation = _pearson(left, right)
        harder_left = set(_harder_ordinals(left))
        harder_right = set(_harder_ordinals(right))
        overlap = len(harder_left & harder_right)
        frozen_scores = [
            (left[index] + right[index]) / 2 for index in range(EXPECTED_GROUPS)
        ]
        frozen_harder = set(_harder_ordinals(frozen_scores))
        prompt_records = [
            {
                "source_pool_ordinal": index,
                "source_prompt_id": left_by_ordinal[index].item.source_prompt_id,
                "canonical_prompt_sha256": left_by_ordinal[
                    index
                ].item.canonical_prompt_sha256,
                "reference_score": frozen_scores[index],
                "fixed_harder": index in frozen_harder,
            }
            for index in range(EXPECTED_GROUPS)
        ]
        private_pools.append(
            {
                "pool_seed": pool_seed,
                "scheduler_generation_seed": scheduler_seed,
                "reference_generation_seeds": list(reference_seeds),
                "prompts": prompt_records,
            }
        )
        public_pools.append(
            {
                "pool_seed": pool_seed,
                "collections": collections,
                "reward_mean_pearson": correlation,
                "harder_half_overlap": overlap,
                "harder_half_size": HARDER_HALF_SIZE,
                "stability_checks": {
                    "pearson_ge_0_6": correlation >= 0.6,
                    "harder_half_overlap_ge_12": overlap >= 12,
                },
            }
        )
    pooled_correlation = _pearson(pooled_left, pooled_right)
    stability_valid = (
        all(
            all(pool["stability_checks"].values())  # type: ignore[union-attr]
            for pool in public_pools
        )
        and pooled_correlation >= 0.75
    )
    private = {
        "schema_version": 1,
        "analysis_status": "frozen_private_dapo_operational_mixture_reference",
        "protocol_sha256": materializer.PROTOCOL_SHA256,
        "contains_prompt_identities": True,
        "pools": private_pools,
    }
    private_sha = hashlib.sha256(
        json.dumps(private, indent=2, sort_keys=True).encode() + b"\n"
    ).hexdigest()
    checks = {
        "all_six_collections_valid": all_collection_valid,
        "all_three_pool_stability_gates_passed": stability_valid,
        "pooled_reward_mean_pearson_ge_0_75": pooled_correlation >= 0.75,
    }
    public = {
        "schema_version": 1,
        "analysis_status": "calibration_only_dapo_operational_mixture_reference",
        "calibration_only": True,
        "confirmatory_eligible": False,
        "scheduler_arm_authorized": False,
        "counterfactual_replay_authorized": False,
        "training_authorized": False,
        "protocol_sha256": materializer.PROTOCOL_SHA256,
        "private_reference_manifest_sha256": private_sha,
        "pools": public_pools,
        "pooled_reward_mean_pearson": pooled_correlation,
        "locked_checks": checks,
        "decision": (
            "reference_valid_freeze_and_draft_scheduler_plan"
            if all(checks.values())
            else "stop_no_scheduler_shadow_arms"
        ),
    }
    return private, public


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--materialization-root", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.private_output, args.public_output):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    parsed = args.run
    runs = dict(parsed)
    _require(len(runs) == len(parsed), "duplicate reference run")
    private, public = analyze(runs, materialization_root=args.materialization_root)
    private_raw = json.dumps(private, indent=2, sort_keys=True) + "\n"
    public_raw = json.dumps(public, indent=2, sort_keys=True) + "\n"
    args.private_output.write_text(private_raw)
    args.private_output.chmod(0o600)
    args.public_output.write_text(public_raw)
    print(json.dumps(public, sort_keys=True))


if __name__ == "__main__":
    main()
