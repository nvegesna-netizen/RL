# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Validate neutral references and freeze DAPO difficulty and generated load."""

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
from tools import materialize_dapo_load_alignment as materializer


EXPECTED_GROUPS = 32
EXPECTED_COMPLETIONS = 16
HALF_SIZE = 16


class DapoLoadAlignmentReferenceError(ValueError):
    """A neutral reference artifact or frozen gate is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DapoLoadAlignmentReferenceError(message)


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
    raise DapoLoadAlignmentReferenceError(f"unknown pool seed {seed}")


def _load_design(
    path: Path, seed: int
) -> tuple[str, tuple[base_analyzer.DesignItem, ...]]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise DapoLoadAlignmentReferenceError("invalid selection design") from error
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
        and value.get("analysis_status") == "controlled_dapo_load_alignment_pool"
        and value.get("calibration_only") is True
        and value.get("confirmatory_eligible") is False
        and value.get("protocol_sha256") == materializer.PROTOCOL_SHA256
        and value.get("selection_seed") == seed
        and value.get("generation_seed") == dispatch_seed,
        "selection design contract mismatch",
    )
    fields = set(base_analyzer.DesignItem.__dataclass_fields__)
    raw_items = value.get("items")
    if not isinstance(raw_items, list):
        raise DapoLoadAlignmentReferenceError("selection design items missing")
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


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        rank = (start + 1 + end) / 2
        for index, _ in ordered[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _lower_ordinals(values: Sequence[float]) -> tuple[int, ...]:
    return tuple(
        sorted(range(len(values)), key=lambda ordinal: (values[ordinal], ordinal))[
            :HALF_SIZE
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
    private_pools: list[dict[str, object]] = []
    public_pools: list[dict[str, object]] = []
    pooled_rewards: tuple[list[float], list[float]] = ([], [])
    pooled_loads: tuple[list[float], list[float]] = ([], [])
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
        by_ordinal = [
            {item.item.source_pool_ordinal: item for item in observations}
            for observations in observation_sets
        ]
        _require(
            set(by_ordinal[0]) == set(by_ordinal[1]) == set(range(EXPECTED_GROUPS))
            and all(
                by_ordinal[0][index].item.source_prompt_id
                == by_ordinal[1][index].item.source_prompt_id
                for index in range(EXPECTED_GROUPS)
            ),
            "reference prompt identity parity mismatch",
        )
        rewards = [
            [by_ordinal[side][index].reward_mean for index in range(EXPECTED_GROUPS)]
            for side in (0, 1)
        ]
        loads = [
            [
                by_ordinal[side][index].mean_generated_tokens
                for index in range(EXPECTED_GROUPS)
            ]
            for side in (0, 1)
        ]
        for side in (0, 1):
            pooled_rewards[side].extend(rewards[side])
            pooled_loads[side].extend(loads[side])
        reward_pearson = _pearson(*rewards)
        load_spearman = _spearman(*loads)
        harder_overlap = len(
            set(_lower_ordinals(rewards[0])) & set(_lower_ordinals(rewards[1]))
        )
        lower_load_overlap = len(
            set(_lower_ordinals(loads[0])) & set(_lower_ordinals(loads[1]))
        )
        frozen_rewards = [statistics.fmean(pair) for pair in zip(*rewards, strict=True)]
        frozen_loads = [statistics.fmean(pair) for pair in zip(*loads, strict=True)]
        harder = set(_lower_ordinals(frozen_rewards))
        lower_load = set(_lower_ordinals(frozen_loads))
        prompts = [
            {
                "source_pool_ordinal": index,
                "source_prompt_id": by_ordinal[0][index].item.source_prompt_id,
                "canonical_prompt_sha256": by_ordinal[0][
                    index
                ].item.canonical_prompt_sha256,
                "difficulty_score": frozen_rewards[index],
                "generated_load_score": frozen_loads[index],
                "fixed_harder": index in harder,
                "fixed_lower_load": index in lower_load,
            }
            for index in range(EXPECTED_GROUPS)
        ]
        private_pools.append(
            {
                "pool_seed": pool_seed,
                "scheduler_generation_seed": scheduler_seed,
                "reference_generation_seeds": list(reference_seeds),
                "prompts": prompts,
            }
        )
        checks = {
            "reward_mean_pearson_ge_0_6": reward_pearson >= 0.6,
            "harder_half_overlap_ge_12": harder_overlap >= 12,
            "generated_load_spearman_ge_0_6": load_spearman >= 0.6,
            "lower_load_half_overlap_ge_10": lower_load_overlap >= 10,
        }
        public_pools.append(
            {
                "pool_seed": pool_seed,
                "collections": collections,
                "reward_mean_pearson": reward_pearson,
                "harder_half_overlap": harder_overlap,
                "generated_load_spearman": load_spearman,
                "lower_load_half_overlap": lower_load_overlap,
                "stability_checks": checks,
            }
        )
    pooled_reward = _pearson(*pooled_rewards)
    pooled_load = _spearman(*pooled_loads)
    all_pool_gates = all(
        all(pool["stability_checks"].values())  # type: ignore[union-attr]
        for pool in public_pools
    )
    private: dict[str, object] = {
        "schema_version": 1,
        "analysis_status": "frozen_private_dapo_load_alignment_reference",
        "protocol_sha256": materializer.PROTOCOL_SHA256,
        "contains_prompt_identities": True,
        "pools": private_pools,
    }
    private_sha = hashlib.sha256(
        (json.dumps(private, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    checks = {
        "all_six_collections_valid": all_collection_valid,
        "all_three_pool_stability_gates_passed": all_pool_gates,
        "pooled_reward_mean_pearson_ge_0_75": pooled_reward >= 0.75,
        "pooled_generated_load_spearman_ge_0_7": pooled_load >= 0.7,
    }
    public: dict[str, object] = {
        "schema_version": 1,
        "analysis_status": "calibration_only_dapo_load_alignment_reference",
        "calibration_only": True,
        "confirmatory_eligible": False,
        "scheduler_arm_authorized": False,
        "counterfactual_replay_authorized": False,
        "training_authorized": False,
        "protocol_sha256": materializer.PROTOCOL_SHA256,
        "private_reference_manifest_sha256": private_sha,
        "pools": public_pools,
        "pooled_reward_mean_pearson": pooled_reward,
        "pooled_generated_load_spearman": pooled_load,
        "locked_checks": checks,
        "decision": (
            "reference_valid_freeze_and_draft_scheduler_plan"
            if all(checks.values())
            else "stop_no_scheduler_arms"
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
    parsed: list[tuple[tuple[int, int], Path]] = args.run
    runs: dict[tuple[int, int], Path] = dict(parsed)
    _require(len(runs) == len(parsed), "duplicate reference run")
    private, public = analyze(runs, materialization_root=args.materialization_root)
    args.private_output.write_text(json.dumps(private, indent=2, sort_keys=True) + "\n")
    args.private_output.chmod(0o600)
    args.public_output.write_text(json.dumps(public, indent=2, sort_keys=True) + "\n")
    print(json.dumps(public, sort_keys=True))


if __name__ == "__main__":
    main()
