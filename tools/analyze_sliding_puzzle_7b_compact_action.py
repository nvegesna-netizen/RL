# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Analyze the frozen exposed-pool 7B compact-action prompt calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Final

from nemo_rl.algorithms.async_utils.fixed_pool import (
    load_fixed_pool_manifest,
    validate_fixed_pool_manifest_design,
    validate_fixed_pool_materialization,
    validate_fixed_pool_trace,
)
from nemo_rl.algorithms.async_utils.scheduler_trace import (
    SchedulerEventType,
    iter_scheduler_trace,
)
from tools import analyze_sliding_puzzle_7b_competence as base_analysis
from tools import materialize_sliding_puzzle_7b_compact_action as materializer


EXPECTED_RUNTIME: Final[dict[str, object]] = {
    **base_analysis.EXPECTED_RUNTIME,
    "fixed_pool_design_id": materializer.DESIGN_ID,
}


class CompactActionAnalysisError(ValueError):
    """The compact-action materialization or trace violates its frozen plan."""


def _validate_lineage(
    *,
    manifest: Any,
    plan: dict[str, Any],
    lineage_path: Path,
    items: tuple[base_analysis.DesignItem, ...],
) -> dict[str, Any]:
    lineage = base_analysis._load_json(lineage_path)
    source = plan["source_pool"]
    expected = {
        "schema_version": 2,
        "analysis_status": plan["analysis_status"],
        "plan_id": materializer.PLAN_ID,
        "plan_sha256": manifest.design_protocol_sha256,
        "source_manifest_sha256": source["fixed_pool_manifest_sha256"],
        "source_pool_id": source["pool_id"],
        "source_selection_design_sha256": source["selection_design_sha256"],
        "board_state_sha256": sorted(item.board_state_sha256 for item in items),
        "materialized_manifest_sha256": manifest.manifest_sha256,
        "materialized_pool_id": manifest.pool_id,
    }
    if lineage != expected:
        raise CompactActionAnalysisError("source-lineage binding mismatch")
    return lineage


def analyze(
    *, trace_path: Path, manifest_path: Path, plan_path: Path, lineage_path: Path
) -> dict[str, object]:
    """Validate the frozen run and compute prompt-competence metrics."""
    plan, _ = materializer._load_plan(plan_path)
    manifest = load_fixed_pool_manifest(manifest_path)
    validate_fixed_pool_materialization(manifest)
    validate_fixed_pool_manifest_design(manifest, materializer.DESIGN_ID)
    if base_analysis._sha256_path(plan_path) != manifest.design_protocol_sha256:
        raise CompactActionAnalysisError("manifest/plan hash mismatch")
    items = base_analysis._design_items(manifest)
    lineage = _validate_lineage(
        manifest=manifest, plan=plan, lineage_path=lineage_path, items=items
    )
    report = validate_fixed_pool_trace(
        trace_path, manifest, expected_completions_per_group=2
    )
    if report.physical_weight_version != 0:
        raise CompactActionAnalysisError("calibration requires weight version zero")
    events = tuple(iter_scheduler_trace(trace_path))
    observations = base_analysis.observations_from_events(
        events,
        items,
        expected_runtime=EXPECTED_RUNTIME,
        generation_seed=63001,
    )
    starts = [
        event for event in events if event.event_type is SchedulerEventType.RUN_STARTED
    ]
    runtime = starts[0].scalar_summaries
    expected_path = str(manifest_path.resolve().parent / "model_snapshot")
    expected_path_sha = hashlib.sha256(
        base_analysis._canonical(expected_path)
    ).hexdigest()
    if (
        runtime.get("policy_model_name_sha256") != expected_path_sha
        or runtime.get("policy_tokenizer_name_sha256") != expected_path_sha
    ):
        raise CompactActionAnalysisError("runtime model snapshot binding mismatch")
    result = base_analysis._summarize(observations)
    result.update(
        {
            "schema_version": 2,
            "analysis_status": plan["analysis_status"],
            "plan_id": materializer.PLAN_ID,
            "interpretation": (
                "paired prompt-competence calibration on an exposed board pool; "
                "no scheduler-latency inference"
            ),
            "runtime_binding": dict(runtime),
            "source_lineage": lineage,
            "source_artifacts": {
                "trace_sha256": base_analysis._sha256_path(trace_path),
                "manifest_sha256": manifest.manifest_sha256,
                "plan_sha256": base_analysis._sha256_path(plan_path),
                "lineage_sha256": base_analysis._sha256_path(lineage_path),
                "pool_id": manifest.pool_id,
                "physical_weight_version": report.physical_weight_version,
            },
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = analyze(
        trace_path=args.trace,
        manifest_path=args.manifest,
        plan_path=args.plan,
        lineage_path=args.lineage,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
