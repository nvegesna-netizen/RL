# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Privacy-safe post-hoc audit of scheduler selection opportunity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemo_rl.algorithms.async_utils.scheduler_trace import (
    SchedulerEventType,
    SchedulerTraceEvent,
    iter_scheduler_trace,
    validate_scheduler_trace,
)


ANALYSIS_STATUS = "posthoc_read_only_selection_opportunity_audit"


class SelectionOpportunityAnalysisError(ValueError):
    """An input trace or audit manifest violates the analysis contract."""


@dataclass(frozen=True, slots=True)
class GroupMetadata:
    source_prompt_id: str
    dispatch_cohort: int
    reference_summaries: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class Decision:
    logical_step: int
    ready_count: int
    eligible_count: int
    buffered_count: int
    selected_count: int
    eligible_source_prompt_ids: tuple[str, ...]
    selected_source_prompt_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TraceAnalysis:
    trace_sha256: str
    groups: Mapping[str, GroupMetadata]
    decisions: tuple[Decision, ...]
    selected_order: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AuditComparison:
    comparison_id: str
    study_id: str
    replication_id: str
    ready_first_trace: Path
    in_order_trace: Path
    reference_summary_key: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SelectionOpportunityAnalysisError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SelectionOpportunityAnalysisError(
            f"summary {key!r} must be a non-negative integer"
        )
    return value


def _numeric_summaries(values: Mapping[str, object]) -> dict[str, float]:
    summaries = {}
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        _require(math.isfinite(number), f"summary {key!r} must be finite")
        summaries[key] = number
    return summaries


def _current_clean_commit(repo_root: Path) -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(not status, "analysis worktree must be clean")
    return commit


def analyze_trace_events(
    events: Sequence[SchedulerTraceEvent], *, trace_sha256: str
) -> TraceAnalysis:
    """Analyze one already schema-validated, complete scheduler trace."""
    starts = [
        event for event in events if event.event_type is SchedulerEventType.RUN_STARTED
    ]
    ends = [
        event for event in events if event.event_type is SchedulerEventType.RUN_ENDED
    ]
    _require(len(starts) == len(ends) == 1, "trace requires one run boundary pair")
    _require(starts[0].run_mode == "scheduler_assay", "trace is not a scheduler assay")
    _require(
        ends[0].terminal_reason == "scheduler_assay_complete", "assay did not complete"
    )
    _require(
        ends[0].scalar_summaries.get("completed_train_steps") == 0
        and ends[0].scalar_summaries.get("final_physical_weight_version") == 0,
        "trace contains a learner update or physical weight change",
    )

    forbidden = {
        SchedulerEventType.ATTEMPT_FAILED,
        SchedulerEventType.ATTEMPT_REMOVED,
        SchedulerEventType.GROUP_EVICTED,
        SchedulerEventType.ABORT_REQUESTED,
        SchedulerEventType.PROMPT_SKIPPED,
        SchedulerEventType.GROUP_REPLACED,
        SchedulerEventType.GROUP_PROMOTED,
    }
    _require(
        not any(event.event_type in forbidden for event in events),
        "trace has forbidden events",
    )

    groups_by_logical_id: dict[str, GroupMetadata] = {}
    summaries_by_logical_id: dict[str, Mapping[str, float]] = {}
    for event in events:
        if event.event_type is SchedulerEventType.ROLLOUT_COMPLETED:
            if event.logical_group_id is None:
                raise SelectionOpportunityAnalysisError(
                    "completion lacks group identity"
                )
            group_id = event.logical_group_id
            _require(
                group_id not in summaries_by_logical_id,
                "group has duplicate completion events",
            )
            summaries_by_logical_id[group_id] = _numeric_summaries(
                event.scalar_summaries
            )
    for event in events:
        if event.event_type is not SchedulerEventType.ATTEMPT_DISPATCHED:
            continue
        if (
            event.logical_group_id is None
            or event.source_prompt_id is None
            or event.dispatch_cohort is None
        ):
            raise SelectionOpportunityAnalysisError(
                "dispatch lacks required metadata identity"
            )
        group_id = event.logical_group_id
        _require(group_id not in groups_by_logical_id, "duplicate dispatched group")
        _require(
            group_id in summaries_by_logical_id, "dispatched group lacks completion"
        )
        groups_by_logical_id[group_id] = GroupMetadata(
            source_prompt_id=event.source_prompt_id,
            dispatch_cohort=event.dispatch_cohort,
            reference_summaries=summaries_by_logical_id[group_id],
        )
    _require(bool(groups_by_logical_id), "trace contains no completed groups")
    _require(
        len({group.source_prompt_id for group in groups_by_logical_id.values()})
        == len(groups_by_logical_id),
        "source prompt identities are not unique",
    )

    decisions = []
    selected_order = []
    for event in events:
        if (
            event.event_type is not SchedulerEventType.SELECT_DECISION
            or not event.selected_logical_group_ids
        ):
            continue
        step = _integer(event.scalar_summaries, "scheduler_assay_step")
        buffered = _integer(event.scalar_summaries, "buffered_prompt_groups")
        selected = _integer(event.scalar_summaries, "selected_prompt_groups")
        if event.ready_prompt_groups is None or event.eligible_prompt_groups is None:
            raise SelectionOpportunityAnalysisError(
                "selection lacks ready or eligible count"
            )
        _require(
            selected == len(event.selected_logical_group_ids)
            and event.eligible_prompt_groups == len(event.eligible_logical_group_ids),
            "selection count does not match identities",
        )
        _require(
            event.ready_prompt_groups >= event.eligible_prompt_groups,
            "eligible count exceeds ready count",
        )
        _require(
            set(event.selected_logical_group_ids)
            <= set(event.eligible_logical_group_ids),
            "selected groups are not eligible",
        )
        _require(
            all(
                group_id in groups_by_logical_id
                for group_id in event.eligible_logical_group_ids
            ),
            "selection references an unknown group",
        )
        eligible_sources = tuple(
            groups_by_logical_id[group_id].source_prompt_id
            for group_id in event.eligible_logical_group_ids
        )
        selected_sources = tuple(
            groups_by_logical_id[group_id].source_prompt_id
            for group_id in event.selected_logical_group_ids
        )
        decisions.append(
            Decision(
                logical_step=step,
                ready_count=event.ready_prompt_groups,
                eligible_count=event.eligible_prompt_groups,
                buffered_count=buffered,
                selected_count=selected,
                eligible_source_prompt_ids=eligible_sources,
                selected_source_prompt_ids=selected_sources,
            )
        )
        selected_order.extend(selected_sources)
    _require(bool(decisions), "trace contains no nonempty selection decisions")
    _require(
        [decision.logical_step for decision in decisions]
        == list(range(len(decisions))),
        "logical selection steps are not contiguous from zero",
    )
    _require(
        len(selected_order) == len(set(selected_order)) == len(groups_by_logical_id),
        "selection does not drain every source prompt exactly once",
    )
    return TraceAnalysis(
        trace_sha256=trace_sha256,
        groups=groups_by_logical_id,
        decisions=tuple(decisions),
        selected_order=tuple(selected_order),
    )


def analyze_trace(path: Path) -> TraceAnalysis:
    """Validate and analyze a scheduler trace without reading prompt text."""
    report = validate_scheduler_trace(path)
    _require(
        not report.incomplete_attempt_ids
        and not report.administratively_censored_group_ids,
        "trace is incomplete or administratively censored",
    )
    return analyze_trace_events(
        tuple(iter_scheduler_trace(path)), trace_sha256=_sha256(path)
    )


def _arm_summary(analysis: TraceAnalysis) -> dict[str, object]:
    decisions = analysis.decisions
    choice = sum(item.eligible_count > item.selected_count for item in decisions)
    ready_but_ineligible = [
        item.ready_count - item.eligible_count for item in decisions
    ]

    def ratio_summary(numerators: Sequence[int]) -> dict[str, float]:
        ratios = [
            numerator / decision.selected_count
            for numerator, decision in zip(numerators, decisions, strict=True)
        ]
        return {
            "minimum": min(ratios),
            "median": statistics.median(ratios),
            "maximum": max(ratios),
        }

    return {
        "trace_sha256": analysis.trace_sha256,
        "prompt_groups": len(analysis.groups),
        "nonempty_selection_decisions": len(decisions),
        "decisions_with_eligible_greater_than_selected": choice,
        "fraction_with_eligible_greater_than_selected": choice / len(decisions),
        "decisions_with_ready_greater_than_selected": sum(
            item.ready_count > item.selected_count for item in decisions
        ),
        "decisions_with_ready_greater_than_eligible": sum(
            count > 0 for count in ready_but_ineligible
        ),
        "total_ready_but_ineligible_groups": sum(ready_but_ineligible),
        "mean_ready_but_ineligible_groups_per_decision": statistics.fmean(
            ready_but_ineligible
        ),
        "decisions_with_buffered_greater_than_selected": sum(
            item.buffered_count > item.selected_count for item in decisions
        ),
        "ready_to_selected_ratio": ratio_summary(
            [item.ready_count for item in decisions]
        ),
        "eligible_to_selected_ratio": ratio_summary(
            [item.eligible_count for item in decisions]
        ),
        "buffered_to_selected_ratio": ratio_summary(
            [item.buffered_count for item in decisions]
        ),
        "per_decision_counts": [
            {
                "logical_step": item.logical_step,
                "ready": item.ready_count,
                "eligible": item.eligible_count,
                "ready_but_ineligible": item.ready_count - item.eligible_count,
                "buffered": item.buffered_count,
                "selected": item.selected_count,
            }
            for item in decisions
        ],
    }


def _reference_metrics(
    *,
    analysis: TraceAnalysis,
    reference: Mapping[str, float],
) -> dict[str, object]:
    decisions_with_variation = 0
    decisions_with_unselected = 0
    selected_minus_unselected = []
    eligible_spans = []
    for decision in analysis.decisions:
        eligible = [
            reference[source_id] for source_id in decision.eligible_source_prompt_ids
        ]
        _require(
            len(eligible) == decision.eligible_count, "reference coverage mismatch"
        )
        if len(set(eligible)) > 1:
            decisions_with_variation += 1
        eligible_spans.append(max(eligible) - min(eligible))
        unselected_ids = set(decision.eligible_source_prompt_ids) - set(
            decision.selected_source_prompt_ids
        )
        if unselected_ids:
            decisions_with_unselected += 1
            selected_mean = statistics.fmean(
                reference[source_id]
                for source_id in decision.selected_source_prompt_ids
            )
            unselected_mean = statistics.fmean(
                reference[source_id] for source_id in unselected_ids
            )
            selected_minus_unselected.append(selected_mean - unselected_mean)
    return {
        "decisions_with_fixed_reference_variation_within_eligible": decisions_with_variation,
        "decisions_with_eligible_but_unselected_groups": decisions_with_unselected,
        "fixed_reference_span_within_eligible": {
            "minimum": min(eligible_spans),
            "median": statistics.median(eligible_spans),
            "maximum": max(eligible_spans),
        },
        "mean_selected_minus_unselected_fixed_reference": (
            statistics.fmean(selected_minus_unselected)
            if selected_minus_unselected
            else None
        ),
    }


def analyze_comparison(
    *,
    comparison_id: str,
    ready_first: TraceAnalysis,
    in_order: TraceAnalysis,
    reference_summary_key: str,
) -> dict[str, Any]:
    """Compare paired ready-first and in-order traces using an in-order reference."""
    _require(
        set(ready_first.selected_order) == set(in_order.selected_order),
        "cross-arm source prompt identities differ",
    )
    _require(
        len(ready_first.decisions) == len(in_order.decisions),
        "cross-arm selection decision counts differ",
    )
    reference = {}
    for group in in_order.groups.values():
        _require(
            reference_summary_key in group.reference_summaries,
            f"in-order group lacks reference summary {reference_summary_key!r}",
        )
        reference[group.source_prompt_id] = group.reference_summaries[
            reference_summary_key
        ]

    ready_ranks = {
        source_id: index
        for index, source_id in enumerate(ready_first.selected_order, 1)
    }
    ordered_ranks = {
        source_id: index for index, source_id in enumerate(in_order.selected_order, 1)
    }
    displacements = {
        source_id: ordered_ranks[source_id] - ready_ranks[source_id]
        for source_id in ordered_ranks
    }
    groups_per_step = [decision.selected_count for decision in in_order.decisions]
    _require(
        len(set(groups_per_step)) == 1, "in-order selection sizes are inconsistent"
    )
    selection_size = groups_per_step[0]
    crossing = {
        source_id
        for source_id in ordered_ranks
        if (ordered_ranks[source_id] - 1) // selection_size
        != (ready_ranks[source_id] - 1) // selection_size
    }
    ready_groups_by_source = {
        group.source_prompt_id: group for group in ready_first.groups.values()
    }
    ordered_groups_by_source = {
        group.source_prompt_id: group for group in in_order.groups.values()
    }
    _require(
        all(
            ready_groups_by_source[source_id].dispatch_cohort
            == ordered_groups_by_source[source_id].dispatch_cohort
            for source_id in ordered_ranks
        ),
        "cross-arm dispatch cohorts differ",
    )
    ready_selection_step = {
        source_id: step
        for step, decision in enumerate(ready_first.decisions)
        for source_id in decision.selected_source_prompt_ids
    }
    ordered_selection_step = {
        source_id: step
        for step, decision in enumerate(in_order.decisions)
        for source_id in decision.selected_source_prompt_ids
    }
    changed_step_by_dispatch_cohort: dict[str, int] = {}
    for source_id in crossing:
        cohort = str(ordered_groups_by_source[source_id].dispatch_cohort)
        changed_step_by_dispatch_cohort[cohort] = (
            changed_step_by_dispatch_cohort.get(cohort, 0) + 1
        )
    lower_count = len(reference) // 2
    lower_reference = {
        source_id
        for source_id, _ in sorted(
            reference.items(), key=lambda item: (item[1], item[0])
        )[:lower_count]
    }
    step_comparisons = []
    for ready_step, ordered_step in zip(
        ready_first.decisions, in_order.decisions, strict=True
    ):
        ready_set = set(ready_step.selected_source_prompt_ids)
        ordered_set = set(ordered_step.selected_source_prompt_ids)
        intersection = len(ready_set & ordered_set)
        step_comparisons.append(
            {
                "logical_step": ordered_step.logical_step,
                "same_selected_set": ready_set == ordered_set,
                "selected_set_intersection": intersection,
                "selected_set_symmetric_difference": len(ready_set ^ ordered_set),
            }
        )
    changed_positions = sum(
        ready_id != ordered_id
        for ready_id, ordered_id in zip(
            ready_first.selected_order, in_order.selected_order, strict=True
        )
    )
    return {
        "comparison_id": comparison_id,
        "privacy": {
            "prompt_or_completion_text_emitted": False,
            "group_or_source_prompt_identifiers_emitted": False,
        },
        "reference": {
            "arm": "in_order",
            "summary_key": reference_summary_key,
            "distinct_values": len(set(reference.values())),
            "lower_reference_group_count": lower_count,
        },
        "arms": {
            "ready_first": {
                **_arm_summary(ready_first),
                "fixed_reference_opportunity": _reference_metrics(
                    analysis=ready_first, reference=reference
                ),
            },
            "in_order": {
                **_arm_summary(in_order),
                "fixed_reference_opportunity": _reference_metrics(
                    analysis=in_order, reference=reference
                ),
            },
        },
        "cross_arm": {
            "global_selection_positions": len(ready_first.selected_order),
            "differing_global_selection_positions": changed_positions,
            "same_selected_step_sets": sum(
                item["same_selected_set"] for item in step_comparisons
            ),
            "selection_steps": len(step_comparisons),
            "step_comparisons": step_comparisons,
            "groups_crossing_selection_step_boundaries": len(crossing),
            "groups_crossing_selection_step_boundaries_by_dispatch_cohort": dict(
                sorted(changed_step_by_dispatch_cohort.items())
            ),
            "ready_first_groups_selected_outside_dispatch_cohort_step": sum(
                ready_selection_step[source_id]
                != ready_groups_by_source[source_id].dispatch_cohort
                for source_id in ready_selection_step
            ),
            "in_order_groups_selected_outside_dispatch_cohort_step": sum(
                ordered_selection_step[source_id]
                != ordered_groups_by_source[source_id].dispatch_cohort
                for source_id in ordered_selection_step
            ),
            "absolute_rank_displacement": sum(
                abs(value) for value in displacements.values()
            ),
            "absolute_rank_displacement_crossing_step_boundaries": sum(
                abs(displacements[source_id]) for source_id in crossing
            ),
            "lower_reference_net_rank_promotion": sum(
                displacements[source_id] for source_id in lower_reference
            ),
            "lower_reference_mean_normalized_rank_promotion": (
                statistics.fmean(
                    displacements[source_id] for source_id in lower_reference
                )
                / (len(reference) - 1)
            ),
        },
    }


def _load_manifest(path: Path) -> tuple[AuditComparison, ...]:
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SelectionOpportunityAnalysisError(
            "audit manifest is unreadable"
        ) from error
    required = {
        "schema_version",
        "analysis_status",
        "calibration_only",
        "confirmatory_eligible",
        "population_claim_authorized",
        "counterfactual_replay_authorized",
        "training_authorized",
        "comparisons",
    }
    valid_header = (
        isinstance(raw, dict)
        and set(raw) == required
        and raw["schema_version"] == 1
        and raw["analysis_status"] == ANALYSIS_STATUS
        and raw["calibration_only"] is True
        and raw["confirmatory_eligible"] is False
        and raw["population_claim_authorized"] is False
        and raw["counterfactual_replay_authorized"] is False
        and raw["training_authorized"] is False
        and isinstance(raw["comparisons"], list)
        and bool(raw["comparisons"])
    )
    _require(valid_header, "audit manifest contract mismatch")
    comparison_fields = {
        "comparison_id",
        "study_id",
        "replication_id",
        "ready_first_trace",
        "in_order_trace",
        "reference_summary_key",
    }
    comparisons = []
    seen_ids = set()
    for value in raw["comparisons"]:
        _require(
            isinstance(value, dict) and set(value) == comparison_fields,
            "comparison fields mismatch",
        )
        comparison_id = value["comparison_id"]
        reference_key = value["reference_summary_key"]
        _require(
            isinstance(comparison_id, str)
            and bool(comparison_id)
            and comparison_id not in seen_ids,
            "comparison ID must be nonempty and unique",
        )
        _require(
            isinstance(reference_key, str) and bool(reference_key),
            "reference summary key must be nonempty",
        )
        for key in (
            "study_id",
            "replication_id",
            "ready_first_trace",
            "in_order_trace",
        ):
            _require(
                isinstance(value[key], str) and bool(value[key]),
                f"comparison {key} must be a nonempty string",
            )
        seen_ids.add(comparison_id)
        comparisons.append(
            AuditComparison(
                comparison_id=comparison_id,
                study_id=value["study_id"],
                replication_id=value["replication_id"],
                ready_first_trace=Path(value["ready_first_trace"]),
                in_order_trace=Path(value["in_order_trace"]),
                reference_summary_key=reference_key,
            )
        )
    return tuple(comparisons)


def run_audit(*, manifest_path: Path, repo_root: Path) -> dict[str, object]:
    """Run all manifest comparisons and return aggregate privacy-safe facts."""
    manifest_comparisons = _load_manifest(manifest_path)
    comparisons = []
    for item in manifest_comparisons:
        comparison = analyze_comparison(
            comparison_id=item.comparison_id,
            ready_first=analyze_trace(item.ready_first_trace),
            in_order=analyze_trace(item.in_order_trace),
            reference_summary_key=item.reference_summary_key,
        )
        comparison["study_id"] = item.study_id
        comparison["replication_id"] = item.replication_id
        comparisons.append(comparison)
    arm_summaries = [
        comparison["arms"][arm_id]
        for comparison in comparisons
        for arm_id in ("ready_first", "in_order")
    ]
    decisions = sum(item["nonempty_selection_decisions"] for item in arm_summaries)
    choices = sum(
        item["decisions_with_eligible_greater_than_selected"] for item in arm_summaries
    )
    return {
        "schema_version": 1,
        "analysis_status": ANALYSIS_STATUS,
        "calibration_only": True,
        "confirmatory_eligible": False,
        "population_claim_authorized": False,
        "counterfactual_replay_authorized": False,
        "training_authorized": False,
        "new_eos_launch_performed": False,
        "analysis_code_commit": _current_clean_commit(repo_root),
        "input_manifest_sha256": _sha256(manifest_path),
        "privacy": {
            "prompt_or_completion_text_emitted": False,
            "group_or_source_prompt_identifiers_emitted": False,
        },
        "comparisons": comparisons,
        "aggregate": {
            "paired_comparisons": len(comparisons),
            "arms": len(arm_summaries),
            "nonempty_selection_decisions": decisions,
            "decisions_with_eligible_greater_than_selected": choices,
            "fraction_with_eligible_greater_than_selected": choices / decisions,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(manifest_path=args.manifest, repo_root=args.repo_root)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
