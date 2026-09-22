#!/usr/bin/env python3
"""Run outcome-aware OARS-v2 reward-composition diagnostics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
PARENT_PATH = HERE / "analyze_offline_autopsy.py"
SPEC = importlib.util.spec_from_file_location("oars_v2_parent", PARENT_PATH)
assert SPEC is not None and SPEC.loader is not None
PARENT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PARENT
SPEC.loader.exec_module(PARENT)


def selection(decision: Any, policy: str) -> tuple[Any, ...]:
    if policy == "m4_risk_band":
        return PARENT.policy_selection(decision, policy)
    if policy == "reward_variance_risk_band":
        return PARENT.select_band(
            decision,
            lambda combination: (
                math.fsum(
                    candidate.reward_variance
                    for candidate in combination
                    if candidate.start_weight_version + 1
                    <= decision.current_learner_version
                ),
                math.fsum(candidate.reward_variance for candidate in combination),
                float(-sum(candidate.valid_actor_tokens for candidate in combination)),
            ),
        )
    if policy == "token_normalized_m4_risk_band":
        return PARENT.select_band(
            decision,
            lambda combination: (
                math.fsum(
                    candidate.l1 / candidate.valid_actor_tokens
                    for candidate in combination
                    if candidate.start_weight_version + 1
                    <= decision.current_learner_version
                ),
                math.fsum(
                    candidate.l1 / candidate.valid_actor_tokens
                    for candidate in combination
                ),
                float(-sum(candidate.valid_actor_tokens for candidate in combination)),
            ),
        )
    raise PARENT.AutopsyError(f"unknown composition policy: {policy}")


def summarize(decisions: Sequence[Any], policy: str) -> dict[str, float]:
    l1 = 0.0
    l2 = 0.0
    imminent_l1 = 0.0
    tokens = 0
    reward_variance = 0.0
    reward_mean = 0.0
    overlaps = 0
    for decision in decisions:
        selected = selection(decision, policy)
        l1 += math.fsum(candidate.l1 for candidate in selected)
        l2 += math.fsum(candidate.l2 for candidate in selected)
        imminent_l1 += math.fsum(
            candidate.l1
            for candidate in selected
            if candidate.start_weight_version + 1
            <= decision.current_learner_version
        )
        tokens += sum(candidate.valid_actor_tokens for candidate in selected)
        reward_variance += math.fsum(candidate.reward_variance for candidate in selected)
        reward_mean += math.fsum(candidate.mean_reward for candidate in selected)
        overlaps += len(
            {candidate.group_id for candidate in selected}
            & set(decision.baseline_group_ids)
        )
    count = len(decisions) * PARENT.SELECTED_GROUPS
    fifo_tokens = sum(decision.baseline_tokens for decision in decisions)
    return {
        "selected_l1": l1,
        "selected_l2": l2,
        "selected_imminent_l1": imminent_l1,
        "selected_valid_actor_tokens": float(tokens),
        "aggregate_token_ratio_to_fifo": tokens / fifo_tokens,
        "mean_group_reward": reward_mean / count,
        "mean_group_reward_variance": reward_variance / count,
        "mean_fifo_overlap_fraction": overlaps / count,
    }


def compare_selections(
    decisions: Sequence[Any], left_policy: str, right_policy: str
) -> dict[str, float]:
    overlap = 0
    identical = 0
    for decision in decisions:
        left = {candidate.group_id for candidate in selection(decision, left_policy)}
        right = {candidate.group_id for candidate in selection(decision, right_policy)}
        overlap += len(left & right)
        identical += left == right
    return {
        "mean_overlap_fraction": overlap / (len(decisions) * PARENT.SELECTED_GROUPS),
        "identical_selection_fraction": identical / len(decisions),
    }


def analyze(
    repository: Path,
    archive_root: Path,
    report_dir: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    protocol = PARENT.load_object(protocol_path)
    PARENT.require(
        protocol.get("schema") == "m4-oars-v2-composition-extension-protocol-v1",
        "composition protocol schema differs",
    )
    policies = (
        "reward_variance_risk_band",
        "token_normalized_m4_risk_band",
        "m4_risk_band",
    )
    arms: list[dict[str, Any]] = []
    for pair in range(1, 11):
        for mode in ("fifo", "oars"):
            identity = f"p{pair:02d}-{mode}"
            _, ledgers = PARENT.archive_ledgers(
                repository, archive_root, report_dir, identity
            )
            opportunities = PARENT.opportunity_map(
                ledgers["opportunity.jsonl"], identity=identity
            )
            decisions, _ = PARENT.reconstruct_decisions(
                identity,
                ledgers["lifecycle.jsonl"],
                ledgers["oars.jsonl"],
                opportunities,
            )
            summaries = {policy: summarize(decisions, policy) for policy in policies}
            m4 = summaries["m4_risk_band"]
            reward = summaries["reward_variance_risk_band"]
            normalized = summaries["token_normalized_m4_risk_band"]
            arms.append(
                {
                    "identity": identity,
                    "policies": summaries,
                    "m4_l1_gain_over_reward_variance_percent": PARENT.percent_gain(
                        m4["selected_l1"], reward["selected_l1"]
                    ),
                    "m4_l1_gain_over_token_normalized_percent": PARENT.percent_gain(
                        m4["selected_l1"], normalized["selected_l1"]
                    ),
                    "m4_vs_reward_variance_selection": compare_selections(
                        decisions, "m4_risk_band", "reward_variance_risk_band"
                    ),
                    "m4_vs_token_normalized_selection": compare_selections(
                        decisions,
                        "m4_risk_band",
                        "token_normalized_m4_risk_band",
                    ),
                }
            )

    aggregate: dict[str, Any] = {}
    for policy in policies:
        aggregate[policy] = {
            metric: math.fsum(float(arm["policies"][policy][metric]) for arm in arms)
            for metric in (
                "selected_l1",
                "selected_l2",
                "selected_imminent_l1",
                "selected_valid_actor_tokens",
            )
        }
        aggregate[policy]["mean_group_reward"] = sum(
            float(arm["policies"][policy]["mean_group_reward"]) for arm in arms
        ) / len(arms)
        aggregate[policy]["mean_group_reward_variance"] = sum(
            float(arm["policies"][policy]["mean_group_reward_variance"])
            for arm in arms
        ) / len(arms)
        fifo_tokens = math.fsum(
            float(arm["policies"][policy]["selected_valid_actor_tokens"])
            / float(arm["policies"][policy]["aggregate_token_ratio_to_fifo"])
            for arm in arms
        )
        aggregate[policy]["aggregate_token_ratio_to_fifo"] = (
            aggregate[policy]["selected_valid_actor_tokens"] / fifo_tokens
        )

    m4 = aggregate["m4_risk_band"]
    reward = aggregate["reward_variance_risk_band"]
    normalized = aggregate["token_normalized_m4_risk_band"]
    return {
        "schema": "m4-oars-v2-composition-extension-result-v1",
        "status": "COMPLETE_EXPLORATORY_DIAGNOSTIC",
        "protocol": {
            "path": str(protocol_path.relative_to(repository)),
            "sha256": PARENT.sha256_file(protocol_path),
        },
        "scope": {"authenticated_arms": len(arms), "reconstructed_decisions": 1280},
        "aggregate": aggregate,
        "diagnostics": {
            "m4_l1_gain_over_reward_variance_percent": PARENT.percent_gain(
                m4["selected_l1"], reward["selected_l1"]
            ),
            "m4_imminent_l1_gain_over_reward_variance_percent": PARENT.percent_gain(
                m4["selected_imminent_l1"], reward["selected_imminent_l1"]
            ),
            "m4_l1_positive_vs_reward_variance_arms": sum(
                arm["m4_l1_gain_over_reward_variance_percent"] > 0 for arm in arms
            ),
            "m4_l1_gain_over_token_normalized_percent": PARENT.percent_gain(
                m4["selected_l1"], normalized["selected_l1"]
            ),
            "m4_l1_positive_vs_token_normalized_arms": sum(
                arm["m4_l1_gain_over_token_normalized_percent"] > 0 for arm in arms
            ),
            "m4_vs_reward_variance_mean_overlap_fraction": sum(
                arm["m4_vs_reward_variance_selection"]["mean_overlap_fraction"]
                for arm in arms
            )
            / len(arms),
            "m4_vs_reward_variance_identical_selection_fraction": sum(
                arm["m4_vs_reward_variance_selection"]["identical_selection_fraction"]
                for arm in arms
            )
            / len(arms),
            "m4_vs_token_normalized_mean_overlap_fraction": sum(
                arm["m4_vs_token_normalized_selection"]["mean_overlap_fraction"]
                for arm in arms
            )
            / len(arms),
        },
        "arms": arms,
        "interpretation_boundary": protocol["interpretation_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.repository.resolve(),
        args.archive_root.resolve(),
        args.report_dir.resolve(),
        args.protocol.resolve(),
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "scope", "diagnostics")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
