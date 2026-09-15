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

"""Clean-room replay of the authenticated live OARS shadow preflight."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import zipfile
from pathlib import Path
from typing import Any

EXPECTED_ARCHIVE_SHA256 = (
    "13af85a685aa51dd6d4f377d0812b303acb007f23d5ba2d4b3bab58027e52677"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(archive: zipfile.ZipFile, suffix: str) -> list[dict[str, Any]]:
    names = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(names) != 1:
        raise ValueError(f"expected one {suffix} member, found {len(names)}")
    return [json.loads(line) for line in archive.read(names[0]).splitlines()]


def _recompute(
    candidates: list[dict[str, Any]],
    *,
    baseline_ids: list[str],
    current_version: int,
    multiplier: float,
) -> dict[str, Any]:
    by_id = {row["group_id"]: row for row in candidates}
    baseline = [by_id[group_id] for group_id in baseline_ids]
    baseline_tokens = sum(row["valid_actor_tokens"] for row in baseline)
    token_budget = math.floor(baseline_tokens * multiplier + 1e-12)
    best_ids: tuple[str, ...] | None = None
    best_score: tuple[float, float, int] | None = None
    combinations = 0
    feasible = 0
    for selected in itertools.combinations(sorted(by_id), len(baseline_ids)):
        combinations += 1
        rows = [by_id[group_id] for group_id in selected]
        tokens = sum(row["valid_actor_tokens"] for row in rows)
        if tokens > token_budget:
            continue
        feasible += 1
        imminent_l1 = math.fsum(
            row["opportunity"]
            for row in rows
            if row["start_weight_version"] + 1 <= current_version
        )
        score = (
            imminent_l1,
            math.fsum(row["opportunity"] for row in rows),
            -tokens,
        )
        if best_score is None or score > best_score:
            best_ids = selected
            best_score = score
    if best_ids is None:
        raise ValueError("no feasible selection")
    selected_rows = [by_id[group_id] for group_id in best_ids]
    return {
        "combination_count": combinations,
        "feasible_combination_count": feasible,
        "proposed_group_ids": list(best_ids),
        "proposed_l1": math.fsum(row["opportunity"] for row in selected_rows),
        "proposed_valid_actor_tokens": sum(
            row["valid_actor_tokens"] for row in selected_rows
        ),
        "token_budget": token_budget,
    }


def replay(archive_path: Path) -> dict[str, Any]:
    archive_sha256 = _sha256(archive_path)
    if archive_sha256 != EXPECTED_ARCHIVE_SHA256:
        raise ValueError(
            f"archive SHA-256 mismatch: {archive_sha256} != {EXPECTED_ARCHIVE_SHA256}"
        )
    with zipfile.ZipFile(archive_path) as archive:
        lifecycle = _rows(archive, "lifecycle.jsonl")
        opportunity_rows = _rows(archive, "opportunity.jsonl")
        oars_rows = _rows(archive, "oars.jsonl")

    header, *decisions = oars_rows
    if header["event_type"] != "header":
        raise ValueError("OARS header missing")
    opportunity = {
        row["group_id"]: row
        for row in opportunity_rows
        if row["event_type"] == "group"
    }
    ready_at = {
        row["group_id"]: row["timestamp_ns"]
        for row in lifecycle
        if row["stage"] == "group_ready"
    }
    removed_at = {
        row["group_id"]: row["timestamp_ns"]
        for row in lifecycle
        if row["stage"] == "removed"
    }

    failures: list[str] = []
    candidate_counts: list[int] = []
    for index, decision in enumerate(decisions, start=1):
        actual_ids = decision["actual_selected_group_ids"]
        decision_ns = min(removed_at[group_id] for group_id in actual_ids)
        current_version = decision["current_learner_version"]
        minimum_version = max(0, current_version - 1)
        candidate_ids = sorted(
            group_id
            for group_id, ready_ns in ready_at.items()
            if ready_ns < decision_ns
            and removed_at.get(group_id, math.inf) >= decision_ns
            and minimum_version
            <= opportunity[group_id]["start_weight_version"]
            <= current_version
        )
        candidates = [opportunity[group_id] for group_id in candidate_ids]
        candidate_counts.append(len(candidates))
        replayed = _recompute(
            candidates,
            baseline_ids=decision["baseline_group_ids"],
            current_version=current_version,
            multiplier=header["service_budget_multiplier"],
        )
        comparisons = {
            "candidate_group_count": len(candidates),
            **replayed,
        }
        for key, value in comparisons.items():
            recorded = decision[key]
            if isinstance(value, float):
                matches = math.isclose(value, recorded, rel_tol=0.0, abs_tol=1e-9)
            else:
                matches = value == recorded
            if not matches:
                failures.append(
                    f"decision {index} {key}: replay={value!r}, recorded={recorded!r}"
                )

    return {
        "archive_sha256": archive_sha256,
        "candidate_group_count_max": max(candidate_counts),
        "candidate_group_count_min": min(candidate_counts),
        "decision_count": len(decisions),
        "failure_count": len(failures),
        "failures": failures,
        "policy": header["policy"],
        "replay_status": "PASS" if not failures else "FAIL",
        "service_budget_multiplier": header["service_budget_multiplier"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = replay(args.archive)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if result["replay_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
