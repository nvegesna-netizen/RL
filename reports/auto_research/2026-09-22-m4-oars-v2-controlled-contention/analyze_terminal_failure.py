# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reproduce the authenticated controlled-frontier failure autopsy."""

from __future__ import annotations

import collections
import hashlib
import json
import math
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
ARCHIVE = (
    ROOT / "session/20260914_m4_opportunity_control/"
    "oars-v2-controlled-frontier-terminal-archives/main-451709979.zip"
)
AUTHENTICATION = REPORT / "terminal_authentication.json"
OUTPUT = REPORT / "terminal_failure_autopsy.json"
PREFIX = "workspace/assets/basic/m4-oars-v2-controlled-frontier-qualification/"
OARS = PREFIX + "m4-oars-v2-controlled-frontier-qualification-oars-v2.jsonl"
RESULT = PREFIX + "m4-oars-v2-controlled-frontier-qualification-result.json"
OBSERVER = PREFIX + "m4-oars-v2-controlled-frontier-qualification-observer-duty.json"


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def p95(values: list[int]) -> int:
    """Return the nearest-rank 95th percentile."""
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def main() -> None:
    """Authenticate the input again, then write a compact failure diagnosis."""
    authentication = json.loads(AUTHENTICATION.read_text())
    if authentication.get("status") != "PASS_AUTHENTICATED_RESULT_UNOPENED":
        raise RuntimeError("terminal authentication did not pass")
    if authentication.get("result_opened") is not False:
        raise RuntimeError("authentication ordering record is invalid")
    if sha256(ARCHIVE) != authentication["archives"]["main_sha256"]:
        raise RuntimeError("main archive no longer matches authentication")

    with zipfile.ZipFile(ARCHIVE) as archive:
        result = json.loads(archive.read(RESULT))
        observer = json.loads(archive.read(OBSERVER))
        lines = archive.read(OARS).splitlines()
    rows = [json.loads(line) for line in lines[1:]]
    selections = [row for row in rows if row.get("actual_selected_group_count") == 4]
    exact_eight = [row for row in rows if row["candidate_group_count"] == 8]
    polling = [row for row in rows if row.get("actual_selected_group_count") == 0]
    scorers = tuple(rows[0]["proposals"])
    fallbacks = [
        {
            "candidate_group_count": row["candidate_group_count"],
            "actual_selected_group_count": row["actual_selected_group_count"],
            "fallback_reason": proposal["fallback_reason"],
            "latency_ns": proposal["decision_latency_ns"],
            "scorer": scorer,
        }
        for row in rows
        for scorer, proposal in row["proposals"].items()
        if proposal["status"] == "fallback"
    ]
    selected_latencies = [row["decision_latency_ns"] for row in selections]
    selected_oars_duty = sum(selected_latencies) / observer["active_window_ns"]
    selected_combined_duty = selected_oars_duty + observer["corrected_observer_duty"]
    selected_contracts = all(
        proposal.get("status") == "proposed"
        and proposal.get("combination_count") == 70
        and proposal.get("search_candidate_count") == 8
        and proposal.get("search_strategy") == "exact"
        for row in selections
        for proposal in row["proposals"].values()
    )
    diagnosis = {
        "schema": "m4-oars-v2-controlled-frontier-terminal-failure-autopsy-v1",
        "status": "IMPLEMENTATION_DEFECT_IDENTIFIED_REPAIR_REQUIRED",
        "source_commit": result["source_commit"],
        "authenticated_result_sha256": authentication["artifacts"][Path(RESULT).name][
            "sha256"
        ],
        "scientific_outcome_acquisition": False,
        "training_quality_analyzed": False,
        "observed_gate_status": result["status"],
        "root_cause": (
            "OpportunityAtRiskV2ShadowSampler observed and recorded every select "
            "poll before WeightFIFO enforced its eight-ready-group watermark."
        ),
        "comparison_to_validated_v1": (
            "The validated v1 observer returns before scoring when ready candidates "
            "are below the WeightFIFO watermark; v2 omitted that guard."
        ),
        "all_observations": {
            "count": len(rows),
            "candidate_count_distribution": dict(
                sorted(
                    collections.Counter(
                        row["candidate_group_count"] for row in rows
                    ).items()
                )
            ),
            "polling_without_selection_count": len(polling),
            "fallbacks": fallbacks,
            "oars_observer_duty": result["oars_observer_duty"],
            "combined_observer_duty": result["combined_observer_duty"],
        },
        "actual_selection_subset": {
            "count": len(selections),
            "equals_exact_eight_candidate_subset": selections == exact_eight,
            "candidate_count_distribution": dict(
                sorted(
                    collections.Counter(
                        row["candidate_group_count"] for row in selections
                    ).items()
                )
            ),
            "fifo_identity_count": sum(
                row.get("baseline_matches_actual") is True for row in selections
            ),
            "proposal_count": len(selections) * len(scorers),
            "exact_proposal_contracts": selected_contracts,
            "fallback_count": sum(
                proposal["status"] == "fallback"
                for row in selections
                for proposal in row["proposals"].values()
            ),
            "oars_latency_sum_ns": sum(selected_latencies),
            "oars_latency_p95_ns": p95(selected_latencies),
            "descriptive_oars_duty": selected_oars_duty,
            "descriptive_combined_duty": selected_combined_duty,
        },
        "interpretation": {
            "qualification_passed": False,
            "strict_gate_correctly_failed": True,
            "selected_subset_is_not_a_retroactive_pass": True,
            "repair": (
                "Mirror the validated watermark guard and restrict controlled "
                "observation to the common first-eight frontier."
            ),
            "requalification_required": True,
        },
    }
    OUTPUT.write_text(json.dumps(diagnosis, indent=2, sort_keys=True) + "\n")
    print(json.dumps(diagnosis, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
