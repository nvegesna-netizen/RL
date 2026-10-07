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

"""Run the frozen out-of-domain M4 causal-risk transport analysis."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PAPER_DIR = ROOT / "reports/auto_research/2026-09-08-m4-opportunity-loss-paper"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PAPER_DIR))

from tools.m4_llama3b_terminal_analysis import THREE_B_CELLS  # noqa: E402
from tools.m4_llama_v5_terminal_analysis import CELLS as ONE_B_CELLS  # noqa: E402
from tools.opportunity_ledger_join import (  # noqa: E402
    LedgerJoinProtocol,
    ReleaseArm,
    join_opportunity_ledgers,
)
from tools.opportunity_loss_pipeline import _parse_protocol  # noqa: E402

from run_publication_synthesis import numina_join_protocol  # noqa: E402


CONVENTIONAL_FEATURES = (
    "start_progress",
    "learner_version_lag",
    "log_valid_actor_tokens",
    "reward_mean",
    "reward_variance",
    "truncated_fraction",
    "log_max_generation_duration_ns",
    "ready_buffer_depth",
    "reserved_buffer_occupancy",
    "active_release_holds",
    "generation_inflight",
    "buffer_admission_stalls",
)
M4_FEATURES = CONVENTIONAL_FEATURES + ("log_m4", "zero_m4")
SCORES = (
    "version_age",
    "reward_variance",
    "conventional_delay_risk",
    "reward_variance_at_risk",
    "m4_at_risk",
    "m4_augmented_at_risk",
)
EXPECTED_ASSIGNMENTS = 106_653


@dataclasses.dataclass(frozen=True)
class Assignment:
    """One authenticated primary-window assignment and pre-hold features."""

    assignment_id: str
    acquisition: str
    family: str
    model: str
    workload: str
    arm: str
    opportunity: float
    delivered: bool
    features: tuple[float, ...]

    @property
    def lost(self) -> bool:
        """Return whether the registered opportunity was not delivered."""
        return not self.delivered


@dataclasses.dataclass(frozen=True)
class SourceCell:
    """One authenticated acquisition input."""

    acquisition: str
    family: str
    model: str
    workload: str
    lifecycle: Path
    opportunity: Path
    lifecycle_sha256: str
    opportunity_sha256: str
    protocol: LedgerJoinProtocol


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    """Read a strict newline-terminated JSONL file."""
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise RuntimeError(f"malformed ledger: {path}")
    return [json.loads(line) for line in raw.splitlines()]


def _qwen_cells(source_root: Path) -> list[SourceCell]:
    synthesis = json.loads(
        (
            ROOT
            / "reports/auto_research/2026-09-08-m4-opportunity-loss-paper/six_cell_synthesis.json"
        ).read_bytes()
    )
    numina_path = (
        ROOT
        / "reports/auto_research/2026-09-07-m4-opportunity-loss-numinamath-generalization/protocol_config.json"
    )
    numina = json.loads(numina_path.read_bytes())
    result = []
    for name in sorted(synthesis["cell_results"]):
        evidence = synthesis["evidence"][name]
        protocol_path = ROOT / evidence["protocol_path"]
        if name.endswith("numinamath"):
            protocol = numina_join_protocol(numina, name)
        else:
            _, protocol, _ = _parse_protocol(protocol_path.read_bytes())
        model, workload = name.rsplit("_", 1)
        result.append(
            SourceCell(
                acquisition=f"qwen-{name}",
                family="qwen",
                model=model,
                workload=workload,
                lifecycle=source_root / evidence["lifecycle_path"],
                opportunity=source_root / evidence["opportunity_path"],
                lifecycle_sha256=evidence["lifecycle_sha256"],
                opportunity_sha256=evidence["opportunity_sha256"],
                protocol=protocol,
            )
        )
    return result


def _llama_cells(source_root: Path) -> list[SourceCell]:
    session = source_root / "session/20260909_m4_llama_lifecycle_derived_transport"
    auth_paths = {
        "1b": ROOT
        / "reports/auto_research/2026-09-09-m4-llama-lifecycle-derived-transport/v5_terminal_authentication.json",
        "3b": ROOT
        / "reports/auto_research/2026-09-10-m4-llama3p2-3b-size-extension/terminal_authentication.json",
    }
    roots = {
        "1b": session / "v5-terminal-artifacts/authenticated",
        "3b": session / "llama3b-terminal-artifacts/authenticated",
    }
    result = []
    for size, cells in (("1b", ONE_B_CELLS), ("3b", THREE_B_CELLS)):
        authentication = json.loads(auth_paths[size].read_bytes())
        for cell, (workload, _, domain, seed, _) in cells.items():
            auth = authentication["cells"][cell.replace("-", "_")]
            source = roots[size] / cell
            result.append(
                SourceCell(
                    acquisition=f"llama-{size}-{cell}",
                    family="llama",
                    model=f"llama3p2_{size}",
                    workload=workload,
                    lifecycle=source / "lifecycle.jsonl",
                    opportunity=source / "opportunity.jsonl",
                    lifecycle_sha256=auth["lifecycle_sha256"],
                    opportunity_sha256=auth["opportunity_sha256"],
                    protocol=LedgerJoinProtocol(
                        assignment_domain=domain,
                        assignment_seed=seed,
                        arms=(
                            ReleaseArm("control", 0.0, 1),
                            ReleaseArm("d5", 5.0, 1),
                        ),
                        primary_start_version=8,
                        primary_end_version=407,
                        siblings_per_group=8,
                        train_batch_size=32,
                    ),
                )
            )
    return result


def source_cells(qwen_root: Path, llama_root: Path) -> list[SourceCell]:
    """Resolve the frozen 14 authenticated acquisition inputs."""
    cells = _qwen_cells(qwen_root) + _llama_cells(llama_root)
    if len(cells) != 14 or len({cell.acquisition for cell in cells}) != 14:
        raise RuntimeError("frozen acquisition identity coverage differs")
    return cells


def _number(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return math.nan
    converted = float(value)
    return converted if math.isfinite(converted) else math.nan


def opportunity_precedes_hold(
    opportunity: Mapping[str, Any], hold: Mapping[str, Any]
) -> bool:
    """Verify the schema-specific pre-hold opportunity ordering contract."""
    if opportunity.get("schema_version") == 2:
        source_sequence = _number(opportunity, "source_controller_sequence_max")
        hold_sequence = _number(hold, "controller_sequence")
        return source_sequence < hold_sequence
    opportunity_time = _number(opportunity, "timestamp_ns")
    hold_time = _number(hold, "timestamp_ns")
    return opportunity_time < hold_time


def _extract_cell(cell: SourceCell) -> list[Assignment]:
    if sha256(cell.lifecycle) != cell.lifecycle_sha256:
        raise RuntimeError(f"{cell.acquisition}: lifecycle digest differs")
    if sha256(cell.opportunity) != cell.opportunity_sha256:
        raise RuntimeError(f"{cell.acquisition}: opportunity digest differs")
    lifecycle = rows(cell.lifecycle)
    opportunity_rows = rows(cell.opportunity)
    joined = join_opportunity_ledgers(
        protocol=cell.protocol,
        lifecycle_rows=lifecycle,
        opportunity_rows=opportunity_rows,
    )
    opportunity_by_group = {
        row["group_id"]: row
        for row in opportunity_rows
        if row.get("event_type") == "group"
    }
    hold_start: dict[str, Mapping[str, Any]] = {}
    max_generation: dict[str, float] = {}
    for row in lifecycle:
        group_id = row.get("group_id")
        if not isinstance(group_id, str):
            continue
        stage = row.get("stage")
        if stage == "release_delay_started":
            if group_id in hold_start:
                raise RuntimeError(f"{cell.acquisition}: duplicate hold-start row")
            hold_start[group_id] = row
        elif stage == "sibling_done":
            duration = _number(row, "generation_duration_ns")
            if math.isfinite(duration):
                max_generation[group_id] = max(
                    duration, max_generation.get(group_id, 0.0)
                )
    assignments = []
    for item in joined:
        group = opportunity_by_group[item.assignment_id]
        hold = hold_start.get(item.assignment_id)
        if hold is None:
            raise RuntimeError(
                f"{cell.acquisition}: primary assignment lacks hold start"
            )
        if not opportunity_precedes_hold(group, hold):
            raise RuntimeError(f"{cell.acquisition}: opportunity is not pre-hold")
        if hold.get("release_arm") != item.arm:
            raise RuntimeError(f"{cell.acquisition}: treatment identity differs")
        siblings = group.get("siblings")
        if not isinstance(siblings, list) or len(siblings) != 8:
            raise RuntimeError(f"{cell.acquisition}: malformed sibling group")
        rewards = [_number(sibling, "reward") for sibling in siblings]
        if not all(math.isfinite(value) for value in rewards):
            raise RuntimeError(f"{cell.acquisition}: non-finite reward")
        reward_mean = math.fsum(rewards) / len(rewards)
        reward_variance = math.fsum(
            (value - reward_mean) ** 2 for value in rewards
        ) / len(rewards)
        truncated = sum(bool(sibling.get("truncated")) for sibling in siblings) / len(
            siblings
        )
        learner_version = _number(hold, "learner_weight_version")
        if not math.isfinite(learner_version):
            raise RuntimeError(f"{cell.acquisition}: hold start lacks learner version")
        window = cell.protocol.primary_end_version - cell.protocol.primary_start_version
        features = (
            (item.start_version - cell.protocol.primary_start_version) / window,
            learner_version - item.start_version,
            math.log1p(_number(group, "valid_actor_tokens")),
            reward_mean,
            reward_variance,
            truncated,
            math.log1p(max_generation.get(item.assignment_id, math.nan)),
            _number(hold, "ready_buffer_depth"),
            _number(hold, "reserved_buffer_occupancy"),
            _number(hold, "active_release_holds"),
            _number(hold, "generation_inflight"),
            _number(hold, "buffer_admission_stalls"),
            math.log1p(item.opportunity),
            float(item.opportunity == 0.0),
        )
        assignments.append(
            Assignment(
                assignment_id=item.assignment_id,
                acquisition=cell.acquisition,
                family=cell.family,
                model=cell.model,
                workload=cell.workload,
                arm=item.arm,
                opportunity=item.opportunity,
                delivered=bool(item.delivered),
                features=features,
            )
        )
    return assignments


def load_assignments(cells: Sequence[SourceCell]) -> list[Assignment]:
    """Authenticate and extract every frozen assignment."""
    result = []
    for cell in cells:
        result.extend(_extract_cell(cell))
    if len(result) != EXPECTED_ASSIGNMENTS:
        raise RuntimeError(
            f"assignment coverage differs: {len(result)} != {EXPECTED_ASSIGNMENTS}"
        )
    if len({(row.acquisition, row.assignment_id) for row in result}) != len(result):
        raise RuntimeError("assignment identities are not unique within acquisition")
    return result


def feature_matrix(rows_: Sequence[Assignment], *, augmented: bool) -> np.ndarray:
    """Build the registered numeric matrix."""
    width = len(M4_FEATURES if augmented else CONVENTIONAL_FEATURES)
    return np.asarray([row.features[:width] for row in rows_], dtype=np.float64)


def _risk_pipeline(width: int) -> Pipeline:
    columns = list(range(width))
    transform = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="median", add_indicator=True),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                columns,
            )
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("features", transform),
            (
                "logistic",
                LogisticRegression(
                    C=1.0,
                    class_weight=None,
                    max_iter=2000,
                    solver="lbfgs",
                    tol=1e-10,
                ),
            ),
        ]
    )


def fit_arm_models(
    training: Sequence[Assignment], *, augmented: bool
) -> dict[str, Pipeline]:
    """Fit one frozen logistic risk model per randomized arm."""
    models = {}
    for arm in ("control", "d5"):
        arm_rows = [row for row in training if row.arm == arm]
        labels = np.asarray([row.lost for row in arm_rows], dtype=np.int64)
        if set(labels.tolist()) != {0, 1}:
            raise RuntimeError(f"training partition arm {arm} lacks both outcomes")
        model = _risk_pipeline(len(M4_FEATURES if augmented else CONVENTIONAL_FEATURES))
        model.fit(feature_matrix(arm_rows, augmented=augmented), labels)
        models[arm] = model
    return models


def predict_counterfactual(
    models: Mapping[str, Pipeline], rows_: Sequence[Assignment], *, augmented: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Predict immediate and delayed non-delivery probabilities."""
    matrix = feature_matrix(rows_, augmented=augmented)
    return (
        models["control"].predict_proba(matrix)[:, 1],
        models["d5"].predict_proba(matrix)[:, 1],
    )


def _capture(target: np.ndarray, score: np.ndarray, fraction: float) -> float:
    count = max(1, math.ceil(len(score) * fraction))
    order = np.lexsort(
        (
            np.asarray([str(index) for index in range(len(score))]),
            -score,
        )
    )
    denominator = float(target.sum())
    return float(target[order[:count]].sum() / denominator) if denominator > 0 else 0.0


def _top_mean(target: np.ndarray, score: np.ndarray, fraction: float) -> float:
    count = max(1, math.ceil(len(score) * fraction))
    order = np.argsort(-score, kind="stable")
    return float(target[order[:count]].mean())


def evaluate_split(
    name: str,
    training: Sequence[Assignment],
    testing: Sequence[Assignment],
) -> dict[str, Any]:
    """Fit on one partition and evaluate the untouched held-out partition."""
    if not training or not testing:
        raise RuntimeError(f"{name}: empty train or test partition")
    conventional = fit_arm_models(training, augmented=False)
    augmented = fit_arm_models(training, augmented=True)
    p0, p1 = predict_counterfactual(conventional, testing, augmented=False)
    q0, q1 = predict_counterfactual(augmented, testing, augmented=True)
    arm = np.asarray([row.arm == "d5" for row in testing], dtype=bool)
    outcome = np.asarray([row.lost for row in testing], dtype=np.int64)
    observed_prediction = np.where(arm, p1, p0)
    augmented_prediction = np.where(arm, q1, q0)
    opportunity = np.asarray([row.opportunity for row in testing], dtype=np.float64)
    lost_opportunity = opportunity * outcome
    reward_variance = feature_matrix(testing, augmented=False)[:, 4]
    lag = feature_matrix(testing, augmented=False)[:, 1]
    delay_risk = np.maximum(0.0, p1 - p0)
    augmented_delay_risk = np.maximum(0.0, q1 - q0)
    scores = {
        "version_age": lag,
        "reward_variance": reward_variance,
        "conventional_delay_risk": delay_risk,
        "reward_variance_at_risk": delay_risk * reward_variance,
        "m4_at_risk": delay_risk * opportunity,
        "m4_augmented_at_risk": augmented_delay_risk * opportunity,
    }
    pseudo_outcome = np.where(arm, 2.0 * lost_opportunity, -2.0 * lost_opportunity)
    ranking = {
        score_name: {
            "observed_lost_m4_capture_top_10pct": _capture(
                lost_opportunity, score, 0.10
            ),
            "observed_lost_m4_capture_top_20pct": _capture(
                lost_opportunity, score, 0.20
            ),
            "ipw_causal_lost_m4_mean_top_10pct": _top_mean(pseudo_outcome, score, 0.10),
        }
        for score_name, score in scores.items()
    }
    return {
        "name": name,
        "training_assignments": len(training),
        "testing_assignments": len(testing),
        "testing_acquisitions": sorted({row.acquisition for row in testing}),
        "conventional_prediction": {
            "log_loss": float(log_loss(outcome, observed_prediction)),
            "brier": float(brier_score_loss(outcome, observed_prediction)),
            "auroc": float(roc_auc_score(outcome, observed_prediction)),
        },
        "m4_augmented_prediction": {
            "log_loss": float(log_loss(outcome, augmented_prediction)),
            "brier": float(brier_score_loss(outcome, augmented_prediction)),
            "auroc": float(roc_auc_score(outcome, augmented_prediction)),
        },
        "ranking": ranking,
    }


def _splits(
    assignments: Sequence[Assignment],
) -> tuple[list[tuple[str, list[Assignment], list[Assignment]]], list[str]]:
    specs: list[tuple[str, Callable[[Assignment], bool]]] = [
        ("family-qwen-to-llama", lambda row: row.family == "llama"),
        ("family-llama-to-qwen", lambda row: row.family == "qwen"),
    ]
    for workload in ("openmath", "gsm8k", "numinamath"):
        specs.append(
            (
                f"workload-holdout-{workload}",
                lambda row, value=workload: row.workload == value,
            )
        )
    acquisitions = sorted({row.acquisition for row in assignments})
    for acquisition in acquisitions:
        specs.append(
            (
                f"acquisition-holdout-{acquisition}",
                lambda row, value=acquisition: row.acquisition == value,
            )
        )
    result = []
    for name, is_test in specs:
        training = [row for row in assignments if not is_test(row)]
        testing = [row for row in assignments if is_test(row)]
        result.append((name, training, testing))
    return result, acquisitions


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def summarize(
    results: Sequence[Mapping[str, Any]], acquisitions: Sequence[str]
) -> dict[str, Any]:
    """Apply the frozen macro decision rule."""
    by_name = {result["name"]: result for result in results}

    def delta(result: Mapping[str, Any]) -> float:
        ranking = result["ranking"]
        return (
            ranking["m4_at_risk"]["observed_lost_m4_capture_top_10pct"]
            - ranking["reward_variance_at_risk"]["observed_lost_m4_capture_top_10pct"]
        )

    acquisition_deltas = [
        delta(by_name[f"acquisition-holdout-{acquisition}"])
        for acquisition in acquisitions
    ]
    family_deltas = {
        name: delta(by_name[name])
        for name in ("family-qwen-to-llama", "family-llama-to-qwen")
    }
    primary = _mean(acquisition_deltas)
    passed = primary > 0.0 and all(value >= 0.0 for value in family_deltas.values())
    return {
        "primary_macro_top_10_capture_difference_m4_minus_reward_variance": primary,
        "leave_one_acquisition_out_differences": dict(
            zip(acquisitions, acquisition_deltas, strict=True)
        ),
        "family_direction_differences": family_deltas,
        "decision": (
            "PASS_AUTHORIZE_GRADIENT_UTILITY_AUDIT"
            if passed
            else "FAIL_RETAIN_MEASUREMENT_ONLY"
        ),
        "authorizes_quality_acquisition": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-source-root", type=Path, required=True)
    parser.add_argument("--llama-source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cells = source_cells(args.qwen_source_root, args.llama_source_root)
    assignments = load_assignments(cells)
    splits, acquisitions = _splits(assignments)
    results = [
        evaluate_split(name, training, testing) for name, training, testing in splits
    ]
    protocol = HERE / "offline_transport_protocol.json"
    output = {
        "schema": "m4-causal-utility-offline-transport-result-v1",
        "status": "COMPLETE_FROZEN_OFFLINE_TRANSPORT_ANALYSIS",
        "protocol_sha256": sha256(protocol),
        "implementation_sha256": sha256(Path(__file__)),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "assignment_count": len(assignments),
        "acquisition_count": len(acquisitions),
        "raw_ledgers_committed": False,
        "results": results,
        "summary": summarize(results, acquisitions),
        "claim_boundary": "This held-out analysis evaluates risk transport and ranking of the registered Q-times-non-delivery endpoint. It is not independent validation of Q as learning utility and does not establish terminal-quality benefit.",
    }
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
