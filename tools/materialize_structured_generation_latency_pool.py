# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Materialize the frozen structured-generation latency feasibility pool."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from nemo_rl.algorithms.async_utils.fixed_pool import compute_fixed_pool_id
from tools import materialize_sliding_puzzle_7b_competence as model_pin


DESIGN_ID: Final[str] = "structured_generation_latency_v1"
PLAN_ID: Final[str] = "161355945ba0d9b44726ce8fe2354b637ec3a75ef5f8fe08a2d21a3c82eaae2c"
PLAN_SHA256: Final[str] = (
    "8aa2b4e40839b37331311eeda2d7875a922d73128930494dae4172102db71cda"
)
SELECTION_SEED: Final[int] = 20260907
ORDER_SEED: Final[int] = 45001
GENERATION_SEED: Final[int] = 64001
PAIR_COUNT: Final[int] = 8
COHORT_SIZE: Final[int] = 4
SOURCE_IDS: Final[tuple[str, str]] = ("structured_short", "structured_long")
CHECK_LINES: Final[dict[str, int]] = {
    "structured_short": 2,
    "structured_long": 16,
}


@dataclass(frozen=True, slots=True)
class StructuredGenerationMaterializationSpec:
    """Parameters that distinguish one immutable structured prompt pool."""

    design_id: str
    analysis_status: str
    protocol_id: str
    selection_seed: int
    order_seed: int
    generation_seed: int
    source_split: str
    protocol_filename: str
    identity_field: str


DEFAULT_SPEC: Final = StructuredGenerationMaterializationSpec(
    design_id=DESIGN_ID,
    analysis_status="preregistered_controlled_generative_demand_feasibility",
    protocol_id=PLAN_ID,
    selection_seed=SELECTION_SEED,
    order_seed=ORDER_SEED,
    generation_seed=GENERATION_SEED,
    source_split="feasibility",
    protocol_filename="feasibility_plan.v1.json",
    identity_field="plan_id",
)


class StructuredGenerationMaterializationError(ValueError):
    """The frozen plan or generated materialization is inconsistent."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _opaque(key: bytes, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode()
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _write(path: Path, value: bytes) -> None:
    path.write_bytes(value)
    path.chmod(0o600)


def _load_plan(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    if _sha_bytes(raw) != PLAN_SHA256:
        raise StructuredGenerationMaterializationError("plan byte hash mismatch")
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as error:
        raise StructuredGenerationMaterializationError(
            "plan is invalid JSON"
        ) from error
    if not isinstance(plan, dict):
        raise StructuredGenerationMaterializationError("plan must be an object")
    without_id = {key: value for key, value in plan.items() if key != "plan_id"}
    if plan.get("plan_id") != PLAN_ID or _sha_bytes(_canonical(without_id)) != PLAN_ID:
        raise StructuredGenerationMaterializationError("plan ID mismatch")
    expected = {
        "analysis_status": "preregistered_controlled_generative_demand_feasibility",
        "attempt_limit": 1,
        "calibration_only": True,
        "confirmatory_eligible": False,
        "schema_version": 1,
        "training_authorized": False,
    }
    if any(plan.get(key) != value for key, value in expected.items()):
        raise StructuredGenerationMaterializationError("plan labels are not frozen")
    model = plan.get("model")
    expected_weights = [
        {"path": name, "bytes": size, "sha256": digest}
        for name, size, digest in model_pin.EXPECTED_WEIGHT_FILES
    ]
    if model != {
        "generation_config_sha256": model_pin.GENERATION_CONFIG_SHA256,
        "repo_id": model_pin.MODEL_REPO,
        "revision": model_pin.MODEL_REVISION,
        "weight_files": expected_weights,
        "weight_manifest_sha256": model_pin.MODEL_WEIGHTS_SHA256,
    }:
        raise StructuredGenerationMaterializationError("model pin mismatch")
    if plan.get("pool") != {
        "cohort_size": COHORT_SIZE,
        "generated_pair_count": PAIR_COUNT,
        "order_seed": ORDER_SEED,
        "pairing": "same arithmetic problem with short and long response contracts",
        "prompt_groups": 2 * PAIR_COUNT,
        "selection_seed": SELECTION_SEED,
        "source_exposure_status": "fresh_pre_generation",
        "task_count_per_cohort": {
            "structured_long": 2,
            "structured_short": 2,
        },
    }:
        raise StructuredGenerationMaterializationError("pool design mismatch")
    runtime = plan.get("runtime")
    if (
        not isinstance(runtime, dict)
        or runtime.get("generation_study_seed") != GENERATION_SEED
    ):
        raise StructuredGenerationMaterializationError("runtime seed mismatch")
    return plan, raw


def _prompt(left: int, right: int, check_lines: int) -> str:
    return (
        f"Compute {left} + {right}. Before the final answer, write exactly "
        f"{check_lines} numbered check lines. Every check line must restate the "
        f"same equality `{left} + {right} = {left + right}`. Number the lines "
        f"from 1 through {check_lines}. Then write a final line containing only "
        f"\\boxed{{{left + right}}}. Do not add any other text."
    )


def _rendered_token_ids(tokenizer: Any, prompt: str) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    return [
        int(value)
        for value in tokenizer(rendered, add_special_tokens=False)["input_ids"]
    ]


def _make_records(
    tokenizer: Any,
    key: bytes,
    spec: StructuredGenerationMaterializationSpec = DEFAULT_SPEC,
) -> dict[str, list[dict[str, object]]]:
    rng = random.Random(spec.selection_seed)
    problems: list[tuple[int, int]] = []
    while len(problems) < PAIR_COUNT:
        candidate = (rng.randrange(101, 900), rng.randrange(101, 900))
        if candidate not in problems:
            problems.append(candidate)
    output = {source_id: [] for source_id in SOURCE_IDS}
    for pair_index, (left, right) in enumerate(problems):
        cluster_id = _opaque(key, spec.design_id, "pair", pair_index, left, right)
        for source_id in SOURCE_IDS:
            check_lines = CHECK_LINES[source_id]
            prompt = _prompt(left, right, check_lines)
            token_ids = _rendered_token_ids(tokenizer, prompt)
            output[source_id].append(
                {
                    "input": prompt,
                    "output": str(left + right),
                    "source_id": source_id,
                    "source_dataset_id": "generated/structured-addition-checks",
                    "source_revision": "v1",
                    "source_split": spec.source_split,
                    "source_dataset_index": pair_index,
                    "source_prompt_id": _opaque(
                        key, spec.design_id, source_id, pair_index, left, right
                    ),
                    "repeated_prompt_cluster_id": cluster_id,
                    "selection_stratum": source_id,
                    "matching_pair_id": f"pair-{pair_index}",
                    "left_operand": left,
                    "right_operand": right,
                    "expected_answer": left + right,
                    "required_check_lines": check_lines,
                    "input_token_count": len(token_ids),
                    "input_token_ids_sha256": _sha_bytes(_canonical(token_ids)),
                    "selection_seed": spec.selection_seed,
                    "model_revision": model_pin.MODEL_REVISION,
                }
            )
    return output


def _build_manifest(
    *,
    key: bytes,
    plan_sha: str,
    records: Mapping[str, Sequence[Mapping[str, object]]],
    raw_sources: Mapping[str, bytes],
    snapshot_sha: str,
    spec: StructuredGenerationMaterializationSpec = DEFAULT_SPEC,
) -> dict[str, object]:
    rng = random.Random(spec.order_seed)
    pair_order = list(range(PAIR_COUNT))
    rng.shuffle(pair_order)
    ordered: list[tuple[str, int]] = []
    for start in range(0, PAIR_COUNT, 2):
        cohort = [
            (source_id, pair_index)
            for pair_index in pair_order[start : start + 2]
            for source_id in SOURCE_IDS
        ]
        rng.shuffle(cohort)
        ordered.extend(cohort)
    offsets = {SOURCE_IDS[0]: 0, SOURCE_IDS[1]: PAIR_COUNT}
    items = []
    for ordinal, (source_id, row) in enumerate(ordered):
        record = records[source_id][row]
        cohort = ordinal // COHORT_SIZE
        items.append(
            {
                "ordinal": ordinal,
                "pool_item_id": _opaque(key, spec.design_id, spec.order_seed, ordinal),
                "source_prompt_id": record["source_prompt_id"],
                "source_id": source_id,
                "source_dataset_index": row,
                "dataset_index": offsets[source_id] + row,
                "task_name": source_id,
                "repeated_prompt_cluster_id": record["repeated_prompt_cluster_id"],
                "dispatch_cohort": cohort,
                "decorrelation_block": f"cohort-{cohort}",
                "matching_pair_id": record["matching_pair_id"],
                "materialized_source_row": row,
                "materialized_record_sha256": _sha_bytes(_canonical(record)),
                "input_token_count": record["input_token_count"],
                "input_token_ids_sha256": record["input_token_ids_sha256"],
            }
        )
    sources = [
        {
            "source_id": source_id,
            "dataset_id": "generated/structured-addition-checks",
            "revision": "v1",
            "split": spec.source_split,
            "materialized_file": f"{source_id}.jsonl",
            "content_sha256": _sha_bytes(raw_sources[source_id]),
        }
        for source_id in SOURCE_IDS
    ]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "id_namespace_fingerprint": _sha_bytes(key),
        "design_protocol_sha256": plan_sha,
        "order_seed": spec.order_seed,
        "model_revision": model_pin.MODEL_REVISION,
        "model_weights_sha256": model_pin.MODEL_WEIGHTS_SHA256,
        "model_snapshot_manifest_sha256": snapshot_sha,
        "model_snapshot_path": "model_snapshot",
        "sources": sources,
        "items": items,
    }
    manifest["pool_id"] = compute_fixed_pool_id(manifest)
    return manifest


def _selection_design(
    manifest: Mapping[str, object],
    records: Mapping[str, Sequence[Mapping[str, object]]],
    spec: StructuredGenerationMaterializationSpec = DEFAULT_SPEC,
) -> dict[str, object]:
    items = manifest["items"]
    if not isinstance(items, list):
        raise StructuredGenerationMaterializationError("manifest items are invalid")
    output = []
    for item in items:
        if not isinstance(item, Mapping):
            raise StructuredGenerationMaterializationError("manifest item is invalid")
        source_id = str(item["source_id"])
        record = records[source_id][int(item["materialized_source_row"])]
        output.append(
            {
                "source_pool_ordinal": item["ordinal"],
                "source_prompt_id": item["source_prompt_id"],
                "task_name": item["task_name"],
                "pair_id": item["matching_pair_id"],
                "stratum": "short" if source_id == SOURCE_IDS[0] else "long",
                "left_operand": record["left_operand"],
                "right_operand": record["right_operand"],
                "expected_answer": record["expected_answer"],
                "required_check_lines": record["required_check_lines"],
                "rendered_prompt_tokens": record["input_token_count"],
            }
        )
    result = {
        "schema_version": 1,
        "analysis_status": spec.analysis_status,
        "calibration_only": True,
        "confirmatory_eligible": False,
        "fixed_pool_id": manifest["pool_id"],
        "items": output,
    }
    result[spec.identity_field] = spec.protocol_id
    return result


def _materialize_from_protocol(
    *,
    output_dir: Path,
    protocol_raw: bytes,
    key: bytes,
    spec: StructuredGenerationMaterializationSpec,
) -> None:
    """Create one private pool from already validated protocol bytes."""
    if len(key) < 32:
        raise StructuredGenerationMaterializationError("HMAC key is too short")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.", dir=output_dir.parent
    ) as temporary_name:
        root = Path(temporary_name)
        root.chmod(0o700)
        tokenizer, snapshot_raw = model_pin._snapshot_model(root)
        records = _make_records(tokenizer, key, spec)
        raw_sources = {
            source_id: b"".join(_canonical(row) + b"\n" for row in rows)
            for source_id, rows in records.items()
        }
        for source_id, raw in raw_sources.items():
            _write(root / f"{source_id}.jsonl", raw)
        _write(root / spec.protocol_filename, protocol_raw)
        manifest = _build_manifest(
            key=key,
            plan_sha=_sha_bytes(protocol_raw),
            records=records,
            raw_sources=raw_sources,
            snapshot_sha=_sha_bytes(snapshot_raw),
            spec=spec,
        )
        manifest_raw = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
        manifest_name = f"fixed_pool_manifest.v1.{spec.order_seed}.json"
        _write(root / manifest_name, manifest_raw)
        design = _selection_design(manifest, records, spec)
        design_raw = json.dumps(design, indent=2, sort_keys=True).encode() + b"\n"
        _write(root / "selection_design.v1.json", design_raw)
        prompt_deltas = []
        for pair_index in range(PAIR_COUNT):
            short = records[SOURCE_IDS[0]][pair_index]
            long = records[SOURCE_IDS[1]][pair_index]
            prompt_deltas.append(
                abs(int(long["input_token_count"]) - int(short["input_token_count"]))
            )
        report = {
            "schema_version": 1,
            "status": "passed",
            "design_id": spec.design_id,
            "selection_design_sha256": _sha_bytes(design_raw),
            "manifest": manifest_name,
            "manifest_sha256": _sha_bytes(manifest_raw),
            "pool_id": manifest["pool_id"],
            "selection_seed": spec.selection_seed,
            "order_seed": spec.order_seed,
            "pairs": PAIR_COUNT,
            "prompt_groups": 2 * PAIR_COUNT,
            "observed_max_input_pair_delta_tokens": max(prompt_deltas),
            "model_repo": model_pin.MODEL_REPO,
            "model_revision": model_pin.MODEL_REVISION,
            "model_weights_sha256": model_pin.MODEL_WEIGHTS_SHA256,
        }
        report[spec.identity_field] = spec.protocol_id
        report[
            "plan_sha256" if spec.identity_field == "plan_id" else "protocol_sha256"
        ] = _sha_bytes(protocol_raw)
        _write(
            root / "materialization_report.v1.json",
            json.dumps(report, indent=2, sort_keys=True).encode() + b"\n",
        )
        hash_lines = [
            f"{_sha_path(path)}  {path.name}\n"
            for path in sorted(root.iterdir())
            if path.is_file() and path.name != "SHA256SUMS"
        ]
        _write(root / "SHA256SUMS", "".join(hash_lines).encode())
        for directory in [root, *(path for path in root.rglob("*") if path.is_dir())]:
            directory.chmod(0o700)
        for path in root.rglob("*"):
            if path.is_file():
                path.chmod(0o600)
        root.rename(output_dir)


def materialize(*, output_dir: Path, plan_path: Path, key: bytes) -> None:
    """Create the frozen feasibility pool and refuse an existing target."""
    _, plan_raw = _load_plan(plan_path)
    _materialize_from_protocol(
        output_dir=output_dir,
        protocol_raw=plan_raw,
        key=key,
        spec=DEFAULT_SPEC,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--hmac-key-env", default="STRUCTURED_GENERATION_PROMPT_HMAC_KEY"
    )
    args = parser.parse_args()
    key = os.environ.get(args.hmac_key_env)
    if key is None:
        raise StructuredGenerationMaterializationError("HMAC key is missing")
    materialize(output_dir=args.output_dir, plan_path=args.plan, key=key.encode())


if __name__ == "__main__":
    main()
