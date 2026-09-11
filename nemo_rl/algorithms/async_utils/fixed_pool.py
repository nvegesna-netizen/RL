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

"""Strict manifests and data plumbing for fixed-pool rollout collection.

The manifest contains only source coordinates and opaque, precomputed IDs. It
must never contain prompt or completion text. Prompt IDs are expected to be
generated upstream with a study-specific keyed hash; the key is not an input to
this module and must not be persisted with the manifest or scheduler trace.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Protocol, TypeAlias, cast

from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.algorithms.async_utils.scheduler_trace import (
    SchedulerEventType,
    SchedulerTraceValidationError,
    iter_scheduler_trace,
    validate_scheduler_trace,
)


FIXED_POOL_MANIFEST_SCHEMA_VERSION: Final[int] = 1
MAX_FIXED_POOL_MANIFEST_BYTES: Final[int] = 16 * 1024 * 1024
MAX_FIXED_POOL_SOURCE_BYTES: Final[int] = 64 * 1024 * 1024
MAX_FIXED_POOL_ITEMS: Final[int] = 1_000_000
OPENMATH_DATASET_ID: Final[str] = "nvidia/OpenMathInstruct-2"
OPENMATH_DATASET_REVISION: Final[str] = "469216e3f46f4dacf476b382e192485ea51a143e"
OPENMATH_DATASET_SPLIT: Final[str] = "train_1M"
OPENMATH_ORDER_SEED: Final[int] = 43001
OPENMATH_SELECTION_SEED: Final[int] = 20260902
OPENMATH_MODEL_REVISION: Final[str] = "8faed761d45a263340a0528343f099c05c9a4323"
OPENMATH_MODEL_WEIGHTS_SHA256: Final[str] = (
    "a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796"
)
OPENMATH_PROMPT_FILE_SHA256: Final[str] = (
    "a3575cc34f8bbd8ed5107a0d58003acf4277baf18d29409c7e61e0946e25b031"
)
OPENMATH_SOURCE_IDS: Final[tuple[str, str]] = (
    "openmath_short",
    "openmath_long",
)
OPENMATH_REFERENCE_SOLUTION_BANDS: Final[dict[str, tuple[int, int]]] = {
    "openmath_short": (32, 96),
    "openmath_long": (256, 384),
}
OPENMATH_INPUT_TOKEN_CALIPER: Final[int] = 8
OPENMATH_ANSWER_TOKEN_CALIPER: Final[int] = 4
OPENMATH_MATERIALIZED_RECORD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "input",
        "output",
        "source_id",
        "source_dataset_id",
        "source_revision",
        "source_split",
        "source_dataset_index",
        "source_prompt_id",
        "repeated_prompt_cluster_id",
        "selection_stratum",
        "problem_source",
        "matching_pair_id",
        "normalized_problem_sha256",
        "input_token_count",
        "input_token_ids_sha256",
        "reference_solution_token_count",
        "reference_answer_token_count",
        "selection_seed",
        "model_revision",
        "prompt_file_sha256",
    }
)
SLIDING_PUZZLE_DATASET_ID: Final[str] = "generated/sliding-puzzle-3x3"
SLIDING_PUZZLE_DATASET_REVISION: Final[str] = "bfs_exact_distance_v1"
SLIDING_PUZZLE_DATASET_SPLIT: Final[str] = "feasibility"
SLIDING_PUZZLE_ORDER_SEED: Final[int] = 44001
SLIDING_PUZZLE_SELECTION_SEED: Final[int] = 20260903
SLIDING_PUZZLE_MODEL_REVISION: Final[str] = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SLIDING_PUZZLE_MODEL_WEIGHTS_SHA256: Final[str] = (
    "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
)
SLIDING_PUZZLE_7B_MODEL_REVISION: Final[str] = (
    "a09a35458c702b33eeacc393d103063234e8bc28"
)
SLIDING_PUZZLE_7B_MODEL_WEIGHTS_SHA256: Final[str] = (
    "456f5eff514d78f7b0ef52a057118046acf15d86606655767db068cabf5f49f7"
)
SLIDING_PUZZLE_SOURCE_IDS: Final[tuple[str, str]] = (
    "sliding_puzzle_easy",
    "sliding_puzzle_hard",
)
SLIDING_PUZZLE_DISTANCE_SCHEDULES: Final[dict[str, tuple[int, ...]]] = {
    "sliding_puzzle_easy": (1, 1, 2, 2, 2, 2, 3, 3),
    "sliding_puzzle_hard": (10, 10, 10, 11, 11, 11, 12, 12),
}
SLIDING_PUZZLE_INPUT_TOKEN_CALIPER: Final[int] = 8
SLIDING_PUZZLE_MAX_INPUT_TOKENS: Final[int] = 512
SLIDING_PUZZLE_PROMPT_BUILDER_VERSION: Final[str] = "legacy_sliding_puzzle_prompt_v1"
SLIDING_PUZZLE_COMPACT_PROMPT_BUILDER_VERSION: Final[str] = "compact_action_only_v2"
SLIDING_PUZZLE_MATERIALIZED_RECORD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "messages",
        "extra_env_info",
        "stop_strings",
        "source_id",
        "source_dataset_id",
        "source_revision",
        "source_split",
        "source_dataset_index",
        "source_prompt_id",
        "repeated_prompt_cluster_id",
        "selection_stratum",
        "matching_pair_id",
        "board_state_sha256",
        "optimal_distance",
        "input_token_count",
        "input_token_ids_sha256",
        "selection_seed",
        "model_revision",
        "prompt_builder_version",
    }
)
STRUCTURED_GENERATION_DATASET_ID: Final[str] = "generated/structured-addition-checks"
STRUCTURED_GENERATION_DATASET_REVISION: Final[str] = "v1"
STRUCTURED_GENERATION_DATASET_SPLIT: Final[str] = "feasibility"
STRUCTURED_GENERATION_ORDER_SEED: Final[int] = 45001
STRUCTURED_GENERATION_SELECTION_SEED: Final[int] = 20260907
STRUCTURED_GENERATION_MODEL_REVISION: Final[str] = (
    "a09a35458c702b33eeacc393d103063234e8bc28"
)
STRUCTURED_GENERATION_MODEL_WEIGHTS_SHA256: Final[str] = (
    "456f5eff514d78f7b0ef52a057118046acf15d86606655767db068cabf5f49f7"
)
STRUCTURED_GENERATION_PLAN_SHA256: Final[str] = (
    "8aa2b4e40839b37331311eeda2d7875a922d73128930494dae4172102db71cda"
)
STRUCTURED_CROSSOVER_PROTOCOL_SHA256: Final[str] = (
    "ddd7536aa3d67351974629c2c56fbe41456b9e3b48b0d26e47ef7f7e2728f7ef"
)
STRUCTURED_CROSSOVER_SELECTION_SEEDS: Final[dict[int, int]] = {
    46001: 2026090801,
    46002: 2026090802,
    46003: 2026090803,
    46004: 2026090804,
}
STRUCTURED_PRESSURE_PROTOCOL_SHA256: Final[str] = (
    "c35c06797483c83bd1a29f8cf856447387729e0e612480026eb84320127576cc"
)
STRUCTURED_PRESSURE_SELECTION_SEEDS: Final[dict[int, int]] = {
    49001: 2026091001,
    49002: 2026091002,
    49003: 2026091003,
    49004: 2026091004,
}
STRUCTURED_GENERATION_SOURCE_IDS: Final[tuple[str, str]] = (
    "structured_short",
    "structured_long",
)
STRUCTURED_GENERATION_CHECK_LINES: Final[dict[str, int]] = {
    "structured_short": 2,
    "structured_long": 16,
}
STRUCTURED_GENERATION_MATERIALIZED_RECORD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "input",
        "output",
        "source_id",
        "source_dataset_id",
        "source_revision",
        "source_split",
        "source_dataset_index",
        "source_prompt_id",
        "repeated_prompt_cluster_id",
        "selection_stratum",
        "matching_pair_id",
        "left_operand",
        "right_operand",
        "expected_answer",
        "required_check_lines",
        "input_token_count",
        "input_token_ids_sha256",
        "selection_seed",
        "model_revision",
    }
)
DAPO_OPERATIONAL_DATASET_ID: Final[str] = "BytedTsinghua-SIA/DAPO-Math-17k"
DAPO_OPERATIONAL_DATASET_REVISION: Final[str] = (
    "65877096c24ffa7abc4e4fa5edb95cf3413a5674"
)
DAPO_OPERATIONAL_MODEL_REVISION: Final[str] = "b101308fe89651ea5ce025f25317fea6fc07e96e"
DAPO_OPERATIONAL_PROTOCOL_SHA256: Final[str] = (
    "4d2c9af4a46a335141c197ad58dec8913ab5290d22c848cda8b978c3b00b4852"
)
DAPO_OPERATIONAL_SELECTION_SEEDS: Final[frozenset[int]] = frozenset(
    {47001, 47002, 47003}
)
DAPO_CROSSOVER_PROTOCOL_SHA256: Final[str] = (
    "c49f9604225db847eded1d298c592938299fb3c3d508cf60d2b598f1e8b604c7"
)
DAPO_CROSSOVER_SELECTION_SEEDS: Final[frozenset[int]] = frozenset(
    {48001, 48002, 48003, 48004}
)
DAPO_OPERATIONAL_SOURCE_IDS: Final[tuple[str, str]] = (
    "dapo_math_a",
    "dapo_math_b",
)
DAPO_OPERATIONAL_MATERIALIZED_RECORD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "input",
        "output",
        "source_id",
        "source_dataset_id",
        "source_revision",
        "source_split",
        "source_dataset_index",
        "source_prompt_id",
        "repeated_prompt_cluster_id",
        "canonical_prompt_sha256",
        "source_extra_index",
        "source_duplicate_count",
        "input_token_count",
        "input_token_ids_sha256",
        "selection_seed",
        "model_revision",
    }
)

Sha256Hex: TypeAlias = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]
Identifier: TypeAlias = Annotated[
    str,
    StringConstraints(min_length=1, max_length=256, pattern=r"^[^\r\n]+$"),
]
SourceIdentifier: TypeAlias = Annotated[
    str,
    StringConstraints(min_length=1, max_length=1024, pattern=r"^[^\r\n]+$"),
]
NonNegativeStrictInt: TypeAlias = Annotated[int, Field(strict=True, ge=0)]


class FixedPoolManifestError(ValueError):
    """A fixed-pool manifest is unreadable or violates its strict contract."""


@dataclass(frozen=True, slots=True)
class FixedPoolTraceValidationReport:
    """Exact lifecycle coverage for an evidentiary source collection."""

    planned_prompt_groups: int
    dispatched_prompt_groups: int
    completed_prompt_groups: int
    ready_prompt_groups: int
    archived_prompt_groups: int
    physical_weight_version: int


class FixedPoolSource(BaseModel, extra="forbid", frozen=True):
    """Immutable source dataset identity for every item in a pool."""

    source_id: Identifier
    dataset_id: SourceIdentifier
    revision: Identifier
    split: Identifier
    materialized_file: Identifier
    content_sha256: Sha256Hex

    @model_validator(mode="after")
    def _validate_materialized_file(self) -> "FixedPoolSource":
        path = Path(self.materialized_file)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("materialized_file must be a safe relative path")
        return self


class FixedPoolManifestItem(BaseModel, extra="forbid", frozen=True):
    """One planned prompt-group occurrence in dispatch order."""

    ordinal: NonNegativeStrictInt
    pool_item_id: Sha256Hex
    source_prompt_id: Sha256Hex
    source_id: Identifier
    source_dataset_index: NonNegativeStrictInt
    dataset_index: NonNegativeStrictInt
    task_name: Identifier
    repeated_prompt_cluster_id: Sha256Hex
    dispatch_cohort: NonNegativeStrictInt
    decorrelation_block: Identifier
    matching_pair_id: Identifier
    materialized_source_row: NonNegativeStrictInt
    materialized_record_sha256: Sha256Hex
    input_token_count: NonNegativeStrictInt
    input_token_ids_sha256: Sha256Hex


class _FixedPoolManifestContent(BaseModel, extra="forbid", frozen=True):
    """Canonical manifest fields covered by ``pool_id``."""

    schema_version: Literal[FIXED_POOL_MANIFEST_SCHEMA_VERSION]
    id_namespace_fingerprint: Sha256Hex
    design_protocol_sha256: Sha256Hex
    order_seed: NonNegativeStrictInt
    model_revision: Identifier
    model_weights_sha256: Sha256Hex
    model_snapshot_manifest_sha256: Sha256Hex
    model_snapshot_path: Identifier
    sources: tuple[FixedPoolSource, ...]
    items: tuple[FixedPoolManifestItem, ...]

    @model_validator(mode="before")
    @classmethod
    def _validate_model_snapshot_path(cls, values: object) -> object:
        if isinstance(values, Mapping):
            raw_path = values.get("model_snapshot_path")
            if isinstance(raw_path, str):
                path = Path(raw_path)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("model_snapshot_path must be a safe relative path")
        return values

    @model_validator(mode="after")
    def _validate_pool_contract(self) -> "_FixedPoolManifestContent":
        if not self.items:
            raise ValueError("fixed-pool manifest must contain at least one item")
        if len(self.items) > MAX_FIXED_POOL_ITEMS:
            raise ValueError(
                f"fixed-pool manifest exceeds {MAX_FIXED_POOL_ITEMS} items"
            )
        if not self.sources:
            raise ValueError("fixed-pool manifest must declare at least one source")
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("fixed-pool source_id values must be unique")
        known_sources = set(source_ids)
        unknown_sources = sorted(
            {item.source_id for item in self.items} - known_sources
        )
        if unknown_sources:
            raise ValueError(
                f"fixed-pool items reference unknown sources: {unknown_sources}"
            )

        ordinals = [item.ordinal for item in self.items]
        if ordinals != list(range(len(self.items))):
            raise ValueError(
                "fixed-pool item ordinals must be contiguous and match file order"
            )

        pool_item_ids = [item.pool_item_id for item in self.items]
        if len(pool_item_ids) != len(set(pool_item_ids)):
            raise ValueError("fixed-pool pool_item_id values must be unique")

        cohorts = [item.dispatch_cohort for item in self.items]
        if cohorts != sorted(cohorts):
            raise ValueError("dispatch cohorts must be contiguous blocks in file order")
        if sorted(set(cohorts)) != list(range(max(cohorts) + 1)):
            raise ValueError("dispatch cohort IDs must be contiguous and start at zero")

        source_identity: dict[int, tuple[str, int, str, str, str]] = {}
        coordinate_identity: dict[tuple[str, int], tuple[str, str, str]] = {}
        prompt_identity: dict[str, tuple[str, int, str, str]] = {}
        for item in self.items:
            identity = (
                item.source_id,
                item.source_dataset_index,
                item.source_prompt_id,
                item.task_name,
                item.repeated_prompt_cluster_id,
            )
            previous = source_identity.setdefault(item.dataset_index, identity)
            if previous != identity:
                raise ValueError(
                    "repeated dataset_index values must preserve source prompt, "
                    "task, and repeated-prompt cluster identity"
                )
            coordinate = (item.source_id, item.source_dataset_index)
            coordinate_value = (
                item.source_prompt_id,
                item.task_name,
                item.repeated_prompt_cluster_id,
            )
            previous_coordinate = coordinate_identity.setdefault(
                coordinate, coordinate_value
            )
            if previous_coordinate != coordinate_value:
                raise ValueError(
                    "repeated source coordinates must preserve prompt, task, and "
                    "repeated-prompt cluster identity"
                )
            prompt_value = (
                item.source_id,
                item.source_dataset_index,
                item.task_name,
                item.repeated_prompt_cluster_id,
            )
            previous_prompt = prompt_identity.setdefault(
                item.source_prompt_id, prompt_value
            )
            if previous_prompt != prompt_value:
                raise ValueError(
                    "source_prompt_id must identify exactly one source coordinate, "
                    "task, and repeated-prompt cluster"
                )

        return self


class _FixedPoolManifestPayload(_FixedPoolManifestContent):
    """On-disk manifest schema before the verified file digest is attached."""

    pool_id: Sha256Hex

    @model_validator(mode="after")
    def _validate_pool_id(self) -> "_FixedPoolManifestPayload":
        if self.pool_id != _compute_pool_id(self):
            raise ValueError("pool_id does not match the canonical manifest payload")
        return self


@dataclass(frozen=True, slots=True)
class FixedPoolManifest:
    """Validated fixed-pool manifest plus its exact raw-file SHA-256."""

    schema_version: int
    pool_id: str
    id_namespace_fingerprint: str
    design_protocol_sha256: str
    order_seed: int
    model_revision: str
    model_weights_sha256: str
    model_snapshot_manifest_sha256: str
    model_snapshot_path: str
    sources: tuple[FixedPoolSource, ...]
    items: tuple[FixedPoolManifestItem, ...]
    manifest_sha256: str
    manifest_path: Path

    @property
    def num_dispatch_cohorts(self) -> int:
        """Return the number of predeclared dispatch cohorts."""
        return self.items[-1].dispatch_cohort + 1

    @property
    def cohort_sizes(self) -> tuple[int, ...]:
        """Return manifest item counts for each contiguous dispatch cohort."""
        counts = [0] * self.num_dispatch_cohorts
        for item in self.items:
            counts[item.dispatch_cohort] += 1
        return tuple(counts)


def _canonical_pool_payload(payload: _FixedPoolManifestContent) -> bytes:
    record = payload.model_dump(mode="json", exclude={"pool_id"})
    return json.dumps(
        record,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _compute_pool_id(payload: _FixedPoolManifestContent) -> str:
    return hashlib.sha256(_canonical_pool_payload(payload)).hexdigest()


def compute_fixed_pool_id(record: Mapping[str, object]) -> str:
    """Compute the canonical pool ID for a manifest record under construction.

    The input may omit ``pool_id``. All other fields are validated before the
    digest is returned, except for cross-checking the digest itself.
    """
    values = {key: value for key, value in record.items() if key != "pool_id"}
    try:
        payload = _FixedPoolManifestContent.model_validate(values)
    except ValidationError as error:
        raise FixedPoolManifestError(str(error)) from error
    return _compute_pool_id(payload)


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    record: dict[str, object] = {}
    for key, value in pairs:
        if key in record:
            raise FixedPoolManifestError(f"duplicate JSON key: {key!r}")
        record[key] = value
    return record


def load_fixed_pool_manifest(path: str | Path) -> FixedPoolManifest:
    """Read and strictly validate a fixed-pool JSON manifest.

    Args:
        path: Manifest path. The exact raw bytes are hashed for provenance.

    Returns:
        A validated immutable manifest.

    Raises:
        FixedPoolManifestError: The file cannot be read or violates the schema.
    """
    manifest_path = Path(path)
    try:
        raw = manifest_path.read_bytes()
    except OSError as error:
        raise FixedPoolManifestError(
            f"cannot read manifest {manifest_path}: {error}"
        ) from error
    if not raw:
        raise FixedPoolManifestError("fixed-pool manifest is empty")
    if len(raw) > MAX_FIXED_POOL_MANIFEST_BYTES:
        raise FixedPoolManifestError(
            f"fixed-pool manifest exceeds {MAX_FIXED_POOL_MANIFEST_BYTES} bytes"
        )
    try:
        record = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FixedPoolManifestError(
            "fixed-pool manifest is not valid UTF-8 JSON"
        ) from error
    if not isinstance(record, dict):
        raise FixedPoolManifestError("fixed-pool manifest root must be a JSON object")
    try:
        payload = _FixedPoolManifestPayload.model_validate(record)
    except ValidationError as error:
        raise FixedPoolManifestError(str(error)) from error
    return FixedPoolManifest(
        schema_version=payload.schema_version,
        pool_id=payload.pool_id,
        id_namespace_fingerprint=payload.id_namespace_fingerprint,
        design_protocol_sha256=payload.design_protocol_sha256,
        order_seed=payload.order_seed,
        model_revision=payload.model_revision,
        model_weights_sha256=payload.model_weights_sha256,
        model_snapshot_manifest_sha256=payload.model_snapshot_manifest_sha256,
        model_snapshot_path=payload.model_snapshot_path,
        sources=payload.sources,
        items=payload.items,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        manifest_path=manifest_path.resolve(),
    )


def _load_materialized_source_rows(
    manifest: FixedPoolManifest,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, int]]:
    """Load and hash-check every manifest-bound materialized JSONL source."""
    base = manifest.manifest_path.parent
    source_rows: dict[str, list[dict[str, object]]] = {}
    global_offsets: dict[str, int] = {}
    offset = 0
    for source in manifest.sources:
        path = (base / source.materialized_file).resolve()
        if path.parent != base and base not in path.parents:
            raise FixedPoolManifestError(
                "materialized source escapes manifest directory"
            )
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise FixedPoolManifestError(
                f"cannot read materialized source {path}: {error}"
            ) from error
        if not raw or len(raw) > MAX_FIXED_POOL_SOURCE_BYTES:
            raise FixedPoolManifestError(
                f"materialized source {path} is empty or exceeds the size limit"
            )
        if hashlib.sha256(raw).hexdigest() != source.content_sha256:
            raise FixedPoolManifestError(
                f"materialized source SHA mismatch for {source.source_id}"
            )
        rows: list[dict[str, object]] = []
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line:
                raise FixedPoolManifestError(
                    f"blank JSONL row in {source.materialized_file}:{line_number}"
                )
            try:
                record = json.loads(
                    line.decode("utf-8"),
                    object_pairs_hook=_reject_duplicate_json_keys,
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise FixedPoolManifestError(
                    f"invalid JSONL row in {source.materialized_file}:{line_number}"
                ) from error
            if not isinstance(record, dict):
                raise FixedPoolManifestError("materialized JSONL rows must be objects")
            rows.append(record)
        global_offsets[source.source_id] = offset
        offset += len(rows)
        source_rows[source.source_id] = rows
    return source_rows, global_offsets


def validate_fixed_pool_materialization(manifest: FixedPoolManifest) -> None:
    """Bind source claims and item coordinates to exact local JSONL bytes."""
    base = manifest.manifest_path.parent
    snapshot_manifest_path = base / "model_snapshot_manifest.v1.json"
    try:
        snapshot_manifest_raw = snapshot_manifest_path.read_bytes()
    except OSError as error:
        raise FixedPoolManifestError(
            f"cannot read model snapshot manifest: {error}"
        ) from error
    if (
        hashlib.sha256(snapshot_manifest_raw).hexdigest()
        != manifest.model_snapshot_manifest_sha256
    ):
        raise FixedPoolManifestError("model snapshot manifest SHA mismatch")
    model_root = (base / manifest.model_snapshot_path).resolve()
    if model_root.parent != base:
        raise FixedPoolManifestError("model snapshot path escapes manifest directory")
    try:
        snapshot_records = json.loads(snapshot_manifest_raw)
    except json.JSONDecodeError as error:
        raise FixedPoolManifestError("invalid model snapshot manifest JSON") from error
    if not isinstance(snapshot_records, list) or not snapshot_records:
        raise FixedPoolManifestError("model snapshot manifest must be a nonempty list")
    weight_records: list[dict[str, object]] = []
    for record in snapshot_records:
        if not isinstance(record, dict):
            raise FixedPoolManifestError("invalid model snapshot manifest record")
        relative = record.get("path")
        expected_size = record.get("bytes")
        expected_sha = record.get("sha256")
        if not isinstance(relative, str):
            raise FixedPoolManifestError("model snapshot record lacks a path")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise FixedPoolManifestError("unsafe model snapshot record path")
        asset = model_root / path
        try:
            stat = asset.stat()
        except OSError as error:
            raise FixedPoolManifestError(
                f"missing model snapshot asset {relative}: {error}"
            ) from error
        if stat.st_size != expected_size:
            raise FixedPoolManifestError(
                f"model snapshot asset size mismatch: {relative}"
            )
        try:
            digest = hashlib.sha256()
            with asset.open("rb") as handle:
                while chunk := handle.read(8 * 1024 * 1024):
                    digest.update(chunk)
            observed_sha = digest.hexdigest()
        except OSError as error:
            raise FixedPoolManifestError(
                f"cannot hash model snapshot asset {relative}: {error}"
            ) from error
        if observed_sha != expected_sha:
            raise FixedPoolManifestError(
                f"model snapshot asset SHA mismatch: {relative}"
            )
        if relative.endswith(".safetensors"):
            weight_records.append(
                {"path": relative, "bytes": expected_size, "sha256": expected_sha}
            )
    if not weight_records:
        raise FixedPoolManifestError("model snapshot contains no safetensor weights")
    if len(weight_records) == 1 and weight_records[0]["path"] == "model.safetensors":
        weights_sha256 = cast(str, weight_records[0]["sha256"])
    else:
        canonical_weights = json.dumps(
            sorted(weight_records, key=lambda value: cast(str, value["path"])),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        weights_sha256 = hashlib.sha256(canonical_weights).hexdigest()
    if weights_sha256 != manifest.model_weights_sha256:
        raise FixedPoolManifestError("pinned model weights SHA mismatch")
    source_rows, global_offsets = _load_materialized_source_rows(manifest)

    for item in manifest.items:
        rows = source_rows[item.source_id]
        if item.materialized_source_row >= len(rows):
            raise FixedPoolManifestError(
                f"materialized row is out of range at ordinal {item.ordinal}"
            )
        record = rows[item.materialized_source_row]
        canonical = json.dumps(
            record,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if hashlib.sha256(canonical).hexdigest() != item.materialized_record_sha256:
            raise FixedPoolManifestError(
                f"materialized record SHA mismatch at ordinal {item.ordinal}"
            )
        expected_identity = {
            "source_id": item.source_id,
            "source_dataset_index": item.source_dataset_index,
            "source_prompt_id": item.source_prompt_id,
            "repeated_prompt_cluster_id": item.repeated_prompt_cluster_id,
        }
        if any(record.get(key) != value for key, value in expected_identity.items()):
            raise FixedPoolManifestError(
                f"materialized source identity mismatch at ordinal {item.ordinal}"
            )
        expected_dataset_index = (
            global_offsets[item.source_id] + item.materialized_source_row
        )
        if item.dataset_index != expected_dataset_index:
            raise FixedPoolManifestError(
                f"merged dataset index mismatch at ordinal {item.ordinal}"
            )


def validate_ready_bias_manifest_design(manifest: FixedPoolManifest) -> None:
    """Enforce the preregistered 48-group heterogeneous calibration design."""
    tasks = ("AIME2024", "gsm8k")
    counts = {
        task: sum(item.task_name == task for item in manifest.items) for task in tasks
    }
    if len(manifest.items) != 48 or counts != {task: 24 for task in tasks}:
        raise FixedPoolManifestError("ready-bias pool must contain 24 prompts per task")
    by_cohort: dict[int, list[FixedPoolManifestItem]] = {}
    by_pair: dict[str, list[FixedPoolManifestItem]] = {}
    for item in manifest.items:
        by_cohort.setdefault(item.dispatch_cohort, []).append(item)
        by_pair.setdefault(item.matching_pair_id, []).append(item)
        if item.input_token_count > 256:
            raise FixedPoolManifestError("ready-bias input exceeds 256-token bound")
    for cohort, items in by_cohort.items():
        task_counts = {
            task: sum(item.task_name == task for item in items) for task in tasks
        }
        blocks = {item.decorrelation_block for item in items}
        if (
            len(items) != 4
            or task_counts != {task: 2 for task in tasks}
            or len(blocks) != 1
        ):
            raise FixedPoolManifestError(
                f"dispatch cohort {cohort} must contain a single balanced 2+2 block"
            )
    if len(by_pair) != 24:
        raise FixedPoolManifestError("ready-bias pool must contain 24 matching pairs")
    for pair_id, items in by_pair.items():
        if {item.task_name for item in items} != set(tasks) or len(items) != 2:
            raise FixedPoolManifestError(f"invalid cross-task matching pair {pair_id}")
        if abs(items[0].input_token_count - items[1].input_token_count) > 16:
            raise FixedPoolManifestError(f"token caliper exceeded for pair {pair_id}")


def _require_nonnegative_record_int(
    record: Mapping[str, object], field: str, *, ordinal: int
) -> int:
    value = record.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise FixedPoolManifestError(
            f"OpenMath materialized record {ordinal} requires nonnegative integer "
            f"{field}"
        )
    return value


def _require_record_sha256(
    record: Mapping[str, object], field: str, *, ordinal: int
) -> str:
    value = record.get(field)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FixedPoolManifestError(
            f"OpenMath materialized record {ordinal} requires lowercase SHA-256 {field}"
        )
    return value


def validate_openmath_latency_feasibility_manifest_design(
    manifest: FixedPoolManifest,
) -> None:
    """Enforce the preregistered 16-prompt OpenMath latency feasibility design."""
    if manifest.order_seed != OPENMATH_ORDER_SEED:
        raise FixedPoolManifestError(
            f"OpenMath order_seed must be {OPENMATH_ORDER_SEED}"
        )
    if (
        manifest.model_revision != OPENMATH_MODEL_REVISION
        or manifest.model_weights_sha256 != OPENMATH_MODEL_WEIGHTS_SHA256
    ):
        raise FixedPoolManifestError(
            "OpenMath manifest must use the pinned model revision and weights"
        )
    source_ids = tuple(source.source_id for source in manifest.sources)
    if source_ids != OPENMATH_SOURCE_IDS:
        raise FixedPoolManifestError(
            "OpenMath sources must be ordered openmath_short, openmath_long"
        )
    expected_files = tuple(f"{source_id}.jsonl" for source_id in OPENMATH_SOURCE_IDS)
    if tuple(source.materialized_file for source in manifest.sources) != expected_files:
        raise FixedPoolManifestError(
            "OpenMath materialized files must match the fixed source ordering"
        )
    expected_source_identity = (
        OPENMATH_DATASET_ID,
        OPENMATH_DATASET_REVISION,
        OPENMATH_DATASET_SPLIT,
    )
    if any(
        (source.dataset_id, source.revision, source.split) != expected_source_identity
        for source in manifest.sources
    ):
        raise FixedPoolManifestError(
            "OpenMath sources must use the pinned dataset revision and split"
        )

    counts = {
        source_id: sum(item.task_name == source_id for item in manifest.items)
        for source_id in OPENMATH_SOURCE_IDS
    }
    if len(manifest.items) != 16 or counts != {
        source_id: 8 for source_id in OPENMATH_SOURCE_IDS
    }:
        raise FixedPoolManifestError(
            "OpenMath feasibility pool must contain 8 short and 8 long prompts"
        )
    if any(item.source_id != item.task_name for item in manifest.items):
        raise FixedPoolManifestError(
            "OpenMath item source_id and task_name must identify the same stratum"
        )
    if len({item.source_prompt_id for item in manifest.items}) != 16:
        raise FixedPoolManifestError(
            "OpenMath feasibility pool must contain 16 distinct prompts"
        )

    by_cohort: dict[int, list[FixedPoolManifestItem]] = {}
    by_pair: dict[str, list[FixedPoolManifestItem]] = {}
    for item in manifest.items:
        by_cohort.setdefault(item.dispatch_cohort, []).append(item)
        by_pair.setdefault(item.matching_pair_id, []).append(item)
        if item.input_token_count > 256:
            raise FixedPoolManifestError(
                "OpenMath feasibility input exceeds 256-token bound"
            )
    if len(by_cohort) != 4:
        raise FixedPoolManifestError(
            "OpenMath feasibility pool must contain four dispatch cohorts"
        )
    for cohort, items in by_cohort.items():
        task_counts = {
            source_id: sum(item.task_name == source_id for item in items)
            for source_id in OPENMATH_SOURCE_IDS
        }
        if (
            len(items) != 4
            or task_counts != {source_id: 2 for source_id in OPENMATH_SOURCE_IDS}
            or len({item.decorrelation_block for item in items}) != 1
        ):
            raise FixedPoolManifestError(
                f"OpenMath dispatch cohort {cohort} must be one balanced 2+2 block"
            )
    if len(by_pair) != 8:
        raise FixedPoolManifestError(
            "OpenMath feasibility pool must contain eight matched pairs"
        )

    source_rows, _ = _load_materialized_source_rows(manifest)
    if any(len(source_rows[source_id]) != 8 for source_id in OPENMATH_SOURCE_IDS):
        raise FixedPoolManifestError(
            "OpenMath materialized sources must contain exactly 8 rows each"
        )
    materialized_records: dict[int, dict[str, object]] = {}
    solution_token_counts: dict[int, int] = {}
    normalized_problem_hashes: set[str] = set()
    for item in manifest.items:
        rows = source_rows[item.source_id]
        if item.materialized_source_row >= len(rows):
            raise FixedPoolManifestError(
                f"OpenMath materialized row is out of range at ordinal {item.ordinal}"
            )
        record = rows[item.materialized_source_row]
        if set(record) != OPENMATH_MATERIALIZED_RECORD_FIELDS:
            raise FixedPoolManifestError(
                f"OpenMath materialized record schema mismatch at ordinal {item.ordinal}"
            )
        canonical = json.dumps(
            record,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if hashlib.sha256(canonical).hexdigest() != item.materialized_record_sha256:
            raise FixedPoolManifestError(
                f"OpenMath materialized record SHA mismatch at ordinal {item.ordinal}"
            )
        solution_tokens = _require_nonnegative_record_int(
            record, "reference_solution_token_count", ordinal=item.ordinal
        )
        lower, upper = OPENMATH_REFERENCE_SOLUTION_BANDS[item.task_name]
        if not lower <= solution_tokens <= upper:
            raise FixedPoolManifestError(
                f"OpenMath reference solution is outside the {item.task_name} band "
                f"at ordinal {item.ordinal}"
            )
        solution_token_counts[item.ordinal] = solution_tokens
        _require_nonnegative_record_int(
            record, "reference_answer_token_count", ordinal=item.ordinal
        )
        normalized_problem_hashes.add(
            _require_record_sha256(
                record, "normalized_problem_sha256", ordinal=item.ordinal
            )
        )
        problem_source = record.get("problem_source")
        if not isinstance(problem_source, str) or not problem_source:
            raise FixedPoolManifestError(
                f"OpenMath materialized record {item.ordinal} requires problem_source"
            )
        expected_record_fields = {
            "source_id": item.source_id,
            "source_dataset_id": OPENMATH_DATASET_ID,
            "source_revision": OPENMATH_DATASET_REVISION,
            "source_split": OPENMATH_DATASET_SPLIT,
            "source_dataset_index": item.source_dataset_index,
            "source_prompt_id": item.source_prompt_id,
            "repeated_prompt_cluster_id": item.repeated_prompt_cluster_id,
            "selection_stratum": item.task_name,
            "matching_pair_id": item.matching_pair_id,
            "input_token_count": item.input_token_count,
            "input_token_ids_sha256": item.input_token_ids_sha256,
            "selection_seed": OPENMATH_SELECTION_SEED,
            "model_revision": OPENMATH_MODEL_REVISION,
            "prompt_file_sha256": OPENMATH_PROMPT_FILE_SHA256,
        }
        if any(
            record.get(field) != value
            for field, value in expected_record_fields.items()
        ):
            raise FixedPoolManifestError(
                f"OpenMath bound source metadata mismatch at ordinal {item.ordinal}"
            )
        materialized_records[item.ordinal] = record
    if len(normalized_problem_hashes) != 16:
        raise FixedPoolManifestError(
            "OpenMath feasibility pool must contain 16 distinct normalized problems"
        )
    median_solution_tokens = {
        source_id: statistics.median(
            solution_token_counts[item.ordinal]
            for item in manifest.items
            if item.task_name == source_id
        )
        for source_id in OPENMATH_SOURCE_IDS
    }
    if (
        median_solution_tokens["openmath_long"]
        / median_solution_tokens["openmath_short"]
        < 3.0
    ):
        raise FixedPoolManifestError(
            "OpenMath long/short reference-solution median ratio is below 3.0"
        )

    for pair_id, items in by_pair.items():
        if len(items) != 2 or {item.task_name for item in items} != set(
            OPENMATH_SOURCE_IDS
        ):
            raise FixedPoolManifestError(f"invalid OpenMath matched pair {pair_id}")
        if len({item.dispatch_cohort for item in items}) != 1:
            raise FixedPoolManifestError(
                f"OpenMath matched pair {pair_id} crosses dispatch cohorts"
            )
        if (
            len(
                {materialized_records[item.ordinal]["problem_source"] for item in items}
            )
            != 1
        ):
            raise FixedPoolManifestError(
                f"OpenMath matched pair {pair_id} crosses problem sources"
            )
        if (
            abs(items[0].input_token_count - items[1].input_token_count)
            > OPENMATH_INPUT_TOKEN_CALIPER
        ):
            raise FixedPoolManifestError(
                f"OpenMath input-token caliper exceeded for pair {pair_id}"
            )
        answer_counts = [
            _require_nonnegative_record_int(
                materialized_records[item.ordinal],
                "reference_answer_token_count",
                ordinal=item.ordinal,
            )
            for item in items
        ]
        if abs(answer_counts[0] - answer_counts[1]) > OPENMATH_ANSWER_TOKEN_CALIPER:
            raise FixedPoolManifestError(
                f"OpenMath answer-token caliper exceeded for pair {pair_id}"
            )


def _sliding_puzzle_distance(state: tuple[int, ...]) -> int:
    goal = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    if state == goal:
        return 0
    frontier = [state]
    visited = {state}
    for distance in range(1, 13):
        next_frontier: list[tuple[int, ...]] = []
        for current in frontier:
            empty = current.index(0)
            row, column = divmod(empty, 3)
            for delta_row, delta_column in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                next_row, next_column = row + delta_row, column + delta_column
                if not (0 <= next_row < 3 and 0 <= next_column < 3):
                    continue
                other = next_row * 3 + next_column
                mutable = list(current)
                mutable[empty], mutable[other] = mutable[other], mutable[empty]
                candidate = tuple(mutable)
                if candidate == goal:
                    return distance
                if candidate not in visited:
                    visited.add(candidate)
                    next_frontier.append(candidate)
        frontier = next_frontier
    raise FixedPoolManifestError("sliding-puzzle board exceeds distance-12 design")


def _sliding_puzzle_permutation_rank(state: tuple[int, ...]) -> int:
    rank = 0
    remaining = list(range(9))
    factorials = [1]
    for value in range(1, 10):
        factorials.append(factorials[-1] * value)
    for index, value in enumerate(state):
        position = remaining.index(value)
        rank += position * factorials[8 - index]
        remaining.pop(position)
    return rank


def _validate_sliding_puzzle_manifest_design(
    manifest: FixedPoolManifest,
    *,
    model_revision: str,
    model_weights_sha256: str,
    prompt_builder_version: str,
) -> None:
    """Enforce the frozen 16-board exact-distance puzzle feasibility design."""
    if manifest.order_seed != SLIDING_PUZZLE_ORDER_SEED:
        raise FixedPoolManifestError(
            f"sliding-puzzle order_seed must be {SLIDING_PUZZLE_ORDER_SEED}"
        )
    if (
        manifest.model_revision != model_revision
        or manifest.model_weights_sha256 != model_weights_sha256
    ):
        raise FixedPoolManifestError(
            "sliding-puzzle manifest must use the pinned model revision and weights"
        )
    source_ids = tuple(source.source_id for source in manifest.sources)
    if source_ids != SLIDING_PUZZLE_SOURCE_IDS:
        raise FixedPoolManifestError(
            "sliding-puzzle sources must be ordered easy, hard"
        )
    source_identity = (
        SLIDING_PUZZLE_DATASET_ID,
        SLIDING_PUZZLE_DATASET_REVISION,
        SLIDING_PUZZLE_DATASET_SPLIT,
    )
    if any(
        (source.dataset_id, source.revision, source.split) != source_identity
        for source in manifest.sources
    ):
        raise FixedPoolManifestError(
            "sliding-puzzle sources must use the frozen generated-source identity"
        )
    if tuple(source.materialized_file for source in manifest.sources) != tuple(
        f"{source_id}.jsonl" for source_id in SLIDING_PUZZLE_SOURCE_IDS
    ):
        raise FixedPoolManifestError(
            "sliding-puzzle materialized files do not match the source order"
        )

    task_counts = {
        source_id: sum(item.task_name == source_id for item in manifest.items)
        for source_id in SLIDING_PUZZLE_SOURCE_IDS
    }
    if len(manifest.items) != 16 or task_counts != {
        source_id: 8 for source_id in SLIDING_PUZZLE_SOURCE_IDS
    }:
        raise FixedPoolManifestError(
            "sliding-puzzle pool must contain exactly 8 easy and 8 hard boards"
        )
    if any(item.source_id != item.task_name for item in manifest.items):
        raise FixedPoolManifestError(
            "sliding-puzzle item source_id and task_name must match"
        )

    by_cohort: dict[int, list[FixedPoolManifestItem]] = {}
    by_pair: dict[str, list[FixedPoolManifestItem]] = {}
    for item in manifest.items:
        by_cohort.setdefault(item.dispatch_cohort, []).append(item)
        by_pair.setdefault(item.matching_pair_id, []).append(item)
        if item.input_token_count > SLIDING_PUZZLE_MAX_INPUT_TOKENS:
            raise FixedPoolManifestError(
                "sliding-puzzle input exceeds the 512-token design bound"
            )
    if len(by_cohort) != 4:
        raise FixedPoolManifestError(
            "sliding-puzzle pool must contain exactly four cohorts"
        )
    for cohort, items in by_cohort.items():
        counts = {
            source_id: sum(item.task_name == source_id for item in items)
            for source_id in SLIDING_PUZZLE_SOURCE_IDS
        }
        if (
            len(items) != 4
            or counts != {source_id: 2 for source_id in SLIDING_PUZZLE_SOURCE_IDS}
            or {item.decorrelation_block for item in items} != {f"cohort-{cohort}"}
        ):
            raise FixedPoolManifestError(
                f"sliding-puzzle cohort {cohort} must be one balanced 2+2 block"
            )
    if len(by_pair) != 8:
        raise FixedPoolManifestError(
            "sliding-puzzle pool must contain eight matched pairs"
        )

    source_rows, _ = _load_materialized_source_rows(manifest)
    if any(len(source_rows[source_id]) != 8 for source_id in SLIDING_PUZZLE_SOURCE_IDS):
        raise FixedPoolManifestError(
            "sliding-puzzle materialized sources must contain eight rows each"
        )
    distances = {source_id: [] for source_id in SLIDING_PUZZLE_SOURCE_IDS}
    board_hashes: set[str] = set()
    for item in manifest.items:
        rows = source_rows[item.source_id]
        if item.materialized_source_row >= len(rows):
            raise FixedPoolManifestError(
                f"sliding-puzzle row out of range at ordinal {item.ordinal}"
            )
        record = rows[item.materialized_source_row]
        if set(record) != SLIDING_PUZZLE_MATERIALIZED_RECORD_FIELDS:
            raise FixedPoolManifestError(
                f"sliding-puzzle record schema mismatch at ordinal {item.ordinal}"
            )
        canonical_record = json.dumps(
            record, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode()
        if (
            hashlib.sha256(canonical_record).hexdigest()
            != item.materialized_record_sha256
        ):
            raise FixedPoolManifestError(
                f"sliding-puzzle record SHA mismatch at ordinal {item.ordinal}"
            )
        bound = {
            "source_id": item.source_id,
            "source_dataset_id": SLIDING_PUZZLE_DATASET_ID,
            "source_revision": SLIDING_PUZZLE_DATASET_REVISION,
            "source_split": SLIDING_PUZZLE_DATASET_SPLIT,
            "source_dataset_index": item.source_dataset_index,
            "source_prompt_id": item.source_prompt_id,
            "repeated_prompt_cluster_id": item.repeated_prompt_cluster_id,
            "selection_stratum": item.task_name,
            "matching_pair_id": item.matching_pair_id,
            "input_token_count": item.input_token_count,
            "input_token_ids_sha256": item.input_token_ids_sha256,
            "selection_seed": SLIDING_PUZZLE_SELECTION_SEED,
            "model_revision": model_revision,
            "prompt_builder_version": prompt_builder_version,
        }
        if any(record.get(key) != value for key, value in bound.items()):
            raise FixedPoolManifestError(
                f"sliding-puzzle source binding mismatch at ordinal {item.ordinal}"
            )
        messages = record.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 1
            or not isinstance(messages[0], dict)
            or set(messages[0]) != {"role", "content"}
            or messages[0].get("role") != "user"
            or not isinstance(messages[0].get("content"), str)
            or not messages[0]["content"]
        ):
            raise FixedPoolManifestError(
                f"invalid sliding-puzzle message at ordinal {item.ordinal}"
            )
        if record.get("stop_strings") != ["</action>"]:
            raise FixedPoolManifestError(
                f"invalid sliding-puzzle stop string at ordinal {item.ordinal}"
            )
        extra = record.get("extra_env_info")
        if not isinstance(extra, dict) or set(extra) != {
            "game_state",
            "num_moves",
            "max_moves",
            "optimal_distance",
        }:
            raise FixedPoolManifestError(
                f"invalid sliding-puzzle environment metadata at ordinal {item.ordinal}"
            )
        if extra.get("num_moves") != 0 or extra.get("max_moves") != 12:
            raise FixedPoolManifestError(
                f"invalid sliding-puzzle move bounds at ordinal {item.ordinal}"
            )
        game_state = extra.get("game_state")
        grid = game_state.get("grid") if isinstance(game_state, dict) else None
        if (
            not isinstance(grid, list)
            or len(grid) != 3
            or any(not isinstance(row, list) or len(row) != 3 for row in grid)
        ):
            raise FixedPoolManifestError(
                f"invalid sliding-puzzle grid at ordinal {item.ordinal}"
            )
        state = tuple(value for row in grid for value in row)
        if any(not isinstance(value, int) for value in state) or set(state) != set(
            range(9)
        ):
            raise FixedPoolManifestError(
                f"invalid sliding-puzzle tile permutation at ordinal {item.ordinal}"
            )
        goal_grid = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        empty_position = list(divmod(state.index(0), 3))
        if (
            set(game_state) != {"size", "grid", "solution", "empty_pos", "commands"}
            or game_state.get("size") != 3
            or game_state.get("solution") != goal_grid
            or game_state.get("empty_pos") != empty_position
            or not isinstance(game_state.get("commands"), dict)
        ):
            raise FixedPoolManifestError(
                f"inconsistent sliding-puzzle game state at ordinal {item.ordinal}"
            )
        distance = _sliding_puzzle_distance(state)
        if (
            record.get("optimal_distance") != distance
            or extra.get("optimal_distance") != distance
        ):
            raise FixedPoolManifestError(
                f"incorrect sliding-puzzle optimal distance at ordinal {item.ordinal}"
            )
        if _sliding_puzzle_permutation_rank(state) != item.source_dataset_index:
            raise FixedPoolManifestError(
                f"incorrect sliding-puzzle source index at ordinal {item.ordinal}"
            )
        board_sha = hashlib.sha256(
            json.dumps(state, separators=(",", ":")).encode()
        ).hexdigest()
        if record.get("board_state_sha256") != board_sha:
            raise FixedPoolManifestError(
                f"incorrect sliding-puzzle board hash at ordinal {item.ordinal}"
            )
        board_hashes.add(board_sha)
        distances[item.task_name].append(distance)
    if len(board_hashes) != 16:
        raise FixedPoolManifestError("sliding-puzzle boards must be unique")
    if any(
        tuple(sorted(distances[source_id]))
        != tuple(sorted(SLIDING_PUZZLE_DISTANCE_SCHEDULES[source_id]))
        for source_id in SLIDING_PUZZLE_SOURCE_IDS
    ):
        raise FixedPoolManifestError("sliding-puzzle distance schedule mismatch")
    for pair_id, items in by_pair.items():
        if (
            len(items) != 2
            or {item.task_name for item in items} != set(SLIDING_PUZZLE_SOURCE_IDS)
            or len({item.dispatch_cohort for item in items}) != 1
        ):
            raise FixedPoolManifestError(
                f"invalid sliding-puzzle matched pair {pair_id}"
            )
        if (
            abs(items[0].input_token_count - items[1].input_token_count)
            > SLIDING_PUZZLE_INPUT_TOKEN_CALIPER
        ):
            raise FixedPoolManifestError(
                f"sliding-puzzle token caliper exceeded for pair {pair_id}"
            )


def validate_sliding_puzzle_latency_feasibility_manifest_design(
    manifest: FixedPoolManifest,
) -> None:
    """Enforce the frozen 1.5B sliding-puzzle feasibility design."""
    _validate_sliding_puzzle_manifest_design(
        manifest,
        model_revision=SLIDING_PUZZLE_MODEL_REVISION,
        model_weights_sha256=SLIDING_PUZZLE_MODEL_WEIGHTS_SHA256,
        prompt_builder_version=SLIDING_PUZZLE_PROMPT_BUILDER_VERSION,
    )


def validate_sliding_puzzle_7b_competence_manifest_design(
    manifest: FixedPoolManifest,
) -> None:
    """Enforce the exposed-pool 7B competence materialization design."""
    _validate_sliding_puzzle_manifest_design(
        manifest,
        model_revision=SLIDING_PUZZLE_7B_MODEL_REVISION,
        model_weights_sha256=SLIDING_PUZZLE_7B_MODEL_WEIGHTS_SHA256,
        prompt_builder_version=SLIDING_PUZZLE_PROMPT_BUILDER_VERSION,
    )


def validate_sliding_puzzle_7b_compact_prompt_manifest_design(
    manifest: FixedPoolManifest,
) -> None:
    """Enforce the exposed-pool 7B compact-action prompt design."""
    _validate_sliding_puzzle_manifest_design(
        manifest,
        model_revision=SLIDING_PUZZLE_7B_MODEL_REVISION,
        model_weights_sha256=SLIDING_PUZZLE_7B_MODEL_WEIGHTS_SHA256,
        prompt_builder_version=SLIDING_PUZZLE_COMPACT_PROMPT_BUILDER_VERSION,
    )


def validate_structured_generation_latency_manifest_design(
    manifest: FixedPoolManifest,
) -> None:
    """Enforce the frozen structured-generation feasibility design."""
    _validate_structured_generation_manifest_design(
        manifest,
        expected_order_seed=STRUCTURED_GENERATION_ORDER_SEED,
        expected_selection_seed=STRUCTURED_GENERATION_SELECTION_SEED,
        expected_protocol_sha256=STRUCTURED_GENERATION_PLAN_SHA256,
        expected_split=STRUCTURED_GENERATION_DATASET_SPLIT,
    )


def validate_structured_generation_scheduler_crossover_manifest_design(
    manifest: FixedPoolManifest,
) -> None:
    """Enforce one confirmed fresh structured scheduler-crossover pool."""
    selection_seed = STRUCTURED_CROSSOVER_SELECTION_SEEDS.get(manifest.order_seed)
    if selection_seed is None:
        raise FixedPoolManifestError("structured crossover order seed mismatch")
    _validate_structured_generation_manifest_design(
        manifest,
        expected_order_seed=manifest.order_seed,
        expected_selection_seed=selection_seed,
        expected_protocol_sha256=STRUCTURED_CROSSOVER_PROTOCOL_SHA256,
        expected_split="scheduler_crossover",
    )


def validate_structured_scheduler_pressure_response_manifest_design(
    manifest: FixedPoolManifest,
) -> None:
    """Enforce one confirmed fresh scheduler pressure-response pool."""
    selection_seed = STRUCTURED_PRESSURE_SELECTION_SEEDS.get(manifest.order_seed)
    if selection_seed is None:
        raise FixedPoolManifestError("structured pressure-response order seed mismatch")
    _validate_structured_generation_manifest_design(
        manifest,
        expected_order_seed=manifest.order_seed,
        expected_selection_seed=selection_seed,
        expected_protocol_sha256=STRUCTURED_PRESSURE_PROTOCOL_SHA256,
        expected_split="scheduler_pressure_response",
        expected_pairs=16,
        expected_cohorts=8,
    )


def validate_dapo_operational_latency_discovery_manifest_design(
    manifest: FixedPoolManifest,
    *,
    expected_selection_seeds: frozenset[int] | None = None,
    expected_protocol_sha256: str | None = None,
    study_label: str = "DAPO discovery",
) -> None:
    """Enforce one deduplicated DAPO operational-latency discovery pool."""
    if expected_selection_seeds is None:
        expected_selection_seeds = DAPO_OPERATIONAL_SELECTION_SEEDS
    if expected_protocol_sha256 is None:
        expected_protocol_sha256 = DAPO_OPERATIONAL_PROTOCOL_SHA256
    if manifest.order_seed not in expected_selection_seeds:
        raise FixedPoolManifestError(f"{study_label} selection seed mismatch")
    if (
        manifest.model_revision != DAPO_OPERATIONAL_MODEL_REVISION
        or manifest.design_protocol_sha256 != expected_protocol_sha256
    ):
        raise FixedPoolManifestError(f"{study_label} model or protocol mismatch")
    if tuple(source.source_id for source in manifest.sources) != (
        DAPO_OPERATIONAL_SOURCE_IDS
    ):
        raise FixedPoolManifestError("DAPO discovery source order mismatch")
    expected_identity = (
        DAPO_OPERATIONAL_DATASET_ID,
        DAPO_OPERATIONAL_DATASET_REVISION,
        "train",
    )
    if any(
        (source.dataset_id, source.revision, source.split) != expected_identity
        for source in manifest.sources
    ):
        raise FixedPoolManifestError("DAPO discovery source identity mismatch")
    expected_files = tuple(
        f"{source_id}_{manifest.order_seed}.jsonl"
        for source_id in DAPO_OPERATIONAL_SOURCE_IDS
    )
    if tuple(source.materialized_file for source in manifest.sources) != expected_files:
        raise FixedPoolManifestError("DAPO discovery source filename mismatch")
    counts = {
        source_id: sum(item.task_name == source_id for item in manifest.items)
        for source_id in DAPO_OPERATIONAL_SOURCE_IDS
    }
    if len(manifest.items) != 16 or counts != {
        source_id: 8 for source_id in DAPO_OPERATIONAL_SOURCE_IDS
    }:
        raise FixedPoolManifestError("DAPO discovery requires two 8-row shards")
    if (
        any(item.source_id != item.task_name for item in manifest.items)
        or len({item.source_prompt_id for item in manifest.items}) != 16
        or len({item.repeated_prompt_cluster_id for item in manifest.items}) != 16
    ):
        raise FixedPoolManifestError("DAPO discovery prompts must be unique")
    by_cohort: dict[int, list[FixedPoolManifestItem]] = {}
    for item in manifest.items:
        by_cohort.setdefault(item.dispatch_cohort, []).append(item)
        if not 0 < item.input_token_count <= 2048:
            raise FixedPoolManifestError("DAPO discovery input-token bound violated")
    if len(by_cohort) != 4 or any(
        len(items) != 4
        or {item.decorrelation_block for item in items} != {f"cohort-{cohort}"}
        for cohort, items in by_cohort.items()
    ):
        raise FixedPoolManifestError("DAPO discovery cohort geometry mismatch")

    source_rows, _ = _load_materialized_source_rows(manifest)
    if any(
        len(source_rows[source_id]) != 8 for source_id in DAPO_OPERATIONAL_SOURCE_IDS
    ):
        raise FixedPoolManifestError("DAPO discovery shard row count mismatch")
    canonical_prompts: set[str] = set()
    for item in manifest.items:
        record = source_rows[item.source_id][item.materialized_source_row]
        if set(record) != DAPO_OPERATIONAL_MATERIALIZED_RECORD_FIELDS:
            raise FixedPoolManifestError(
                f"DAPO discovery record schema mismatch at {item.ordinal}"
            )
        canonical_prompt_sha = _require_record_sha256(
            record, "canonical_prompt_sha256", ordinal=item.ordinal
        )
        prompt = record.get("input")
        ground_truth = record.get("output")
        duplicate_count = record.get("source_duplicate_count")
        source_extra_index = record.get("source_extra_index")
        if (
            not isinstance(prompt, str)
            or not prompt
            or not isinstance(ground_truth, str)
            or not ground_truth
            or not isinstance(duplicate_count, int)
            or isinstance(duplicate_count, bool)
            or duplicate_count < 1
            or not isinstance(source_extra_index, str)
        ):
            raise FixedPoolManifestError(
                f"DAPO discovery record value mismatch at {item.ordinal}"
            )
        observed_prompt_sha = hashlib.sha256(
            json.dumps(
                [{"role": "user", "content": prompt}],
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        expected = {
            "source_id": item.source_id,
            "source_dataset_id": DAPO_OPERATIONAL_DATASET_ID,
            "source_revision": DAPO_OPERATIONAL_DATASET_REVISION,
            "source_split": "train",
            "source_dataset_index": item.source_dataset_index,
            "source_prompt_id": item.source_prompt_id,
            "repeated_prompt_cluster_id": item.repeated_prompt_cluster_id,
            "input_token_count": item.input_token_count,
            "input_token_ids_sha256": item.input_token_ids_sha256,
            "selection_seed": manifest.order_seed,
            "model_revision": DAPO_OPERATIONAL_MODEL_REVISION,
            "canonical_prompt_sha256": observed_prompt_sha,
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise FixedPoolManifestError(
                f"DAPO discovery source binding mismatch at {item.ordinal}"
            )
        if item.matching_pair_id != f"unique-prompt-{canonical_prompt_sha}":
            raise FixedPoolManifestError(
                f"DAPO discovery unique-prompt identity mismatch at {item.ordinal}"
            )
        canonical_prompts.add(canonical_prompt_sha)
    if len(canonical_prompts) != 16:
        raise FixedPoolManifestError("DAPO discovery canonical prompts are not unique")


def validate_dapo_scheduler_crossover_manifest_design(
    manifest: FixedPoolManifest,
) -> None:
    """Enforce one untouched-holdout DAPO scheduler-crossover pool."""
    validate_dapo_operational_latency_discovery_manifest_design(
        manifest,
        expected_selection_seeds=DAPO_CROSSOVER_SELECTION_SEEDS,
        expected_protocol_sha256=DAPO_CROSSOVER_PROTOCOL_SHA256,
        study_label="DAPO crossover",
    )


def _validate_structured_generation_manifest_design(
    manifest: FixedPoolManifest,
    *,
    expected_order_seed: int,
    expected_selection_seed: int,
    expected_protocol_sha256: str,
    expected_split: str,
    expected_pairs: int = 8,
    expected_cohorts: int = 4,
) -> None:
    """Validate shared immutable structure for generated short/long pairs."""
    if manifest.order_seed != expected_order_seed:
        raise FixedPoolManifestError("structured-generation order seed mismatch")
    if (
        manifest.model_revision != STRUCTURED_GENERATION_MODEL_REVISION
        or manifest.model_weights_sha256 != STRUCTURED_GENERATION_MODEL_WEIGHTS_SHA256
        or manifest.design_protocol_sha256 != expected_protocol_sha256
    ):
        raise FixedPoolManifestError("structured-generation model or plan mismatch")
    source_ids = tuple(source.source_id for source in manifest.sources)
    if source_ids != STRUCTURED_GENERATION_SOURCE_IDS:
        raise FixedPoolManifestError("structured-generation source order mismatch")
    expected_identity = (
        STRUCTURED_GENERATION_DATASET_ID,
        STRUCTURED_GENERATION_DATASET_REVISION,
        expected_split,
    )
    if any(
        (source.dataset_id, source.revision, source.split) != expected_identity
        for source in manifest.sources
    ):
        raise FixedPoolManifestError("structured-generation source identity mismatch")
    if tuple(source.materialized_file for source in manifest.sources) != tuple(
        f"{source_id}.jsonl" for source_id in STRUCTURED_GENERATION_SOURCE_IDS
    ):
        raise FixedPoolManifestError("structured-generation filenames mismatch")
    counts = {
        source_id: sum(item.task_name == source_id for item in manifest.items)
        for source_id in STRUCTURED_GENERATION_SOURCE_IDS
    }
    if len(manifest.items) != 2 * expected_pairs or counts != {
        source_id: expected_pairs for source_id in STRUCTURED_GENERATION_SOURCE_IDS
    }:
        raise FixedPoolManifestError("structured-generation pool is not balanced")
    if any(item.source_id != item.task_name for item in manifest.items):
        raise FixedPoolManifestError("structured-generation source/task mismatch")
    by_cohort: dict[int, list[FixedPoolManifestItem]] = {}
    by_pair: dict[str, list[FixedPoolManifestItem]] = {}
    for item in manifest.items:
        by_cohort.setdefault(item.dispatch_cohort, []).append(item)
        by_pair.setdefault(item.matching_pair_id, []).append(item)
        if item.input_token_count > 256:
            raise FixedPoolManifestError(
                "structured-generation input exceeds 256 tokens"
            )
    if len(by_cohort) != expected_cohorts or len(by_pair) != expected_pairs:
        raise FixedPoolManifestError("structured-generation cohort/pair count mismatch")
    for cohort, items in by_cohort.items():
        cohort_counts = {
            source_id: sum(item.task_name == source_id for item in items)
            for source_id in STRUCTURED_GENERATION_SOURCE_IDS
        }
        if (
            len(items) != 4
            or cohort_counts
            != {source_id: 2 for source_id in STRUCTURED_GENERATION_SOURCE_IDS}
            or {item.decorrelation_block for item in items} != {f"cohort-{cohort}"}
        ):
            raise FixedPoolManifestError(
                f"structured-generation cohort {cohort} is not balanced"
            )
    source_rows, _ = _load_materialized_source_rows(manifest)
    if any(
        len(source_rows[source_id]) != expected_pairs
        for source_id in STRUCTURED_GENERATION_SOURCE_IDS
    ):
        raise FixedPoolManifestError("structured-generation source row count mismatch")
    records: dict[int, dict[str, object]] = {}
    for item in manifest.items:
        record = source_rows[item.source_id][item.materialized_source_row]
        if set(record) != STRUCTURED_GENERATION_MATERIALIZED_RECORD_FIELDS:
            raise FixedPoolManifestError(
                f"structured-generation record schema mismatch at {item.ordinal}"
            )
        bound = {
            "source_id": item.source_id,
            "source_dataset_id": STRUCTURED_GENERATION_DATASET_ID,
            "source_revision": STRUCTURED_GENERATION_DATASET_REVISION,
            "source_split": expected_split,
            "source_dataset_index": item.source_dataset_index,
            "source_prompt_id": item.source_prompt_id,
            "repeated_prompt_cluster_id": item.repeated_prompt_cluster_id,
            "selection_stratum": item.task_name,
            "matching_pair_id": item.matching_pair_id,
            "input_token_count": item.input_token_count,
            "input_token_ids_sha256": item.input_token_ids_sha256,
            "selection_seed": expected_selection_seed,
            "model_revision": STRUCTURED_GENERATION_MODEL_REVISION,
            "required_check_lines": STRUCTURED_GENERATION_CHECK_LINES[item.task_name],
        }
        if any(record.get(key) != value for key, value in bound.items()):
            raise FixedPoolManifestError(
                f"structured-generation source binding mismatch at {item.ordinal}"
            )
        left = _require_nonnegative_record_int(
            record, "left_operand", ordinal=item.ordinal
        )
        right = _require_nonnegative_record_int(
            record, "right_operand", ordinal=item.ordinal
        )
        answer = _require_nonnegative_record_int(
            record, "expected_answer", ordinal=item.ordinal
        )
        if answer != left + right or record.get("output") != str(answer):
            raise FixedPoolManifestError(
                f"structured-generation arithmetic mismatch at {item.ordinal}"
            )
        if not isinstance(record.get("input"), str) or not record["input"]:
            raise FixedPoolManifestError(
                f"structured-generation prompt missing at {item.ordinal}"
            )
        records[item.ordinal] = record
    for pair_id, items in by_pair.items():
        if (
            len(items) != 2
            or {item.task_name for item in items}
            != set(STRUCTURED_GENERATION_SOURCE_IDS)
            or len({item.dispatch_cohort for item in items}) != 1
            or len({item.repeated_prompt_cluster_id for item in items}) != 1
        ):
            raise FixedPoolManifestError(
                f"invalid structured-generation pair {pair_id}"
            )
        pair_records = [records[item.ordinal] for item in items]
        if (
            len(
                {
                    (
                        record["left_operand"],
                        record["right_operand"],
                        record["expected_answer"],
                    )
                    for record in pair_records
                }
            )
            != 1
        ):
            raise FixedPoolManifestError(
                f"structured-generation arithmetic pair mismatch in {pair_id}"
            )
        if abs(items[0].input_token_count - items[1].input_token_count) > 4:
            raise FixedPoolManifestError(
                f"structured-generation input-token caliper exceeded in {pair_id}"
            )


def validate_fixed_pool_manifest_design(
    manifest: FixedPoolManifest, design_id: str
) -> None:
    """Dispatch strict validation for a declared fixed-pool study design."""
    if design_id == "ready_bias_v1":
        validate_ready_bias_manifest_design(manifest)
    elif design_id in {
        "openmath_latency_feasibility_v1",
        "openmath_termination_headroom_v1",
    }:
        validate_openmath_latency_feasibility_manifest_design(manifest)
    elif design_id == "sliding_puzzle_latency_feasibility_v1":
        validate_sliding_puzzle_latency_feasibility_manifest_design(manifest)
    elif design_id == "sliding_puzzle_7b_competence_v1":
        validate_sliding_puzzle_7b_competence_manifest_design(manifest)
    elif design_id == "sliding_puzzle_7b_compact_prompt_v2":
        validate_sliding_puzzle_7b_compact_prompt_manifest_design(manifest)
    elif design_id == "structured_generation_latency_v1":
        validate_structured_generation_latency_manifest_design(manifest)
    elif design_id == "structured_generation_scheduler_crossover_v1":
        validate_structured_generation_scheduler_crossover_manifest_design(manifest)
    elif design_id == "structured_scheduler_pressure_response_v1":
        validate_structured_scheduler_pressure_response_manifest_design(manifest)
    elif design_id == "dapo_math_operational_latency_discovery_v2":
        validate_dapo_operational_latency_discovery_manifest_design(manifest)
    elif design_id == "dapo_math_scheduler_crossover_v1":
        validate_dapo_scheduler_crossover_manifest_design(manifest)
    else:
        raise FixedPoolManifestError(f"unsupported fixed-pool design_id: {design_id!r}")


class _DatumDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> DatumSpec: ...


class FixedPoolDataset(Sequence[DatumSpec]):
    """Select and label a processed dataset in exact manifest order."""

    def __init__(self, dataset: _DatumDataset, manifest: FixedPoolManifest) -> None:
        dataset_size = len(dataset)
        for item in manifest.items:
            if item.dataset_index >= dataset_size:
                raise FixedPoolManifestError(
                    f"dataset_index {item.dataset_index} is outside dataset size "
                    f"{dataset_size}"
                )
        self._dataset = dataset
        self._manifest = manifest

    def __len__(self) -> int:
        return len(self._manifest.items)

    def __getitem__(self, index: int | slice) -> DatumSpec | Sequence[DatumSpec]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        item = self._manifest.items[index]
        datum = dict(self._dataset[item.dataset_index])
        actual_idx = datum.get("idx")
        if actual_idx != item.dataset_index:
            raise FixedPoolManifestError(
                f"manifest item {item.ordinal} expected dataset idx "
                f"{item.dataset_index}, got {actual_idx!r}"
            )
        actual_task = datum.get("task_name")
        if actual_task != item.task_name:
            raise FixedPoolManifestError(
                f"manifest item {item.ordinal} expected task {item.task_name!r}, "
                f"got {actual_task!r}"
            )
        token_ids: list[int] = []
        for message in datum["message_log"]:
            values = message["token_ids"]
            if hasattr(values, "tolist"):
                values = values.tolist()
            token_ids.extend(int(value) for value in values)
        token_digest = hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if len(token_ids) != item.input_token_count:
            raise FixedPoolManifestError(
                f"input token count mismatch at ordinal {item.ordinal}"
            )
        if token_digest != item.input_token_ids_sha256:
            raise FixedPoolManifestError(
                f"input token IDs SHA mismatch at ordinal {item.ordinal}"
            )
        datum.update(
            source_pool_ordinal=item.ordinal,
            source_prompt_id=item.source_prompt_id,
            repeated_prompt_cluster_id=item.repeated_prompt_cluster_id,
            dispatch_cohort=item.dispatch_cohort,
        )
        return cast(DatumSpec, datum)


def fixed_pool_collate_fn(data_batch: list[DatumSpec]) -> BatchedDataDict[Any]:
    """Collate normal RL fields plus required fixed-pool identity metadata."""
    if not data_batch:
        raise FixedPoolManifestError("cannot collate an empty fixed-pool batch")
    required = (
        "source_pool_ordinal",
        "source_prompt_id",
        "repeated_prompt_cluster_id",
        "dispatch_cohort",
    )
    for datum in data_batch:
        missing = [field for field in required if field not in datum]
        if missing:
            raise FixedPoolManifestError(
                f"fixed-pool datum is missing identity fields: {missing}"
            )
    cohorts = {datum["dispatch_cohort"] for datum in data_batch}
    if len(cohorts) != 1:
        raise FixedPoolManifestError(
            "a fixed-pool dataloader batch must not cross a dispatch-cohort boundary"
        )
    batch = rl_collate_fn(data_batch)
    for field in required:
        batch[field] = [datum[field] for datum in data_batch]  # type: ignore[literal-required]
    return batch


def validate_fixed_pool_trace(
    trace_path: str | Path,
    manifest: FixedPoolManifest,
    *,
    expected_completions_per_group: int,
) -> FixedPoolTraceValidationReport:
    """Validate a complete scheduler-neutral, fixed-policy source collection."""
    if expected_completions_per_group < 1:
        raise ValueError("expected_completions_per_group must be positive")
    generic_report = validate_scheduler_trace(trace_path)
    if generic_report.incomplete_attempt_ids:
        raise SchedulerTraceValidationError("fixed-pool trace has incomplete attempts")
    if generic_report.administratively_censored_group_ids:
        raise SchedulerTraceValidationError(
            "fixed-pool trace has administrative horizon censoring"
        )
    events = list(iter_scheduler_trace(trace_path))
    run_start, run_end = events[0], events[-1]
    for boundary in (run_start, run_end):
        if boundary.run_mode != "fixed_pool":
            raise SchedulerTraceValidationError("fixed-pool run mode is missing")
        if boundary.pool_id != manifest.pool_id:
            raise SchedulerTraceValidationError("fixed-pool pool ID mismatch")
        if boundary.pool_manifest_sha256 != manifest.manifest_sha256:
            raise SchedulerTraceValidationError("fixed-pool manifest SHA mismatch")
        if boundary.model_revision != manifest.model_revision:
            raise SchedulerTraceValidationError("fixed-pool model revision mismatch")
        if boundary.model_weights_sha256 != manifest.model_weights_sha256:
            raise SchedulerTraceValidationError("fixed-pool model weights SHA mismatch")
    if run_end.terminal_reason != "fixed_pool_complete":
        raise SchedulerTraceValidationError("fixed-pool run did not complete cleanly")
    if run_end.live_logical_group_ids:
        raise SchedulerTraceValidationError("fixed-pool run ended with live groups")
    if run_end.scalar_summaries.get("completed_train_steps") != 0:
        raise SchedulerTraceValidationError("fixed-pool collection performed training")
    if run_start.scalar_summaries.get("planned_prompt_groups") != len(manifest.items):
        raise SchedulerTraceValidationError("fixed-pool planned count mismatch")
    if run_end.scalar_summaries.get("collected_prompt_groups") != len(manifest.items):
        raise SchedulerTraceValidationError("fixed-pool collected count mismatch")

    forbidden = {
        SchedulerEventType.SELECT_DECISION,
        SchedulerEventType.GROUP_EVICTED,
        SchedulerEventType.ATTEMPT_FAILED,
        SchedulerEventType.ATTEMPT_REMOVED,
        SchedulerEventType.ABORT_REQUESTED,
        SchedulerEventType.PROMPT_SKIPPED,
        SchedulerEventType.GROUP_REPLACED,
        SchedulerEventType.GROUP_PROMOTED,
    }
    bad_events = [
        event.event_type.value for event in events if event.event_type in forbidden
    ]
    if bad_events:
        raise SchedulerTraceValidationError(
            f"fixed-pool trace contains forbidden lifecycle events: {bad_events}"
        )

    manifest_by_ordinal = {item.ordinal: item for item in manifest.items}
    expected_cohorts: dict[int, list[int]] = {}
    for item in manifest.items:
        expected_cohorts.setdefault(item.dispatch_cohort, []).append(item.ordinal)
    admission_cohorts: dict[str, int] = {}
    lifecycle: dict[int, list[SchedulerEventType]] = {}
    dispatch_order: list[int] = []
    physical_versions: set[int] = set()
    trainer_versions = {
        event.trainer_version for event in events if event.trainer_version is not None
    }
    if len(trainer_versions) != 1:
        raise SchedulerTraceValidationError(
            "fixed-pool trainer version changed during collection"
        )

    for event in events:
        if event.event_type is SchedulerEventType.ADMISSION_GRANTED:
            assert event.admission_id is not None
            cohort = event.sampler_dispatch_index
            if cohort is None or cohort not in expected_cohorts:
                raise SchedulerTraceValidationError(
                    "fixed-pool admission has an unknown dispatch cohort"
                )
            if event.admission_id in admission_cohorts:
                raise SchedulerTraceValidationError(
                    "fixed-pool dispatch cohort was admitted more than once"
                )
            if event.scalar_summaries.get("expected_prompt_groups") != len(
                expected_cohorts[cohort]
            ):
                raise SchedulerTraceValidationError(
                    "fixed-pool admission cohort size mismatch"
                )
            admission_cohorts[event.admission_id] = cohort
        if event.event_type not in {
            SchedulerEventType.ATTEMPT_DISPATCHED,
            SchedulerEventType.ROLLOUT_COMPLETED,
            SchedulerEventType.GROUP_READY,
            SchedulerEventType.GROUP_ARCHIVED,
        }:
            continue
        ordinal = event.source_pool_ordinal
        if ordinal is None or ordinal not in manifest_by_ordinal:
            raise SchedulerTraceValidationError(
                "fixed-pool lifecycle event is not in the manifest"
            )
        item = manifest_by_ordinal[ordinal]
        if (
            event.prompt_idx != item.dataset_index
            or event.task_name != item.task_name
            or event.source_prompt_id != item.source_prompt_id
            or event.repeated_prompt_cluster_id != item.repeated_prompt_cluster_id
            or event.dispatch_cohort != item.dispatch_cohort
        ):
            raise SchedulerTraceValidationError(
                f"fixed-pool manifest identity mismatch at ordinal {ordinal}"
            )
        if admission_cohorts.get(event.admission_id) != item.dispatch_cohort:
            raise SchedulerTraceValidationError(
                f"fixed-pool admission mismatch at ordinal {ordinal}"
            )
        lifecycle.setdefault(ordinal, []).append(event.event_type)
        if event.event_type is SchedulerEventType.ATTEMPT_DISPATCHED:
            dispatch_order.append(ordinal)
            if event.start_weight_version is None:
                raise SchedulerTraceValidationError(
                    "fixed-pool dispatch lacks physical weight version"
                )
            physical_versions.add(event.start_weight_version)
        elif event.event_type is SchedulerEventType.ROLLOUT_COMPLETED:
            if event.scalar_summaries.get("completion_count") != (
                expected_completions_per_group
            ):
                raise SchedulerTraceValidationError(
                    f"completion count mismatch at ordinal {ordinal}"
                )
        elif event.event_type is SchedulerEventType.GROUP_READY:
            if (
                event.start_weight_version is None
                or event.end_weight_version is None
                or event.start_weight_version != event.end_weight_version
            ):
                raise SchedulerTraceValidationError(
                    f"physical policy changed during ordinal {ordinal}"
                )
            physical_versions.add(event.start_weight_version)

    if set(admission_cohorts.values()) != set(expected_cohorts):
        raise SchedulerTraceValidationError("fixed-pool cohort coverage mismatch")
    if dispatch_order != list(range(len(manifest.items))):
        raise SchedulerTraceValidationError("fixed-pool dispatch order mismatch")
    expected_lifecycle = [
        SchedulerEventType.ATTEMPT_DISPATCHED,
        SchedulerEventType.ROLLOUT_COMPLETED,
        SchedulerEventType.GROUP_READY,
        SchedulerEventType.GROUP_ARCHIVED,
    ]
    if any(
        lifecycle.get(item.ordinal) != expected_lifecycle for item in manifest.items
    ):
        raise SchedulerTraceValidationError("fixed-pool lifecycle coverage mismatch")
    if len(physical_versions) != 1:
        raise SchedulerTraceValidationError(
            "fixed-pool source collection used multiple physical policy versions"
        )
    physical_weight_version = next(iter(physical_versions))
    return FixedPoolTraceValidationReport(
        planned_prompt_groups=len(manifest.items),
        dispatched_prompt_groups=len(dispatch_order),
        completed_prompt_groups=len(manifest.items),
        ready_prompt_groups=len(manifest.items),
        archived_prompt_groups=len(manifest.items),
        physical_weight_version=physical_weight_version,
    )
