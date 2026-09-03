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

"""Materialize a deterministic exact-distance sliding-puzzle feasibility pool."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import random
import stat
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from nemo_rl.algorithms.async_utils.fixed_pool import (
    compute_fixed_pool_id,
    load_fixed_pool_manifest,
    validate_fixed_pool_manifest_design,
    validate_fixed_pool_materialization,
)
from nemo_rl.environments.games.sliding_puzzle import SlidingPuzzleGameLogic


MODEL_REPO: Final[str] = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION: Final[str] = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
MODEL_WEIGHTS_SHA256: Final[str] = (
    "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
)
DATASET_ID: Final[str] = "generated/sliding-puzzle-3x3"
DATASET_REVISION: Final[str] = "bfs_exact_distance_v1"
DATASET_SPLIT: Final[str] = "feasibility"
SELECTION_SEED: Final[int] = 20260903
ORDER_SEED: Final[int] = 44001
GENERATION_STUDY_SEED: Final[int] = 53001
BOOTSTRAP_SEED: Final[int] = 20260905
BOOTSTRAP_REPLICATES: Final[int] = 10_000
SOURCE_IDS: Final[tuple[str, str]] = (
    "sliding_puzzle_easy",
    "sliding_puzzle_hard",
)
EASY_DISTANCES: Final[tuple[int, ...]] = (1, 1, 2, 2, 2, 2, 3, 3)
HARD_DISTANCES: Final[tuple[int, ...]] = (10, 10, 10, 11, 11, 11, 12, 12)
PAIRS: Final[int] = 8
COHORT_SIZE: Final[int] = 4
INPUT_TOKEN_CALIPER: Final[int] = 8
MAX_INPUT_TOKENS: Final[int] = 512
MAX_MOVES: Final[int] = 12
MAX_ROLLOUT_TURNS: Final[int] = 12
MAX_TOTAL_SEQUENCE_LENGTH: Final[int] = 2048
MAX_NEW_TOKENS_PER_TURN: Final[int] = 128
NUM_GENERATIONS_PER_PROMPT: Final[int] = 2
TEMPERATURE: Final[float] = 1.0
TOP_P: Final[float] = 0.999
TOP_K: Final[int] = 10_000
PROMPT_BUILDER_VERSION: Final[str] = "legacy_sliding_puzzle_prompt_v1"
GOAL_STATE: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6, 7, 8, 0)
ACTION_STOP_STRING: Final[str] = "</action>"


@dataclass(frozen=True, slots=True)
class PuzzleCandidate:
    """One deterministic board and its exact rendered prompt."""

    state: tuple[int, ...]
    optimal_distance: int
    source_dataset_index: int
    rendered_prompt: str
    input_token_ids: tuple[int, ...]


def _canonical(record: object) -> bytes:
    return json.dumps(
        record, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _opaque(key: bytes, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.chmod(0o600)
    temporary.replace(path)


def _neighbors(state: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    empty = state.index(0)
    row, column = divmod(empty, 3)
    values: list[tuple[int, ...]] = []
    for delta_row, delta_column in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        next_row = row + delta_row
        next_column = column + delta_column
        if 0 <= next_row < 3 and 0 <= next_column < 3:
            other = next_row * 3 + next_column
            mutable = list(state)
            mutable[empty], mutable[other] = mutable[other], mutable[empty]
            values.append(tuple(mutable))
    return tuple(values)


def _states_by_distance(max_distance: int) -> dict[int, tuple[tuple[int, ...], ...]]:
    """Enumerate every state through ``max_distance`` by exact BFS distance."""
    distance_by_state = {GOAL_STATE: 0}
    queue = deque([GOAL_STATE])
    by_distance: dict[int, list[tuple[int, ...]]] = {0: [GOAL_STATE]}
    while queue:
        state = queue.popleft()
        distance = distance_by_state[state]
        if distance == max_distance:
            continue
        for neighbor in _neighbors(state):
            if neighbor in distance_by_state:
                continue
            next_distance = distance + 1
            distance_by_state[neighbor] = next_distance
            by_distance.setdefault(next_distance, []).append(neighbor)
            queue.append(neighbor)
    return {distance: tuple(sorted(states)) for distance, states in by_distance.items()}


def _permutation_rank(state: tuple[int, ...]) -> int:
    """Return the stable Lehmer rank of a board permutation."""
    rank = 0
    remaining = list(range(9))
    factorial = 1
    factorials = [1]
    for value in range(1, 10):
        factorial *= value
        factorials.append(factorial)
    for index, value in enumerate(state):
        position = remaining.index(value)
        rank += position * factorials[8 - index]
        remaining.pop(position)
    return rank


def _game_state(state: tuple[int, ...]) -> dict[str, object]:
    grid = [list(state[row * 3 : (row + 1) * 3]) for row in range(3)]
    solution = [list(GOAL_STATE[row * 3 : (row + 1) * 3]) for row in range(3)]
    empty = state.index(0)
    return {
        "size": 3,
        "grid": grid,
        "solution": solution,
        "empty_pos": list(divmod(empty, 3)),
        "commands": {
            "up": "Slide tile below empty space up",
            "down": "Slide tile above empty space down",
            "left": "Slide tile to the right of empty space left",
            "right": "Slide tile to the left of empty space right",
            "view": "View the current state of the board",
        },
    }


def _prompt_for_state(state: tuple[int, ...]) -> str:
    game_state = _game_state(state)
    initial_render = SlidingPuzzleGameLogic.render(game_state)
    welcome_message = SlidingPuzzleGameLogic.init(game_state)
    return (
        f"{welcome_message}\n\n"
        f"Current Board State:\n{initial_render}\n\n"
        "Reach the goal state where numbers are ordered 1 through 8 "
        "with the empty space (0) at the bottom right.\n"
        "Valid actions: 'up', 'down', 'left', 'right', or 'slide row col' "
        "(e.g., 'slide 1 2').\n"
        "After thinking, output your chosen action on a new line starting with "
        "'<action></action>' like this:\n<action>your_action</action>"
        "\nIf you just want to see the board, output <action>view</action>"
        "\nThink carefully step-by-step before acting.\n"
    )


def _render_candidate(
    tokenizer: Any, state: tuple[int, ...], distance: int
) -> PuzzleCandidate:
    prompt = _prompt_for_state(state)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    ).strip()
    input_ids = tokenizer(
        rendered,
        return_tensors=None,
        add_special_tokens=False,
    )["input_ids"]
    token_ids = tuple(int(value) for value in input_ids)
    if len(token_ids) > MAX_INPUT_TOKENS:
        raise RuntimeError("sliding-puzzle prompt exceeds the frozen input bound")
    return PuzzleCandidate(
        state=state,
        optimal_distance=distance,
        source_dataset_index=_permutation_rank(state),
        rendered_prompt=rendered,
        input_token_ids=token_ids,
    )


def _rank_key(state: tuple[int, ...], label: str) -> str:
    return _sha(_canonical([SELECTION_SEED, label, state]))


def _select_candidates(
    tokenizer: Any,
) -> tuple[list[tuple[PuzzleCandidate, PuzzleCandidate]], dict[str, object]]:
    states = _states_by_distance(max(HARD_DISTANCES))
    easy_by_distance = {
        distance: sorted(values, key=lambda state: _rank_key(state, "easy"))
        for distance, values in states.items()
        if distance in set(EASY_DISTANCES)
    }
    hard_by_distance = {
        distance: sorted(values, key=lambda state: _rank_key(state, "hard"))
        for distance, values in states.items()
        if distance in set(HARD_DISTANCES)
    }
    easy: list[PuzzleCandidate] = []
    easy_offsets: dict[int, int] = {}
    for distance in EASY_DISTANCES:
        offset = easy_offsets.get(distance, 0)
        candidates = easy_by_distance[distance]
        if offset >= len(candidates):
            raise RuntimeError(f"insufficient exact-distance-{distance} easy boards")
        easy.append(_render_candidate(tokenizer, candidates[offset], distance))
        easy_offsets[distance] = offset + 1

    hard: list[PuzzleCandidate] = []
    used_hard: set[tuple[int, ...]] = set()
    for pair_index, (easy_candidate, distance) in enumerate(zip(easy, HARD_DISTANCES)):
        candidates = [
            _render_candidate(tokenizer, state, distance)
            for state in hard_by_distance[distance]
            if state not in used_hard
        ]
        compatible = [
            candidate
            for candidate in candidates
            if abs(len(candidate.input_token_ids) - len(easy_candidate.input_token_ids))
            <= INPUT_TOKEN_CALIPER
        ]
        if not compatible:
            raise RuntimeError(
                f"no prompt-length-compatible hard board for pair {pair_index}"
            )
        selected = min(
            compatible,
            key=lambda candidate: (
                abs(
                    len(candidate.input_token_ids) - len(easy_candidate.input_token_ids)
                ),
                _rank_key(candidate.state, f"hard-pair-{pair_index}"),
            ),
        )
        used_hard.add(selected.state)
        hard.append(selected)
    pairs = list(zip(easy, hard))
    telemetry = {
        "bfs_state_counts_by_distance": {
            str(distance): len(values) for distance, values in states.items()
        },
        "easy_distance_schedule": list(EASY_DISTANCES),
        "hard_distance_schedule": list(HARD_DISTANCES),
        "observed_max_input_pair_delta_tokens": max(
            abs(len(short.input_token_ids) - len(long.input_token_ids))
            for short, long in pairs
        ),
    }
    return pairs, telemetry


def _source_record(
    *, key: bytes, source_id: str, candidate: PuzzleCandidate, pair_index: int
) -> dict[str, object]:
    board_sha = _sha(_canonical(candidate.state))
    prompt_id = _opaque(
        key,
        "sliding-puzzle-prompt-v1",
        DATASET_REVISION,
        candidate.source_dataset_index,
        board_sha,
    )
    return {
        "messages": [{"role": "user", "content": candidate.rendered_prompt}],
        "extra_env_info": {
            "game_state": _game_state(candidate.state),
            "num_moves": 0,
            "max_moves": MAX_MOVES,
            "optimal_distance": candidate.optimal_distance,
        },
        "stop_strings": [ACTION_STOP_STRING],
        "source_id": source_id,
        "source_dataset_id": DATASET_ID,
        "source_revision": DATASET_REVISION,
        "source_split": DATASET_SPLIT,
        "source_dataset_index": candidate.source_dataset_index,
        "source_prompt_id": prompt_id,
        "repeated_prompt_cluster_id": _opaque(key, "board-cluster-v1", board_sha),
        "selection_stratum": source_id,
        "matching_pair_id": f"pair-{pair_index}",
        "board_state_sha256": board_sha,
        "optimal_distance": candidate.optimal_distance,
        "input_token_count": len(candidate.input_token_ids),
        "input_token_ids_sha256": _sha(_canonical(candidate.input_token_ids)),
        "selection_seed": SELECTION_SEED,
        "model_revision": MODEL_REVISION,
        "prompt_builder_version": PROMPT_BUILDER_VERSION,
    }


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> bytes:
    raw = b"".join(_canonical(record) + b"\n" for record in records)
    _atomic_write(path, raw)
    return raw


def _snapshot_model(output_dir: Path) -> tuple[Any, bytes]:
    model_dir = output_dir / "model_snapshot"
    snapshot_download(repo_id=MODEL_REPO, revision=MODEL_REVISION, local_dir=model_dir)
    for path in model_dir.rglob("*"):
        if path.is_symlink():
            raise RuntimeError("pinned model snapshot contains a symbolic link")
    for directory in [
        model_dir,
        *(path for path in model_dir.rglob("*") if path.is_dir()),
    ]:
        directory.chmod(0o700)
    for path in model_dir.rglob("*"):
        if path.is_file():
            path.chmod(0o600)
    weights_path = model_dir / "model.safetensors"
    if _sha_file(weights_path) != MODEL_WEIGHTS_SHA256:
        raise RuntimeError("pinned Qwen Instruct model weights SHA-256 mismatch")
    records = [
        {
            "path": str(path.relative_to(model_dir)),
            "bytes": path.stat().st_size,
            "sha256": _sha_file(path),
        }
        for path in sorted(model_dir.rglob("*"))
        if path.is_file() and ".cache" not in path.parts
    ]
    manifest = json.dumps(records, indent=2, sort_keys=True).encode() + b"\n"
    _atomic_write(output_dir / "model_snapshot_manifest.v1.json", manifest)
    return AutoTokenizer.from_pretrained(model_dir), manifest


def _build_manifest(
    *,
    key: bytes,
    design_sha: str,
    records: Mapping[str, Sequence[Mapping[str, object]]],
    source_raw: Mapping[str, bytes],
    snapshot_manifest_sha: str,
) -> dict[str, object]:
    records_by_pair = {
        (source_id, str(record["matching_pair_id"])): record
        for source_id, source_records in records.items()
        for record in source_records
    }
    row_by_prompt = {
        str(record["source_prompt_id"]): row
        for source_records in records.values()
        for row, record in enumerate(source_records)
    }
    offsets = {SOURCE_IDS[0]: 0, SOURCE_IDS[1]: len(records[SOURCE_IDS[0]])}
    rng = random.Random(ORDER_SEED)
    pair_order = list(range(PAIRS))
    rng.shuffle(pair_order)
    ordered: list[tuple[str, Mapping[str, object], int]] = []
    for pair_start in range(0, PAIRS, 2):
        cohort: list[tuple[str, Mapping[str, object], int]] = []
        for pair_index in pair_order[pair_start : pair_start + 2]:
            pair_id = f"pair-{pair_index}"
            cohort.extend(
                (
                    (
                        SOURCE_IDS[0],
                        records_by_pair[(SOURCE_IDS[0], pair_id)],
                        pair_index,
                    ),
                    (
                        SOURCE_IDS[1],
                        records_by_pair[(SOURCE_IDS[1], pair_id)],
                        pair_index,
                    ),
                )
            )
        rng.shuffle(cohort)
        ordered.extend(cohort)

    items: list[dict[str, object]] = []
    for ordinal, (source_id, record, pair_index) in enumerate(ordered):
        row = row_by_prompt[str(record["source_prompt_id"])]
        cohort = ordinal // COHORT_SIZE
        items.append(
            {
                "ordinal": ordinal,
                "pool_item_id": _opaque(
                    key, "pool-item-v1", ORDER_SEED, ordinal, record["source_prompt_id"]
                ),
                "source_prompt_id": record["source_prompt_id"],
                "source_id": source_id,
                "source_dataset_index": record["source_dataset_index"],
                "dataset_index": offsets[source_id] + row,
                "task_name": source_id,
                "repeated_prompt_cluster_id": record["repeated_prompt_cluster_id"],
                "dispatch_cohort": cohort,
                "decorrelation_block": f"cohort-{cohort}",
                "matching_pair_id": f"pair-{pair_index}",
                "materialized_source_row": row,
                "materialized_record_sha256": _sha(_canonical(record)),
                "input_token_count": record["input_token_count"],
                "input_token_ids_sha256": record["input_token_ids_sha256"],
            }
        )
    sources = [
        {
            "source_id": source_id,
            "dataset_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
            "materialized_file": f"{source_id}.jsonl",
            "content_sha256": _sha(source_raw[source_id]),
        }
        for source_id in SOURCE_IDS
    ]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "id_namespace_fingerprint": _sha(key),
        "design_protocol_sha256": design_sha,
        "order_seed": ORDER_SEED,
        "model_revision": MODEL_REVISION,
        "model_weights_sha256": MODEL_WEIGHTS_SHA256,
        "model_snapshot_manifest_sha256": snapshot_manifest_sha,
        "model_snapshot_path": "model_snapshot",
        "sources": sources,
        "items": items,
    }
    manifest["pool_id"] = compute_fixed_pool_id(manifest)
    return manifest


def _validate_private_modes(output_dir: Path) -> None:
    if stat.S_IMODE(output_dir.stat().st_mode) != 0o700:
        raise RuntimeError("materialization directory mode is not 0700")
    for path in output_dir.rglob("*"):
        expected = 0o700 if path.is_dir() else 0o600
        if stat.S_IMODE(path.stat().st_mode) != expected:
            raise RuntimeError(
                f"private artifact mode mismatch for {path.relative_to(output_dir)}"
            )


def materialize(output_dir: Path, key: bytes) -> None:
    """Write the complete private puzzle pool and provenance bundle."""
    if len(key) < 32:
        raise RuntimeError("prompt-ID HMAC key is shorter than 32 bytes")
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    output_dir.chmod(0o700)
    design_sha = _sha(Path(__file__).read_bytes())
    tokenizer, snapshot_manifest = _snapshot_model(output_dir)
    selected_pairs, selection_telemetry = _select_candidates(tokenizer)

    records: dict[str, list[dict[str, object]]] = {
        SOURCE_IDS[0]: [],
        SOURCE_IDS[1]: [],
    }
    for pair_index, (easy, hard) in enumerate(selected_pairs):
        records[SOURCE_IDS[0]].append(
            _source_record(
                key=key,
                source_id=SOURCE_IDS[0],
                candidate=easy,
                pair_index=pair_index,
            )
        )
        records[SOURCE_IDS[1]].append(
            _source_record(
                key=key,
                source_id=SOURCE_IDS[1],
                candidate=hard,
                pair_index=pair_index,
            )
        )
    source_raw = {
        source_id: _write_jsonl(output_dir / f"{source_id}.jsonl", source_records)
        for source_id, source_records in records.items()
    }
    manifest = _build_manifest(
        key=key,
        design_sha=design_sha,
        records=records,
        source_raw=source_raw,
        snapshot_manifest_sha=_sha(snapshot_manifest),
    )
    manifest_name = f"fixed_pool_manifest.v1.{ORDER_SEED}.json"
    manifest_raw = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    _atomic_write(output_dir / manifest_name, manifest_raw)

    records_by_prompt = {
        str(record["source_prompt_id"]): record
        for source_records in records.values()
        for record in source_records
    }
    design_items = []
    for item in manifest["items"]:  # type: ignore[union-attr]
        record = records_by_prompt[str(item["source_prompt_id"])]
        design_items.append(
            {
                "source_pool_ordinal": item["ordinal"],
                "source_prompt_id": item["source_prompt_id"],
                "task_name": item["task_name"],
                "pair_id": item["matching_pair_id"],
                "stratum": ("easy" if item["task_name"] == SOURCE_IDS[0] else "hard"),
                "optimal_distance": record["optimal_distance"],
                "rendered_prompt_tokens": record["input_token_count"],
                "board_state_sha256": record["board_state_sha256"],
            }
        )
    selection_design = {
        "schema_version": 1,
        "analysis_status": "preregistered_pre_generation",
        "calibration_only": True,
        "confirmatory_eligible": False,
        "fixed_pool_id": manifest["pool_id"],
        "fixed_pool_manifest_sha256": _sha(manifest_raw),
        "selection_seed": SELECTION_SEED,
        "order_seed": ORDER_SEED,
        "generation_study_seed": GENERATION_STUDY_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "max_moves": MAX_MOVES,
        "max_rollout_turns": MAX_ROLLOUT_TURNS,
        "max_total_sequence_length": MAX_TOTAL_SEQUENCE_LENGTH,
        "max_new_tokens_per_turn": MAX_NEW_TOKENS_PER_TURN,
        "num_generations_per_prompt": NUM_GENERATIONS_PER_PROMPT,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "locked_gates": {
            "action_format_valid_rate_min": 0.9,
            "truncation_rate_max_exclusive": 0.1,
            "easy_max_turn_rate_max": 0.25,
            "turn_median_ratio_min": 2.0,
            "generated_token_median_ratio_min": 1.5,
            "pair_turn_sign_count_min": 6,
            "ready_latency_median_ratio_min": 1.5,
            "ready_latency_rank_biserial_min": 0.3,
            "pair_ready_sign_count_min": 6,
        },
        "items": design_items,
    }
    selection_raw = (
        json.dumps(selection_design, indent=2, sort_keys=True).encode() + b"\n"
    )
    _atomic_write(output_dir / "selection_design.v1.json", selection_raw)

    exclusion = {
        "schema_version": 1,
        "excluded_board_state_sha256": sorted(
            str(record["board_state_sha256"])
            for source_records in records.values()
            for record in source_records
        ),
    }
    exclusion_raw = json.dumps(exclusion, indent=2, sort_keys=True).encode() + b"\n"
    _atomic_write(output_dir / "private_exclusion_ledger.v1.json", exclusion_raw)
    report = {
        "schema_version": 1,
        "design_protocol_sha256": design_sha,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_split": DATASET_SPLIT,
        "selection_seed": SELECTION_SEED,
        "order_seed": ORDER_SEED,
        "pairs": PAIRS,
        "cohort_size": COHORT_SIZE,
        "source_counts": {
            source_id: len(records[source_id]) for source_id in SOURCE_IDS
        },
        "input_token_caliper": INPUT_TOKEN_CALIPER,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_moves": MAX_MOVES,
        "selection_telemetry": selection_telemetry,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_weights_sha256": MODEL_WEIGHTS_SHA256,
        "manifest": manifest_name,
        "manifest_sha256": _sha(manifest_raw),
        "selection_design": "selection_design.v1.json",
        "selection_design_sha256": _sha(selection_raw),
        "private_exclusion_ledger_sha256": _sha(exclusion_raw),
    }
    _atomic_write(
        output_dir / "materialization_report.v1.json",
        json.dumps(report, indent=2, sort_keys=True).encode() + b"\n",
    )
    hash_lines = [
        f"{_sha_file(path)}  {path.name}\n"
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    _atomic_write(output_dir / "SHA256SUMS", "".join(hash_lines).encode())
    _validate_private_modes(output_dir)
    loaded_manifest = load_fixed_pool_manifest(output_dir / manifest_name)
    validate_fixed_pool_materialization(loaded_manifest)
    validate_fixed_pool_manifest_design(
        loaded_manifest, "sliding_puzzle_latency_feasibility_v1"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--hmac-key-env",
        default="SLIDING_PUZZLE_PROMPT_HMAC_KEY",
        help="Environment variable containing the non-persisted ID key",
    )
    args = parser.parse_args()
    key_value = os.environ.get(args.hmac_key_env)
    if key_value is None:
        raise RuntimeError("prompt-ID HMAC key is missing")
    materialize(args.output_dir, key_value.encode())


if __name__ == "__main__":
    main()
