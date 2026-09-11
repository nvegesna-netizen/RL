#!/usr/bin/env python3
"""Extract the authenticated V8 prompt-correctness covariate from its ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


ARCHIVE_SHA256 = "26cfc6ea86c4959d7123b51d7ea1355f54430298c3bc3847fabb03473701f9c0"
EVALUATION_SHA256 = "5a552cbeee8e14bb57ed9ab6d55f30caecb68c08a7271f314bb237b5a27f2a05"
PROMPT_MANIFEST_SHA256 = (
    "469a34a176febe9d15c5c6d967ff21f135a43cb20a7cb7d5ae1a53066e6b3a0a"
)
EVALUATION_MEMBER = (
    "workspace/assets/basic/m4-downstream-quality-no-training-preflight-v8/"
    "base-evaluation/evaluation_data.json"
)
PROMPT_MEMBER = (
    "workspace/assets/basic/m4-downstream-quality-no-training-preflight-v8/"
    "m4-downstream-quality-prompt-manifest.json"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract(*, archive: Path, output: Path) -> dict[str, object]:
    archive_data = archive.read_bytes()
    if sha256(archive_data) != ARCHIVE_SHA256:
        raise RuntimeError("V8 workload archive moved")
    with zipfile.ZipFile(archive) as source:
        evaluation_data = source.read(EVALUATION_MEMBER)
        prompt_data = source.read(PROMPT_MEMBER)
    if sha256(evaluation_data) != EVALUATION_SHA256:
        raise RuntimeError("V8 evaluation artifact moved")
    if sha256(prompt_data) != PROMPT_MANIFEST_SHA256:
        raise RuntimeError("V8 prompt manifest moved")
    evaluation = json.loads(evaluation_data)
    prompt_manifest = json.loads(prompt_data)
    rows = evaluation["evaluation_data"]
    row_hashes = prompt_manifest["row_sha256"]
    if (
        len(rows) != 1024
        or len(row_hashes) != 1024
        or len(set(row_hashes)) != 1024
        or sorted(row["sample_index"] for row in rows) != list(range(1024))
    ):
        raise RuntimeError("V8 prompt topology differs")
    by_index = {row["sample_index"]: row for row in rows}
    scores = []
    for index, prompt_hash in enumerate(row_hashes):
        reward = by_index[index]["reward"]
        if isinstance(reward, bool) or reward not in (0, 1, 0.0, 1.0):
            raise RuntimeError("V8 reward is nonbinary")
        scores.append({"prompt_row_sha256": prompt_hash, "reward": int(reward)})
    if sum(row["reward"] for row in scores) != 369:
        raise RuntimeError("V8 success count differs")
    result: dict[str, object] = {
        "schema": "m4-downstream-quality-v8-base-prompt-covariate-v1",
        "source_workload_archive_sha256": ARCHIVE_SHA256,
        "source_evaluation_data_sha256": EVALUATION_SHA256,
        "prompt_manifest_sha256": PROMPT_MANIFEST_SHA256,
        "prompt_count": 1024,
        "successes": 369,
        "accuracy": 0.3603515625,
        "role": "pretreatment_sensitivity_only_not_primary_estimator_input",
        "scores": scores,
    }
    if output.exists():
        raise FileExistsError(f"base covariate output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = extract(archive=args.archive, output=args.output)
    print("M4_DOWNSTREAM_QUALITY_V8_BASE_COVARIATE_PASS")
    print(f"successes={result['successes']}")
    print(f"output_sha256={sha256(args.output.read_bytes())}")


if __name__ == "__main__":
    main()
