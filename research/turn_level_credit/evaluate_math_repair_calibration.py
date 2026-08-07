#!/usr/bin/env python3
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

"""Evaluate one frozen-policy math-repair validation artifact."""

import argparse
import json
from pathlib import Path
from typing import Any

from turn_level_credit.math_repair_calibration import (
    evaluate_math_repair_calibration,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise TypeError(f"line {line_number} is not a JSON object")
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("validation_jsonl", type=Path)
    parser.add_argument("--generations-per-prompt", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_math_repair_calibration(
        _read_jsonl(args.validation_jsonl),
        generations_per_prompt=args.generations_per_prompt,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    if result["decision"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
