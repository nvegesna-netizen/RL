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

"""Evaluate the predeclared frozen-policy dense-signal gates."""

import argparse
import json
from pathlib import Path

from turn_level_credit.dense_calibration import evaluate_dense_calibration


def parse_args() -> argparse.Namespace:
    """Parse paired calibration logs and rollout artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-log", type=Path, required=True)
    parser.add_argument("--treatment-log", type=Path, required=True)
    parser.add_argument("--control-rollouts", type=Path, required=True)
    parser.add_argument("--treatment-rollouts", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Print a structured gate result and fail when any check is unmet."""
    args = parse_args()
    result = evaluate_dense_calibration(
        control_log_text=args.control_log.read_text(),
        treatment_log_text=args.treatment_log.read_text(),
        control_rollouts=args.control_rollouts.read_bytes(),
        treatment_rollouts=args.treatment_rollouts.read_bytes(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
