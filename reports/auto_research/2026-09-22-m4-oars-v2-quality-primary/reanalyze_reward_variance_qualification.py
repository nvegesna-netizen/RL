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

"""Re-evaluate the authenticated reward-variance systems ledger offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

from tools.m4_oars_v2_actuation_qualification import assess_v2_actuation

PREFIX = "workspace/assets/basic/m4-oars-v2-reward-variance-actuation-qualification/"
STEM = "m4-oars-v2-reward-variance-actuation-qualification"


def sha256(value: bytes) -> str:
    """Return a SHA-256 hexadecimal digest."""
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    """Authenticate compact inputs and write the corrected offline gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--authentication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    authentication = json.loads(args.authentication.read_bytes())
    if authentication["status"] != "PASS_AUTHENTICATED_RESULT_UNOPENED":
        raise RuntimeError("terminal authentication did not pass")
    if sha256(args.archive.read_bytes()) != authentication["archives"]["main_sha256"]:
        raise RuntimeError("authenticated main archive moved")
    with zipfile.ZipFile(args.archive) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("main archive failed ZIP integrity")
        files = {
            PurePosixPath(name).name: archive.read(name)
            for name in archive.namelist()
            if name.startswith(PREFIX)
            and not name.endswith("/")
            and PurePosixPath(name).parent.as_posix() == PREFIX.rstrip("/")
        }
    for name, record in authentication["artifacts"].items():
        if name not in files or sha256(files[name]) != record["sha256"]:
            raise RuntimeError(f"authenticated compact artifact moved: {name}")
    oars_name = f"{STEM}-oars-v2.jsonl"
    lifecycle_name = f"{STEM}-lifecycle.jsonl"
    duty_name = f"{STEM}-observer-duty.json"
    original_name = f"{STEM}-result.json"
    oars_rows = [json.loads(line) for line in files[oars_name].splitlines()]
    lifecycle_rows = [json.loads(line) for line in files[lifecycle_name].splitlines()]
    observer_duty = json.loads(files[duty_name])
    original = json.loads(files[original_name])
    runtime_ns = round(float(original["runtime_seconds"]) * 1_000_000_000)
    result = assess_v2_actuation(
        scorer="reward_variance_risk",
        oars_rows=oars_rows,
        lifecycle_rows=lifecycle_rows,
        observer_duty=observer_duty,
        source_commit=original["source_commit"],
        run_start_ns=0,
        run_end_ns=runtime_ns,
    )
    result.update(
        {
            "analysis_revision": "bounded_credit_accounting_v2",
            "original_result_sha256": sha256(files[original_name]),
            "original_status": original["status"],
            "repair_scope": (
                "offline gate correction only; runtime code, assignments, "
                "selections, and artifacts are unchanged"
            ),
        }
    )
    if result["status"] != "PASS_OARS_V2_ACTUATION_QUALIFIED":
        raise RuntimeError(f"corrected offline gate failed: {result['checks']}")
    args.output.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
