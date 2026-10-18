"""Produce compact authenticated classifications, without reading any outcomes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from research.m4_paired_replay.capture_authentication import (
    CaptureClient,
    collect_cohort,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = CaptureClient(os.environ["JET_GITLAB_TOKEN"])
    rows = collect_cohort(
        client,
        plan=json.loads(args.plan.read_bytes()),
        bindings=json.loads(args.bindings.read_bytes()),
        output=args.output,
    )
    print(
        json.dumps(
            {
                "status": "ALL_CAPTURES_AUTHENTICATED",
                "classifications": {
                    name: sum(r["classification"] == name for r in rows)
                    for name in ("CAPSULE", "NO_ELIGIBLE", "FAILED")
                },
                "outcomes_opened": False,
            }
        )
    )


if __name__ == "__main__":
    main()
