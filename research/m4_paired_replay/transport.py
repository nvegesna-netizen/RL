"""Stream selected members from a gate-admitted EOS job; never a collector archive."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, build_opener

from research.m4_eligible_replay.contracts import (
    canonical_json,
    digest_file,
    safe_member,
    stream_member,
)
from research.m4_eligible_replay.transport import NoRedirect
from research.m4_paired_replay.contracts import admit, members, capture_gate


def fetch_arm(
    *,
    plan: dict,
    captures: list[dict],
    identity: str,
    arm: str,
    artifact_prefix: str,
    output: Path,
    token: str,
    job_token: bool,
    opener: Any = None,
) -> dict:
    """Fetch bounded individual members after all twenty captures are classified."""
    gate = capture_gate(plan, captures)
    sources = [r for r in gate["admitted"] if r["identity"] == identity]
    if len(sources) != 1:
        raise ValueError("identity not admitted")
    source = sources[0]
    # Canonical deterministic workload identity prevents collector substitution.
    expected = f"workspace/assets/basic/m4-eligible-live-acquisition-{identity}/m4-eligible-live-capsule"
    if str(safe_member(artifact_prefix)) != expected:
        raise ValueError("noncanonical direct-EOS artifact prefix")
    base = f"https://gitlab-master.nvidia.com/api/v4/projects/70847/jobs/{source['eos_job_id']}/artifacts/{quote(expected, safe='/')}/"
    opener = build_opener(NoRedirect()) if opener is None else opener
    headers = {"JOB-TOKEN" if job_token else "PRIVATE-TOKEN": token}
    with opener.open(
        Request(base + "manifest.json", headers=headers), timeout=60
    ) as response:
        manifest_bytes = response.read((1 << 20) + 1)
    if len(manifest_bytes) > 1 << 20:
        raise ValueError("oversized capsule manifest")
    _, manifest = admit(plan, captures, identity, manifest_bytes)
    allowed = members(manifest, arm)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "manifest.json").open("xb") as stream:
        stream.write(manifest_bytes)
    records = []
    for row in allowed:
        target = output / row["path"]
        with opener.open(
            Request(base + quote(row["path"], safe="/"), headers=headers), timeout=300
        ) as response:
            stream_member(
                response,
                target,
                expected_bytes=row["bytes"],
                expected_hash=row["sha256"],
            )
        records.append(
            {
                "path": row["path"],
                "bytes": target.stat().st_size,
                "sha256": digest_file(target),
            }
        )
    receipt = {
        "schema": "m4-paired-transport-v1",
        "identity": identity,
        "arm": arm,
        "source_eos_job_id": source["eos_job_id"],
        "manifest_sha256": source["manifest_sha256"],
        "capture_gate_sha256": hashlib.sha256(canonical_json(gate)).hexdigest(),
        "members": records,
        "large_archive_downloaded": False,
        "logs_after_used": False,
    }
    with (output / "transport-receipt.json").open("xb") as stream:
        stream.write(canonical_json(receipt))
    return receipt
