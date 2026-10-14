"""Fetch individual capsule members directly from their EOS job artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener

from research.m4_eligible_replay.contracts import (
    allowed_members,
    canonical_json,
    digest_file,
    safe_member,
    stream_member,
)


class NoRedirect(HTTPRedirectHandler):
    """Never forward job credentials through an artifact redirect."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("artifact redirected; inspect transport before continuing")


def fetch_capsule(
    *, protocol_path: Path, output: Path, token: str, job_token: bool
) -> dict:
    """Authenticate the compact manifest before streaming its allowlisted members."""
    protocol = json.loads(protocol_path.read_bytes())
    source = protocol["source_capsule"]
    if source["project_id"] != 70847 or source["eos_job_id"] != 474952005:
        raise ValueError("qualification artifact identity changed")
    prefix = safe_member(source["artifact_prefix"])
    if (
        str(prefix)
        != "workspace/assets/basic/m4-eligible-live-capture-qualification/m4-eligible-live-capsule"
    ):
        raise ValueError("qualification artifact prefix changed")
    base = f"https://gitlab-master.nvidia.com/api/v4/projects/70847/jobs/474952005/artifacts/{quote(str(prefix), safe='/')}/"
    opener = build_opener(NoRedirect())
    headers = {"JOB-TOKEN" if job_token else "PRIVATE-TOKEN": token}
    with opener.open(
        Request(base + "manifest.json", headers=headers), timeout=60
    ) as response:
        manifest_bytes = response.read(1 << 20)
        if response.read(1):
            raise ValueError("oversized capsule manifest")
    if hashlib.sha256(manifest_bytes).hexdigest() != source["manifest_sha256"]:
        raise ValueError("capsule manifest identity differs")
    manifest = json.loads(manifest_bytes)
    if (
        manifest["learner_version"] != 7
        or manifest["captured_update_executed"]
        or manifest["post_update_outcomes_opened"]
    ):
        raise ValueError("historical capsule boundary differs")
    members = allowed_members(manifest)
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_bytes(manifest_bytes)
    records = []
    for member in members:
        destination = output / member["path"]
        with opener.open(
            Request(base + quote(member["path"], safe="/"), headers=headers),
            timeout=300,
        ) as response:
            stream_member(
                response,
                destination,
                expected_bytes=member["bytes"],
                expected_hash=member["sha256"],
            )
        records.append(
            {
                "path": member["path"],
                "bytes": destination.stat().st_size,
                "sha256": digest_file(destination),
            }
        )
    result = {
        "status": "PASS_DIRECT_EOS_MEMBER_TRANSPORT",
        "source_job_id": 474952005,
        "manifest_sha256": source["manifest_sha256"],
        "members": records,
        "large_archive_downloaded": False,
        "logs_after_used": False,
        "shield_only_tensors_downloaded": False,
    }
    (output / "transport-receipt.json").write_bytes(canonical_json(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-token", action="store_true")
    args = parser.parse_args()
    token = os.environ["CI_JOB_TOKEN" if args.job_token else "JET_GITLAB_TOKEN"]
    result = fetch_capsule(
        protocol_path=args.protocol,
        output=args.output,
        token=token,
        job_token=args.job_token,
    )
    print(
        json.dumps({"status": result["status"], "member_count": len(result["members"])})
    )


if __name__ == "__main__":
    main()
