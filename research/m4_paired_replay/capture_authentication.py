"""Authenticate terminal EOS capture evidence before producing admission rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, build_opener
from urllib.error import HTTPError

from research.m4_eligible_replay.contracts import canonical_json, safe_member
from research.m4_eligible_replay.transport import NoRedirect
from research.m4_paired_replay.contracts import (
    DOMAIN,
    TERMINAL,
    arm_groups,
    object_hash,
    validate_plan,
)


# Exact checks emitted by the frozen independent capture analyzer.
CAPTURE_CHECKS = {
    "protocol_schema",
    "manifest_schema",
    "status",
    "trajectory_identity",
    "trajectory_seed",
    "assignment_domain",
    "learner_version_matches_updates",
    "parameter_hash_unchanged",
    "captured_update_not_executed",
    "outcomes_unopened",
    "captured_optimizer_steps_zero",
    "captured_scheduler_steps_zero",
    "row_count",
    "group_authentication",
    "paths",
    "raw_file_hashes",
    "tensor_content_hashes",
    "required_fields",
    "tensor_shapes",
    "unique_group_ids",
    "unique_sample_ids",
    "heldout_disjoint",
    "base_from_frontier",
    "shield_from_frontier",
    "actions_differ",
    "minimum_overlap",
    "roles_match",
    "exact_search",
    "actual_is_shield",
    "actual_ids_are_shield",
    "token_band",
    "reward_variance_preserved",
    "strict_imminent_m4_gain",
    "checkpoint_within_capsule",
    "checkpoint_nonempty",
    "checkpoint_inventory",
    "checkpoint_authentication",
}
CAUSAL_PROTOCOL_HASH = (
    "716ba0eb8f980ca5af0ef3d57e2415befc1fb1a315afeb01b9bc326a60341b48"
)


def classify(
    *,
    assignment: dict,
    parent: dict,
    child: dict | None,
    jobs: list[dict],
    expected_parent_id: int,
    expected_child_id: int | None,
    expected_eos_job_id: int | None,
    result_bytes: bytes | None,
    manifest_bytes: bytes | None,
    inventory_bytes: bytes | None,
    progress_bytes: bytes | None,
) -> tuple[dict, dict]:
    """Use complete API topology and hash-bound compact files, never success alone."""
    identity, seed = assignment["identity"], assignment["seed"]
    workload = f"m4-eligible-live-acquisition-{identity}"
    exact_name = f"basic/{workload} dgxh100_eos 00 [2 dgxh100_eos]"
    if parent["id"] != expected_parent_id or (
        (child is None and expected_child_id is not None)
        or (child is not None and child["id"] != expected_child_id)
    ):
        raise ValueError("capture parent/child identity changed")
    if (
        parent["status"] not in TERMINAL
        or (child is not None and child["status"] not in TERMINAL)
        or not jobs
        or any(j["status"] not in TERMINAL for j in jobs)
    ):
        raise ValueError("every capture chain job must be terminal")
    candidates = [j for j in jobs if j["name"] == exact_name]
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("duplicate job metadata")
    if expected_eos_job_id is None:
        if candidates or parent["status"] == "success":
            raise ValueError("pre-workload failure topology inconsistent")
        eos = None
    else:
        if child is None:
            raise ValueError("EOS capture cannot exist without a child pipeline")
        if len(candidates) != 1 or candidates[0]["id"] != expected_eos_job_id:
            raise ValueError("exact capture EOS workload not authenticated")
        eos = candidates[0]

    evidence = {
        "schema": "m4-authenticated-capture-evidence-v1",
        "identity": identity,
        "seed": seed,
        "assignment_domain": DOMAIN,
        "parent": parent,
        "child": child,
        "jobs": jobs,
        "compact_sha256": {},
        "classification": "FAILED",
    }

    def verified(data: bytes | None, relative: str) -> dict:
        if data is None or inventory_bytes is None:
            raise ValueError("required compact artifact missing")
        if len(data) > 2 << 20:
            raise ValueError("oversized compact capture artifact")
        target = f"/jet/assets/basic/{workload}/{relative}"
        entries = [
            line.split("  ", 1) for line in inventory_bytes.decode().splitlines()
        ]
        hashes = [h for h, p in entries if p == target]
        digest = hashlib.sha256(data).hexdigest()
        if hashes != [digest]:
            raise ValueError("compact artifact not authenticated by workload inventory")
        evidence["compact_sha256"][relative] = digest
        return json.loads(data)

    manifest_hash = None
    classification = "FAILED"
    if eos is not None and eos["status"] == "success":
        result = verified(
            result_bytes, f"m4-eligible-live-{identity}-capture-result.json"
        )
        if (
            result.get("trajectory_identity"),
            result.get("trajectory_seed"),
            result.get("assignment_domain"),
        ) != (identity, seed, DOMAIN):
            raise ValueError("capture result assignment differs")
        if result.get("post_update_outcomes_opened") is not False:
            raise ValueError("capture result outcome embargo differs")
        if result["status"] == "PASS_OUTCOME_EXCLUDED_ELIGIBLE_CAPTURE":
            if set(result["checks"]) != CAPTURE_CHECKS or not all(
                v is True for v in result["checks"].values()
            ):
                raise ValueError("independent capture checks incomplete")
            if result["protocol_sha256"] != CAUSAL_PROTOCOL_HASH:
                raise ValueError("capture protocol changed")
            manifest = verified(
                manifest_bytes, "m4-eligible-live-capsule/manifest.json"
            )
            manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
            if manifest_hash != result["manifest_sha256"]:
                raise ValueError("capture analysis and manifest differ")
            if (
                manifest["trajectory_identity"],
                manifest["trajectory_seed"],
                manifest["assignment_domain"],
            ) != (identity, seed, DOMAIN):
                raise ValueError("historical manifest substituted")
            if (
                manifest["captured_update_executed"] is not False
                or manifest["post_update_outcomes_opened"] is not False
            ):
                raise ValueError("capsule boundary differs")
            arm_groups(manifest, "reward_variance_base")
            arm_groups(manifest, "m4_shield")
            classification = "CAPSULE"
        elif result["status"] == "NO_ELIGIBLE_DECISION":
            progress = verified(
                progress_bytes, f"m4-eligible-live-{identity}-progress.json"
            )
            if (
                result.get("paired_replay_started") is not False
                or result["maximum_completed_updates"] != 64
                or (
                    progress["train_steps"] != 64
                    or progress["learner_version"] != 64
                    or progress["run_exit_code"] != 0
                    or progress["capsule_written"] is not False
                )
            ):
                raise ValueError(
                    "no-eligibility requires complete authenticated window"
                )
            if manifest_bytes is not None:
                raise ValueError("no-eligibility record unexpectedly has a capsule")
            classification = "NO_ELIGIBLE"
        else:
            raise ValueError(
                "successful EOS workload lacks a valid capture classification"
            )
    evidence["classification"] = classification
    evidence["inventory_sha256"] = (
        hashlib.sha256(inventory_bytes).hexdigest()
        if inventory_bytes is not None
        else None
    )
    row = {
        "identity": identity,
        "seed": seed,
        "assignment_domain": DOMAIN,
        "terminal_status": eos["status"] if eos else parent["status"],
        "eos_job_id": expected_eos_job_id,
        "classification": classification,
        "manifest_sha256": manifest_hash,
        "evidence_sha256": object_hash(evidence),
    }
    return row, evidence


class CaptureClient:
    """Read-only GitLab transport; job credentials cannot cross a redirect."""

    def __init__(self, token: str, *, opener: Any = None) -> None:
        self.headers = {"PRIVATE-TOKEN": token}
        self.opener = build_opener(NoRedirect()) if opener is None else opener

    def read(self, route: str, *, maximum_bytes: int = 2 << 20) -> bytes:
        if not route.startswith("/") or "://" in route:
            raise ValueError("API route must be project-relative")
        url = "https://gitlab-master.nvidia.com/api/v4/projects/70847" + route
        with self.opener.open(
            Request(url, headers=self.headers), timeout=90
        ) as response:
            data = response.read(maximum_bytes + 1)
        if len(data) > maximum_bytes:
            raise ValueError("oversized capture API response")
        return data

    def artifact(self, job_id: int, prefix: str, relative: str) -> bytes:
        path = str(safe_member(prefix) / safe_member(relative))
        return self.read(f"/jobs/{job_id}/artifacts/{quote(path, safe='/')}")

    def collect(self, assignment: dict, binding: dict) -> tuple[dict, dict]:
        """Authenticate source topology including retried jobs; never fetch checkpoints."""
        parent_id, child_id, eos_id = (
            binding[k]
            for k in ("parent_pipeline_id", "child_pipeline_id", "eos_job_id")
        )
        parent = json.loads(self.read(f"/pipelines/{parent_id}"))
        child = (
            json.loads(self.read(f"/pipelines/{child_id}"))
            if child_id is not None
            else None
        )
        bridges = json.loads(self.read(f"/pipelines/{parent_id}/bridges?per_page=100"))
        downstream = [
            b["downstream_pipeline"]["id"]
            for b in bridges
            if b.get("downstream_pipeline")
        ]
        if (
            parent["ref"] != "dlalgo-ci/eos"
            or (
                child is not None
                and (
                    downstream != [child_id]
                    or parent["sha"] != child["sha"]
                    or child["ref"] != parent["ref"]
                )
            )
            or (
                child is None
                and (
                    downstream
                    or eos_id is not None
                    or parent["status"] not in {"failed", "canceled", "skipped"}
                )
            )
        ):
            raise ValueError("pipeline relationship differs from submitted chain")
        jobs = []
        for page in range(1, 11):
            rows = json.loads(
                self.read(
                    f"/pipelines/{child_id if child is not None else parent_id}/jobs?include_retried=true&per_page=100&page={page}"
                )
            )
            jobs.extend(rows)
            if len(rows) < 100:
                break
        else:
            raise ValueError("unexpectedly large capture topology")
        # Retain only source identity/status fields, not API user free text.
        parent = {k: parent[k] for k in ("id", "status")}
        child = {k: child[k] for k in ("id", "status")} if child is not None else None
        jobs = [{k: j[k] for k in ("id", "name", "status")} for j in jobs]
        args = dict(
            assignment=assignment,
            parent=parent,
            child=child,
            jobs=jobs,
            expected_parent_id=parent_id,
            expected_child_id=child_id,
            expected_eos_job_id=eos_id,
        )
        files = dict(
            result_bytes=None,
            manifest_bytes=None,
            inventory_bytes=None,
            progress_bytes=None,
        )
        eos = next((j for j in jobs if j["id"] == eos_id), None)
        if eos and eos["status"] == "success":
            identity = assignment["identity"]
            prefix = f"workspace/assets/basic/m4-eligible-live-acquisition-{identity}"
            files["inventory_bytes"] = self.artifact(eos_id, prefix, "artifacts.sha256")
            files["result_bytes"] = self.artifact(
                eos_id, prefix, f"m4-eligible-live-{identity}-capture-result.json"
            )
            status = json.loads(files["result_bytes"])["status"]
            if status == "PASS_OUTCOME_EXCLUDED_ELIGIBLE_CAPTURE":
                files["manifest_bytes"] = self.artifact(
                    eos_id, prefix, "m4-eligible-live-capsule/manifest.json"
                )
            elif status == "NO_ELIGIBLE_DECISION":
                files["progress_bytes"] = self.artifact(
                    eos_id, prefix, f"m4-eligible-live-{identity}-progress.json"
                )
                try:
                    files["manifest_bytes"] = self.artifact(
                        eos_id, prefix, "m4-eligible-live-capsule/manifest.json"
                    )
                except HTTPError as error:
                    if error.code != 404:
                        raise
        return classify(**args, **files)


def collect_cohort(
    client: CaptureClient, *, plan: dict, bindings: list[dict], output: Path
) -> list[dict]:
    """Produce all twenty immutable evidence packets; no replacement or outcome access."""
    validate_plan(plan)
    if len(bindings) != 20 or [b["identity"] for b in bindings] != [
        r["identity"] for r in plan["trajectories"]
    ]:
        raise ValueError("all fixed capture bindings required")
    results = [
        client.collect(assignment, binding)
        for assignment, binding in zip(plan["trajectories"], bindings)
    ]
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    rows = []
    for row, evidence in results:
        with (output / f"{row['identity']}.json").open("xb") as stream:
            stream.write(canonical_json(evidence))
        rows.append(row)
    with (output / "classifications.json").open("xb") as stream:
        stream.write(canonical_json(rows))
    verify_admission_bundle(plan, rows, output)
    return rows


def verify_admission_bundle(plan: dict, rows: list[dict], evidence_dir: Path) -> None:
    """Recheck provenance packets before either transfer or GPU dispatch."""
    validate_plan(plan)
    if len(rows) != 20:
        raise ValueError("all twenty classifications required")
    for expected, row in zip(plan["trajectories"], rows):
        if (row["identity"], row["seed"], row["assignment_domain"]) != (
            expected["identity"],
            expected["seed"],
            DOMAIN,
        ):
            raise ValueError("capture evidence differs from fixed assignment order")
        path = evidence_dir / f"{expected['identity']}.json"
        if path.is_symlink():
            raise ValueError("linked capture evidence forbidden")
        data = path.read_bytes()
        proof = json.loads(data)
        if hashlib.sha256(data).hexdigest() != row[
            "evidence_sha256"
        ] or data != canonical_json(proof):
            raise ValueError("capture evidence packet differs")
        if (
            proof["identity"],
            proof["seed"],
            proof["assignment_domain"],
            proof["classification"],
        ) != (
            row["identity"],
            row["seed"],
            DOMAIN,
            row["classification"],
        ):
            raise ValueError("capture evidence assignment differs")
        if proof["parent"]["status"] not in TERMINAL or (
            proof["child"] is not None and proof["child"]["status"] not in TERMINAL
        ):
            raise ValueError("capture evidence not terminal")
        jobs = proof["jobs"]
        if (
            not jobs
            or any(j["status"] not in TERMINAL for j in jobs)
            or len({j["id"] for j in jobs}) != len(jobs)
        ):
            raise ValueError("capture job set not fully terminal or unique")
        eos = [
            j
            for j in jobs
            if j["name"]
            == f"basic/m4-eligible-live-acquisition-{expected['identity']} dgxh100_eos 00 [2 dgxh100_eos]"
        ]
        if row["eos_job_id"] is None:
            if (
                eos
                or row["classification"] != "FAILED"
                or proof["parent"]["status"] == "success"
                or row["terminal_status"] != proof["parent"]["status"]
            ):
                raise ValueError("capture pre-workload failure differs")
        elif (
            proof["child"] is None
            or len(eos) != 1
            or (eos[0]["id"], eos[0]["status"])
            != (
                row["eos_job_id"],
                row["terminal_status"],
            )
        ):
            raise ValueError("capture EOS evidence binding differs")
        if (
            row["classification"] in {"CAPSULE", "NO_ELIGIBLE"}
            and row["terminal_status"] != "success"
        ):
            raise ValueError("valid capture requires successful source workload")
        if row["classification"] == "CAPSULE":
            if (
                proof["compact_sha256"].get("m4-eligible-live-capsule/manifest.json")
                != row["manifest_sha256"]
            ):
                raise ValueError("manifest not bound by evidence packet")
