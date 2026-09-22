#!/usr/bin/env python3
"""Build the M4 confirmatory completion gate without opening outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


EXPECTED_RUNS = 20
EXPECTED_PAIRS = 10
EXPECTED_MANIFEST_SHA256 = (
    "1f59d8cf3339d70b58b084af48bd2f192a880d7a43fca8190fb1d54649db5b39"
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class CompletionGateError(ValueError):
    """Raised when the frozen completion contract is not satisfied."""


def require(condition: bool, message: str) -> None:
    """Raise a gate error when a required condition is false."""
    if not condition:
        raise CompletionGateError(message)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Read a compact JSON provenance record."""
    value = json.loads(path.read_bytes())
    require(isinstance(value, dict), f"{path}: expected an object")
    return value


def valid_sha256(value: object) -> bool:
    """Return whether a value is a lowercase SHA-256 digest."""
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def validate_terminal_authentication(
    identity: str, receipt: dict[str, Any]
) -> None:
    """Validate one hash-only terminal authentication receipt."""
    require(
        receipt.get("schema")
        == "m4-oars-confirmatory-arm-terminal-authentication-v1",
        f"{identity}: terminal authentication schema differs",
    )
    require(
        receipt.get("status") == "PASS_AUTHENTICATED_OUTCOME_EMBARGOED",
        f"{identity}: terminal authentication did not pass",
    )
    require(receipt.get("identity") == identity, f"{identity}: identity differs")
    require(
        receipt.get("pipeline_status") == {"upstream": "success", "child": "success"},
        f"{identity}: pipeline topology is not terminal-success",
    )
    job_status = receipt.get("job_status")
    require(
        isinstance(job_status, dict)
        and set(job_status) == {"generator", "logs_before", "main", "logs_after"}
        and all(value == "success" for value in job_status.values()),
        f"{identity}: job topology is not terminal-success",
    )
    archives = receipt.get("archives")
    require(
        isinstance(archives, dict)
        and set(archives) == {"main", "logs_after"}
        and all(value.get("zip_integrity_passed") is True for value in archives.values()),
        f"{identity}: terminal ZIP integrity failed",
    )
    artifact_hashes = receipt.get("artifact_sha256")
    require(
        receipt.get("declared_artifact_count") == 14
        and isinstance(artifact_hashes, dict)
        and len(artifact_hashes) == 14
        and all(valid_sha256(value) for value in artifact_hashes.values()),
        f"{identity}: declared artifact hash contract differs",
    )
    require(
        "arm-result.json" in artifact_hashes
        and "evaluation/evaluation_data.json" in artifact_hashes,
        f"{identity}: outcome hashes are absent",
    )
    require(
        receipt.get("declared_hashes_match") is True
        and receipt.get("terminal_copies_byte_identical") is True,
        f"{identity}: terminal copies are not authenticated",
    )
    require(
        receipt.get("workload_exit_codes") == {"main": 0, "logs_after": 0},
        f"{identity}: workload exit code differs",
    )
    require(
        receipt.get("result_opened") is False
        and receipt.get("evaluation_data_opened") is False
        and receipt.get("scientific_outcome_accessed") is False
        and receipt.get("complete_outcome_embargo_preserved") is True,
        f"{identity}: complete outcome embargo was not preserved",
    )


def validate_regular_prelaunch(
    run: dict[str, Any], receipt: dict[str, Any], predecessor_sha256: str
) -> None:
    """Validate an ordinary strict-sequence prelaunch receipt."""
    identity = run["identity"]
    require(
        receipt.get("schema") == "m4-oars-confirmatory-arm-prelaunch-validation-v1"
        and receipt.get("status") == "PASS_AUTHORIZED_UNSUBMITTED",
        f"{identity}: prelaunch gate differs",
    )
    for key in ("identity", "global_sequence", "pair", "arm", "mode", "predecessor"):
        require(receipt.get(key) == run[key], f"{identity}: prelaunch {key} differs")
    frozen = receipt.get("frozen_assignment")
    require(
        isinstance(frozen, dict)
        and frozen.get("training_seed") == run["training_seed"]
        and frozen.get("assignment_seed") == run["assignment_seed"]
        and frozen.get("assignment_domain") == run["assignment_domain"],
        f"{identity}: frozen assignment differs",
    )
    predecessor = receipt.get("predecessor_terminal_authentication")
    require(
        isinstance(predecessor, dict)
        and predecessor.get("sha256") == predecessor_sha256
        and predecessor.get("status") == "PASS_AUTHENTICATED_OUTCOME_EMBARGOED",
        f"{identity}: predecessor authentication differs",
    )
    checks = receipt.get("checks")
    require(
        isinstance(checks, dict)
        and checks
        and all(value is True for value in checks.values())
        and receipt.get("scientific_outcome_accessed") is False
        and receipt.get("launch_permitted") is True,
        f"{identity}: prelaunch checks did not pass",
    )


def validate_regular_submission(
    run: dict[str, Any],
    receipt: dict[str, Any],
    prelaunch_sha256: str,
    predecessor_sha256: str,
    candidate_sha256: str,
    authentication: dict[str, Any],
) -> None:
    """Validate an ordinary one-shot submission receipt and topology."""
    identity = run["identity"]
    require(
        receipt.get("schema") == "m4-oars-confirmatory-arm-submission-receipt-v1"
        and receipt.get("status") == "SUBMITTED_NO_WAIT_TOPOLOGY_PRESENT",
        f"{identity}: submission receipt differs",
    )
    for key in ("identity", "global_sequence", "pair", "arm", "mode", "predecessor"):
        require(receipt.get(key) == run[key], f"{identity}: submission {key} differs")
    require(
        receipt.get("prelaunch_validation_sha256") == prelaunch_sha256
        and receipt.get("predecessor_terminal_authentication_sha256")
        == predecessor_sha256
        and receipt.get("manifest_sha256") == candidate_sha256
        and receipt.get("launcher") == "runllm.py --no_wait"
        and receipt.get("launcher_return_code") == 0
        and receipt.get("submission_attempts_consumed") == 1
        and receipt.get("submission_attempt_limit") == 1
        and receipt.get("scheduler_time_limit_seconds") == 14400
        and receipt.get("queue_deadline_override") is None,
        f"{identity}: one-shot launch contract differs",
    )
    require(
        receipt.get("automatic_retry") is False
        and receipt.get("automatic_replacement") is False
        and receipt.get("automatic_extension") is False
        and receipt.get("logs_opened") is False
        and receipt.get("artifacts_opened") is False
        and receipt.get("scientific_outcome_accessed") is False
        and receipt.get("complete_outcome_embargo_preserved") is True,
        f"{identity}: submission embargo or retry contract differs",
    )
    pipeline = receipt.get("pipeline")
    require(isinstance(pipeline, dict), f"{identity}: pipeline topology missing")
    require(
        pipeline.get("upstream_id") == authentication["pipeline_ids"]["upstream"]
        and pipeline.get("child_id") == authentication["pipeline_ids"]["child"]
        and pipeline.get("generator_job_id") == authentication["jobs"]["generator"]
        and pipeline.get("logs_before_job_id") == authentication["jobs"]["logs_before"]
        and pipeline.get("main_job_id") == authentication["jobs"]["main"]
        and pipeline.get("logs_after_job_id") == authentication["jobs"]["logs_after"],
        f"{identity}: submitted and terminal topology differ",
    )


def validate_first_replacement(
    run: dict[str, Any], validation: dict[str, Any], submission: dict[str, Any],
    authentication: dict[str, Any], validation_sha256: str
) -> None:
    """Validate the frozen transport-only replacement used by sequence one."""
    identity = run["identity"]
    require(
        validation.get("schema")
        == "m4-oars-randomized-confirmatory-transport-replacement-validation-v1"
        and validation.get("status") == "PASS_REPLACEMENT_AUTHORIZED_UNSUBMITTED"
        and validation.get("identity") == identity
        and validation.get("global_sequence") == 1
        and validation.get("predecessor") is None,
        f"{identity}: transport replacement validation differs",
    )
    manifest = validation.get("manifest")
    prior_attempt = validation.get("prior_attempt")
    replacement = validation.get("replacement")
    require(
        isinstance(manifest, dict)
        and manifest.get("scientific_or_runtime_change") is False
        and manifest.get("scheduler_time_limit_seconds") == 14400
        and manifest.get("queue_deadline_present") is False
        and isinstance(prior_attempt, dict)
        and prior_attempt.get("guard_consumed") is True
        and prior_attempt.get("returned_pipeline_id") is None
        and prior_attempt.get("launch_log_created") is False
        and prior_attempt.get("scientific_run_started") is False
        and isinstance(replacement, dict)
        and replacement.get("attempt_limit") == 1
        and replacement.get("required_launcher") == "runllm.py --no_wait"
        and replacement.get("outcome_embargo_preserved") is True,
        f"{identity}: transport-only replacement contract differs",
    )
    require(
        submission.get("schema")
        == "m4-oars-randomized-confirmatory-submission-receipt-v1"
        and submission.get("status") == "SUBMITTED_PENDING_STATUS_ONLY"
        and submission.get("identity") == identity
        and submission.get("global_sequence") == 1
        and submission.get("predecessor") is None
        and submission.get("manifest_sha256") == manifest.get("sha256")
        and submission.get("transport_replacement_validation_sha256")
        == validation_sha256,
        f"{identity}: replacement submission receipt differs",
    )
    launcher = submission.get("launcher")
    guard = submission.get("guard")
    upstream = submission.get("upstream_pipeline")
    downstream = submission.get("downstream_pipeline")
    require(
        isinstance(launcher, dict)
        and launcher.get("required_launcher") == "runllm.py --no_wait"
        and launcher.get("return_code") == 0
        and isinstance(guard, dict)
        and guard.get("replacement_attempt_limit") == 1
        and guard.get("replacement_attempts_consumed") == 1
        and guard.get("automatic_retry") is False
        and guard.get("automatic_replacement") is False
        and guard.get("automatic_extension") is False
        and submission.get("artifacts_opened") is False
        and submission.get("scientific_outcome_accessed") is False,
        f"{identity}: replacement launch contract differs",
    )
    require(
        isinstance(upstream, dict)
        and isinstance(downstream, dict)
        and upstream.get("id") == authentication["pipeline_ids"]["upstream"]
        and downstream.get("id") == authentication["pipeline_ids"]["child"]
        and upstream.get("generate_job_id") == authentication["jobs"]["generator"]
        and downstream.get("logs_before_job_id") == authentication["jobs"]["logs_before"]
        and downstream.get("workload_job_id") == authentication["jobs"]["main"]
        and downstream.get("logs_after_job_id") == authentication["jobs"]["logs_after"],
        f"{identity}: replacement and terminal topology differ",
    )


def build_gate(root: Path, manifest_path: Path, audited_at: str) -> dict[str, Any]:
    """Audit the full frozen sequence and return its compact completion gate."""
    require(sha256(manifest_path) == EXPECTED_MANIFEST_SHA256, "run manifest hash differs")
    manifest = load_json(manifest_path)
    runs = manifest.get("runs")
    require(
        manifest.get("run_count") == EXPECTED_RUNS
        and isinstance(runs, list)
        and len(runs) == EXPECTED_RUNS,
        "run manifest is not the frozen 20-run design",
    )
    require(
        [run.get("global_sequence") for run in runs] == list(range(1, EXPECTED_RUNS + 1)),
        "strict global sequence differs",
    )
    identities = [run.get("identity") for run in runs]
    require(len(set(identities)) == EXPECTED_RUNS, "run identities are not unique")
    require(
        {(run.get("pair"), run.get("arm")) for run in runs}
        == {(pair, arm) for pair in range(1, EXPECTED_PAIRS + 1) for arm in ("fifo", "oars")},
        "paired arm coverage differs",
    )
    require(
        runs[0].get("predecessor") is None
        and all(runs[index].get("predecessor") == runs[index - 1].get("identity")
                for index in range(1, EXPECTED_RUNS)),
        "predecessor chain differs",
    )

    results: dict[str, Any] = {}
    authentication_paths: dict[str, Path] = {}
    for run in runs:
        identity = run["identity"]
        stem = identity.replace("-", "_")
        authentication_path = root / f"confirmatory_{stem}_terminal_authentication.json"
        authentication = load_json(authentication_path)
        validate_terminal_authentication(identity, authentication)
        authentication_paths[identity] = authentication_path

        submission_path = root / f"confirmatory_{stem}_submission_receipt.json"
        submission = load_json(submission_path)
        if run["global_sequence"] == 1:
            prelaunch_path = root / "confirmatory_p01_oars_transport_replacement_validation.json"
            prelaunch = load_json(prelaunch_path)
            validate_first_replacement(
                run, prelaunch, submission, authentication, sha256(prelaunch_path)
            )
        else:
            prelaunch_path = root / f"confirmatory_{stem}_prelaunch_validation.json"
            prelaunch = load_json(prelaunch_path)
            predecessor_path = authentication_paths[run["predecessor"]]
            predecessor_sha256 = sha256(predecessor_path)
            validate_regular_prelaunch(run, prelaunch, predecessor_sha256)
            validate_regular_submission(
                run,
                submission,
                sha256(prelaunch_path),
                predecessor_sha256,
                prelaunch["candidate"]["sha256"],
                authentication,
            )

        artifact_hashes = authentication["artifact_sha256"]
        results[identity] = {
            "sha256": artifact_hashes["arm-result.json"],
            "evaluation_data_sha256": artifact_hashes[
                "evaluation/evaluation_data.json"
            ],
            "authentication_receipt": {
                "path": str(authentication_path.relative_to(root.parent.parent.parent)),
                "sha256": sha256(authentication_path),
            },
            "prelaunch_receipt": {
                "path": str(prelaunch_path.relative_to(root.parent.parent.parent)),
                "sha256": sha256(prelaunch_path),
            },
            "submission_receipt": {
                "path": str(submission_path.relative_to(root.parent.parent.parent)),
                "sha256": sha256(submission_path),
            },
            "pipeline_ids": authentication["pipeline_ids"],
            "jobs": authentication["jobs"],
        }

    return {
        "schema": "m4-oars-randomized-confirmatory-terminal-authentication-v1",
        "status": "PASS_ALL_20_AUTHENTICATED_RESULTS_UNOPENED",
        "audited_at_utc": audited_at,
        "run_manifest": {
            "path": str(manifest_path.relative_to(root.parent.parent.parent)),
            "sha256": EXPECTED_MANIFEST_SHA256,
        },
        "protocol_sha256": manifest["protocol_sha256"],
        "run_count": EXPECTED_RUNS,
        "pair_count": EXPECTED_PAIRS,
        "ordered_identities": identities,
        "checks": {
            "exact_frozen_identity_coverage": True,
            "strict_global_sequence_complete": True,
            "predecessor_chain_authenticated_before_successor": True,
            "all_prelaunch_and_submission_receipts_authenticated": True,
            "all_terminal_authentications_pass": True,
            "all_declared_artifact_hashes_match": True,
            "all_terminal_copies_byte_identical": True,
            "all_workload_exit_codes_zero": True,
            "complete_outcome_embargo_preserved_through_gate": True,
        },
        "scientific_outcome_accessed": False,
        "outcome_embargo_state": "COMPLETE_GATE_SATISFIED_OUTCOMES_NOT_YET_OPENED",
        "results": results,
        "next_step": "commit_and_push_gate_before_any_separately_authorized_analysis",
    }


def main() -> None:
    """Parse arguments, audit receipts, and write the completion gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--audited-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    gate = build_gate(root, args.run_manifest.resolve(), args.audited_at)
    args.output.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
