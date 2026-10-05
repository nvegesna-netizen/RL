"""Synthetic clean-room tests for quality-primary compact-result extraction."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "quality_primary_extraction", ROOT / "extract_quality_primary_compact_results.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sha256(path: Path) -> str:
    """Return a test file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fixture_tree(tmp_path: Path) -> dict[str, Path | str]:
    """Create a complete synthetic archive/authentication topology."""
    repository = tmp_path / "repository"
    authentication_dir = repository / "authentication"
    archive_root = repository / "archives"
    authentication_dir.mkdir(parents=True)
    archive_root.mkdir(parents=True)
    source_manifest = ROOT / "quality_primary_run_manifest.json"
    run_manifest = repository / "quality_primary_run_manifest.json"
    run_manifest.write_bytes(source_manifest.read_bytes())
    manifest = json.loads(run_manifest.read_bytes())
    for run in manifest["runs"]:
        identity = run["identity"]
        main_job = 100000 + run["global_sequence"]
        member = (
            "workspace/assets/basic/"
            f"m4-oars-v2-quality-primary-{identity}-acquisition/arm-result.json"
        )
        payload = json.dumps({"identity": identity}, sort_keys=True).encode() + b"\n"
        archive_dir = (
            archive_root / f"oars-v2-quality-primary-{identity}-terminal-archives"
        )
        archive_dir.mkdir()
        archive = archive_dir / f"main-{main_job}.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr(member, payload)
        authentication = {
            "schema": "m4-oars-v2-quality-primary-arm-terminal-authentication-v1",
            "identity": identity,
            "status": "PASS_AUTHENTICATED_OUTCOME_EMBARGOED",
            "successor_gate": "PASS_AUTHENTICATED_TERMINAL_PREDECESSOR",
            "complete_outcome_embargo_preserved": True,
            "scientific_outcome_accessed": False,
            "result_opened": False,
            "evaluation_data_opened": False,
            "declared_hashes_match": True,
            "terminal_copies_byte_identical": True,
            "jobs": {"main": main_job, "logs_after": 200000 + run["global_sequence"]},
            "archives": {"main": {"sha256": sha256(archive)}},
            "artifact_sha256": {"arm-result.json": hashlib.sha256(payload).hexdigest()},
        }
        auth_path = MODULE.authentication_path(authentication_dir, identity)
        auth_path.write_text(json.dumps(authentication, sort_keys=True) + "\n")
    audit = {
        "schema": MODULE.COMPLETION_AUDIT_SCHEMA,
        "status": MODULE.COMPLETION_AUDIT_STATUS,
        "completion_gate": {
            "status": "PASS_ALL_54_AUTHENTICATED_OUTCOME_EMBARGOED",
            "expected_identities": 54,
            "actual_authenticated_identities": 54,
            "missing_identities": [],
            "unexpected_identities": [],
        },
        "embargo": {
            "complete_outcome_embargo_preserved": True,
            "scientific_outcome_accessed": False,
            "result_opened": False,
            "evaluation_data_opened": False,
        },
        "frozen_evidence_sha256": {"run_manifest": MODULE.RUN_MANIFEST_SHA256},
    }
    completion_audit = repository / "completion_audit.json"
    completion_audit.write_text(json.dumps(audit, sort_keys=True) + "\n")
    return {
        "repository": repository,
        "authentication_dir": authentication_dir,
        "archive_root": archive_root,
        "run_manifest": run_manifest,
        "completion_audit": completion_audit,
        "completion_audit_sha256": sha256(completion_audit),
    }


def run_extraction(paths: dict[str, Path | str], output_dir: Path) -> dict:
    """Run the extractor against a synthetic complete gate."""
    return MODULE.extract(
        repository=paths["repository"],
        completion_audit_path=paths["completion_audit"],
        run_manifest_path=paths["run_manifest"],
        archive_root=paths["archive_root"],
        authentication_dir=paths["authentication_dir"],
        output_dir=output_dir,
        extracted_at="2026-10-05T00:00:00Z",
        expected_completion_audit_sha256=paths["completion_audit_sha256"],
    )


def test_complete_gate_extracts_only_compact_results(tmp_path: Path) -> None:
    paths = write_fixture_tree(tmp_path)
    output_dir = tmp_path / "released"
    receipt = run_extraction(paths, output_dir)
    assert receipt["status"] == "PASS_54_RESULTS_EXTRACTED_AND_HASH_AUTHENTICATED"
    assert receipt["result_count"] == 54
    assert receipt["scientific_outcome_accessed"] is True
    assert receipt["evaluation_data_accessed"] is False
    assert len(list(output_dir.glob("*.json"))) == 54


def test_gate_mutation_refuses_before_output_creation(tmp_path: Path) -> None:
    paths = write_fixture_tree(tmp_path)
    audit_path = paths["completion_audit"]
    audit = json.loads(audit_path.read_bytes())
    audit["embargo"]["scientific_outcome_accessed"] = True
    audit_path.write_text(json.dumps(audit, sort_keys=True) + "\n")
    output_dir = tmp_path / "released"
    with pytest.raises(MODULE.ExtractionError, match="completion gate"):
        MODULE.extract(
            repository=paths["repository"],
            completion_audit_path=audit_path,
            run_manifest_path=paths["run_manifest"],
            archive_root=paths["archive_root"],
            authentication_dir=paths["authentication_dir"],
            output_dir=output_dir,
            extracted_at="2026-10-05T00:00:00Z",
            expected_completion_audit_sha256=sha256(audit_path),
        )
    assert not output_dir.exists()


def test_archive_mutation_refuses_before_member_read(tmp_path: Path) -> None:
    paths = write_fixture_tree(tmp_path)
    archive = next(paths["archive_root"].glob("*/main-*.zip"))
    with archive.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(MODULE.ExtractionError, match="main archive hash differs"):
        run_extraction(paths, tmp_path / "released")
