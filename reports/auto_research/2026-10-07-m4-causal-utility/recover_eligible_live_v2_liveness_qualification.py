#!/usr/bin/env python3
"""Replay a header-counting repair on authenticated qualification artifacts."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SESSION = ROOT / "session/20260914_m4_opportunity_control"
ARTIFACTS = SESSION / "eligible-live-v2-liveness-terminal"
MANIFESTS = SESSION / "m4-eligible-live-v2-liveness-qualification-manifests"
PROTOCOL = HERE / "eligible_live_v2_liveness_qualification_protocol.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    authentication_path = ARTIFACTS / "authentication.json"
    authentication = json.loads(authentication_path.read_bytes())
    package = json.loads((HERE / "eligible_live_v2_liveness_qualification_package_receipt.json").read_bytes())
    assert sha256(PROTOCOL.read_bytes()) == package["protocol_sha256"]
    expected_manifests = {row["identity"]: row["sha256"] for row in package["manifests"]}
    protocol = json.loads(PROTOCOL.read_bytes())
    expected_seeds = {row["identity"]: row["seed"] for row in protocol["qualification"]["identities"]}
    assert {row["identity"] for row in authentication["chains"]} == set(expected_seeds)
    replay_dir = ARTIFACTS / "offline-recovery"
    replay_dir.mkdir(exist_ok=True)
    records = []
    for chain in authentication["chains"]:
        identity = chain["identity"]
        assert chain["seed"] == expected_seeds[identity]
        assert chain["artifact_copies_identical"]
        assert chain["workload_owned_artifacts_authenticated"]
        assert not chain["qualification_result_present"]
        directory = ARTIFACTS / identity / "eos"
        for relative, record in chain["artifact_records"].items():
            content = (directory / relative).read_bytes()
            assert sha256(content) == record["sha256"]
            assert content == (ARTIFACTS / identity / "logs_after" / relative).read_bytes()
            if not relative.startswith("jet_assets/"):
                assert record["manifest_hash_matches"]
        prefix = f"m4-eligible-live-v2-{identity}"
        oars = directory / f"{prefix}-oars-v2.jsonl"
        ledger = [json.loads(line) for line in oars.read_text().splitlines() if line]
        assert ledger[0]["event_type"] == "header"
        assert ledger[0]["actuation_scorer"] == "m4_shield"
        assert len([row for row in ledger if row["event_type"] == "header"]) == 1
        decisions = [row for row in ledger if row["event_type"] == "decision"]
        assert len(decisions) == 64 and len(ledger) == 65
        for row in decisions:
            assert row["proposal_matches_actual"] is True
            assert row["actual_selected_group_ids"] == row["proposals"]["m4_shield"]["proposed_group_ids"]
        log = directory / f"{prefix}-run.log"
        match = re.search(r"(?m)^SC run complete: (\{.*\})$", log.read_text())
        assert match
        controller = ast.literal_eval(match.group(1))
        assert controller["train_steps"] == controller["trainer_version"] == 64
        tests = (directory / f"{prefix}-tests.log").read_text()
        assert len(re.findall(r"(?m)^2 passed,", tests)) == 1
        assert len(re.findall(r"(?m)^7 passed,", tests)) == 1
        assert len(re.findall(r"(?m)^16 passed,", tests)) == 1
        manifest_path = MANIFESTS / f"{identity}.json"
        assert sha256(manifest_path.read_bytes()) == expected_manifests[identity]
        script = json.loads(manifest_path.read_bytes())["spec"]["script"]
        script = script.replace("{{", "{").replace("}}", "}")
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", script, re.S)
        original = next(block for block in blocks if "qualification=controller" in block)
        original_line = "oars_rows=[json.loads(line) for line in oars.read_text().splitlines() if line]"
        repaired_line = "oars_rows=[row for row in (json.loads(line) for line in oars.read_text().splitlines() if line) if row.get('event_type')=='decision']"
        assert original.count(original_line) == 1
        repaired = original.replace(original_line, repaired_line)
        paths = [
            PROTOCOL, identity, str(chain["seed"]), "0",
            directory / f"{prefix}-liveness.json",
            directory / f"{prefix}-lifecycle.jsonl",
            directory / f"{prefix}-opportunity.jsonl",
            directory / f"{prefix}-observer-duty.json",
            oars, directory / f"{prefix}-forbidden-capsule", log,
        ]
        outcomes = {}
        for label, source in (("original", original), ("repaired", repaired)):
            source_path = replay_dir / f"{identity}-{label}-classifier.py"
            source_path.write_text(source + "\n")
            result_path = replay_dir / f"{identity}-{label}-result.json"
            process = subprocess.run(
                [sys.executable, str(source_path), *map(str, paths), str(result_path)],
                capture_output=True, text=True,
            )
            outcomes[label] = {"exit_code": process.returncode, "source_sha256": sha256(source.encode())}
            if label == "original":
                assert process.returncode != 0
                assert "line 42" in process.stderr and "AssertionError" in process.stderr
                assert not result_path.exists()
            else:
                assert process.returncode == 0, process.stderr
                result = json.loads(result_path.read_bytes())
                assert result["status"] == protocol["per_identity_gate"]["pass_status"]
                outcomes[label]["result_sha256"] = sha256(result_path.read_bytes())
        records.append({
            "identity": identity, "seed": chain["seed"],
            "eos_job_id": chain["eos_job_id"], "logs_after_job_id": chain["logs_after_job_id"],
            "original_manifest_sha256": expected_manifests[identity],
            "artifact_manifest_sha256": chain["artifact_manifest_sha256"],
            "artifact_records": chain["artifact_records"],
            "original_wrapper_status": "ASSERTION_FAILED_ON_HEADER_COUNT",
            "classifier_replay": outcomes,
            "recovered_qualification": result,
        })
        print(identity, result["status"], "64/64 actions", flush=True)
    record = {
        "schema": "m4-eligible-live-liveness-offline-recovery-v1",
        "recorded_at": "2026-10-13",
        "status": "PASS_ALL_SIX_ORIGINAL_LIVENESS_REQUIREMENTS_OFFLINE_RECOVERED",
        "original_pipeline_states": "FAILED",
        "original_wrapper_results_present": False,
        "raw_authentication_sha256": sha256(authentication_path.read_bytes()),
        "protocol_sha256": package["protocol_sha256"],
        "repair": {
            "single_classifier_change": "Exclude the single ledger header when counting OARS decisions.",
            "scientific_or_engineering_thresholds_changed": False,
            "training_or_scheduler_code_changed": False,
            "all_six_original_failures_reproduced": True,
            "all_six_repaired_classifiers_pass": True,
            "harness_log_issue": "JET appends to its output log after the recursive checksum sweep; all six workload-owned artifacts match their declared hashes and both copies are identical.",
        },
        "chains": records,
        "new_eos_launches": 0,
        "v2_acquisition_started": False,
        "paired_replay_started": False,
        "post_update_heldout_outcomes_opened": False,
        "causal_effect_estimated": False,
    }
    destination = HERE / "eligible_live_v2_liveness_qualification_offline_recovery.json"
    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(record["status"])


if __name__ == "__main__":
    main()
