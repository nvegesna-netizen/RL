from __future__ import annotations

import ast
import json
from pathlib import Path

from analyze_eligible_live_restore_qualification import analyze


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True))


def _row(protocol_sha: str, label: str, loss: float) -> dict:
    parameter = "e0765eb28e40eb7308c7c9b2e936179dc12f5da7deb38bdb467dc0e0de075e9e"
    return {
        "schema": "m4-shield-eligible-live-single-restore-result-v1",
        "status": "PASS_SINGLE_RESTORE",
        "restore_label": label,
        "protocol_sha256": protocol_sha,
        "capsule_manifest_sha256": "184004bfd24a47e7bd4f80ceee9f11f2891fdbae70ee60cbce5b8ca5ec4284c7",
        "checkpoint_authentication_sha256": "df9fd2aa0f2b7f35edef1d5601cc8baff599504f4ff09b67fb9c20f3f4407ded",
        "heldout_group_ids": ["h0", "h1", "h2", "h3"],
        "heldout_groups": 4,
        "heldout_samples": 32,
        "parameter_sha256_before_evaluation": parameter,
        "parameter_sha256_after_evaluation": parameter,
        "parameter_hash_unchanged": True,
        "heldout_loss": loss,
        "optimizer_restore_requested": True,
        "optimizer_path_equals_weights_path": True,
        "eval_mode": True,
        "torch_no_grad": True,
        "optimizer_steps": 0,
        "scheduler_steps": 0,
        "arm_group_files_loaded": False,
        "causal_arm_outcomes_opened": False,
        "causal_effect_estimated": False,
        "paired_acquisition_started": False,
    }


def test_restore_runtime_disables_optional_ray_dashboard() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "nemo_rl/distributed/virtual_cluster.py"
    ).read_text()
    tree = ast.parse(source)
    init_ray = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "init_ray"
    )
    local_initializers = [
        node
        for node in ast.walk(init_ray)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "init"
        and any(keyword.arg == "resources" for keyword in node.keywords)
    ]
    assert len(local_initializers) == 1
    dashboard = next(
        keyword.value
        for keyword in local_initializers[0].keywords
        if keyword.arg == "include_dashboard"
    )
    assert isinstance(dashboard, ast.Constant) and dashboard.value is False


def test_gate_passes_reproducible_fresh_restores(tmp_path: Path) -> None:
    protocol = Path(__file__).with_name(
        "eligible_live_restore_qualification_protocol.json"
    )
    import hashlib

    protocol_sha = hashlib.sha256(protocol.read_bytes()).hexdigest()
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write(a, _row(protocol_sha, "a", 0.125))
    _write(b, _row(protocol_sha, "b", 0.12500001))
    result = analyze(protocol, a, b)
    assert result["status"] == "PASS_RESTORE_AND_EVALUATION_GATE"
    assert all(result["checks"].values())


def test_gate_fails_parameter_drift(tmp_path: Path) -> None:
    protocol = Path(__file__).with_name(
        "eligible_live_restore_qualification_protocol.json"
    )
    import hashlib

    protocol_sha = hashlib.sha256(protocol.read_bytes()).hexdigest()
    row_a = _row(protocol_sha, "a", 0.125)
    row_b = _row(protocol_sha, "b", 0.125)
    row_b["parameter_sha256_after_evaluation"] = "0" * 64
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write(a, row_a)
    _write(b, row_b)
    result = analyze(protocol, a, b)
    assert result["status"] == "FAIL_RESTORE_AND_EVALUATION_GATE"
    assert result["checks"]["parameter_hashes"] is False


def test_gate_fails_opened_arm_or_irreproducible_loss(tmp_path: Path) -> None:
    protocol = Path(__file__).with_name(
        "eligible_live_restore_qualification_protocol.json"
    )
    import hashlib

    protocol_sha = hashlib.sha256(protocol.read_bytes()).hexdigest()
    row_a = _row(protocol_sha, "a", 0.125)
    row_b = _row(protocol_sha, "b", 0.25)
    row_b["causal_arm_outcomes_opened"] = True
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write(a, row_a)
    _write(b, row_b)
    result = analyze(protocol, a, b)
    assert result["status"] == "FAIL_RESTORE_AND_EVALUATION_GATE"
    assert result["checks"]["losses_reproducible"] is False
    assert result["checks"]["outcome_exclusion"] is False
