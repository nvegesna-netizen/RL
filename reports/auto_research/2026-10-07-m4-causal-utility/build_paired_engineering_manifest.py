"""Freeze and clean-room test exactly one synthetic native-model EOS workload."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

from build_paired_integration_package import build as build_overlay
from build_paired_source_projection import validate as validate_projection
from research.m4_eligible_replay.contracts import canonical_json, digest_file
from research.m4_eligible_replay.deterministic_backward import MARKER, REQUIRED_ENV
from research.m4_paired_replay.quarantine import validate_public_key


ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
SESSION = ROOT / "session/20260914_m4_opportunity_control"
BASE_HASH = "de2ee7b5c1b4437b1d6ed53b9f930a3bf96dca471f31aee780aea3844fab238e"
NAME = "m4-synthetic-paired-v0-engineering"


def build(
    *, materials: Path, output: Path, source_projection: Path | None = None
) -> dict:
    protocol_path = REPORT / "paired_engineering_protocol.json"
    protocol = json.loads(protocol_path.read_bytes())
    public_materials = json.loads(materials.read_bytes())
    if (
        set(public_materials)
        != {"schema", "public_pem_base64", "public_key_sha256", "wheels"}
        or public_materials["schema"] != "m4-engineering-public-materials-v1"
    ):
        raise ValueError("unexpected engineering public material fields")
    pem = base64.b64decode(public_materials["public_pem_base64"], validate=True)
    validate_public_key(pem)
    if (
        hashlib.sha256(pem).hexdigest() != protocol["privacy"]["public_key_sha256"]
        or public_materials["public_key_sha256"]
        != protocol["privacy"]["public_key_sha256"]
    ):
        raise ValueError("public escrow key does not match frozen engineering protocol")
    overlay_path = output.with_suffix(".overlay.json")
    overlay_receipt = build_overlay(output=overlay_path)
    frozen = json.loads(overlay_path.read_bytes())
    for path in (protocol_path,):
        frozen["overlay"][str(path.relative_to(ROOT))] = {
            "sha256": digest_file(path),
            "base64": base64.b64encode(path.read_bytes()).decode(),
        }
    base_path = SESSION / "m4-eligible-live-minimal-observer-verified-manifest.json"
    if digest_file(base_path) != BASE_HASH:
        raise ValueError("accepted base manifest changed")
    base = json.loads(base_path.read_bytes())
    old_script = base["spec"]["script"].replace("{{", "{").replace("}}", "}")
    prefix = old_script.split('cd "$RUN_REPO"\n', 1)[0]
    projection_proof = None
    if source_projection is not None:
        original = (
            SESSION
            / "m4-eligible-live-v2-liveness-qualification-source-11143be69.tar.gz"
        )
        projection_proof = validate_projection(original, source_projection)
        old_payload = base64.b64encode(original.read_bytes()).decode()
        new_payload = base64.b64encode(source_projection.read_bytes()).decode()
        old_hash = projection_proof["original_source_sha256"]
        if prefix.count(old_payload) != 1 or prefix.count(old_hash) != 1:
            raise ValueError("accepted source transport boundary changed")
        prefix = prefix.replace(old_payload, new_payload).replace(
            old_hash, projection_proof["projected_source_sha256"]
        )
    overlay_payload = json.dumps(
        frozen["overlay"], sort_keys=True, separators=(",", ":")
    )
    material_payload = json.dumps(
        public_materials, sort_keys=True, separators=(",", ":")
    )
    extraction = r"""import base64,hashlib,io,json,sys,zipfile
from pathlib import Path,PurePosixPath
root,crypto=map(Path,sys.argv[1:])
overlay=json.loads('@OVERLAY@')
for name,row in overlay.items():
    relative=PurePosixPath(name)
    assert not relative.is_absolute() and ".." not in relative.parts
    path=root/name; assert not path.is_symlink()
    data=base64.b64decode(row["base64"],validate=True)
    assert hashlib.sha256(data).hexdigest()==row["sha256"]
    path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
materials=json.loads('@MATERIALS@')
pem=base64.b64decode(materials["public_pem_base64"],validate=True)
assert pem.startswith(b"-----BEGIN PUBLIC KEY-----")
assert hashlib.sha256(pem).hexdigest()==materials["public_key_sha256"]
(root/"engineering-public.pem").write_bytes(pem)
crypto.mkdir(parents=True,exist_ok=False)
assert len(materials["wheels"])==3
for wheel in materials["wheels"]:
    data=base64.b64decode(wheel["base64"],validate=True)
    assert len(data)==wheel["bytes"] and hashlib.sha256(data).hexdigest()==wheel["sha256"]
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names=[PurePosixPath(i.filename) for i in archive.infolist()]
        assert len(names)==len(set(names)) and not any(n.is_absolute() or ".." in n.parts for n in names)
        assert not any((i.external_attr>>16)&0o170000==0o120000 for i in archive.infolist())
        archive.extractall(crypto)
print("PASS_PUBLIC_ONLY_OVERLAY_AND_PINNED_CRYPTO")
""".replace("@OVERLAY@", overlay_payload).replace("@MATERIALS@", material_payload)
    old_test_line = next(
        line for line in old_script.splitlines() if '"$PYTHON" -m unittest ' in line
    )
    modules = [
        w
        for w in shlex.split(old_test_line)
        if w.startswith("research.m4_eligible_replay.test_")
    ]
    modules += [
        "research.m4_paired_replay.test_package",
        "research.m4_paired_replay.test_integration",
        "research.m4_paired_replay.test_engineering",
    ]
    script = (
        prefix
        + '\nreadonly CRYPTO_ROOT=/workspace/m4-engineering-crypto\n"$PYTHON" - "$RUN_REPO" "$CRYPTO_ROOT" <<\'PY\'\n'
        + extraction
        + "PY\n"
    )
    script += r"""cd "$RUN_REPO"
export PYTHONPATH="$CRYPTO_ROOT:$RUN_REPO"
export CUDA_VISIBLE_DEVICES=0
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1
export VIRTUAL_ENV=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
export UV_PROJECT_ENVIRONMENT="$VIRTUAL_ENV"
export M4_PAIRED_SYNTHETIC_ENGINEERING=1
"$PYTHON" - <<'PY'
from pathlib import Path
import cryptography
from research.m4_paired_replay.quarantine import crypto
crypto()
assert cryptography.__version__=="48.0.1"
assert Path(cryptography.__file__).is_relative_to("/workspace/m4-engineering-crypto")
print("PASS_PATCHED_CRYPTO_RUNTIME")
PY
"$PYTHON" -m unittest @MODULES@ -v > "$ASSETS/tests.log" 2>&1
"$PYTHON" -m research.m4_eligible_replay.preflight --output "$ASSETS/environment-preflight.json" > "$ASSETS/environment-preflight.log" 2>&1
"$PYTHON" -m research.m4_paired_replay.tensor_preflight --output "$ASSETS/secondary-preflight.json"
set +e
"$PYTHON" -m research.m4_paired_replay.engineering --private-root /workspace/m4-paired-private/qualification --public "$ASSETS" --public-key "$RUN_REPO/engineering-public.pem" > /workspace/m4-engineering-controller.log 2>&1
runtime_rc=$?
set -e
find "$ASSETS" -maxdepth 1 -type f ! -name artifacts.sha256 -print0 | sort -z | xargs -0 sha256sum > "$ASSETS/artifacts.sha256"
echo "ENGINEERING_EXIT_CODE=$runtime_rc"
exit "$runtime_rc"
""".replace("@MODULES@", " ".join(modules))
    subprocess.run(
        ["bash", "-n"],
        input=script.replace("{assets_dir}", "/tmp/m4-engineering-assets"),
        text=True,
        check=True,
    )
    blocks = [block.split("\nPY\n", 1)[0] for block in script.split("<<'PY'\n")[1:]]
    for block in blocks:
        compile(block, "engineering-embedded.py", "exec")
    with tempfile.TemporaryDirectory(prefix="m4-synthetic-cleanroom-") as directory:
        root = Path(directory) / "repo"
        subprocess.run(
            [sys.executable, "-", str(root)],
            input=blocks[0],
            text=True,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            [sys.executable, "-", str(root), str(Path(directory) / "linux-crypto")],
            input=extraction,
            text=True,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        env = {**os.environ, "PYTHONPATH": str(root), MARKER: "1", **REQUIRED_ENV}
        code = "import json,sys,unittest; suite=unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:]); r=unittest.TextTestRunner().run(suite); print(json.dumps({'run':r.testsRun,'skipped':len(r.skipped),'failures':len(r.failures),'errors':len(r.errors)})); raise SystemExit(0 if r.wasSuccessful() else 1)"
        process = subprocess.run(
            [sys.executable, "-c", code, *modules],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        tests = json.loads(process.stdout.splitlines()[-1])
        subprocess.run(
            [sys.executable, "-m", "compileall", "-q", "research/m4_paired_replay"],
            cwd=root,
            env=env,
            check=True,
        )
    manifest = {
        **base,
        "spec": {
            **base["spec"],
            "name": NAME,
            "time_limit": 14400,
            "script": script.replace("{", "{{")
            .replace("}", "}}")
            .replace("{{assets_dir}}", "{assets_dir}"),
        },
    }
    with output.open("xb") as stream:
        stream.write(canonical_json(manifest))
    reference = json.loads(
        (REPORT / "eligible_live_minimal_observer_terminal_result.json").read_bytes()
    )
    guards = reference["guard_sha256"]
    if not all(
        digest_file(SESSION / name) == expected for name, expected in guards.items()
    ):
        raise ValueError("consumed predecessor guard changed")
    return {
        "schema": "m4-synthetic-engineering-package-v1",
        "status": "PASS_OFFLINE_UNSUBMITTED",
        "manifest_sha256": digest_file(output),
        "workload_name": NAME,
        "protocol_sha256": digest_file(protocol_path),
        "integration_overlay_sha256": overlay_receipt["overlay_sha256"],
        "source_sha256": {
            name: row["sha256"] for name, row in frozen["overlay"].items()
        },
        "clean_room_tests": tests,
        "prior_guard_sha256": guards,
        "public_key_sha256": public_materials["public_key_sha256"],
        "crypto_wheel_sha256": {
            w["filename"]: w["sha256"] for w in public_materials["wheels"]
        },
        "custom_config_sha256": digest_file(
            REPORT / "eligible_live_restore_eos_custom_config_deadline_suppressed.yaml"
        ),
        "variables_sha256": digest_file(
            REPORT / "eligible_live_capture_variables.yaml"
        ),
        "private_key_uploaded": False,
        "scientific_outcomes_accessed": False,
        "new_launcher_invocations": 0,
        "gpu_qualification_passed": False,
        "source_projection": projection_proof,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-projection", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                materials=args.materials,
                output=args.output,
                source_projection=args.source_projection,
            ),
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
