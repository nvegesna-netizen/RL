"""Package one engineering workload using the qualified source plus explicit overlays."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import tarfile
from pathlib import Path

from research.m4_eligible_replay.contracts import canonical_json, digest_file


REPORT = Path(__file__).resolve().parent
REPO = REPORT.parents[2]
SOURCE_HASH = "ff8cb7f09b64b6ff5d781c12c52d566c8c8821ebab559d569e322ec2fabe691f"
MEGATRON_HASH = "98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
BRIDGE_HASH = "1429945d1d50045900e40314fa283fa5f66484e945aad4920da137d7ad2c9313"
BRIDGE_COMMIT = "573e088c9c6740082c39744e03dc5b009e730ed4"
NAME = "m4-eligible-live-duplicate-update-qualification"
CONFIG = "examples/configs/grpo_math_1B_megatron_single_controller_m4_eligible_live_capture.yaml"
PROTOCOL = "reports/auto_research/2026-10-07-m4-causal-utility/eligible_live_duplicate_update_qualification_protocol.json"


def build(
    *,
    source: Path,
    megatron: Path,
    output: Path,
    bootstrap_replacement: bool = False,
    full_environment: bool = False,
    bridge: Path | None = None,
    ray_startup_repair: bool = False,
) -> dict:
    """Embed hash-bound source and overlays, then test expanded shell syntax."""
    if ray_startup_repair and not full_environment:
        raise ValueError("Ray startup repair requires the full pinned environment")
    if digest_file(source) != SOURCE_HASH or digest_file(megatron) != MEGATRON_HASH:
        raise ValueError("qualified base archives changed")
    if full_environment:
        if bridge is None or digest_file(bridge) != BRIDGE_HASH:
            raise ValueError("qualified Bridge archive missing or changed")
        with tarfile.open(bridge, "r:gz") as archive:
            if archive.pax_headers.get("comment") != BRIDGE_COMMIT:
                raise ValueError("Bridge archive does not match the qualified gitlink")
    paths = [
        str(path.relative_to(REPO))
        for path in sorted((REPO / "research/m4_eligible_replay").glob("*.py"))
    ] + [PROTOCOL]
    overlay = {
        name: {
            "sha256": digest_file(REPO / name),
            "base64": base64.b64encode((REPO / name).read_bytes()).decode(),
        }
        for name in paths
    }
    script = r"""set -euo pipefail
readonly PYTHON=/opt/nemo_rl_venv/bin/python
readonly RUN_REPO=/workspace/m4-duplicate-update-repo
readonly CAPSULE=/workspace/m4-duplicate-update-input-capsule
readonly ASSETS={assets_dir}
test "${NEMO_RL_COMMIT:-unknown}" = "ae07eafe8035b5b2e84efa7234e70e7fd7e493c1"
test -n "${CI_JOB_TOKEN:-}"
test ! -e "$RUN_REPO"
test ! -e "$CAPSULE"
"$PYTHON" - "$RUN_REPO" <<'PY'
import base64,hashlib,io,json,sys,tarfile
from pathlib import Path,PurePosixPath
root=Path(sys.argv[1])
for payload,expected,destination in (("@SOURCE@","@SOURCE_HASH@",root),("@MEGATRON@","@MEGATRON_HASH@",root/"3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty")):
    data=base64.b64decode(payload); assert hashlib.sha256(data).hexdigest()==expected
    with tarfile.open(fileobj=io.BytesIO(data),mode="r:gz") as archive:
        members=archive.getmembers(); names=[PurePosixPath(m.name) for m in members]
        assert len(names)==len(set(names)) and not any(n.is_absolute() or ".." in n.parts for n in names)
        links={n for n,m in zip(names,members) if m.issym()}
        assert not any(any(p in links for p in n.parents) for n in names)
        destination.mkdir(parents=True,exist_ok=True); archive.extractall(destination,filter="data")
overlays=json.loads('@OVERLAY@')
for name,row in overlays.items():
    path=root/name; path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists(); data=base64.b64decode(row["base64"])
    assert hashlib.sha256(data).hexdigest()==row["sha256"]; path.write_bytes(data)
print("PASS_AUTHENTICATED_BASE_AND_REPLAY_OVERLAY")
PY
cd "$RUN_REPO"
export PYTHONPATH="$RUN_REPO"
export CUDA_VISIBLE_DEVICES=0
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1
"$PYTHON" -m unittest research.m4_eligible_replay.test_contracts -v > "$ASSETS/tests.log" 2>&1
"$PYTHON" - <<'PY'
import torch
from research.m4_eligible_replay.worker import state_digest
assert state_digest({"x":torch.tensor([1.,2.])})==state_digest({"x":torch.tensor([1.,2.])})
assert state_digest({"x":torch.tensor([1.,2.])})!=state_digest({"x":torch.tensor([1.,3.])})
print("PASS_TRAINING_STATE_DIGEST_SMOKE")
PY
"$PYTHON" -m research.m4_eligible_replay.transport --protocol "@PROTOCOL@" --output "$CAPSULE" --job-token > "$ASSETS/transport.log" 2>&1
cp "$CAPSULE/transport-receipt.json" "$ASSETS/transport-receipt.json"
for replicate in a b; do
  "$PYTHON" -m research.m4_eligible_replay.run_qualification --capsule-dir "$CAPSULE" --config "@CONFIG@" --protocol "@PROTOCOL@" --replicate "$replicate" --ray-log-dir "/workspace/m4-duplicate-ray-$replicate" --output "$ASSETS/replay-$replicate.json" > "$ASSETS/replay-$replicate.log" 2>&1
done
"$PYTHON" - "$ASSETS" "@PROTOCOL@" <<'PY'
import json,sys
from pathlib import Path
from research.m4_eligible_replay.contracts import canonical_json,digest_file,validate_duplicate_replays
assets,protocol=map(Path,sys.argv[1:]); p=json.loads(protocol.read_bytes())
records=[json.loads((assets/f"replay-{label}.json").read_bytes()) for label in ("a","b")]
result=validate_duplicate_replays(records,protocol_hash=digest_file(protocol),manifest_hash=p["source_capsule"]["manifest_sha256"])
result["protocol_sha256"]=digest_file(protocol)
result["records_sha256"]={label:digest_file(assets/f"replay-{label}.json") for label in ("a","b")}
(assets/"qualification-result.json").write_bytes(canonical_json(result))
print(result["status"])
PY
find "$ASSETS" -maxdepth 1 -type f ! -name artifacts.sha256 -print0 | sort -z | xargs -0 sha256sum > "$ASSETS/artifacts.sha256"
"""
    if bootstrap_replacement:
        script = script.replace(
            '"$PYTHON" -m unittest research.m4_eligible_replay.test_contracts -v',
            '"$PYTHON" -m unittest research.m4_eligible_replay.test_contracts research.m4_eligible_replay.test_bootstrap -v',
        )
    if full_environment:
        script = script.replace(
            "readonly PYTHON=/opt/nemo_rl_venv/bin/python",
            'readonly PYTHON=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python\ntest -x "$PYTHON"',
        )
        original = '(("@SOURCE@","@SOURCE_HASH@",root),("@MEGATRON@","@MEGATRON_HASH@",root/"3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty"))'
        complete = '(("@SOURCE@","@SOURCE_HASH@",root),("@BRIDGE@","@BRIDGE_HASH@",root/"3rdparty/Megatron-Bridge-workspace"),("@MEGATRON@","@MEGATRON_HASH@",root/"3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty"))'
        if original not in script:
            raise ValueError("source extraction template moved")
        script = script.replace(original, complete)
        script = script.replace(
            "export NEMO_RL_PY_EXECUTABLES_SYSTEM=1",
            'export NEMO_RL_PY_EXECUTABLES_SYSTEM=1\nexport VIRTUAL_ENV=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker\nexport UV_PROJECT_ENVIRONMENT="$VIRTUAL_ENV"',
        )
        begin = script.index('"$PYTHON" -m unittest')
        end = script.index('"$PYTHON" -m research.m4_eligible_replay.transport', begin)
        script = (
            script[:begin]
            + """"$PYTHON" -m unittest research.m4_eligible_replay.test_contracts research.m4_eligible_replay.test_bootstrap research.m4_eligible_replay.test_preflight -v > "$ASSETS/tests.log" 2>&1
"$PYTHON" -m research.m4_eligible_replay.preflight --output "$ASSETS/environment-preflight.json" > "$ASSETS/environment-preflight.log" 2>&1
"""
            + script[end:]
        )
    if ray_startup_repair:
        script = script.replace(
            ' --ray-log-dir "/workspace/m4-duplicate-ray-$replicate"', ""
        )
        script = script.replace(
            '"$PYTHON" -m unittest',
            '"$PYTHON" -m research.m4_eligible_replay.ray_startup > "$ASSETS/ray-layout.json"\n"$PYTHON" -m unittest',
            1,
        ).replace(
            "research.m4_eligible_replay.test_preflight -v",
            "research.m4_eligible_replay.test_preflight research.m4_eligible_replay.test_ray_startup -v",
        )
    replacements = {
        "@SOURCE@": base64.b64encode(source.read_bytes()).decode(),
        "@SOURCE_HASH@": SOURCE_HASH,
        "@MEGATRON@": base64.b64encode(megatron.read_bytes()).decode(),
        "@MEGATRON_HASH@": MEGATRON_HASH,
        "@OVERLAY@": json.dumps(overlay, sort_keys=True, separators=(",", ":")),
        "@PROTOCOL@": PROTOCOL,
        "@CONFIG@": CONFIG,
    }
    if full_environment:
        replacements.update(
            {
                "@BRIDGE@": base64.b64encode(bridge.read_bytes()).decode(),
                "@BRIDGE_HASH@": BRIDGE_HASH,
            }
        )
    for marker, value in replacements.items():
        script = script.replace(marker, value)
    subprocess.run(
        ["bash", "-n"],
        input=script.replace("{assets_dir}", "/tmp/m4-qualification-assets"),
        text=True,
        check=True,
    )
    for block in script.split("<<'PY'\n")[1:]:
        compile(block.split("\nPY\n", 1)[0], "embedded_runtime.py", "exec")
    escaped = (
        script.replace("{", "{{")
        .replace("}", "}}")
        .replace("{{assets_dir}}", "{assets_dir}")
    )
    manifest = {
        "type": "basic",
        "format_version": 1,
        "maintainers": ["nvegesna"],
        "loggers": ["stdout"],
        "labels": {"target": "silicon"},
        "launchers": {"type:slurm": {"nodes": 1, "ntasks_per_node": 1}},
        "spec": {
            "name": NAME
            + (
                "-ray-startup-repair"
                if ray_startup_repair
                else (
                    "-full-environment"
                    if full_environment
                    else ("-bootstrap-replacement" if bootstrap_replacement else "")
                )
            ),
            "workspace": "/workspace",
            "nodes": 1,
            "time_limit": 14400,
            "image_source": {
                "local_path": "/lustre/fsw/coreai_dlalgo_ci/nvegesna/nemo_rl_images/nemo-rl-nightly-5802754.sqsh"
            },
            "script": escaped,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(canonical_json(manifest))
    result = {
        "status": "PASS_STATIC_PACKAGE_CHECKS_UNSUBMITTED",
        "name": manifest["spec"]["name"],
        "manifest_sha256": digest_file(output),
        "source_archive_sha256": SOURCE_HASH,
        "megatron_archive_sha256": MEGATRON_HASH,
        "protocol_sha256": digest_file(REPO / PROTOCOL),
        "overlay_sha256": {name: row["sha256"] for name, row in overlay.items()},
        "capsule_outside_assets": True,
        "four_hour_allocated_limit": True,
        "new_eos_launches": 0,
    }
    if full_environment:
        result.update(
            {
                "bridge_archive_sha256": BRIDGE_HASH,
                "bridge_commit": BRIDGE_COMMIT,
                "driver_and_actor_environment": "proven prefetched MCORE",
                "full_import_preflight_before_transport": True,
            }
        )
    if ray_startup_repair:
        result["ray_startup_repair"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replacement", action="store_true")
    parser.add_argument("--full-environment", action="store_true")
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--ray-startup-repair", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                source=args.source,
                megatron=args.megatron,
                output=args.output,
                bootstrap_replacement=args.bootstrap_replacement,
                full_environment=args.full_environment,
                bridge=args.bridge,
                ray_startup_repair=args.ray_startup_repair,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
