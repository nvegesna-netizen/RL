"""Check the complete proven MCORE driver and isolated actor before input retrieval."""

from __future__ import annotations

import argparse
import importlib
from importlib.metadata import distribution
import json
import os
from pathlib import Path
import sys

from research.m4_eligible_replay.bootstrap import initialize_bundled_megatron
from research.m4_eligible_replay.contracts import canonical_json


MCORE_PREFIX = Path(
    "/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
)
MCORE_PYTHON = MCORE_PREFIX / "bin/python"
TRANSFERQUEUE_COMMIT = "b266d39a15aae114730de36cf8317b6285436f7f"


def inspect_environment() -> dict:
    """Import real backend modules and verify interpreter, origins, and pinned TQ."""
    if Path(sys.prefix) != MCORE_PREFIX:
        raise ValueError(f"expected proven MCORE prefix, got {sys.prefix}")
    core = initialize_bundled_megatron(require_bridge=True)
    bridge = core.parent.parent / "src"
    modules = {
        name: importlib.import_module(name)
        for name in (
            "torch",
            "transformer_engine.pytorch",
            "megatron.core",
            "megatron.bridge.training.checkpointing",
            "nemo_rl.models.policy.tq_policy",
            "research.m4_eligible_replay.worker",
        )
    }
    if (
        Path(modules["megatron.core"].__file__).resolve()
        != core / "megatron/core/__init__.py"
    ):
        raise ValueError("Core import did not use the pinned bundle")
    expected_bridge = bridge / "megatron/bridge/training/checkpointing.py"
    if (
        Path(modules["megatron.bridge.training.checkpointing"].__file__).resolve()
        != expected_bridge
    ):
        raise ValueError("Bridge import did not use the pinned bundle")
    direct = json.loads(distribution("TransferQueue").read_text("direct_url.json"))
    if direct["vcs_info"]["commit_id"] != TRANSFERQUEUE_COMMIT:
        raise ValueError("MCORE TransferQueue differs from the frozen runtime")
    torch = modules["torch"]
    digest = modules["research.m4_eligible_replay.worker"].state_digest
    original = {"x": torch.tensor([1.0, 2.0])}
    if digest(original) != digest({"x": torch.tensor([1.0, 2.0])}) or digest(
        original
    ) == digest({"x": torch.tensor([1.0, 3.0])}):
        raise ValueError("state digest smoke check failed")
    registry = importlib.import_module(
        "nemo_rl.distributed.ray_actor_environment_registry"
    )
    if registry.MCORE_EXECUTABLE != sys.executable:
        raise ValueError("actor environment must use the verified MCORE interpreter")
    return {
        "python": sys.executable,
        "prefix": sys.prefix,
        "torch_version": torch.__version__,
        "core_path": str(core),
        "bridge_checkpointing_path": str(expected_bridge),
        "transferqueue_commit": TRANSFERQUEUE_COMMIT,
        "full_worker_import": True,
        "state_digest_smoke": True,
    }


def run(output: Path) -> None:
    """Check real modules in driver and a fresh Ray initializer-like process."""
    driver = inspect_environment()
    # Imports are intentionally delayed until bundled sources are initialized.
    import ray
    from nemo_rl.distributed.virtual_cluster import init_ray

    init_ray(log_dir="/workspace/m4-full-environment-preflight-ray")
    try:
        probe = ray.remote(num_cpus=1, num_gpus=0)(inspect_environment)
        actor = ray.get(
            probe.options(
                runtime_env={
                    "py_executable": str(MCORE_PYTHON),
                    "env_vars": {
                        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
                        "NEMO_RL_PY_EXECUTABLES_SYSTEM": "1",
                        "VIRTUAL_ENV": str(MCORE_PREFIX),
                        "UV_PROJECT_ENVIRONMENT": str(MCORE_PREFIX),
                    },
                }
            ).remote(),
            timeout=300,
        )
        for key in (
            "prefix",
            "torch_version",
            "core_path",
            "bridge_checkpointing_path",
            "transferqueue_commit",
        ):
            if driver[key] != actor[key]:
                raise ValueError(f"driver/actor mismatch: {key}")
        result = {
            "status": "PASS_FULL_DRIVER_AND_ACTOR_IMPORT_PREFLIGHT",
            "driver": driver,
            "actor": actor,
            "checkpoint_download_started": False,
            "replay_started": False,
        }
        with output.open("xb") as stream:
            stream.write(canonical_json(result))
        print(result["status"])
    finally:
        ray.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("NEMO_RL_PY_EXECUTABLES_SYSTEM") != "1":
        raise ValueError("explicit unified MCORE actor selection required")
    run(args.output)


if __name__ == "__main__":
    main()
