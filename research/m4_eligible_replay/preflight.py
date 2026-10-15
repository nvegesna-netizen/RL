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
from research.m4_eligible_replay.evaluation_rng import isolate_evaluation_tracker
from research.m4_eligible_replay.ray_startup import (
    allocate_ray_root,
    ray_cleanup,
    validate_ray_root,
)


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
            "research.m4_eligible_replay.run_qualification",
            "ray",
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
    # Optional backend is available here; use a separate Core tracker, never
    # initialize or replace the model's global tracker during preflight.
    from megatron.core.tensor_parallel.random import CudaRNGStatesTracker

    tracker = CudaRNGStatesTracker()
    tracker.set_states({"smoke": torch.tensor([1, 2], dtype=torch.uint8)})
    expected = digest(tracker.get_states())
    for exceptional in (False, True):
        failure = RuntimeError("intentional RNG cleanup smoke")
        try:
            with isolate_evaluation_tracker(
                tracker, clone_state=lambda value: value.clone(), digest=digest
            ):
                tracker.get_states()["smoke"].add_(1)
                if exceptional:
                    raise failure
        except RuntimeError as error:
            if not exceptional or error is not failure:
                raise
        if digest(tracker.get_states()) != expected:
            raise ValueError("real tensor RNG restoration smoke failed")
    registry = importlib.import_module(
        "nemo_rl.distributed.ray_actor_environment_registry"
    )
    if registry.MCORE_EXECUTABLE != sys.executable:
        raise ValueError("actor environment must use the verified MCORE interpreter")
    return {
        "python": sys.executable,
        "prefix": sys.prefix,
        "torch_version": torch.__version__,
        "ray_version": modules["ray"].__version__,
        "python_version": sys.version,
        "core_path": str(core),
        "bridge_checkpointing_path": str(expected_bridge),
        "transferqueue_commit": TRANSFERQUEUE_COMMIT,
        "full_worker_import": True,
        "qualification_module_import": True,
        "state_digest_smoke": True,
        "real_tensor_tracker_restore_smoke": True,
    }


def run(output: Path) -> None:
    """Check real modules in driver and a fresh Ray initializer-like process."""
    driver = inspect_environment()
    # Imports are intentionally delayed until bundled sources are initialized.
    import ray
    from nemo_rl.distributed.virtual_cluster import init_ray

    ray_root = allocate_ray_root("preflight")
    print(
        json.dumps(
            {
                "stage": "PASS_DRIVER_IMPORT",
                "driver": driver,
                "ray_root": str(ray_root),
                "socket_budget_bytes": validate_ray_root(ray_root),
            }
        ),
        flush=True,
    )
    with ray_cleanup(ray.shutdown):
        init_ray(log_dir=str(ray_root))
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
            "ray_version",
            "python_version",
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
            "ray_root": str(ray_root),
            "socket_budget_bytes": validate_ray_root(ray_root),
        }
        with output.open("xb") as stream:
            stream.write(canonical_json(result))
        print(result["status"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("NEMO_RL_PY_EXECUTABLES_SYSTEM") != "1":
        raise ValueError("explicit unified MCORE actor selection required")
    run(args.output)


if __name__ == "__main__":
    main()
