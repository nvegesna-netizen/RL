"""Run one duplicate comparator update; never evaluate a historical arm contrast."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import ray
import torch
from omegaconf import OmegaConf

from nemo_rl.algorithms.async_utils.conditional_m4_capsule import tensor_content_sha256
from nemo_rl.algorithms.loss import ClippedPGLossFn
from nemo_rl.algorithms.single_controller_utils.config import MasterConfig
from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data_plane.column_io import kv_first_write
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
    MCORE_EXECUTABLE,
)
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.models.policy.tq_policy import TQPolicy
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from research.m4_eligible_replay.contracts import (
    allowed_members,
    canonical_json,
    digest_file,
)
from research.m4_eligible_replay.ray_startup import (
    allocate_ray_root,
    ray_cleanup,
    validate_ray_root,
)
from research.m4_eligible_replay.state_diagnostics import write_diagnostic


WORKER = "research.m4_eligible_replay.worker.EligibleLiveReplayWorker"


def scalar(value: Any) -> float:
    """Convert and reject nonfinite scalar metrics."""
    result = float(value.item()) if isinstance(value, torch.Tensor) else float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite qualification loss")
    return result


def run(
    *,
    capsule: Path,
    config_path: Path,
    protocol_path: Path,
    replicate: str,
    ray_log_dir: Path | None = None,
    state_diagnostics_dir: Path | None = None,
    training_diagnostics: bool = False,
) -> dict[str, Any]:
    """Cover all setup failures with Ray cleanup and validate the socket budget."""
    if replicate not in ("a", "b"):
        raise ValueError("wrong qualification replicate")
    if training_diagnostics and state_diagnostics_dir is None:
        raise ValueError("training diagnostics requires a persistent state directory")
    root = allocate_ray_root(replicate) if ray_log_dir is None else ray_log_dir
    size = validate_ray_root(root)
    print(
        json.dumps(
            {
                "stage": "RAY_ROOT_VALIDATED",
                "replicate": replicate,
                "ray_root": str(root),
                "socket_budget_bytes": size,
            }
        ),
        flush=True,
    )
    with ray_cleanup(ray.shutdown):
        return _run(
            capsule=capsule,
            config_path=config_path,
            protocol_path=protocol_path,
            replicate=replicate,
            ray_log_dir=root,
            state_diagnostics_dir=state_diagnostics_dir,
            training_diagnostics=training_diagnostics,
        )


def _run(
    *,
    capsule: Path,
    config_path: Path,
    protocol_path: Path,
    replicate: str,
    ray_log_dir: Path,
    state_diagnostics_dir: Path | None = None,
    training_diagnostics: bool = False,
) -> dict[str, Any]:
    """Restore, forward-evaluate withheld data, and update only the comparator."""
    protocol = json.loads(protocol_path.read_bytes())
    if protocol[
        "schema"
    ] != "m4-eligible-live-duplicate-update-qualification-v1" or replicate not in (
        "a",
        "b",
    ):
        raise ValueError("wrong qualification protocol or replicate")
    source = protocol["source_capsule"]
    manifest_path = capsule / "manifest.json"
    if digest_file(manifest_path) != source["manifest_sha256"]:
        raise ValueError("wrong historical capsule")
    manifest = json.loads(manifest_path.read_bytes())
    expected_members = allowed_members(manifest)
    expected_paths = {row["path"] for row in expected_members}
    actual_paths = {
        str(path.relative_to(capsule)) for path in capsule.rglob("*") if path.is_file()
    } - {"manifest.json", "transport-receipt.json"}
    if actual_paths != expected_paths or any(
        path.is_symlink() for path in capsule.rglob("*")
    ):
        raise ValueError("capsule does not match the qualification-only allowlist")
    if any(
        digest_file(capsule / row["path"]) != row["sha256"] for row in expected_members
    ):
        raise ValueError("capsule member failed authentication")

    def load_groups(role: str) -> tuple[list[dict], list[str], list[str]]:
        rows = [row for row in manifest["groups"] if role in row["roles"]]
        if len(rows) != 4:
            raise ValueError("expected four frozen groups per role")
        batches, samples, groups = [], [], []
        for row in rows:
            fields = torch.load(
                capsule / row["raw_file"], weights_only=True, map_location="cpu"
            )
            if tensor_content_sha256(fields) != row["tensor_content_sha256"]:
                raise ValueError("group tensor content changed")
            batches.append(fields)
            samples.extend(row["sample_ids"])
            groups.append(row["group_id"])
        if len(samples) != 32 or len(set(samples)) != 32:
            raise ValueError("qualification sample count or uniqueness differs")
        return batches, samples, groups

    base_batches, base_samples, base_groups = load_groups("base")
    heldout_batches, heldout_samples, heldout_groups = load_groups("heldout")
    if set(base_samples) & set(heldout_samples):
        raise ValueError("comparator and held-out samples overlap")
    heldout_hashes = {
        row["raw_file"]: row["raw_file_sha256"]
        for row in manifest["groups"]
        if row["roles"] == ["heldout"]
    }
    register_omegaconf_resolvers()
    config = MasterConfig(
        **OmegaConf.to_container(load_config(config_path), resolve=True)
    )
    config.policy["megatron_cfg"]["train_iters"] = 64
    if training_diagnostics:
        if state_diagnostics_dir is None:
            raise ValueError(
                "training diagnostics requires a persistent state directory"
            )
        config.policy["eligible_replay_training_trace"] = {
            "output": str(
                (
                    state_diagnostics_dir / f"replay-{replicate}-training-trace.jsonl"
                ).resolve()
            ),
            "replicate": replicate,
            "protocol_sha256": digest_file(protocol_path),
            "capsule_manifest_sha256": source["manifest_sha256"],
        }
    if (
        config.policy["model_name"] != "meta-llama/Llama-3.2-1B-Instruct"
        or config.policy["train_global_batch_size"] != 32
        or config.policy["train_micro_batch_size"] != 1
    ):
        raise ValueError("frozen model or batch configuration differs")
    for key in (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
    ):
        if config.policy["megatron_cfg"][key] != 1:
            raise ValueError("qualification requires single-worker parallelism")
    ACTOR_ENVIRONMENT_REGISTRY[WORKER] = MCORE_EXECUTABLE
    set_seed(20261101)
    init_ray(log_dir=str(ray_log_dir))
    tokenizer = get_tokenizer(config.policy["tokenizer"])
    cluster = RayVirtualCluster(
        name=f"m4_replay_qualification_{replicate}",
        bundle_ct_per_node_list=[1],
        use_gpus=True,
        num_gpus_per_node=1,
        max_colocated_worker_groups=1,
        port_range_low=config.cluster.get("master_port_range_low"),
        port_range_high=config.cluster.get("master_port_range_high"),
    )
    policy = None
    metas = []
    try:
        weights = capsule / "checkpoint/policy/weights"
        policy = TQPolicy(
            cluster=cluster,
            config=config.policy,
            tokenizer=tokenizer,
            weights_path=str(weights),
            optimizer_path=str(weights),
            init_optimizer=True,
            init_reference_model=False,
            dp_cfg=config.data_plane,
            tq_partition_id=f"qualification-{replicate}",
            worker_extension_cls_fqn=WORKER,
        )
        policy.prepare_step(num_samples=32, group_size=8)
        for label, batches, samples in (
            ("heldout", heldout_batches, heldout_samples),
            ("base", base_batches, base_samples),
        ):
            combined = BatchedDataDict.from_batches(
                batches, pad_value_dict={"input_ids": int(tokenizer.pad_token_id)}
            )
            metas.append(
                kv_first_write(
                    combined,
                    sample_ids=samples,
                    dp_client=policy.dp_client,
                    partition_id=policy.tq_partition_id,
                    task_name="train",
                    tags=[{"role": label} for _ in samples],
                )
            )
        policy.prepare_for_training()

        def snapshot() -> dict:
            states = policy.run_all_workers_single_data("replay_state")
            if len(states) != 1:
                raise ValueError("expected exactly one worker snapshot")
            return states[0]

        before = snapshot()

        def record_state(stage: str, states: dict[str, dict]) -> None:
            if state_diagnostics_dir is not None:
                write_diagnostic(
                    state_diagnostics_dir,
                    replicate=replicate,
                    stage=stage,
                    states=states,
                    protocol_hash=digest_file(protocol_path),
                    manifest_hash=source["manifest_sha256"],
                )

        record_state("before", {"before": before})
        if (
            before["parameter_sha256"] != source["captured_parameter_sha256"]
            or before["optimizer_step"] != 7
            or before["scheduler_num_steps"] != 224
        ):
            raise ValueError("captured training state was not fully restored")
        idle = snapshot() if state_diagnostics_dir is not None else before
        record_state("idle", {"idle": idle})
        evaluation = policy.train_from_meta(
            metas[0], ClippedPGLossFn(config.loss_fn), eval_mode=True, gbs=32, mbs=1
        )
        pre_loss = scalar(evaluation["loss"])
        before_update = snapshot()
        record_state("after-eval", {"after_eval": before_update})
        record_state(
            "comparison", {"before": before, "idle": idle, "after_eval": before_update}
        )
        if before != before_update:
            raise ValueError("held-out evaluation mutated training state")
        # Discard comparator training metrics; do not compute a held-out
        # post-update loss or run the Shield action in qualification.
        policy.train_from_meta(
            metas[1], ClippedPGLossFn(config.loss_fn), eval_mode=False, gbs=32, mbs=1
        )
        after = snapshot()
        if (
            after["optimizer_step"] - before_update["optimizer_step"] != 1
            or after["scheduler_num_steps"] - before_update["scheduler_num_steps"] != 32
        ):
            raise ValueError("one-update execution counters differ")
        if after["parameter_sha256"] == before_update["parameter_sha256"]:
            raise ValueError("comparator update did not change parameters")
        unchanged = all(
            digest_file(capsule / name) == expected
            for name, expected in heldout_hashes.items()
        )
        if not unchanged:
            raise ValueError("held-out tensor files changed")
        return {
            "replicate": replicate,
            "protocol_sha256": digest_file(protocol_path),
            "capsule_manifest_sha256": source["manifest_sha256"],
            "action": "reward_variance_base",
            "base_group_ids": base_groups,
            "heldout_group_ids": heldout_groups,
            "state_before": before,
            "state_before_update": before_update,
            "state_after": after,
            "pre_update_heldout_loss": pre_loss,
            "optimizer_updates": after["optimizer_step"]
            - before_update["optimizer_step"],
            "scheduler_sample_increment": after["scheduler_num_steps"]
            - before_update["scheduler_num_steps"],
            "optimizer_restored": before["load_optim"],
            "scheduler_restored": before["scheduler_num_steps"] == 224,
            "rng_restored": before["load_rng"],
            "heldout_files_unchanged": unchanged,
            "post_update_heldout_loss_computed": False,
            "historical_capsule_excluded": True,
        }
    finally:
        if policy is not None:
            try:
                for meta in metas:
                    policy.dp_client.clear_samples(
                        sample_ids=list(meta.sample_ids), partition_id=meta.partition_id
                    )
            finally:
                policy.shutdown()
        cluster.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--replicate", choices=("a", "b"), required=True)
    parser.add_argument("--ray-log-dir", type=Path)
    parser.add_argument("--state-diagnostics-dir", type=Path)
    parser.add_argument("--training-diagnostics", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        capsule=args.capsule_dir,
        config_path=args.config,
        protocol_path=args.protocol,
        replicate=args.replicate,
        ray_log_dir=args.ray_log_dir,
        state_diagnostics_dir=args.state_diagnostics_dir,
        training_diagnostics=args.training_diagnostics,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(canonical_json(result))
    print(
        json.dumps(
            {
                "replicate": args.replicate,
                "status": "COMPLETED_QUALIFICATION_UPDATE",
                "post_update_heldout_loss_computed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
