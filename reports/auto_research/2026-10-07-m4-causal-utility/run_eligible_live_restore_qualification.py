#!/usr/bin/env python3
"""Restore one eligible-live checkpoint and evaluate held-out loss forward-only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from nemo_rl.algorithms.async_utils.conditional_m4_capsule import (
    tensor_content_sha256,
)
from nemo_rl.algorithms.loss import ClippedPGLossFn
from nemo_rl.algorithms.single_controller_utils.config import MasterConfig
from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data_plane.column_io import kv_first_write
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.models.policy.tq_policy import TQPolicy
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


def canonical_json(value: Any) -> bytes:
    """Serialize a compact record deterministically."""
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def file_sha256(path: Path, *, chunk_bytes: int = 16 << 20) -> str:
    """Hash a potentially large file without reading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _as_number(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"expected scalar tensor, got {tuple(value.shape)}")
        return float(value.detach().cpu().item())
    return float(value)


def _metric_range(metrics: dict[str, Any], key: str) -> dict[str, float] | None:
    values = metrics.get("all_mb_metrics", {}).get(key)
    if not values:
        return None
    numbers = [_as_number(value) for value in values]
    if not all(math.isfinite(value) for value in numbers):
        raise ValueError(f"nonfinite metric {key}")
    return {"minimum": min(numbers), "maximum": max(numbers)}


def authenticate_capsule_subset(
    capsule_dir: Path, protocol: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, torch.Tensor]], list[str], list[str]]:
    """Authenticate the full checkpoint and only the four allowed held-out files."""
    source = protocol["source_capsule"]
    allowlist = protocol["download_allowlist"]
    manifest_path = capsule_dir / "manifest.json"
    if file_sha256(manifest_path) != source["manifest_sha256"]:
        raise ValueError("capsule manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_bytes())
    if (
        manifest.get("schema") != "m4-shield-eligible-live-capsule-v1"
        or manifest.get("status") != "QUALIFIED_OUTCOME_EXCLUDED_ELIGIBLE_CAPTURE"
        or manifest.get("checkpoint_authentication_sha256")
        != source["checkpoint_authentication_sha256"]
        or manifest.get("group_authentication_sha256")
        != source["group_authentication_sha256"]
        or manifest.get("parameter_sha256_before")
        != source["captured_parameter_sha256"]
        or manifest.get("parameter_sha256_after") != source["captured_parameter_sha256"]
        or manifest.get("learner_version") != source["learner_version"]
        or manifest.get("captured_update_executed") is not False
        or manifest.get("post_update_outcomes_opened") is not False
    ):
        raise ValueError("capsule manifest violates the frozen source contract")

    checkpoint_root = capsule_dir / manifest["checkpoint_root"]
    expected_checkpoint = {row["path"]: row for row in manifest["checkpoint_files"]}
    actual_checkpoint = {
        str(path.relative_to(checkpoint_root))
        for path in checkpoint_root.rglob("*")
        if path.is_file()
    }
    if actual_checkpoint != set(expected_checkpoint):
        raise ValueError("downloaded checkpoint inventory is not exact")
    for relative, row in expected_checkpoint.items():
        path = checkpoint_root / relative
        if path.is_symlink() or path.stat().st_size != row["bytes"]:
            raise ValueError(f"unsafe or wrong-sized checkpoint member: {relative}")
        if file_sha256(path) != row["sha256"]:
            raise ValueError(f"checkpoint member SHA-256 mismatch: {relative}")

    indices = list(allowlist["heldout_group_indices"])
    if indices != [8, 9, 10, 11]:
        raise ValueError("held-out allowlist changed")
    if set(indices) & set(allowlist["frontier_group_indices_forbidden"]):
        raise ValueError("held-out and forbidden frontier indices overlap")
    groups_root = capsule_dir / "groups"
    expected_group_files = {f"group-{index:02d}.pt" for index in indices}
    actual_group_files = {path.name for path in groups_root.iterdir() if path.is_file()}
    if actual_group_files != expected_group_files:
        raise ValueError("downloaded group inventory is not heldout-only")

    batches: list[dict[str, torch.Tensor]] = []
    sample_ids: list[str] = []
    heldout_group_ids: list[str] = []
    for index in indices:
        row = manifest["groups"][index]
        if row["index"] != index or row["roles"] != ["heldout"]:
            raise ValueError(f"group {index} is not exclusively held out")
        path = capsule_dir / row["raw_file"]
        if path.name not in expected_group_files or path.is_symlink():
            raise ValueError(f"invalid held-out group path: {path}")
        if file_sha256(path) != row["raw_file_sha256"]:
            raise ValueError(f"held-out group SHA-256 mismatch: {index}")
        fields = torch.load(path, map_location="cpu", weights_only=True)
        if not all(isinstance(value, torch.Tensor) for value in fields.values()):
            raise TypeError(f"held-out group {index} contains a non-tensor field")
        if tensor_content_sha256(fields) != row["tensor_content_sha256"]:
            raise ValueError(f"held-out tensor-content hash mismatch: {index}")
        if {name: list(value.shape) for name, value in sorted(fields.items())} != row[
            "tensor_shapes"
        ]:
            raise ValueError(f"held-out tensor shapes mismatch: {index}")
        batches.append(fields)
        sample_ids.extend(row["sample_ids"])
        heldout_group_ids.append(row["group_id"])

    if len(batches) != protocol["evaluation"]["groups"]:
        raise ValueError("held-out group count mismatch")
    if len(sample_ids) != protocol["evaluation"]["samples"]:
        raise ValueError("held-out sample count mismatch")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("held-out sample IDs are not unique")
    return manifest, batches, sample_ids, heldout_group_ids


def run_restore(
    *,
    capsule_dir: Path,
    config_path: Path,
    protocol_path: Path,
    restore_label: str,
    ray_log_dir: Path,
) -> dict[str, Any]:
    """Run one fresh restoration and one non-mutating held-out evaluation."""
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    if protocol.get("schema") != "m4-shield-eligible-live-restore-qualification-v1":
        raise ValueError("wrong restore qualification protocol")
    manifest, batches, sample_ids, heldout_group_ids = authenticate_capsule_subset(
        capsule_dir, protocol
    )

    register_omegaconf_resolvers()
    raw_config = OmegaConf.to_container(load_config(config_path), resolve=True)
    config = MasterConfig(**raw_config)
    config.policy["megatron_cfg"]["train_iters"] = int(config.grpo.max_num_steps)
    if (
        config.policy["model_name"] != "meta-llama/Llama-3.2-1B-Instruct"
        or config.policy["train_global_batch_size"] != 32
        or config.policy["train_micro_batch_size"] != 1
        or config.policy["megatron_cfg"]["tensor_model_parallel_size"] != 1
        or config.policy["megatron_cfg"]["pipeline_model_parallel_size"] != 1
        or config.policy["megatron_cfg"]["context_parallel_size"] != 1
        or not config.data_plane["enabled"]
        or config.data_plane["impl"] != "transfer_queue"
    ):
        raise ValueError("runtime config differs from the frozen one-GPU contract")

    set_seed(20261101)
    init_ray(log_dir=str(ray_log_dir))
    tokenizer = get_tokenizer(config.policy["tokenizer"])
    combined = BatchedDataDict.from_batches(
        batches,
        pad_value_dict={"input_ids": int(tokenizer.pad_token_id)},
    )
    weights_path = capsule_dir / manifest["checkpoint_root"] / "policy" / "weights"
    if not weights_path.is_dir():
        raise FileNotFoundError(f"missing checkpoint weights: {weights_path}")

    cluster = RayVirtualCluster(
        name=f"eligible_live_restore_{restore_label}",
        bundle_ct_per_node_list=[1],
        use_gpus=True,
        num_gpus_per_node=1,
        max_colocated_worker_groups=1,
        port_range_low=config.cluster.get("master_port_range_low"),
        port_range_high=config.cluster.get("master_port_range_high"),
    )
    policy: TQPolicy | None = None
    meta = None
    try:
        policy = TQPolicy(
            cluster=cluster,
            config=config.policy,
            tokenizer=tokenizer,
            processor=None,
            weights_path=str(weights_path),
            optimizer_path=str(weights_path),
            init_optimizer=True,
            init_reference_model=False,
            dp_cfg=config.data_plane,
            tq_partition_id=f"eligible-live-restore-{restore_label}",
        )
        policy.prepare_step(num_samples=len(sample_ids), group_size=8)
        meta = kv_first_write(
            combined,
            sample_ids=sample_ids,
            dp_client=policy.dp_client,
            partition_id=policy.tq_partition_id,
            task_name="train",
            tags=[{"role": "heldout"} for _ in sample_ids],
        )
        policy.prepare_for_training()
        parameter_before = policy.model_parameter_sha256()
        metrics = policy.train_from_meta(
            meta,
            ClippedPGLossFn(config.loss_fn),
            eval_mode=True,
            gbs=32,
            mbs=1,
        )
        parameter_after = policy.model_parameter_sha256()
        loss = _as_number(metrics["loss"])
        if not math.isfinite(loss):
            raise ValueError("held-out loss is nonfinite")
        expected_parameter = protocol["restore"]["expected_parameter_sha256"]
        passed = parameter_before == expected_parameter == parameter_after
        return {
            "schema": "m4-shield-eligible-live-single-restore-result-v1",
            "status": "PASS_SINGLE_RESTORE" if passed else "FAIL_PARAMETER_GATE",
            "restore_label": restore_label,
            "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "capsule_manifest_sha256": protocol["source_capsule"]["manifest_sha256"],
            "checkpoint_authentication_sha256": manifest[
                "checkpoint_authentication_sha256"
            ],
            "heldout_group_ids": heldout_group_ids,
            "heldout_groups": len(batches),
            "heldout_samples": len(sample_ids),
            "parameter_sha256_expected": expected_parameter,
            "parameter_sha256_before_evaluation": parameter_before,
            "parameter_sha256_after_evaluation": parameter_after,
            "parameter_hash_unchanged": parameter_before == parameter_after,
            "heldout_loss": loss,
            "learning_rate_range": _metric_range(metrics, "lr"),
            "weight_decay_range": _metric_range(metrics, "wd"),
            "global_valid_tokens_range": _metric_range(metrics, "global_valid_toks"),
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
    finally:
        if policy is not None:
            if meta is not None:
                try:
                    policy.dp_client.clear_samples(
                        sample_ids=list(meta.sample_ids),
                        partition_id=meta.partition_id,
                    )
                except Exception:
                    pass
            policy.shutdown()
        cluster.shutdown()
        import ray

        ray.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--restore-label", choices=("a", "b"), required=True)
    parser.add_argument("--ray-log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_restore(
        capsule_dir=args.capsule_dir,
        config_path=args.config,
        protocol_path=args.protocol,
        restore_label=args.restore_label,
        ray_log_dir=args.ray_log_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(result))
    print(json.dumps({"output": str(args.output), "status": result["status"]}))
    if result["status"] != "PASS_SINGLE_RESTORE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
