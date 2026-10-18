"""Actual NeMo adapter; imported only inside an admitted arm's private runtime."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

from research.m4_eligible_replay import run_qualification as qualified
from research.m4_eligible_replay.contracts import digest_file
from research.m4_eligible_replay.deterministic_backward import apply_torch_controls
from research.m4_eligible_replay.ray_startup import allocate_ray_root, validate_ray_root
from research.m4_paired_replay.contracts import arm_groups, members
from research.m4_paired_replay.runtime import verify_reference_runtime
from nemo_rl.algorithms.async_utils.eligible_live_decision_capsule import (
    validate_eligible_decision,
)
import os


class NeMoBackend:
    """Restore one action into a fresh process, using the accepted worker unchanged."""

    def __init__(
        self,
        *,
        capsule: Path,
        manifest: dict,
        arm: str,
        config_path: Path,
        expected_manifest_sha256: str,
    ) -> None:
        self.policy = None
        self.cluster = None
        self.metas = []
        self.capsule = capsule
        self.manifest = manifest
        self.arm = arm
        self.manifest_sha256 = expected_manifest_sha256
        self.inventory = members(manifest, arm)
        try:
            self._initialize(config_path)
        except BaseException:
            # Cleanup must retain the primary setup failure.
            from research.m4_eligible_replay.training_trace import observer_cleanup

            with observer_cleanup() as cleanup:
                cleanup.callback(self.close)
                raise

    def _initialize(self, config_path: Path) -> None:
        validate_eligible_decision(
            self.manifest["decision"],
            base_group_ids=self.manifest["base_group_ids"],
            shield_group_ids=self.manifest["shield_group_ids"],
        )
        self.verify_inputs()
        qualified.register_omegaconf_resolvers()
        self.config = qualified.MasterConfig(
            **qualified.OmegaConf.to_container(
                qualified.load_config(config_path), resolve=True
            )
        )
        policy_config = self.config.policy
        if (
            policy_config["model_name"] != "meta-llama/Llama-3.2-1B-Instruct"
            or policy_config["train_global_batch_size"] != 32
            or policy_config["train_micro_batch_size"] != 1
            or self.config.loss_fn["token_level_loss"] is not True
            or policy_config.get("eligible_replay_training_trace") is not None
            or any(
                policy_config["megatron_cfg"][k] != 1
                for k in (
                    "tensor_model_parallel_size",
                    "pipeline_model_parallel_size",
                    "context_parallel_size",
                )
            )
        ):
            raise ValueError("frozen replay model/batch/loss/parallelism differs")
        policy_config["megatron_cfg"]["train_iters"] = 64
        policy_config["eligible_replay_deterministic"] = True
        policy_config["eligible_replay_minimal_observer"] = True
        apply_torch_controls(qualified.torch, os.environ)
        qualified.ACTOR_ENVIRONMENT_REGISTRY[qualified.WORKER] = (
            qualified.MCORE_EXECUTABLE
        )
        qualified.set_seed(
            20261101
        )  # Qualified process setup seed; saved RNG is restored afterward.
        self.ray_root = allocate_ray_root(
            "a" if self.arm == "reward_variance_base" else "b"
        )
        validate_ray_root(self.ray_root)
        qualified.init_ray(log_dir=str(self.ray_root))
        tokenizer = qualified.get_tokenizer(policy_config["tokenizer"])
        self.cluster = qualified.RayVirtualCluster(
            name="m4_paired_replay",
            bundle_ct_per_node_list=[1],
            use_gpus=True,
            num_gpus_per_node=1,
            max_colocated_worker_groups=1,
            port_range_low=self.config.cluster.get("master_port_range_low"),
            port_range_high=self.config.cluster.get("master_port_range_high"),
        )
        weights = self.capsule / "checkpoint/policy/weights"
        self.policy = qualified.TQPolicy(
            cluster=self.cluster,
            config=policy_config,
            tokenizer=tokenizer,
            weights_path=str(weights),
            optimizer_path=str(weights),
            init_optimizer=True,
            init_reference_model=False,
            dp_cfg=self.config.data_plane,
            tq_partition_id="paired-" + self.arm,
            worker_extension_cls_fqn=qualified.WORKER,
        )
        self.policy.prepare_step(num_samples=32, group_size=8)
        for label, rows in (
            ("heldout", self.manifest["groups"][8:]),
            ("selected", arm_groups(self.manifest, self.arm)),
        ):
            batches, samples = [], []
            for row in rows:
                fields = qualified.torch.load(
                    self.capsule / row["raw_file"],
                    weights_only=True,
                    map_location="cpu",
                )
                if (
                    qualified.tensor_content_sha256(fields)
                    != row["tensor_content_sha256"]
                ):
                    raise ValueError("selected/heldout tensor content differs")
                batches.append(fields)
                samples.extend(row["sample_ids"])
            combined = qualified.BatchedDataDict.from_batches(
                batches, pad_value_dict={"input_ids": int(tokenizer.pad_token_id)}
            )
            self.metas.append(
                qualified.kv_first_write(
                    combined,
                    sample_ids=samples,
                    dp_client=self.policy.dp_client,
                    partition_id=self.policy.tq_partition_id,
                    task_name="train",
                    tags=[{"role": label} for _ in samples],
                )
            )
        self.policy.prepare_for_training()

    def verify_inputs(self) -> None:
        if digest_file(self.capsule / "manifest.json") != self.manifest_sha256:
            raise ValueError("admitted manifest mutated")
        expected = {r["path"] for r in self.inventory}
        actual = {
            str(p.relative_to(self.capsule))
            for p in self.capsule.rglob("*")
            if p.is_file()
        } - {"manifest.json", "transport-receipt.json"}
        if actual != expected or any(p.is_symlink() for p in self.capsule.rglob("*")):
            raise ValueError("capsule contains unexpected or linked members")
        for row in self.inventory:
            path = self.capsule / row["path"]
            if digest_file(path) != row["sha256"] or (
                row["bytes"] is not None and path.stat().st_size != row["bytes"]
            ):
                raise ValueError("capsule member mutated")

        # Authenticate the complete manifest's metadata inventories as captured.
        def capture_hash(value: object) -> str:
            return hashlib.sha256(
                (
                    json.dumps(
                        value,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode()
            ).hexdigest()

        if (
            capture_hash(self.manifest["groups"])
            != self.manifest["group_authentication_sha256"]
            or capture_hash(self.manifest["checkpoint_files"])
            != self.manifest["checkpoint_authentication_sha256"]
        ):
            raise ValueError("capture metadata inventory differs")

    def snapshot(self) -> dict:
        rows = self.policy.run_all_workers_single_data("replay_state")
        if len(rows) != 1:
            raise ValueError("expected one model worker")
        return rows[0]

    def runtime(self) -> dict:
        rows = self.policy.run_all_workers_single_data("replay_execution_boundary")
        if len(rows) != 1 or rows[0]["training_trace_enabled"] is not False:
            raise ValueError("deep trace not disabled")
        verify_reference_runtime(rows[0]["runtime"])
        return rows[0]["runtime"]

    def evaluate(self) -> float:
        result = self.policy.train_from_meta(
            self.metas[0],
            qualified.ClippedPGLossFn(self.config.loss_fn),
            eval_mode=True,
            gbs=32,
            mbs=1,
        )
        return qualified.scalar(result["loss"])

    def update(self) -> None:
        self.policy.train_from_meta(
            self.metas[1],
            qualified.ClippedPGLossFn(self.config.loss_fn),
            eval_mode=False,
            gbs=32,
            mbs=1,
        )

    def close(self) -> None:
        # Nested cleanup mirrors the accepted runner and always shuts Ray down.
        try:
            if self.policy is not None:
                try:
                    for meta in self.metas:
                        self.policy.dp_client.clear_samples(
                            sample_ids=list(meta.sample_ids),
                            partition_id=self.policy.tq_partition_id,
                        )
                finally:
                    self.policy.shutdown()
        finally:
            try:
                if self.cluster is not None:
                    self.cluster.shutdown()
            finally:
                qualified.ray.shutdown()
