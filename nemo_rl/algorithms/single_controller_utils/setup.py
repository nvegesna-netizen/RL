# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Driver-side factory for the SingleController (async-RL) training path.

setup builds the full SingleControllerActorArgs on the driver and the caller passes it to
SingleControllerActor.remote. Everything lives on the driver because driver-side
TQPolicy owns the worker group directly — running this inside another Ray actor nests
runtime_envs and breaks Ray's resource resolution (see the PR #2692 follow-up).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast

from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.async_utils.replay_buffer import TQReplayBuffer
from nemo_rl.algorithms.async_utils.scheduler_assay import (
    SchedulerAssayArm,
)
from nemo_rl.algorithms.async_utils.structured_scheduler_crossover import (
    DapoSchedulerCrossoverPlan,
    SchedulerPressureResponseArm,
    SchedulerPressureResponsePlan,
    SchedulerProtocolArm,
    SchedulerProtocolPlan,
    StructuredSchedulerCrossoverPlan,
    load_scheduler_protocol,
)
from nemo_rl.algorithms.async_utils.fixed_pool import (
    FixedPoolDataset,
    FixedPoolManifest,
    fixed_pool_collate_fn,
    load_fixed_pool_manifest,
    validate_fixed_pool_manifest_design,
    validate_fixed_pool_materialization,
)
from nemo_rl.algorithms.grpo import MasterConfig as GrpoMasterConfig
from nemo_rl.algorithms.grpo import (
    _create_advantage_estimator,
    _should_use_nemo_gym,
)
from nemo_rl.algorithms.loss import ClippedPGLossFn
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    validate_single_controller_config,
)
from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.utils import setup_response_data
from nemo_rl.data_plane import DataPlaneClient, build_data_plane_client
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.environments.nemo_gym import spinup_nemo_gym_actor
from nemo_rl.experience.rollout_manager import RolloutManager
from nemo_rl.experience.rollouts import should_mask_flagged_samples
from nemo_rl.models.generation.interfaces import (
    resolve_routed_experts_dtype_name_for_model,
)
from nemo_rl.models.generation.sglang.config import SGLangConfig
from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.megatron.router_replay import (
    configure_vllm_for_router_replay,
    router_replay_enabled,
)
from nemo_rl.models.policy.tq_policy import TQPolicy
from nemo_rl.weight_sync import WeightSynchronizer, create_weight_synchronizer


@dataclass
class SingleControllerActorArgs:
    """All inputs SingleControllerActor needs, built driver-side by setup_single_controller().

    Passed as a single arg to SingleControllerActor.remote so the actor's __init__ does
    no construction work — every heavy object is cloudpickled in.
    """

    gen_handle: Any
    trainer_handle: Any  # driver-side TQPolicy
    env_handles: dict[str, EnvironmentInterface]
    train_cluster: RayVirtualCluster
    inference_cluster: RayVirtualCluster
    dp_client: DataPlaneClient
    dataloader: StatefulDataLoader
    weight_synchronizer: WeightSynchronizer
    advantage_estimator: Any
    loss_fn: LossFunction
    rollout_manager: RolloutManager
    tq_buffer: TQReplayBuffer
    partition_id: str
    fixed_pool_manifest: Optional[FixedPoolManifest] = None
    scheduler_assay_plan: Optional[SchedulerProtocolPlan] = None
    scheduler_assay_arm: Optional[SchedulerProtocolArm] = None


def _build_clusters(
    master_config: MasterConfig,
) -> tuple[RayVirtualCluster, RayVirtualCluster]:
    """Allocate train + inference clusters; one shared cluster when colocated."""
    cluster_config = master_config.cluster
    generation_config = master_config.policy["generation"]
    colocated = generation_config["colocated"]["enabled"]
    backend = generation_config["backend"]
    num_nodes = cluster_config["num_nodes"]
    gpus_per_node = cluster_config["gpus_per_node"]
    port_range_low = cluster_config.get("master_port_range_low")
    port_range_high = cluster_config.get("master_port_range_high")

    if colocated:
        # Policy + generation share GPUs — one cluster.
        cluster = RayVirtualCluster(
            name="sc_policy_cluster",
            bundle_ct_per_node_list=[gpus_per_node] * num_nodes,
            use_gpus=True,
            num_gpus_per_node=gpus_per_node,
            max_colocated_worker_groups=1 if backend == "megatron" else 2,
            port_range_low=port_range_low,
            port_range_high=port_range_high,
        )
        return cluster, cluster

    # Non-colocated: split node into train + inference clusters.
    assert backend != "megatron", (
        "The Megatron generation backend does not support non-colocated inference "
        "in SingleController."
    )
    inference_resources = generation_config["colocated"]["resources"]
    inference_gpus_per_node = inference_resources["gpus_per_node"]
    if inference_gpus_per_node is None:
        raise ValueError(
            "Non-colocated generation requires "
            "policy.generation.colocated.resources.gpus_per_node."
        )
    inference_nodes = inference_resources["num_nodes"] or 1
    if num_nodes == 1:
        train_gpus_per_node = gpus_per_node - inference_gpus_per_node
        train_nodes = 1
        assert train_gpus_per_node > 0, (
            f"Not enough GPUs for training: {gpus_per_node} - {inference_gpus_per_node} = {train_gpus_per_node}"
        )
    else:
        train_gpus_per_node = gpus_per_node
        train_nodes = num_nodes - inference_nodes
        assert train_nodes > 0, (
            f"train_nodes must be > 0: {num_nodes} - {inference_nodes} = {train_nodes}"
        )

    train_cluster = RayVirtualCluster(
        name="sc_train_cluster",
        bundle_ct_per_node_list=[train_gpus_per_node] * train_nodes,
        use_gpus=True,
        num_gpus_per_node=train_gpus_per_node,
        max_colocated_worker_groups=1,
        port_range_low=port_range_low,
        port_range_high=port_range_high,
    )
    inference_cluster = RayVirtualCluster(
        name="sc_inference_cluster",
        bundle_ct_per_node_list=[inference_gpus_per_node] * inference_nodes,
        use_gpus=True,
        num_gpus_per_node=inference_gpus_per_node,
        max_colocated_worker_groups=1,
        port_range_low=port_range_low,
        port_range_high=port_range_high,
    )
    return train_cluster, inference_cluster


def _build_generation(
    inference_cluster: RayVirtualCluster,
    master_config: MasterConfig,
):
    """Spin up the generation backend (vLLM or SGLang)."""
    generation_config = master_config.policy["generation"]
    generation_config["model_name"] = master_config.policy["model_name"]
    backend = generation_config["backend"]
    if backend == "vllm":
        vllm_config = cast(VllmConfig, generation_config)
        vllm_config.setdefault("vllm_kwargs", {})["hf_overrides"] = (
            master_config.policy.get("hf_config_overrides", {})
        )
        configure_vllm_for_router_replay(master_config.policy)
        gen = VllmGeneration(cluster=inference_cluster, config=vllm_config)
    elif backend == "sglang":
        sglang_config = cast(SGLangConfig, generation_config)
        sglang_config["sglang_cfg"].setdefault(
            "model_path", master_config.policy["model_name"]
        )
        gen = SGLangGeneration(
            cluster=inference_cluster,
            sglang_cfg=sglang_config,
        )
    else:
        raise ValueError(
            f"single_controller_utils.setup only supports vllm or sglang generation; got {backend!r}"
        )
    gen.finish_generation()
    return gen


def _build_trainer(
    train_cluster: RayVirtualCluster,
    master_config: MasterConfig,
    tokenizer,
    processor,
):
    """Build the TQ-mediated trainer (driver-side TQPolicy).

    Driver-side on purpose: instantiating TQPolicy inside another Ray
    actor nests runtime_envs and triggers Ray's
    get_accelerator_ids_for_accelerator_resource IndexError. Keep this
    here until PolicyTrainerActor (PR #2692) lands.
    """
    loss_config = master_config.loss_fn
    init_reference_model = loss_config.reference_policy_kl_penalty > 0
    return TQPolicy(
        cluster=train_cluster,
        config=master_config.policy,
        tokenizer=tokenizer,
        processor=processor,
        weights_path=None,
        optimizer_path=None,
        init_optimizer=True,
        init_reference_model=init_reference_model,
        dp_cfg=master_config.data_plane,
    )


def _generation_max_seq_len(generation_config) -> int:
    """Return the per-backend max sequence length.

    vllm uses vllm_cfg.max_model_len; sglang uses sglang_cfg.context_length;
    megatron generation has no dedicated field and routes max_new_tokens
    through as max_sequence_length on the inference worker.
    """
    backend = generation_config["backend"]
    if backend == "vllm":
        return generation_config["vllm_cfg"]["max_model_len"]
    if backend == "sglang":
        return generation_config["sglang_cfg"]["context_length"]
    if backend == "megatron":
        return generation_config["max_new_tokens"]
    raise ValueError(f"Unknown generation backend: {backend!r}")


def _clamp_max_num_steps(
    master_config: MasterConfig, dataloader: StatefulDataLoader
) -> None:
    """Clamp grpo.max_num_steps to max_num_epochs * len(dataloader)."""
    grpo_config = master_config.grpo
    max_num_epochs = grpo_config.max_num_epochs
    if max_num_epochs is None:
        return
    grpo_config.max_num_steps = min(
        grpo_config.max_num_steps,
        max_num_epochs * len(dataloader),
    )


def _maybe_inject_megatron_train_iters(master_config: MasterConfig) -> None:
    """Set train_iters from max_num_steps after its dataloader clamp."""
    policy_config = master_config.policy
    if not policy_config.get("megatron_cfg", {}).get("enabled", False):
        return
    grpo_config = master_config.grpo
    megatron_config = cast(dict[str, Any], policy_config["megatron_cfg"])
    megatron_config["train_iters"] = grpo_config.max_num_steps


def setup_single_controller(
    master_config: MasterConfig,
    tokenizer: PreTrainedTokenizerBase,
    *,
    processor: Optional[AutoProcessor] = None,
    partition_id: str = "rollout_data",
) -> SingleControllerActorArgs:
    """Build the full SC actor args driver-side.

    Args:
        master_config: SC MasterConfig.
        tokenizer: Tokenizer used by the policy.
        processor: Optional AutoProcessor for VLM paths.
        partition_id: TQ partition the rollout writer + sampler share.

    Returns:
        SingleControllerActorArgs ready to be passed to SingleControllerActor.
    """
    validate_single_controller_config(master_config)

    # short names for config sections
    grpo_config = master_config.grpo
    dp_config = master_config.data_plane
    policy_config = master_config.policy
    generation_config = policy_config["generation"]
    data_config = master_config.data

    if grpo_config.val_period > 0 or grpo_config.val_at_start or grpo_config.val_at_end:
        raise NotImplementedError(
            "SingleController doesn't support validation now, will support "
            "later. Set grpo.val_period=0, val_at_start=false, val_at_end=false."
        )
    if master_config.checkpointing["enabled"]:
        raise NotImplementedError(
            "SingleController doesn't support checkpointing now, will support "
            "later. Set checkpointing.enabled=false."
        )

    if dp_config is None or not dp_config.get("enabled", False):
        raise ValueError(
            "single_controller_utils.setup requires "
            "master_config.data_plane.enabled=True. The async-RL "
            "SingleController path is built on the TransferQueue data plane."
        )

    assert generation_config is not None, (
        "single_controller_utils.setup requires policy.generation in master_config"
    )

    if data_config["use_multiple_dataloader"]:
        raise NotImplementedError(
            "single_controller_utils does not support "
            "data.use_multiple_dataloader=True yet."
        )

    set_seed(grpo_config.seed)

    # ==========================
    # Setup Dataset & Environments
    # ==========================
    fixed_pool_config = master_config.async_rl.fixed_pool
    assay_config = master_config.async_rl.scheduler_assay
    manifest: Optional[FixedPoolManifest] = None
    assay_plan: Optional[SchedulerProtocolPlan] = None
    assay_arm: Optional[SchedulerProtocolArm] = None
    if fixed_pool_config.enabled:
        manifest = load_fixed_pool_manifest(fixed_pool_config.manifest_path)  # type: ignore[arg-type]
        validate_fixed_pool_materialization(manifest)
        validate_fixed_pool_manifest_design(manifest, fixed_pool_config.design_id)
        expected_model_path = (
            manifest.manifest_path.parent / manifest.model_snapshot_path
        ).resolve()
        observed_model_path = Path(master_config.policy["model_name"]).resolve()
        observed_tokenizer_path = Path(
            master_config.policy["tokenizer"]["name"]
        ).resolve()
        if observed_model_path != expected_model_path or (
            observed_tokenizer_path != expected_model_path
        ):
            raise ValueError(
                "fixed-pool policy.model_name and tokenizer.name must reference "
                "the manifest-bound local model snapshot"
            )
        raw_train_configs = cast(Any, data_config["train"])
        train_configs: list[dict[str, Any]] = (
            [cast(dict[str, Any], config) for config in raw_train_configs]
            if isinstance(raw_train_configs, list)
            else [cast(dict[str, Any], raw_train_configs)]
        )
        expected_sources = [
            (
                source.source_id,
                (manifest.manifest_path.parent / source.materialized_file).resolve(),
            )
            for source in manifest.sources
        ]
        observed_sources = [
            (
                config.get("task_name"),
                Path(config["data_path"]).resolve()
                if config.get("data_path") is not None
                else None,
            )
            for config in train_configs
        ]
        if observed_sources != expected_sources or any(
            config.get("dataset_name") != "ResponseDataset" for config in train_configs
        ):
            raise ValueError(
                "fixed-pool data.train must list the manifest materialized files "
                "in source order as ResponseDataset entries with exact task_name values"
            )
        if assay_config.enabled:
            assert assay_config.plan_path is not None
            assert assay_config.arm_id is not None
            assert assay_config.order_seed is not None
            assay_plan = load_scheduler_protocol(assay_config.plan_path)
            if isinstance(
                assay_plan,
                (
                    StructuredSchedulerCrossoverPlan,
                    DapoSchedulerCrossoverPlan,
                    SchedulerPressureResponsePlan,
                ),
            ):
                assay_arm = assay_plan.arm(assay_config.arm_id)
                pool = assay_plan.pool(assay_config.order_seed)
                expected_generation_seed = pool.generation_study_seed
            else:
                assay_arm = assay_plan.arm(assay_config.arm_id)
                pool = assay_plan.pool(assay_config.order_seed)
                expected_generation_seed = assay_plan.generation_study_seed
            if (
                manifest.order_seed != pool.order_seed
                or manifest.pool_id != pool.pool_id
                or manifest.manifest_sha256 != pool.manifest_sha256
            ):
                raise ValueError("scheduler assay plan/source manifest mismatch")
            if master_config.async_rl.sampler.name != assay_arm.sampler:
                raise ValueError("scheduler assay arm/sampler mismatch")
            if (
                grpo_config.num_generations_per_prompt
                != assay_plan.completions_per_group
                or master_config.policy["max_total_sequence_length"]
                != assay_plan.max_total_sequence_length
                or generation_config["temperature"] != assay_plan.temperature
                or generation_config["top_p"] != assay_plan.top_p
                or cast(dict[str, Any], generation_config)
                .get("vllm_cfg", {})
                .get("study_seed")
                != expected_generation_seed
            ):
                raise ValueError(
                    "scheduler assay generation config does not match frozen plan"
                )
            if isinstance(assay_plan, StructuredSchedulerCrossoverPlan) and (
                generation_config.get("top_k") != assay_plan.top_k
                or generation_config.get("repetition_penalty")
                != assay_plan.repetition_penalty
                or generation_config.get("max_new_tokens") != assay_plan.max_new_tokens
                or master_config.async_rl.max_inflight_prompts
                != assay_plan.max_inflight_prompts
                or master_config.async_rl.max_buffered_rollouts
                != assay_plan.max_buffered_rollouts
            ):
                raise ValueError(
                    "structured crossover runtime does not match frozen plan"
                )
            if isinstance(assay_plan, DapoSchedulerCrossoverPlan) and (
                generation_config.get("max_new_tokens") != assay_plan.max_new_tokens
                or master_config.async_rl.max_inflight_prompts
                != assay_plan.max_inflight_prompts
                or master_config.async_rl.max_buffered_rollouts
                != assay_plan.max_buffered_rollouts
                or data_config["max_input_seq_length"]
                != assay_plan.data_max_input_seq_length
                or policy_config.get("hf_config_overrides", {}).get(
                    "max_position_embeddings"
                )
                != assay_plan.hf_config_override_max_position_embeddings
            ):
                raise ValueError("DAPO crossover runtime does not match frozen plan")
            if isinstance(assay_plan, SchedulerPressureResponsePlan) and (
                generation_config.get("top_k") != assay_plan.top_k
                or generation_config.get("repetition_penalty")
                != assay_plan.repetition_penalty
                or generation_config.get("max_new_tokens") != assay_plan.max_new_tokens
                or master_config.async_rl.max_inflight_prompts
                != assay_plan.max_inflight_prompts
                or master_config.async_rl.max_buffered_rollouts
                != assay_arm.max_buffered_rollouts
                or (
                    getattr(
                        master_config.async_rl.sampler, "max_staleness_versions", None
                    )
                    if assay_arm.sampler == "ready_first"
                    else getattr(
                        master_config.async_rl.sampler, "max_lookahead_versions", None
                    )
                )
                != assay_arm.sampler_lookahead_versions
            ):
                raise ValueError(
                    "scheduler pressure-response runtime does not match frozen arm"
                )

    # TODO: add validate dataset wiring.
    use_nemo_gym = _should_use_nemo_gym(cast(GrpoMasterConfig, master_config))
    if use_nemo_gym and generation_config["backend"] != "vllm":
        raise NotImplementedError(
            "SC NeMo-Gym integration currently supports the vllm backend "
            f"only; got {generation_config['backend']!r}"
        )
    if use_nemo_gym:
        # NeMo-Gym creates the env actor outside setup_response_data; we wire
        # it in after generation is up (it needs the OpenAI server URLs).
        response_data = setup_response_data(tokenizer, data_config, env_configs=None)
        assert len(response_data) == 2
        dataset, _val_dataset = response_data
        env_handles: dict[str, EnvironmentInterface] = {}
    else:
        response_data = setup_response_data(
            tokenizer, data_config, env_configs=master_config.env
        )
        assert len(response_data) == 4
        dataset, _val_dataset, env_handles, _val_env_handles = response_data
    if fixed_pool_config.enabled:
        assert manifest is not None
        if any(
            cohort_size != grpo_config.num_prompts_per_step
            for cohort_size in manifest.cohort_sizes
        ):
            raise ValueError(
                "every fixed-pool dispatch cohort must contain exactly "
                f"grpo.num_prompts_per_step={grpo_config.num_prompts_per_step} "
                f"items; got {manifest.cohort_sizes}"
            )
        dataset = FixedPoolDataset(cast(Any, dataset), manifest)
        dataloader = StatefulDataLoader(
            cast(Any, dataset),
            batch_size=grpo_config.num_prompts_per_step,
            shuffle=False,
            collate_fn=fixed_pool_collate_fn,
            drop_last=False,
            num_workers=data_config["num_workers"],
        )
    else:
        dataloader = StatefulDataLoader(
            cast(Any, dataset),
            batch_size=grpo_config.num_prompts_per_step,
            shuffle=data_config["shuffle"],
            collate_fn=rl_collate_fn,
            drop_last=True,
            num_workers=data_config["num_workers"],
        )

    _clamp_max_num_steps(master_config, dataloader)
    _maybe_inject_megatron_train_iters(master_config)

    # ==========================
    # Setup Clusters & Workers
    # ==========================
    train_cluster, inference_cluster = _build_clusters(master_config)
    colocated = generation_config["colocated"]["enabled"]
    if colocated:
        # Colocated: vLLM prefers a clean GPU at load time, so generation
        # comes up before the policy.
        generation = _build_generation(inference_cluster, master_config)
        policy = _build_trainer(train_cluster, master_config, tokenizer, processor)
    else:
        # Non-colocated: generation + policy run on disjoint GPUs, so
        # bring them up in parallel.
        with ThreadPoolExecutor(max_workers=2) as executor:
            gen_future = executor.submit(
                _build_generation, inference_cluster, master_config
            )
            policy_future = executor.submit(
                _build_trainer, train_cluster, master_config, tokenizer, processor
            )
            generation = gen_future.result()
            policy = policy_future.result()

    # ==========================
    # NeMo-Gym actor (after generation is up so OpenAI URLs are available)
    # ==========================
    if use_nemo_gym:
        # TODO(#2625): Mirror GRPO's deferred vLLM load so NeMo-Gym spinup
        # overlaps model loading instead of running serially afterward.
        enable_router_replay = router_replay_enabled(policy_config)
        routed_experts_dtype = (
            resolve_routed_experts_dtype_name_for_model(generation_config["model_name"])
            if enable_router_replay
            else "int16"
        )
        vllm_generation = cast(VllmGeneration, generation)
        env_handles["nemo_gym"] = spinup_nemo_gym_actor(
            env_configs=master_config.env,
            base_urls=vllm_generation.dp_openai_server_base_urls,
            model_name=generation_config["model_name"],
            enable_router_replay=enable_router_replay,
            routed_experts_dtype=routed_experts_dtype,
            use_fastokens=bool(policy_config["tokenizer"].get("use_fastokens")),
        )

    # ==========================
    # Setup Data Plane Client & Weight Sync
    # ==========================
    # Connect-only DP client; TQPolicy already bootstrapped the controller.
    dp_client = build_data_plane_client(dp_config, bootstrap=False)

    backend = generation_config["backend"]
    weight_synchronizer = create_weight_synchronizer(
        policy=policy,
        generation=generation,
        generation_backend=backend,
        colocated=colocated,
        train_cluster=train_cluster,
        inference_cluster=inference_cluster,
        refit_buffer_size_gb=policy_config.get("refit_buffer_size_gb"),
    )
    weight_synchronizer.init_communicator()

    # ==========================
    # Setup Algorithm + Rollout Wiring
    # ==========================
    advantage_estimator = _create_advantage_estimator(
        cast(GrpoMasterConfig, master_config)
    )
    loss_fn: LossFunction = ClippedPGLossFn(master_config.loss_fn)

    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    tq_buffer = TQReplayBuffer(
        dp_client,
        partition_id=partition_id,
        pad_value_dict={"token_ids": pad_id, "input_ids": pad_id},
        require_routed_experts=router_replay_enabled(policy_config),
    )
    rollout_manager = RolloutManager(
        tokenizer=tokenizer,
        task_to_env=env_handles,
        num_generations_per_prompt=grpo_config.num_generations_per_prompt,
        max_seq_len=_generation_max_seq_len(generation_config),
        max_rollout_turns=grpo_config.max_rollout_turns,
        policy_generation=generation,
        generation_config=generation_config,
        use_nemo_gym=use_nemo_gym,
        mask_env_flagged_samples=should_mask_flagged_samples(master_config.env),
        tq_buffer=tq_buffer,
        scheduler_assay_arm=(
            assay_arm
            if isinstance(assay_arm, (SchedulerAssayArm, SchedulerPressureResponseArm))
            else None
        ),
        scheduler_assay_delay_seconds=(
            assay_arm.release_delay_seconds
            if isinstance(assay_arm, SchedulerPressureResponseArm)
            else assay_plan.release_delay_seconds
            if assay_plan is not None and isinstance(assay_arm, SchedulerAssayArm)
            else None
        ),
    )

    return SingleControllerActorArgs(
        gen_handle=generation,
        trainer_handle=policy,
        env_handles=env_handles,
        train_cluster=train_cluster,
        inference_cluster=inference_cluster,
        dp_client=dp_client,
        dataloader=dataloader,
        weight_synchronizer=weight_synchronizer,
        advantage_estimator=advantage_estimator,
        loss_fn=loss_fn,
        rollout_manager=rollout_manager,
        tq_buffer=tq_buffer,
        partition_id=partition_id,
        fixed_pool_manifest=manifest if fixed_pool_config.enabled else None,
        scheduler_assay_plan=assay_plan,
        scheduler_assay_arm=assay_arm,
    )
