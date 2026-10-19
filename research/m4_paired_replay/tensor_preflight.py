"""Real-Torch secondary mechanics smoke before any native model is allocated."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from research.m4_eligible_replay.contracts import canonical_json
from research.m4_paired_replay.run_arm import create_private_root
from research.m4_paired_replay.secondary import exact_gradient_dot, update_kl
from research.m4_paired_replay.worker_support import (
    gradient_probe,
    parameter_begin,
    parameter_end,
)


def packing_smoke() -> bool:
    """Exercise native packing preparation on real CPU tensors on EOS only."""
    # Optional model-stack imports stay inside the EOS-only tensor preflight.
    import torch
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.policy.lm_policy import Policy
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
    from research.m4_paired_replay.engineering import CONFIG

    register_omegaconf_resolvers()
    cfg = load_config(CONFIG).policy
    assert cfg.sequence_packing.enabled and not cfg.dynamic_batching.enabled
    packing = {
        "algorithm": cfg.sequence_packing.algorithm,
        "input_key": "input_ids",
        "input_lengths_key": "input_lengths",
        "sequence_length_pad_multiple": cfg.make_sequence_length_divisible_by,
    }
    order = cfg.sequence_packing.get("microbatch_order")
    if order is not None:
        packing["microbatch_order"] = order
    policy = SimpleNamespace(
        cfg=cfg,
        data_parallel_size=1,
        use_dynamic_batches=False,
        use_sequence_packing=True,
        sequence_packing_args=packing,
    )
    batch = BatchedDataDict(
        {
            "input_ids": torch.arange(64).repeat(96, 1),
            "input_lengths": torch.full((96,), 64, dtype=torch.int64),
            "token_mask": torch.ones((96, 64)),
            "sample_mask": torch.ones(96),
        }
    )
    assert batch.micro_batch_indices is None
    shards, _ = Policy._shard_for_logprob(policy, batch)
    assert len(shards) == 1 and shards[0].micro_batch_indices is not None
    count, max_length = shards[0].get_microbatch_iterator_for_packable_sequences_len()
    assert count > 0 and max_length >= 64
    assert (
        sum(
            mb.size
            for mb in shards[0].make_microbatch_iterator_for_packable_sequences()
        )
        == 96
    )
    return True


def smoke() -> dict:
    # Torch/CUDA is optional until the EOS-only real-tensor preflight runs.
    import torch
    from research.m4_eligible_replay.bootstrap import initialize_bundled_megatron

    initialize_bundled_megatron(require_bridge=True)
    from megatron.core.optimizer import Adam, OptimizerConfig
    from research.m4_paired_replay.optimizer_zero import materialize_native_zero

    packing_ready = packing_smoke()
    # Exercise the actual installed TE initialization API, with no step/backward.
    parameter = torch.nn.Parameter(
        torch.tensor([1.0, -2.0], device="cuda", dtype=torch.bfloat16)
    )
    cfg = OptimizerConfig(
        use_precision_aware_optimizer=True, use_distributed_optimizer=True, bf16=True
    )
    raw = Adam(
        [parameter],
        lr=0.001,
        master_weights=True,
        use_decoupled_grad=True,
        master_weight_dtype=cfg.main_params_dtype,
        exp_avg_dtype=cfg.exp_avg_dtype,
        exp_avg_sq_dtype=cfg.exp_avg_sq_dtype,
        store_param_remainders=cfg.store_param_remainders,
    )
    leaf = SimpleNamespace(
        optimizer=raw,
        config=cfg,
        model_param_group_index_map={parameter: (0, 0)},
    )
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    leaf._get_main_param_and_optimizer_states = lambda p: (
        DistributedOptimizer._get_main_param_and_optimizer_states(leaf, p)
    )

    def zero_snapshot() -> dict:
        return {
            "parameter": parameter.detach().cpu().tolist(),
            "cpu_rng": torch.get_rng_state().tolist(),
            "cuda_rng": torch.cuda.get_rng_state().tolist(),
            "steps": [g.get("step") for g in raw.param_groups],
        }

    initialized = materialize_native_zero(
        [leaf], torch=torch, adam_class=Adam, snapshot=zero_snapshot
    )
    # State serialization/restoration must preserve the zero-step boundary too.
    restored = Adam(
        [parameter],
        lr=0.001,
        master_weights=True,
        use_decoupled_grad=True,
        master_weight_dtype=cfg.main_params_dtype,
        exp_avg_dtype=cfg.exp_avg_dtype,
        exp_avg_sq_dtype=cfg.exp_avg_sq_dtype,
        store_param_remainders=cfg.store_param_remainders,
    )
    restored.load_state_dict(raw.state_dict())
    from research.m4_paired_replay.optimizer_zero import validate_native_zero

    assert (
        validate_native_zero(
            [SimpleNamespace(optimizer=restored, config=cfg)],
            torch=torch,
            adam_class=Adam,
        )
        == 1
    )
    root = Path("/workspace/m4-paired-private/preflight-smoke")
    create_private_root(root)
    model = torch.nn.Linear(2, 1, bias=False)
    proxy = SimpleNamespace(
        distributed=SimpleNamespace(get_world_size=lambda: 1),
        cuda=torch.cuda,
        sum=torch.sum,
        save=torch.save,
    )
    subject = {
        "identity": "engineering-tensor-smoke",
        "arm": "base",
        "manifest_sha256": "a" * 64,
    }
    worker = SimpleNamespace(
        cfg={
            "eligible_replay_paired_integration": True,
            "eligible_replay_gradient_probe": True,
            "eligible_replay_probe_subject": subject,
        },
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        _paired_parameter_snapshot=None,
        _fetch=lambda meta: meta,
        _attach_or_repack_pack_metadata=lambda data, meta: data,
    )

    def snapshot() -> dict:
        return {
            "parameter_sha256": hashlib.sha256(
                model.weight.detach().numpy().tobytes()
            ).hexdigest(),
            "optimizer_sha256": "b" * 64,
            "scheduler_sha256": "c" * 64,
            "optimizer_step": 0,
            "scheduler_num_steps": 0,
        }

    worker.replay_state = snapshot
    parameter_begin(worker, proxy)
    with torch.no_grad():
        model.weight.add_(1)
    assert math.isclose(parameter_end(worker, proxy), math.sqrt(2), abs_tol=1e-6)
    before = snapshot()
    original_step = worker.optimizer.step
    had_local_step = "step" in vars(worker.optimizer)

    def native_train(data: object, **kwargs: object) -> None:
        worker.optimizer.zero_grad()
        model(torch.tensor([[1.0, 2.0]])).sum().backward()
        worker.optimizer.step()

    result = gradient_probe(
        worker,
        object(),
        None,
        torch=proxy,
        native_train=native_train,
        output=str(root / "probe.pt"),
        subject={**subject, "role": "selected"},
    )
    assert result["optimizer_updates"] == 0 and snapshot() == before
    assert worker.optimizer.step == original_step
    assert ("step" in vars(worker.optimizer)) == had_local_step
    captured = torch.load(root / "probe.pt", weights_only=True)
    assert torch.equal(captured["gradients"][0][1], torch.tensor([[1.0, 2.0]]))
    heldout = {
        **captured,
        "subject": {**subject, "role": "heldout"},
        "gradients": [("weight", torch.tensor([[3.0, -4.0]]))],
    }
    torch.save(heldout, root / "heldout.pt")
    dot = exact_gradient_dot(
        root / "probe.pt", root / "heldout.pt", torch, expected_subject=subject
    )
    assert dot["heldout_gradient_dot_train_gradient"] == -5
    drift = update_kl(
        torch.tensor([0.0, 0.0], dtype=torch.float64),
        torch.tensor([math.log(2), 0.0], dtype=torch.float64),
        torch.tensor([1.0, 0.0]),
        torch,
    )
    assert math.isclose(drift, 1 - math.log(2), abs_tol=1e-12)
    return {
        "status": "PASS_REAL_TORCH_SECONDARY_MECHANICS",
        "checks": {
            "native_sequence_packing_preparation": packing_ready,
            "native_zero_step_optimizer_initialization_and_restore": (
                initialized["optimizer_updates"] == 0
                and initialized["native_checkpoint_extractor_passed"]
            ),
            "parameter_delta": True,
            "probe_stops_before_optimizer": True,
            "optimizer_method_restored": True,
            "gradient_role_and_state_bound": True,
            "exact_gradient_dot": True,
            "frozen_token_k3_formula": True,
        },
        "native_model_qualified": False,
        "native_optimizer_initialization": {
            "source_sha256": initialized["native_adam_source_sha256"],
            "signatures": initialized["native_initialization_signatures"],
            "optimizer_updates": 0,
        },
        "scientific_outcomes_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = smoke()
    with args.output.open("xb") as stream:
        stream.write(canonical_json(result))
    print(json.dumps({"status": result["status"]}))


if __name__ == "__main__":
    main()
