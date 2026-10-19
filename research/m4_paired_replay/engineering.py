"""One bounded synthetic native-model qualification, never a scientific cohort."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import traceback
from typing import Callable, Protocol

from research.m4_eligible_replay.contracts import canonical_json, digest_file
from research.m4_eligible_replay.training_trace import observer_cleanup
from research.m4_paired_replay.contracts import (
    ARMS,
    members,
    object_hash,
    validate_transition,
)
from research.m4_paired_replay.engineering_contracts import (
    DOMAIN,
    IDENTITY,
    PRIMARIES,
    PROBES,
    SEED,
    STAGES,
    compare_primaries,
    synthetic_decision,
    validate_synthetic_manifest,
)
from research.m4_paired_replay.execution import (
    finite,
    strict_runtime,
    validate_secondaries,
)
from research.m4_paired_replay.quarantine import export_files, validate_public_key
from research.m4_paired_replay.run_arm import create_private_root


CONFIG = "examples/configs/grpo_math_1B_megatron_single_controller_m4_eligible_live_capture.yaml"


class BootstrapCheckpointPolicy(Protocol):
    """Native policy checkpoint operations used by the synthetic bootstrap."""

    def save_checkpoint(self, weights_path: str, *, optimizer_path: str) -> None: ...

    def finalize_async_save(self) -> None: ...


def save_bootstrap_checkpoint(
    policy: BootstrapCheckpointPolicy,
    checkpoint: Path,
    *,
    load_train_state: Callable[[Path], object],
) -> None:
    """Wait for native save completion before authenticating genuine v0 metadata.

    Native finalization blocks for asynchronous saves and is a no-op for
    synchronous saves. Never inspect metadata or publish a capsule on failure.
    """
    policy.save_checkpoint(str(checkpoint), optimizer_path=str(checkpoint))
    policy.finalize_async_save()
    metadata = checkpoint / "latest_train_state.pt"
    if not metadata.is_file():
        raise ValueError("native Bridge v0 metadata missing")
    train_state = load_train_state(metadata)
    if (
        not isinstance(train_state, dict)
        or "step" not in train_state
        or train_state["step"] != 0
    ):
        raise ValueError("native bootstrap checkpoint is not version zero")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(canonical_json(value))


def bootstrap(root: Path) -> None:
    """Initialize once from native pretrained weights and save the real v0 state."""
    # Real CUDA/model dependencies are loaded only inside a private stage process.
    from research.m4_eligible_replay import run_qualification as q
    from research.m4_eligible_replay.deterministic_backward import apply_torch_controls
    from research.m4_eligible_replay.ray_startup import (
        allocate_ray_root,
        validate_ray_root,
    )
    from nemo_rl.algorithms.async_utils.eligible_live_decision_capsule import (
        recursive_file_inventory,
    )
    from nemo_rl.data_plane.schema import DP_TRAIN_FIELDS

    q.register_omegaconf_resolvers()
    config = q.MasterConfig(
        **q.OmegaConf.to_container(q.load_config(CONFIG), resolve=True)
    )
    cfg = config.policy
    cfg["megatron_cfg"]["train_iters"] = 64
    cfg["eligible_replay_deterministic"] = True
    cfg["eligible_replay_minimal_observer"] = True
    cfg["eligible_replay_paired_integration"] = True
    cfg["eligible_replay_engineering_cold_start"] = True
    cfg["eligible_replay_expected_optimizer_step"] = 0
    cfg["eligible_replay_probe_subject"] = {"identity": IDENTITY}
    apply_torch_controls(q.torch, os.environ)
    q.ACTOR_ENVIRONMENT_REGISTRY[q.WORKER] = q.MCORE_EXECUTABLE
    q.set_seed(SEED)
    ray_root = allocate_ray_root("a")
    validate_ray_root(ray_root)
    with observer_cleanup() as cleanup:
        cleanup.callback(q.ray.shutdown)
        q.init_ray(log_dir=str(ray_root))
        tokenizer = q.get_tokenizer(cfg["tokenizer"])
        cluster = q.RayVirtualCluster(
            name="m4_synthetic_bootstrap",
            bundle_ct_per_node_list=[1],
            use_gpus=True,
            num_gpus_per_node=1,
            max_colocated_worker_groups=1,
            port_range_low=config.cluster.get("master_port_range_low"),
            port_range_high=config.cluster.get("master_port_range_high"),
        )
        cleanup.callback(cluster.shutdown)
        if cluster.world_size() != 1:
            raise ValueError("synthetic bootstrap requires one worker")
        policy = q.TQPolicy(
            cluster=cluster,
            config=cfg,
            tokenizer=tokenizer,
            weights_path=None,
            optimizer_path=None,
            init_optimizer=True,
            init_reference_model=False,
            dp_cfg=config.data_plane,
            tq_partition_id="synthetic-bootstrap",
            worker_extension_cls_fqn=q.WORKER,
        )
        cleanup.callback(policy.shutdown)
        policy.prepare_for_training()
        canary = json.loads((root / "canary.json").read_bytes())["canary"]
        policy.run_all_workers_single_data("paired_engineering_canary", canary=canary)
        generator = q.torch.Generator(device="cpu").manual_seed(SEED)
        ids = q.torch.randint(
            1000, 10000, (96, 64), generator=generator, dtype=q.torch.int64
        )
        mask = q.torch.zeros((96, 64), dtype=q.torch.float32)
        mask[:, 32:] = 1
        seed_fields = q.BatchedDataDict(
            {
                "input_ids": ids,
                "input_lengths": q.torch.full((96,), 64, dtype=q.torch.int64),
                "token_mask": mask,
                "sample_mask": q.torch.ones(96, dtype=q.torch.float32),
            }
        )
        outputs = policy.get_logprobs(data=seed_fields)
        logprobs = outputs["logprobs"].detach().cpu().float()
        if logprobs.shape != ids.shape or not bool(
            q.torch.isfinite(logprobs).all().item()
        ):
            raise ValueError("synthetic pre-update logprob shape or values invalid")
        capsule = root / "capsule"
        checkpoint = capsule / "checkpoint/policy/weights"
        checkpoint.mkdir(mode=0o700, parents=True)
        parameter_sha = policy.run_all_workers_single_data("model_parameter_sha256")[0]
        initialization = policy.run_all_workers_single_data(
            "paired_engineering_initialize_optimizer"
        )
        if len(initialization) != 1:
            raise ValueError("native initialization requires exactly one worker")
        if (
            initialization[0]["before"]["parameters"] != parameter_sha
            or initialization[0]["before"] != initialization[0]["after"]
            or initialization[0]["optimizer_updates"] != 0
        ):
            raise ValueError("native bootstrap initialization changed training state")
        write_json(root / "optimizer-initialization.json", initialization[0])
        save_bootstrap_checkpoint(
            policy,
            checkpoint,
            load_train_state=lambda path: q.torch.load(
                path, map_location="cpu", weights_only=False
            ),
        )
        groups = []
        decision = synthetic_decision()
        base = decision["proposals"]["reward_variance_risk"]["proposed_group_ids"]
        shield = decision["proposals"]["m4_shield"]["proposed_group_ids"]
        (capsule / "groups").mkdir(mode=0o700)
        for i in range(12):
            sl = slice(i * 8, (i + 1) * 8)
            advantages = q.torch.tensor([1.0, -1.0] * 4).reshape(8, 1).expand(
                8, 64
            ).contiguous() * (1 + i / 12)
            values = {
                "input_ids": ids[sl].clone(),
                "input_lengths": seed_fields["input_lengths"][sl].clone(),
                "generation_logprobs": logprobs[sl].clone(),
                "prev_logprobs": logprobs[sl].clone(),
                "reference_policy_logprobs": logprobs[sl].clone(),
                "advantages": advantages,
                "token_mask": mask[sl].clone(),
                "sample_mask": seed_fields["sample_mask"][sl].clone(),
            }
            fields = {name: values[name] for name in DP_TRAIN_FIELDS}
            name = f"groups/group-{i:02d}.pt"
            q.torch.save(fields, capsule / name, pickle_protocol=2)
            group_id = f"engineering-g{i}"
            groups.append(
                {
                    "group_id": group_id,
                    "index": i,
                    "raw_file": name,
                    "raw_file_sha256": digest_file(capsule / name),
                    "tensor_content_sha256": q.tensor_content_sha256(fields),
                    "sample_ids": [f"engineering-g{i}-s{j}" for j in range(8)],
                    "roles": (["base"] if group_id in base else [])
                    + (["shield"] if group_id in shield else [])
                    + (["heldout"] if i >= 8 else []),
                }
            )
        inventory = recursive_file_inventory(capsule / "checkpoint")
        manifest = {
            "schema": "m4-shield-eligible-live-capsule-v1",
            "status": "QUALIFIED_OUTCOME_EXCLUDED_ELIGIBLE_CAPTURE",
            "trajectory_identity": IDENTITY,
            "trajectory_seed": SEED,
            "assignment_domain": DOMAIN,
            "engineering_synthetic_only": True,
            "learner_version": 0,
            "train_steps_before_capture": 0,
            "captured_update_executed": False,
            "post_update_outcomes_opened": False,
            "parameter_hash_unchanged": True,
            "parameter_sha256_before": parameter_sha,
            "parameter_sha256_after": parameter_sha,
            "base_group_ids": base,
            "shield_group_ids": shield,
            "decision": decision,
            "groups": groups,
            "checkpoint_root": "checkpoint",
            "checkpoint_files": inventory,
            "group_authentication_sha256": object_hash(groups),
            "checkpoint_authentication_sha256": object_hash(inventory),
        }
        validate_synthetic_manifest(manifest)
        write_json(capsule / "manifest.json", manifest)
    # Read-only hard links avoid copying a full checkpoint per arm. Native loads
    # restore tensors into fresh processes; no checkpoint file is modified.
    manifest_bytes = (root / "capsule/manifest.json").read_bytes()
    for arm in ARMS:
        target = root / "arms" / arm
        target.mkdir(mode=0o700, parents=True)
        for row in members(manifest, arm):
            source = root / "capsule" / row["path"]
            os.chmod(source, 0o400)
            destination = target / row["path"]
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.link(source, destination)
        with (target / "manifest.json").open("xb") as stream:
            stream.write(manifest_bytes)


def run_model_stage(stage: str, root: Path) -> None:
    # Optional CUDA imports remain inside the private model-stage process.
    from research.m4_paired_replay.backend import NeMoBackend

    arm = (PROBES if stage in PROBES else PRIMARIES)[stage][0]
    capsule = root / "arms" / arm
    manifest = json.loads((capsule / "manifest.json").read_bytes())
    validate_synthetic_manifest(manifest)
    prefix = "base" if arm == ARMS[0] else "shield"
    probes = (
        root / f"probes/{prefix}-train/gradient.pt",
        root / f"probes/{prefix}-heldout/gradient.pt",
    )
    with observer_cleanup() as cleanup:
        backend = NeMoBackend(
            capsule=capsule,
            manifest=manifest,
            arm=arm,
            config_path=Path(CONFIG),
            expected_manifest_sha256=digest_file(capsule / "manifest.json"),
            gradient_probe_files=probes
            if stage in PRIMARIES and PRIMARIES[stage][1]
            else None,
            probe_mode=stage in PROBES,
        )
        cleanup.callback(backend.close)
        canary = json.loads((root / "canary.json").read_bytes())["canary"]
        backend.policy.run_all_workers_single_data(
            "paired_engineering_canary", canary=canary
        )
        runtime = backend.runtime()
        strict_runtime(runtime)
        before = backend.snapshot()
        if stage in PROBES:
            backend.gradient_probe(
                role=PROBES[stage][1], output=root / f"probes/{stage}/gradient.pt"
            )
            record = {
                "stage": stage,
                "status": "CAPTURED_NO_OPTIMIZER_UPDATE",
                "optimizer_updates": 0,
                "state_before": before,
            }
        else:
            observed = PRIMARIES[stage][1]
            pre = finite(backend.evaluate())
            if observed:
                backend.begin_measurements()
            if before != backend.snapshot():
                raise ValueError("synthetic pre-evaluation/measurement mutated state")
            backend.update()
            after = backend.snapshot()
            post = finite(backend.evaluate())
            secondaries = backend.finish_measurements() if observed else None
            validate_transition(before, before, after, backend.snapshot(), manifest)
            if runtime != backend.runtime():
                raise ValueError("synthetic runtime changed")
            backend.verify_inputs()
            if observed:
                validate_secondaries(secondaries)
            outcome = {
                "synthetic_only": True,
                "pre_loss": pre,
                "post_loss": post,
                "change": post - pre,
                "secondaries": secondaries,
            }
            write_json(root / f"outcomes/{stage}.json", outcome)
            record = {
                "stage": stage,
                "status": "COMPLETE_ENGINEERING_PRIMARY",
                "arm": arm,
                "observed": observed,
                "state_before": before,
                "state_after": after,
                "runtime": runtime,
                "secondaries_complete": observed,
                "outcome_sha256": digest_file(root / f"outcomes/{stage}.json"),
            }
    # Cleanup must succeed before a completion record exists.
    write_json(root / f"records/{stage}.json", record)


def execute_stages(root: Path) -> dict:
    completed = []
    result = {
        "status": "FAILED_PRIVATE_DIAGNOSTICS",
        "synthetic_only": True,
        "fresh_acquisition_unlocked": False,
    }
    for stage in STAGES:
        log = root / f"{stage}.log"
        with log.open("xb") as stream:
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "research.m4_paired_replay.engineering",
                    "--stage",
                    stage,
                    "--private-root",
                    str(root),
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if process.returncode:
            result["failed_stage"] = stage
            break
        completed.append(stage)
    else:
        manifest = json.loads((root / "capsule/manifest.json").read_bytes())
        records = {
            stage: json.loads((root / f"records/{stage}.json").read_bytes())
            for stage in PRIMARIES
        }
        result = compare_primaries(records, manifest)
        result["manifest_sha256"] = digest_file(root / "capsule/manifest.json")
    result["completed_stages"] = completed
    return result


def coordinate(root: Path, public: Path, public_key: Path) -> int:
    """Run ten fresh processes, sealing logs even if a stage fails."""
    if os.environ.get("M4_PAIRED_SYNTHETIC_ENGINEERING") != "1":
        raise ValueError("synthetic qualification environment marker missing")
    pem = public_key.read_bytes()
    validate_public_key(pem)
    create_private_root(root)
    public.mkdir(parents=True, exist_ok=True)
    write_json(
        root / "canary.json", {"canary": secrets.token_hex(32), "synthetic_only": True}
    )
    try:
        result = execute_stages(root)
    except Exception:
        # The privacy boundary retains diagnostic details without publishing metrics.
        with (root / "controller-error.log").open("x") as stream:
            traceback.print_exc(file=stream)
        result = {
            "status": "FAILED_PRIVATE_DIAGNOSTICS",
            "synthetic_only": True,
            "failed_stage": "runtime-comparison-or-infrastructure",
            "fresh_acquisition_unlocked": False,
        }
    names = ["canary.json"] + [
        f"{stage}.log" for stage in STAGES if (root / f"{stage}.log").exists()
    ]
    names += (
        ["controller-error.log"] if (root / "controller-error.log").exists() else []
    )
    if (root / "optimizer-initialization.json").exists():
        names.append("optimizer-initialization.json")
    names += [
        str(p.relative_to(root))
        for folder in ("records", "outcomes")
        for p in sorted((root / folder).glob("*.json"))
    ]
    export_files(
        private_root=root,
        names=names,
        public_pem=pem,
        identity={"identity": IDENTITY, "domain": DOMAIN},
        envelope_path=public / "engineering-private-envelope.json",
        receipt_path=public / "engineering-encrypted-receipt.json",
    )
    write_json(public / "engineering-runtime-result.json", result)
    return (
        0
        if result["status"] == "PASS_RUNTIME_ENGINEERING_COLLECTOR_CHECK_PENDING"
        else 1
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--public", type=Path)
    parser.add_argument("--public-key", type=Path)
    args = parser.parse_args()
    if args.stage == "bootstrap":
        bootstrap(args.private_root)
    elif args.stage is not None:
        run_model_stage(args.stage, args.private_root)
    elif args.public is not None and args.public_key is not None:
        raise SystemExit(coordinate(args.private_root, args.public, args.public_key))
    else:
        parser.error("coordinator requires --public and --public-key")


if __name__ == "__main__":
    main()
