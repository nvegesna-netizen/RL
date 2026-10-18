"""Opt-in paired diagnostics; gradient probes never execute in a primary process."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from research.m4_eligible_replay.evaluation_rng import isolate_evaluation_tracker
from research.m4_eligible_replay.training_trace import observer_cleanup
from research.m4_paired_replay.secondary import gradient_rows, tensor_squared_difference


class ProbeComplete(Exception):
    """Internal single-process signal after capture and before optimizer.step."""


def require_paired(worker: Any, torch: Any) -> None:
    if (
        worker.cfg.get("eligible_replay_paired_integration") is not True
        or torch.distributed.get_world_size() != 1
    ):
        raise ValueError(
            "paired diagnostic requires explicit single-worker integration mode"
        )
    if worker.cfg.get("eligible_replay_training_trace") is not None:
        raise ValueError("paired primary/diagnostic mode forbids deep training tracing")


def parameter_begin(worker: Any, torch: Any) -> None:
    require_paired(worker, torch)
    if worker._paired_parameter_snapshot is not None:
        raise ValueError("parameter delta snapshot already open")
    worker._paired_parameter_snapshot = [
        (name, p.detach().cpu().contiguous().clone())
        for name, p in worker.model.named_parameters()
        if p.requires_grad
    ]
    if not worker._paired_parameter_snapshot:
        raise ValueError("no trainable parameters in delta snapshot")


def parameter_end(worker: Any, torch: Any) -> float:
    require_paired(worker, torch)
    before = worker._paired_parameter_snapshot
    if before is None:
        raise ValueError("parameter delta snapshot not open")
    try:
        after = [
            (name, p) for name, p in worker.model.named_parameters() if p.requires_grad
        ]
        if [n for n, _ in before] != [n for n, _ in after]:
            raise ValueError("parameter identity/order changed")
        return math.sqrt(
            math.fsum(
                tensor_squared_difference(a, b, torch)
                for (_, a), (_, b) in zip(before, after)
            )
        )
    finally:
        worker._paired_parameter_snapshot = None


def heldout_logprobs(
    worker: Any, meta: Any, *, torch: Any, tracker: Any, digest: Any
) -> dict:
    require_paired(worker, torch)
    data = worker._fetch(meta)
    data = worker._attach_or_repack_pack_metadata(data, meta)

    def clone(value: Any) -> Any:
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.uint8
            or value.ndim != 1
        ):
            raise ValueError("unsupported RNG tracker state")
        return value.clone()

    with isolate_evaluation_tracker(tracker, clone_state=clone, digest=digest):
        output = worker.get_logprobs(
            data=data, micro_batch_size=1, require_router_replay=False
        )
    logprobs = output["logprobs"].detach().cpu()
    mask = (data["token_mask"] * data["sample_mask"].reshape(-1, 1)).detach().cpu()
    if logprobs.shape != mask.shape:
        raise ValueError("heldout logprob output/mask shape differs")
    return {"logprobs": logprobs, "mask": mask}


def gradient_probe(
    worker: Any,
    meta: Any,
    loss_fn: Any,
    *,
    torch: Any,
    native_train: Any,
    output: str,
    subject: dict,
) -> dict:
    require_paired(worker, torch)
    if worker.cfg.get("eligible_replay_gradient_probe") is not True:
        raise ValueError("gradient probe cannot run in the primary arm process")
    path = Path(output)
    private_root = Path("/workspace/m4-paired-private")
    if (
        not path.is_absolute()
        or ".." in path.parts
        or not path.is_relative_to(private_root)
        or path.exists()
        or any(p.is_symlink() for p in (path, *path.parents))
    ):
        raise ValueError("unexpected private gradient destination")
    expected_subject = worker.cfg.get("eligible_replay_probe_subject")
    if (
        not isinstance(expected_subject, dict)
        or subject.get("role") not in {"selected", "heldout"}
        or subject != {**expected_subject, "role": subject["role"]}
    ):
        raise ValueError("gradient probe subject differs from restored process")
    before = worker.replay_state()
    data = worker._fetch(meta)
    data = worker._attach_or_repack_pack_metadata(data, meta)
    captured = False
    had_local_step = "step" in vars(worker.optimizer)
    original_step = worker.optimizer.step

    def restore() -> None:
        if had_local_step:
            worker.optimizer.step = original_step
        else:
            del worker.optimizer.step

    def capture() -> None:
        nonlocal captured
        if captured:
            raise ValueError("gradient probe reached more than one optimizer boundary")
        torch.cuda.synchronize()
        rows = gradient_rows(worker.model, torch)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            torch.save(
                {
                    "subject": subject,
                    "state_before": before,
                    "parameter_sha256": before["parameter_sha256"],
                    "gradients": rows,
                },
                stream,
            )
        captured = True
        raise ProbeComplete()

    with observer_cleanup() as cleanup:
        cleanup.callback(restore)
        worker.optimizer.step = capture
        try:
            native_train(data, loss_fn=loss_fn, eval_mode=False, gbs=32, mbs=1)
        except ProbeComplete:
            if not captured:
                raise
        else:
            raise ValueError("gradient probe did not stop before optimizer update")
    after = worker.replay_state()
    for key in (
        "parameter_sha256",
        "optimizer_sha256",
        "scheduler_sha256",
        "optimizer_step",
        "scheduler_num_steps",
    ):
        if before[key] != after[key]:
            raise ValueError(
                "gradient probe mutated parameter/optimizer/scheduler state"
            )
    return {"status": "CAPTURED_NO_OPTIMIZER_UPDATE", "optimizer_updates": 0}
