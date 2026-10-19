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


def smoke() -> dict:
    # Torch/CUDA is optional until the EOS-only real-tensor preflight runs.
    import torch

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
            "parameter_delta": True,
            "probe_stops_before_optimizer": True,
            "optimizer_method_restored": True,
            "gradient_role_and_state_bound": True,
            "exact_gradient_dot": True,
            "frozen_token_k3_formula": True,
        },
        "native_model_qualified": False,
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
