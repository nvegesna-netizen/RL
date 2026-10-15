"""Real Torch hook/gradient/update observation smoke for the EOS preflight."""

from types import SimpleNamespace
from typing import Any

from research.m4_eligible_replay.training_trace import observe_training


def smoke(torch: Any, digest: Any) -> None:
    """Require traced and untraced CPU updates to agree without random seeding."""

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2.0, 3.0]))

        def forward(self, *, input_ids: Any) -> Any:
            return (input_ids * self.weight).sum()

    traced, control = Model(), Model()
    optimizer = torch.optim.SGD(traced.parameters(), lr=0.125)
    control_optimizer = torch.optim.SGD(control.parameters(), lr=0.125)
    original_step = optimizer.step
    original_local_step = "step" in vars(optimizer)
    inputs = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    before_rng = torch.get_rng_state().clone()

    def iterator(*args: Any, **kwargs: Any) -> tuple:
        return iter(inputs), 2, 1, 2, 2

    def forward_backward(**kwargs: Any) -> None:
        for item in kwargs["data_iterator"]:
            traced(input_ids=item).backward()

    backend = SimpleNamespace(
        get_microbatch_iterator=iterator, megatron_forward_backward=forward_backward
    )
    events = []
    with observe_training(
        backend=backend,
        model=traced,
        optimizer=optimizer,
        data=inputs,
        fingerprint=digest,
        gradients=lambda: digest(traced.weight.grad),
        optimizer_state=lambda: {
            "inner_optimizer_sha256": digest(optimizer.state_dict()),
            "master_parameters_sha256": digest(list(traced.parameters())),
            "parameter_sha256": digest(list(traced.parameters())),
        },
        emit=lambda phase, payload: events.append((phase, payload)),
    ):
        data_iterator, count, mbs, _, seq = backend.get_microbatch_iterator()
        backend.megatron_forward_backward(
            data_iterator=data_iterator,
            num_microbatches=count,
            mbs=mbs,
            seq_length=seq,
            forward_only=False,
            global_valid_seqs=torch.tensor(2),
            global_valid_toks=torch.tensor(4),
        )
        optimizer.step()
    for item in inputs:
        control(input_ids=item).backward()
    control_optimizer.step()
    if (
        not torch.equal(traced.weight, control.weight)
        or not torch.equal(traced.weight.grad, control.weight.grad)
        or not torch.equal(before_rng, torch.get_rng_state())
        or events[-1][0] != "complete"
        or traced._forward_hooks
        or traced._forward_pre_hooks
        or ("step" in vars(optimizer)) != original_local_step
        or optimizer.step != original_step
        or backend.get_microbatch_iterator is not iterator
        or backend.megatron_forward_backward is not forward_backward
    ):
        raise ValueError("real Torch training trace smoke failed")
