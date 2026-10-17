# Strict deterministic-backward engineering qualification

2026-10-17. User request: “do it”, following the authenticated gradient
divergence diagnosis. Scope: one excluded-capsule comparator-only engineering
qualification. No fresh acquisition, terminal-quality outcome, automatic retry
or weakening of the equality gate is authorized by this package.

## Evidence motivating the change

Parent 74228841 / child 74229277 / EOS 490485546 completed both one-update
replays and the diagnostic comparison. All eight packed microbatches, actual
model inputs and raw model outputs matched. Gradients first differed after
backward, while pre-step model/master parameters and inner optimizer state
still matched. Post-step parameters and optimizer state differed. Recorded
Torch/model deterministic modes were false. This localizes an interval; it
does not identify an offending kernel or quantify the difference.

Both terminal copies and the original gate failure are authenticated in the
preserved terminal result and analysis. Actual actor Bridge paths differed
from preflight paths, but the two audited file hashes matched the pinned
archive bytes. No module-version cause is established by that path difference.

## Pinned loss and initialization audit

Resolved all seven YAML parents in the actual packaged source archive. The
configuration remains Llama-3.2-1B-Instruct, BF16, packing enabled, packer order,
no `sequence_packing.fuse_loss` override (native default false), fused-linear
log-probabilities false, logprob chunk size null, top-k null and top-p 1.0.
The runner constructs the unchanged `ClippedPGLossFn`, whose input type is
LOGPROB. `LossPostProcessor` therefore uses `SequencePackingLossWrapper` and
ordinary log-probability preparation, not the native GPT cross-entropy endpoint.
The audited log-probability path includes distributed log-softmax/gather and
custom autograd backward. Do not attribute the prior failure to any one of
these operations or to attention backward without further evidence.

Source member SHA-256 values:

| Packaged member | SHA-256 |
| --- | --- |
| NeMo `models/megatron/setup.py` | `4945e932b9d5a70faf6698269f10cebdcec132d7606ef8579e50705e754b6ad0` |
| NeMo `algorithms/loss/loss_functions.py` | `e744785d4c2ef73d85a958a0eb6db5b0f060c4cdcd9af7f464b4293c6f1e9b73` |
| NeMo `algorithms/loss/utils.py` | `4ebec7b0a9f2eab9690c85750307d1668fbc8d445367d8008441f58137d03a1b` |
| NeMo `algorithms/loss/wrapper.py` | `815195f7c5ce7e3c6ae745456f7e3208450ac54bf542ff9efcc68fc9c572baf4` |
| NeMo `distributed/model_utils.py` | `0d8651cb355e8a40397a22983e89057bb9139bad352ab569650071cafe52e832` |
| Core `models/gpt/gpt_model.py` | `f42663bf44c39273ae4febfac9ce9e567c5d4b31bf60aa3c6df99372eff45cc1` |

Pinned native model setup calls `_apply_performance_config` before model
`__post_init__`, Bridge ConfigContainer validation and model construction.
The custom worker temporarily wraps that call during its constructor to set
model deterministic mode true and cross-entropy fusion false. The latter is
required by the pinned Bridge validator, not asserted to be the root cause.
The original binding is restored on normal and exceptional exit. No source
change is made to NeMo, Core, Bridge or the loss implementation.

## Execution controls and verification

The manifest exports these values before the first Python process:

- `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`
- `NCCL_ALGO=Ring`
- a qualification-specific mode marker.

Torch deterministic algorithms are enabled with `warn_only=False`; cuDNN
deterministic is true and benchmarking false. The existing matmul TF32=false
condition is checked, not silently changed. Attention selection stays auto;
BF16, dropout, model, batches, loss, optimizer, checkpoint and update counts
are unchanged. Transformer Engine may select a different permitted algorithm
under the deterministic constraint; this is an explicit execution variant,
not proof that one prior algorithm caused the discrepancy.

The custom worker preserves native resource configuration and explicitly
propagates the environment to the model actor. It applies strict controls
before inherited construction, verifies actual model/process flags afterward,
and rechecks before both evaluation and training. Driver and isolated-process
preflight also apply and recheck the flags after their real Torch smoke tests.
Unsupported deterministic operations must raise; there is no warn-only mode,
automatic fallback, reseeding or approximate hash comparison.

These controls are consistent with the [official Megatron deterministic
training guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/deterministic-training.html),
[Transformer Engine environment reference](https://docs.nvidia.com/deeplearning/transformer-engine/envvars.html),
and [Torch strict deterministic API](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html).
The actual pinned Bridge and Core/TE checks are additionally executed from
their ASTs in the extracted-package tests; newer documentation is not a
substitute for those pinned compatibility checks.

## Gates and tests

Final manifest SHA-256:
`228febc07ffee199cd12dc990c79b885b63d8e33b5b52c41d55c78a929746b73`.

- Exact extracted package: 95 tests pass, zero skipped locally, under the
  submitted deterministic environment. Twelve shell-boundary tests pass.
- Tests cover early environment rejection, strict-not-warn flags, constructor
  and resource argument forwarding, native setup ordering, cleanup on failure,
  actual worker downgrade rejection, explicit Ray-probe environment propagation,
  pinned Bridge validation and the pinned Core/TE attention environment guard.
- Existing restore, evaluation isolation, trace ordering/outcome-embargo and
  full-state mutation-negative tests continue to pass.
- `--require-identical` writes the trace comparison first, then rejects any
  observed runtime or execution difference. This is an additional engineering
  requirement, including gradients; it does not replace or relax the frozen
  duplicate post-update state validator. Matching traces alone are not a pass.
- Independent rebuild is byte-identical. JET schema, allocated-time/queue
  configuration, Bash and embedded Python syntax, extracted compile checks,
  Ruff and scoped whitespace checks pass.
- Frozen protocol, contracts, transport, evaluation RNG helper, trace recorder,
  state diagnostic schema, bootstrap and Ray startup are byte-identical.
  Removing only the new mode argument/config assignment leaves the scientific
  `_run` AST identical. All seven consumed predecessor guards are hash-bound.

This Mac lacks the full Torch/Ray/CUDA runtime. The actual strict model
construction, backend support and numerical A/B equality remain unverified
until EOS executes. Full preflight and the real Torch observer/control-update
smoke remain mandatory before checkpoint transport.

## Launch and interpretation boundary

Use refreshed `setup_jet.sh`, a new exclusive one-shot guard, and
`runllm.py --no_wait`. Keep one four-hour allocated limit and no client queue
deadline. Authenticate terminal traces, mode flags, state records and result
before interpreting the run. No fresh cohort is launched automatically.

A pass would establish duplicate reproducibility only for this strict,
instrumented execution variant. It would not identify the original kernel,
prove uninstrumented reproducibility, quantify a quality gain or generalize
the scientific finding. Review the synchronization observer effect and any
runtime change before planning a separate acquisition. Preserve every prior
failure record and all unrelated manuscript edits.
