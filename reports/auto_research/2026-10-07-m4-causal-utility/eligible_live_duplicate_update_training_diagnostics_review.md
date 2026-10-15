# Training divergence diagnostic: audit, implementation and review

2026-10-15. User requested the next step after the duplicate-update mismatch.
The task is to localize divergence, not to relax reproducibility requirements
or run another scientific acquisition. Standing EOS authorization applies to
one separately guarded engineering qualification, with no automatic retry.

## Evidence and source audit

Parent 74196188 / child 74196812 / EOS 490229782 completed both comparator
updates but failed `duplicate replay mismatch: state_after`. Both initial
states agree, both evaluations preserve every recorded state field, both
updates advance Adam 7 to 8 and scheduler samples 224 to 256, and held-out
files are unchanged. Only post-update parameter and optimizer fingerprints
differ. All RNG components and scheduler state agree. The accompanying
terminal result and analysis preserve matching hashes from both terminal
artifact copies. Difference magnitude and the first divergent operator are
not established. No post-update held-out loss or Shield contrast was opened.

Inspected the exact archived code, whose member hashes are in the package
receipt. The TQ shim fetches and prepares data before dynamically calling
`self.train`. Native Megatron worker `train` calls `get_microbatch_iterator`
at lines 794–802, `megatron_forward_backward` at 850–870, and optimizer step
at 885–887. The iterator returns five values, including a lazy iterator of
`ProcessedMicrobatch` dataclasses. These carry the actual loss data, token
order and packed-sequence metadata. Native model forward calls the root
module at `nemo_rl/models/megatron/train.py:128`; model hooks can therefore
observe inputs and raw outputs before later temperature/loss processing.

Pinned Bridge exposes deterministic-mode validation, but previous artifacts
did not record whether it was active in the actual model actor. Preflight
origins alone are not a complete record of later actor imports. No backend
switch or deterministic-mode change is justified by the current hashes alone.

## Scoped implementation

- Default-off `--training-diagnostics` passes an explicit protocol/capsule,
  replicate and exclusive output-path configuration to the custom worker.
  Normal uninstrumented training and evaluation remain on their prior paths.
- An actor-local context temporarily wraps the native microbatch factory,
  forward/backward call and optimizer step, and installs root-model pre/post
  hooks. Every original argument, yielded object and return value is passed
  through. Context cleanup restores bindings and removes hooks, including
  cancellation; a cleanup failure remains fatal and retains a primary error.
- Persist only hashes of fetched ordered data, each processed microbatch,
  root-model inputs/outputs, gradients after backward and before the optimizer,
  inner optimizer state, master parameters and model parameters before/after
  step. No tensor or numerical loss value is written to this trace.
- Record actual actor Torch/backend flags, selected environment variables,
  model modes, loaded module paths/source hashes and native callable origins.
  Missing modules are recorded as missing rather than imported speculatively.
- Write exclusive JSONL into the asset directory and flush every event, so
  partial evidence survives an exception. Strict analysis rejects raw outcome
  fields, wrong identities, incomplete traces, reordered events or bad hashes.
  It reports the earliest *observed* divergence, not a proven kernel-level cause.
- Compare both traces and persist that report before the unchanged duplicate
  state gate. All original state checks, protocol and outcome embargo remain.

The context is restricted to one training worker and one forward/backward
plus one optimizer update. It is not a reusable distributed tracing facility.
Hashing CPU copies and explicit gradient-boundary synchronization can affect
timing. A matching instrumented run would not establish that the uninstrumented
backend is reproducible; it must be reviewed as evidence of a possible observer
effect. Do not automatically advance to the fresh cohort in either case.

## Verification

Final manifest SHA-256:
`9ae5af0835edc29250042f60a6e795241d201524bd33cac387249a1daa8dfd26`.

- Exact clean-room package: 77 tests passed, zero skipped locally. Twenty-four
  new tests cover actual worker opt-in, a full file-writing actor-adapter
  simulation, exclusive trace creation, lazy object-preserving iteration,
  native return forwarding, binding/hook cleanup on exceptions/cancellation,
  cleanup failures, payload normalization, strict trace validation and earliest
  divergence classification. Existing scientific and mutation-negative tests pass.
- All six shell-boundary tests pass. The package enforces full backend preflight
  before transport and trace comparison after both replays but before the gate.
- Independent rebuild is byte-identical. JET schema, Bash/embedded-Python syntax,
  extracted module compilation, Ruff and scoped whitespace checks pass.
- Protocol, contracts, transport, evaluation RNG helper, diagnostic state schema,
  Ray startup and bootstrap are byte-identical. Removing only the explicit opt-in
  configuration/wrapper additions leaves the original `_run` and worker `train`
  ASTs identical. No other worker method changes.
- All six consumed launch guards are preserved and separately hash-bound.
- A real Torch traced-versus-untraced CPU update smoke is included in the actual
  EOS driver and isolated-process preflight. It checks parameter/gradient and
  Torch CPU RNG equality plus restoration of hooks and native method bindings.
  This Mac lacks Torch/Ray/CUDA, so that runtime smoke and actual GPU tracing
  have not passed locally. The upcoming engineering run must establish them.

## Execution and interpretation

Use refreshed `setup_jet.sh`, exactly one new guard and `runllm.py --no_wait`.
Allocated runtime remains 14,400 seconds with no client queue deadline. Preserve
both trace files, their comparison, ordinary replay records and environment
preflight before interpreting any outcome. No automatic repair/retry and no
fresh acquisition are included. Unrelated manuscript edits are excluded.
