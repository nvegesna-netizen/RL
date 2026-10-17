# Divergence localized to gradient production

2026-10-17. Parent 74228841 / child 74229277 / EOS 490485546 failed at the
retained duplicate post-update state equality gate. The diagnostic code ran
successfully and preserved both complete training traces and their comparison.
Collector 490485548 succeeded. All twelve retrieved evidence files match
byte-for-byte across EOS and collector copies. The accompanying JSON records
their SHA-256 values. Final qualification result is absent in both copies.

## Independently established observations

- Strict trace schema, identities, event ordering and completion checks pass
  for A and B. Recomputing the comparison reproduces the recorded result.
  Running the original qualification validator reproduces
  `duplicate replay mismatch: state_after`.
- Initial/idle/after-evaluation state snapshots are identical within both
  replays. Initial A and B states agree. Evaluation RNG isolation still works.
- Actual ordered worker data and all eight processed packed microbatches agree.
  Each corresponding root-model input and raw model output hash agrees.
- Earliest observed difference is at event 27, `forward_backward_end`:
  gradient fingerprints differ. At event 28, immediately before optimizer step,
  only gradient fingerprints differ; model parameters, master parameters and
  inner optimizer state still agree.
- At event 29, after optimizer step, model parameters, master parameters and
  inner optimizer state differ. This is not an optimizer-only first divergence.
- Each replay completes one update, Adam 7 to 8 and scheduler samples 224 to 256.
  Held-out files remain unchanged. No post-update held-out loss or treatment
  contrast was computed. Final RNG and scheduler state agree across replays.
- EOS ran 77 tests: 76 passed and one expected interpreter test skipped. Actual
  driver/isolated-process preflight and real Torch trace/control-update smoke pass.
- Slurm 6259417 on eos0170 queued 33h15m44s, then ran 18m03s (workload 15m26s).
  It exited 1:0, not by timeout. The long queue wait did not prevent allocation.

## What the runtime records add

A and B record identical runtime modes and audited module origins. Torch
deterministic algorithms and model deterministic mode are both false.
Attention backend is auto; attention/hidden dropout are zero; BF16 is enabled;
FP8, CUDA graphs and recomputation are not enabled in the recorded modes.
`NVTE_ALLOW_NONDETERMINISTIC_ALGO`, `CUBLAS_WORKSPACE_CONFIG` and `NCCL_ALGO`
are unset. Matmul TF32 is false; cuDNN deterministic is false.

The model configuration reports cross-entropy fusion true. That flag alone
does not establish that fused cross entropy ran in this policy-gradient path.
Do not identify it, attention backward or another kernel as the cause without
tracing the actual loss/backward dispatch.

The actual model actor loads Bridge checkpointing and AutoBridge from image
paths, whereas preflight reports packaged paths. Independently hashing those
two pinned archive members produces exactly the actual actor hashes:
checkpointing `175a84e0...c5e05ee8`, AutoBridge `01cd6c31...5723ab25`.
Those path differences therefore do not establish differing implementations
for the audited files. This is not authentication of the entire Bridge tree.

## Interpretation and next step

The first observed divergence is between matching raw model outputs and
completed training gradients. That interval includes loss preparation,
loss-gradient computation, model backward and gradient accumulation. Exact
operator and difference magnitude are still unknown. Matching RNG states
do not guarantee deterministic floating-point gradient production.

The justified next step is a fail-closed deterministic-execution audit and
separately packaged engineering qualification focused on this interval:

1. Verify actual policy-gradient loss/backward dispatch and supported controls
   against the packaged source and installed backend, not configuration flags
   alone. Keep the existing input/output/gradient evidence.
2. Apply and verify deterministic controls in the actual model actor at the
   appropriate initialization stage. Pinned Bridge deterministic validation
   requires cross-entropy fusion disabled, an allowed NCCL algorithm, and
   enables Torch deterministic algorithms. Treat those as compatibility
   requirements, not proof of the offending operation. Verify additional
   backend requirements before selecting them; do not silently swap attention.
3. Fail on unsupported deterministic operations rather than warn-only execution
   or a looser hash comparison. Preserve full restoration, exact equality,
   one-update counts, outcome embargo, four-hour allocation and no queue cap.
4. Review any matching instrumented result for the hashing/synchronization
   observer effect before claiming uninstrumented reproducibility or advancing
   to a fresh cohort. No unchanged retry is justified by the current evidence.

Diagnosis only this turn: no runtime repair, new manifest, guard reset, commit,
push or launch. Prior scientific evidence and unrelated manuscript edits remain
unchanged. The consumed diagnostic guard must never be reused.
