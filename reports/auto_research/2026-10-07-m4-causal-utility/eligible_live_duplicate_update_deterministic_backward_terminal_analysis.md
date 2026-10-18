# Strict duplicate-update qualification: authenticated pass

2026-10-18. The two fresh processes reproduce the same one-update comparator
transition under the explicit strict deterministic, instrumented execution
variant. The original full-state equality gate passes without relaxation.
This resolves the observed duplicate-update reproducibility blocker for this
tested variant and capsule, not a scientific treatment-effect question.

## Authentication

Parent 74643360, child 74643465, EOS 493556318 and collector 493556319 are
terminal success. Source commit:
`e385fd177a729d495cd69a24c89df67366b2f640`.
Manifest SHA-256:
`228febc07ffee199cd12dc990c79b885b63d8e33b5b52c41d55c78a929746b73`.
The frozen protocol and excluded historical capsule are unchanged.

Downloaded only compact artifact members, not checkpoint archives or new
scientific outcomes. All 22 members in the published SHA-256 inventory match
their listed hashes and match byte-for-byte between EOS and collector. The
inventory itself also matches between copies. The accompanying terminal JSON
records each member's size/hash and the job-trace hash. Independent replay
validation and strict trace recomputation exactly reproduce the saved results.
All seven predecessor guard hashes remain unchanged; no guard was reset.

## What passed

- Both processes restore identical parameter, optimizer, scheduler and RNG
  fingerprints. Before, idle and after-evaluation states agree in each process.
- All eight packed microbatches, actual model inputs and outputs match.
  All 31 ordered trace events match by phase/payload after excluding the
  deliberately different replicate identity. Runtime metadata also agrees.
- Unlike the preceding diagnostic, gradients after backward and before the
  optimizer match. Post-step parameters, master parameters and inner optimizer
  state match, as do complete post-update training-state fingerprints.
- Both optimizer counters advance 7 to 8 and scheduler counters 224 to 256.
  Parameters actually change: this is not a no-op passing an equality check.
- Actual worker flags show strict Torch determinism enabled, warn-only false,
  model deterministic mode true, cuDNN deterministic true, benchmarking false,
  matmul TF32 false, CUBLAS workspace `:4096:8`, NCCL `Ring` and NVTE
  nondeterministic algorithms disabled. Model cross-entropy fusion is false,
  as required by the pinned Bridge compatibility check.
- Actual worker source hash matches the frozen overlay. Full driver and
  isolated-actor preflight, including real-tensor restore and traced/control
  update smoke checks, pass. EOS ran 95 unit tests: 94 passed and one expected
  base-interpreter test skipped because the real MCORE interpreter was active;
  the actual full preflight checks that interpreter.
- Held-out files remain unchanged. No post-update held-out loss or historical
  treatment contrast is computed. The historical capsule remains excluded.

The trace-only comparison intentionally retains `qualification_passed=false`:
it is not the qualification authority. The separately recomputed original
full-state validator returns `PASS_DUPLICATE_ONE_UPDATE_REPLAY`; both gates
pass together.

Slurm 6270656 ran on eos0031. Queue time was 13:46:37; allocated duration
00:15:57; workload step 00:15:22; exit 0:0. The allocation retained the
four-hour limit and no queue deadline. This was not a timeout.

## Interpretation and next step

The explicit deterministic execution variant removes the previously observed
divergence in this instrumented duplicate replay. Several execution controls
changed together; the evidence does not isolate the original offending
operator. It also does not show an M4 action benefit or terminal-quality gain:
both processes deliberately executed the same comparator action.

Before a new causal acquisition, prepare a separately frozen minimal-observer
bridge qualification: preserve the strict execution controls, restored capsule,
comparator action, one-update counters and full-state equality requirement, but
remove per-microbatch/model/gradient tracing from the training hot path. Keep
boundary-state authentication and outcome embargo. The current trace recorder
explicitly synchronizes GPU tensors; its CPU traced/control smoke is useful
but does not prove noninterference for the actual Megatron GPU update.

This bridge should first establish duplicate equality without deep tracing.
Compare its common post-update state to this pass as an additional
cross-instrumentation check, with any discrepancy requiring investigation
rather than a relaxed gate. Freeze and review that engineering package
separately. Only afterward consider a prospectively specified fresh eligible
decision acquisition. No new launch or acquisition occurred during this
authentication, and no manuscript result was changed.
