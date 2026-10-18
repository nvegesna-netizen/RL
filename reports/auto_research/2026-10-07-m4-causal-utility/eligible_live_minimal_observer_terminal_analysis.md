# Minimal-observer bridge: authenticated pass

2026-10-18. Two fresh comparator-only replays without deep training tracing
produce exactly the same complete state transition as one another and the
accepted traced run. The original duplicate gate and additional frozen
cross-instrumentation gate both pass independently.

## Evidence

Parent 74769312 / child 74769382 / EOS 494463544 / collector 494463548 are
terminal success. Source: `20e3b1edf6b5a85afbceca04a7f6b1684c577a1c`.
Manifest SHA-256:
`de2ee7b5c1b4437b1d6ed53b9f930a3bf96dca471f31aee780aea3844fab238e`.
The bridge protocol binds reference EOS 493556318 and its authenticated
terminal evidence. The original duplicate protocol is unchanged.

All 22 published compact members authenticate against their SHA-256 inventory
and match byte-for-byte between EOS and collector. The inventory itself also
matches. No raw checkpoint archive or new scientific outcome was downloaded.
The terminal JSON preserves individual hashes, runtime metadata, common state,
job-trace hash and all nine consumed-guard hashes.

Independently recomputed results:

- `PASS_DUPLICATE_ONE_UPDATE_REPLAY`: unchanged complete-state duplicate gate.
- `PASS_MINIMAL_OBSERVER_DUPLICATE_AND_REFERENCE_EQUALITY`: strict runtime,
  authenticated worker source and exact reference-state gate.
- Both complete replay JSON files are byte-identical to their respective
  counterparts from the traced reference, not merely equal on selected fields.
- Before, idle and after-evaluation states match within each process.
  Parameter, optimizer, scheduler and RNG fingerprints match between processes
  before and after the update, including optimizer master-state coverage.
- Both optimizer counters advance 7 to 8, scheduler counters 224 to 256.
  Parameters actually change; this is not a no-op.
- Deep training tracing is disabled in both actual workers' before/after
  boundary records. No deep-trace artifact is listed in the complete published
  compact inventory. Packaged mode guards and superclass dispatch were tested
  before launch.
- Actual strict runtime metadata agrees across all boundaries and with the
  traced reference, except the prospectively allowed custom worker file hash.
  That hash matches the frozen overlay. Torch strict determinism, not warn-only,
  model deterministic mode, cuDNN controls, TF32 setting and early CUBLAS/NVTE/
  NCCL controls are retained.
- Held-out tensors remain unchanged. No post-update held-out loss or treatment
  contrast was computed. The historical capsule remains scientifically excluded.

EOS ran 106 tests: 105 passed and one expected base-interpreter test skipped
because it was running in the real MCORE interpreter. Actual full driver and
isolated actor preflight, including real-tensor restore and observer/control
smokes, pass. The local exact-package test count was 106 with no skips.

Slurm 6275376 ran on eos0536, queued 05:53:22, allocated 00:13:11, with workload
step 00:12:34 and exit 0:0. The four-hour allocated limit and no client queue
deadline remain intact. These job durations are execution provenance, not an
estimate of performance improvement from removing tracing.

## Meaning

The previously passing traced update can be reproduced without deep hot-path
observation in this strict deterministic setting. The matching state transition
also crosses the two tested EOS allocations/nodes. This closes the specific
engineering concern that the passing duplicate result required deep tracing.
Boundary hashes/runtime reads remain, so this is not a zero-observation claim.
The original divergent operator is still not isolated.

Both actions here were the comparator. This does not estimate whether M4-Shield
helps learning, establish terminal accuracy improvement, or validate production
performance. Do not add this engineering pass to the paper as a positive
treatment effect or reinterpret previous scientific acquisitions.

## Next step

Prepare and review the fresh cohort's capture/replay execution package offline.
Explicitly bind the qualified strict minimal-observer replay variant into a
prospective runtime amendment, validate the frozen two-action replay and sealed
outcome handling, and clean-room test direct EOS checkpoint transport. Preserve
the existing 20 fresh seeds, at most one first-eligible capsule per trajectory,
16-of-20 completion gate before paired replay/outcome opening, zero replacement
seeds, historical exclusions and held-out surrogate endpoint.

This pass satisfies the same-action reproducibility prerequisite; it does not
by itself qualify the full two-action runner or mark the cohort launch-ready.
Complete those package checks before any fresh scientific acquisition. No new
launch, runtime source edit, causal contrast or manuscript change occurred
during this verification.
