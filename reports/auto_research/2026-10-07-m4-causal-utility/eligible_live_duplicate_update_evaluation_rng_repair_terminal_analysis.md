# Duplicate-update mismatch after successful evaluation isolation

Parent 74196188 / child 74196812 / EOS 490229782 is terminal failed.
The collector 490229790 succeeded. All 15 retrieved compact evidence files
match byte-for-byte across the EOS and collector copies; their sizes and
SHA-256 values are recorded in the accompanying terminal result JSON.
The final qualification result is absent in both copies because its gate failed.

## Established result

The evaluation-local RNG repair works in the actual model worker. Independently
compared initial, idle and after-evaluation diagnostic snapshots: all three
are identical in both A and B. The initial states also match between A and B.
The runtime reports the ordinary Core CudaRNGStatesTracker with tensor states.
The actual driver/isolated-process preflight and real tensor restoration smoke
pass. EOS ran 53 tests, with 52 passes and one expected interpreter skip.

Both fresh processes finish exactly one comparator update. Each changes model
parameters, advances Adam from 7 to 8 and scheduler samples from 224 to 256,
and preserves the held-out files. No post-update held-out loss is computed.
Base group IDs, held-out group IDs and pre-update loss within the frozen
1e-6 bound agree. No Shield contrast or scientific acquisition was performed.

The post-update states disagree in exactly two recorded fields:
`parameter_sha256` and `optimizer_sha256`. Scheduler state, counters,
restoration flags and every RNG component agree. The original validator
rejects this as `duplicate replay mismatch: state_after`; independently
rerunning that validator against the preserved records reproduces the error.

Slurm 6258859 on eos0349 queued for one second and ran for 13m35s, with
TimeLimit=04:00:00 and Deadline=N/A. It exited 1:0, not by timeout.

## What is not yet established

Hash disagreement establishes nonidentical results, not the size of the
parameter difference or its first cause. Matching RNG fingerprints do not
prove deterministic floating-point kernels; matching group IDs do not prove
identical microbatch ordering or actual worker tensors. These records do not
yet distinguish backend numerical nondeterminism, an execution-order change,
or another unrecorded training-path difference. Do not call the difference
tiny or benign, attribute it to a particular kernel, or relax the equality gate.

Pinned Bridge source has a deterministic-mode validator and Torch deterministic
algorithm control. That establishes an available audit path, not that these
controls were enabled in the actual training worker. The import preflight is
not a complete inventory of modules subsequently loaded in the model actor.

## Next diagnostic sequence

1. Audit the exact packaged training path, batching/order and existing
   determinism controls. Verify actual worker module origins as well as driver
   preflight origins; do not assume the module inventory is identical.
2. Add outcome-free fingerprints for the actual ordered training inputs and
   forward/backward boundaries, plus gradient fingerprints before the optimizer
   step and separate optimizer component hashes. Record deterministic settings
   and relevant backend modes in the actual training actor.
3. Use those records to localize the earliest A/B divergence. Preserve normal
   model, loss, optimizer, counter and full-state equality checks. A deterministic
   backend change, if justified, needs explicit packaging and verification;
   do not silently swap numerical implementations or edit the frozen protocol.
4. Freeze and verify a new one-shot engineering package before any later
   authorized rerun. Do not repeat this consumed guard or launch the fresh cohort.

This turn diagnoses and records evidence only: no runtime edits, new manifest,
guard reset, commit, push or launch. Unrelated paper edits remain untouched.
