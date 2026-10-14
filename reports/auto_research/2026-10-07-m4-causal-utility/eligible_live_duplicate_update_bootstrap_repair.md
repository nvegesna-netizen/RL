# Standalone worker bootstrap repair

2026-10-14. User authorized repair and relaunch after terminal failure of
parent `73740465` / EOS job `486620007`.

The qualification worker now exposes the hash-bound bundled Megatron Core
source before importing Megatron. This reproduces the source-path setup that
normally happens during NeMo-RL initialization, without requiring a prior
NeMo-RL import. Missing source fails explicitly. Repeated bootstrap calls do
not duplicate paths. No optimizer, RNG-restoration, scheduler, action-selection,
checkpoint, held-out-loss, or training logic is changed.

The new fresh-interpreter regression verifies exact bundled module resolution
without importing NeMo-RL or Torch. A second regression checks bootstrap
precedes Megatron imports in the actual worker. The exact replacement source
and overlay were extracted in a clean directory; all 10 tests pass. Expanded
Bash, embedded Python, lint, and compilation checks pass. An early unsubmitted
candidate mistakenly required Bridge source in the base archive; the clean-room
test caught that assumption, and the final repair uses only the bundled Core
path that the original NeMo-RL bootstrap configures. No launcher invocation was
used for the rejected local candidate.

The existing full worker/state-digest smoke check remains before checkpoint
transport in the exact EOS image. The local Mac does not establish CUDA worker
import success or replay success. These are enforced by the replacement job.

Same frozen qualification protocol SHA-256:
`a57986017e250fff4140ab3e078e6cc48f1a9a6f2a6a418f9b56c66de860037d`.
Same excluded checkpoint, comparator-only duplicate update, complete-state
hash requirements, counters, and outcome embargo. New workload name and
guard, one additional `runllm.py --no_wait` invocation, four-hour allocation,
no queue deadline, no automatic retry, and no fresh acquisition launch.
Original manifest, receipt, and consumed guard remain unchanged. See the
replacement package receipt for exact source and final manifest hashes.
