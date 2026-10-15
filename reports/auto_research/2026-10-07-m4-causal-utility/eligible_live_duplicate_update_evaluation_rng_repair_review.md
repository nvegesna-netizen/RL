# Evaluation-local RNG repair and qualification review

2026-10-15. User request: “make the repair and launch”. One replacement
engineering qualification is authorized; no fresh acquisition or automatic retry.

## Evidence and repair

The predecessor parent 74169247, child 74170029, EOS 490015455 failed the
unchanged state-equality gate. Authenticated diagnostic snapshots establish
that the idle state is identical to the initial state and evaluation changes
only the Megatron tracked RNG and its combined fingerprint. No comparator
update or replay B occurred. See the preserved terminal evidence record and
`eligible_live_evaluation_rng_repair_evidence.md` for exact hashes and pinned
source provenance. The exact random operator was not identified.

The custom qualification worker now overrides the verified `train` hook.
Only `eval_mode=True` uses independently cloned named tracker states and
`set_states` restoration on normal and exceptional exit. A pre-forward
round-trip check verifies the installed tracker API. Nonempty initialized
trackers with one-dimensional byte-tensor states are supported; other
representations fail before evaluation. The actual runtime tracker class and
state representation are logged without raw RNG contents. Unexpected stream
names or types and restoration failures remain fatal. If evaluation and
cleanup both fail, the original exception is retained with a cleanup note.

Training delegates directly to the inherited implementation with every
argument unchanged. It does not enter the isolation helper or restore RNG.
No reseeding, tracker reset, default CUDA generator reset, hash substitution,
tolerance change or scientific gate relaxation was introduced.

## Verification

- Exact extracted manifest: 53 tests passed, zero skipped locally. Fourteen
  new tests execute the actual worker override with mutable tensor stand-ins,
  including successful and exceptional restoration, argument/return
  forwarding, repeated evaluation, unsupported state rejection, aliasing,
  restoration failures, and training RNG advancement. A test compiles the
  actual packaged TQ shim and verifies dynamic dispatch into this override.
- All six generated-shell negative tests pass. Socket/import gates remain
  before transport; the two fresh-process comparator replay loop is retained.
- The actual EOS preflight additionally exercises real Torch byte tensors
  with a separate pinned Core tracker on normal and exceptional exits, in
  both driver and isolated process. This never replaces the global tracker.
  It has not run locally: this Mac lacks the CUDA/Torch/Ray runtime.
- Deterministic independent rebuild is byte-identical; JET schema validates.
  Bash syntax, embedded Python compilation, extracted-module compilation,
  Ruff and scoped whitespace checks pass.
- Frozen protocol, contracts, transport, diagnostic schema, Ray startup,
  bootstrap and complete qualification runner are byte-identical to the
  predecessor. Every pre-existing worker function is AST-identical.
- Manifest SHA-256:
  `09275496967cab72edb7cc698729ab2a27015648090a08c37efd26fb9609ee56`.
  Protocol SHA-256:
  `a57986017e250fff4140ab3e078e6cc48f1a9a6f2a6a418f9b56c66de860037d`.

## Launch and stop boundary

Use a new manifest identity and new exclusive one-shot guard. Verify all five
consumed predecessor guards without modifying them. Source rotated
`setup_jet.sh` and submit through `runllm.py --no_wait`. Keep the allocated
limit at 14,400 seconds and the client queue deadline suppressed.

Qualification is pending, not passed by offline tests or job creation. It
must establish complete evaluation-state equality and identical one-update
state hashes across fresh A/B restores, including real parameter change and
optimizer/scheduler advancement. No Shield contrast or post-update held-out
quality is opened. No 20-seed cohort may launch before this gate passes.
Unrelated manuscript edits are preserved and excluded from this repair.
