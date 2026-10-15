# Evaluation-state diagnostic qualification

2026-10-15. User authorized the next diagnostic step and one launch after
verification. The previous failure identifies a before/after state mismatch,
not the changed component. This package diagnoses that gap; it does not claim
to have repaired evaluation state neutrality.

## Diagnostic change

The worker retains every original fingerprint and counter and adds SHA-256
fingerprints for Python, NumPy, Torch CPU, CUDA, and Megatron tracker RNG state,
from the same captured RNG dictionary. No raw RNG bytes, tensor values, loss,
accuracy or reward are included in diagnostic records.

With the explicit diagnostic-directory flag, the runner writes exclusive
compact records for the initial snapshot, an additional idle snapshot, the
post-evaluation snapshot, and their field differences. Each record binds the
frozen protocol and capsule manifest. Files close before the original equality
gate executes, so a repeated mismatch leaves evidence in terminal assets.

The comparison distinguishes initial-to-idle, initial-to-evaluation, and
idle-to-evaluation changes. Idle drift is evidence of instability between
snapshots, not proof of a specific background consumer. Likewise an evaluation
difference does not alone establish which internal operation caused it.

The original `before != before_update` exception remains before the comparator
update. No field is removed, no RNG is reset, and no numerical tolerance is
relaxed. If it fails again, replay B does not start. If all gates pass, the
unchanged duplicate comparator qualification can finish. No Shield action,
historical treatment contrast, post-update held-out loss or fresh acquisition
is introduced.

## Verification

Exact clean-room extraction: 39 tests passed, zero skipped locally. Six shell
boundary tests passed. The new tests reject raw/outcome fields and invalid
identities, enforce exclusive writes, persist diagnostics before a synthetic
failure, distinguish idle drift, identify RNG components, and inspect actual
runner ordering before the unchanged gate and update.

AST comparison of the worker is identical after removing only the additive RNG
fingerprint return field. The inner replay AST is identical after removing
only record-writing instrumentation, the extra idle snapshot and the optional
diagnostic argument. Protocol, contracts, transport, backend imports, short
Ray roots and state-digest implementation are preserved.

Byte-identical package rebuild, JET manifest schema, Bash and embedded Python
syntax, module compilation, Ruff and diff checks pass. Frozen manifest SHA-256:
`155617b25518767c3c6a2b70ca83c52c90c2a247f984dd397819f7ccdf1fd893`.
The package receipt binds every overlay. No full CUDA diagnostic execution is
claimed locally. The extra snapshot/hashing is measurement instrumentation,
not a scientific arm or evidence of causal utility.

## Execution boundary

Exactly one new guarded `runllm.py --no_wait` invocation after sourcing the
rotated setup. Allocated limit remains 14400 seconds, with no client queue
deadline. All four prior consumed guards remain intact. No automatic retry or
fresh 20-seed acquisition. Authenticate terminal records before choosing a
state-neutral evaluation repair; do not treat an expected diagnostic mismatch
as justification to remove the equality gate.
