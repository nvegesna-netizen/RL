# Lifecycle-derived no-training EOS preflight

Status: `FROZEN_PENDING_IMPLEMENTATION_AND_PACKAGE_LOCK`.

One credential-free EOS submission is allowed. It must use
`runllm.py --no_wait`, disable retriers, fit within 30 minutes, and perform no
training, qualification, acquisition, retry, or extension.

The preflight must authenticate its exact source and package hashes, safely
extract the source, compile the embedded/runtime code, run the complete
instrument test suite, prove exhaustive and randomized arithmetic equivalence,
prove treatment-blind metamorphism and topology fail-closed behavior, validate
the new configs, and perform metadata-only access checks for the gated Llama
model. It may not download model weights.

The preflight passes only if every command exits zero and the artifact preserves
the full test output and hash ledger. Any failure terminally closes the
successor sequence without a second submission.
