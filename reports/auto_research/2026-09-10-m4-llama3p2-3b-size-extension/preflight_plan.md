# Llama 3.2 3B no-training preflight plan

Status: `LOCAL_PACKAGE_PENDING_NO_LAUNCH_AUTHORITY`.

The first executable boundary is exactly one no-training preflight submitted
with `runllm.py --no_wait`, only after a dedicated authorization record exists.

It may authenticate the pinned source, dependency archive, protocol, and all
six configuration hashes; import the runtime; resolve the two neutral
qualification configs; verify the exact model and tokenizer identities; query
repository metadata/tokenizer access; and confirm one-node/two-GPU topology.

It must fail before model-weight loading, trainer construction, trainer steps,
qualification, acquisition, nested submission, retry, or extension. A green
preflight permits only a later authorization decision for the paired neutral
qualification. It never qualifies the scientific study by itself.
