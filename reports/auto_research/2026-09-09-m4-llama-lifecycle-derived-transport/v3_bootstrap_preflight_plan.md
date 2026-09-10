# Llama lifecycle-derived transport v3 bootstrap preflight

## Authorized scope

Exactly one credential-free, no-training EOS preflight may be submitted with
`runllm.py --no_wait`. It may authenticate the frozen scientific source,
container fingerprint, v3 operational amendment, and pinned Megatron-LM source
tree; execute the repaired `import nemo_rl` then `import megatron` bootstrap;
and emit a compact result plus hashes.

It may not download model weights, run a training entrypoint, start a paired
qualification or acquisition, submit nested work, retry automatically, or
extend the study. The candidate must use one node, a 900-second workload limit,
a 30-minute scheduler cap, and disabled JET retrier.

## Success gate

Success requires terminal-green parent, child, workload, and logs-after jobs;
zero workload exit code; exact source and dependency hashes; exact normalized
fingerprint authentication; the expected pinned Megatron path in
`megatron.__path__`; and the explicit terminal marker
`M4_LLAMA_V3_BOOTSTRAP_NO_TRAINING_PREFLIGHT_GREEN`.

Any failure ends this authorization. No resubmission or qualification follows
without new explicit user authorization and a documented scientific decision.
