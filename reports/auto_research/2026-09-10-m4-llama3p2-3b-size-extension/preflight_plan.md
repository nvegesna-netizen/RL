# Llama 3.2 3B no-training preflight plan

Status: `EOS_PACKAGE_AUTHORIZED_VALIDATED_NOT_SUBMITTED`.

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

The source-bound local candidate was built and clean-room validated on
2026-09-10. Its SHA-256 is recorded in
`local_preflight_package_receipt.json`; the 15.6 MB candidate remains outside
Git. Because the embedded authorization has `eos_submission_authorized=false`,
the candidate exits at the explicit no-launch boundary before runtime config
resolution or model metadata access if it is submitted accidentally. A later
EOS preflight requires a new, narrowly scoped authorization and a newly built
manifest whose authorization payload permits exactly that one submission.

That authorization was received on 2026-09-10. The launchable manifest was
derived from the validated candidate by changing only its name and embedded
authorization surface, then schema-checked and deterministically rebuilt. Its
hash is recorded in `eos_preflight_package_receipt.json`. This status does not
mean the one-shot submission has been consumed; submission provenance must be
recorded separately after invoking `runllm.py --no_wait`.
