# M4 downstream-quality no-training EOS preflight v8 result

V8 passed. Parent `67419324`, child `67419903`, and all three child jobs are
terminal successful.

The run validated the complete no-training measurement path: exact dependency
provenance, zero-step Megatron export, conversion to Hugging Face, vLLM reload,
deterministic OpenMath prompt construction, greedy evaluation, serialization,
and compact artifact production. It initialized no optimizer, performed no
training, and started no qualification, pilot, or scientific acquisition.

Independent artifact verification found 1,024 unique prompt rows and 369 binary
successes, for baseline accuracy `0.3603515625` (36.04%). The compact result's
headline value matches the recomputation. The prompt-manifest SHA-256 is
`469a34a176febe9d15c5c6d967ff21f135a43cb20a7cb7d5ae1a53066e6b3a0a`,
identical to v7, so the serializer repair did not alter prompt selection.

This is an operational feasibility result, not evidence that measured M4
opportunity loss changes final training quality. Answering that causal question
requires a separately preregistered trained paired acquisition with explicit
authorization and resource accounting.

Large raw artifacts remain outside Git and are referenced by SHA-256 in
`eos_preflight_v8_terminal_result.json`.
