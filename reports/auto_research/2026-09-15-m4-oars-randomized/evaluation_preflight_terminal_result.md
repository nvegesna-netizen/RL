# Llama terminal-export and evaluation preflight result

Status: `PASS_OPERATIONAL_PREFLIGHT`

Parent pipeline `68680309`, child pipeline `68680637`, and all three child jobs
finished successfully. Both terminal artifact archives passed ZIP integrity.
Every core workload artifact matched its declared SHA-256 and was byte-identical
between the workload and logs-after copies before the compact result or
evaluation data was opened.

The run exercised the exact pre-acquisition path for
`meta-llama/Llama-3.2-1B-Instruct`: model loading, zero-step non-resumable
Megatron terminal export, Megatron-to-Hugging-Face conversion, vLLM reload,
deterministic construction and evaluation of all 1,319 `openai/gsm8k`
`main/test` prompts, binary-reward serialization, and compact result creation.
All independently recomputed gates passed. The run initialized no optimizer,
performed no optimizer or trainer step, advanced no learner version, and began
no qualification, pilot, confirmatory acquisition, or scientific endpoint.

As an evaluator sanity check only, the zero-step/base exported policy answered
478 of 1,319 prompts correctly (`0.36239575435936316`). This value is not a
scientific outcome and is not part of the FIFO-versus-OARS confirmatory design.

## Artifact-finalization finding

The workload hash manifest exactly covered all expected files, but the JET
output log was still being appended after the exit trap hashed it. The manifest
recorded SHA-256 `bd69781b...`; both preserved terminal copies are byte-identical
at the later SHA-256 `804a70de...`. All core artifacts—including the compact
result, prompt manifest, evaluation data, terminal-export metadata, resolved
config, and converted-model hash list—match both their declared hashes and each
other across copies.

This is an operational live-log checksum race, not a model, conversion,
evaluation, or scientific failure. It exposes the same avoidable finalization
surface in the frozen 20-run confirmatory candidates. Before acquisition, the
package should be amended to exclude only JET-managed live-log paths from the
workload-owned `artifacts.sha256`, then all candidates should be rebuilt and
clean-room revalidated. No scientific input or analysis rule needs to change.

Large raw archives remain outside Git and are referenced by SHA-256 in
`evaluation_preflight_terminal_authentication.json`.
