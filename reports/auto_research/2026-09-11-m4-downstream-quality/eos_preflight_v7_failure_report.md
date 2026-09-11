# M4 downstream-quality no-training EOS preflight v7 failure

V7 is terminal failed without an authenticated scientific result. Parent
`67395534`, child `67395685`, and workload `435677894` failed; both log-transfer
jobs succeeded. No retry or extension is authorized.

V7 cleared every earlier blocker. It passed the symlink-aware Transformer Engine
provenance gate, actual converter import, protocol checks, model load, zero-step
terminal export, Megatron-to-HF conversion, vLLM reload, deterministic 1,024-row
prompt construction, and evaluation generation/scoring.

The first causal failure occurred while saving the evaluation data:

```text
KeyError: 'dataset_name'
nemo_rl/evals/eval.py:476
```

The serializer reads `master_config.data["dataset_name"]`. The preflight placed
the same identity at `master_config.data["train"]["dataset_name"]` only. It
therefore failed before writing `evaluation_data.json` or the compact preflight
result. The prompt manifest survives with SHA-256
`469a34a176febe9d15c5c6d967ff21f135a43cb20a7cb7d5ae1a53066e6b3a0a`, but
no accuracy can be authenticated from in-memory scores.

The narrow successor repair is to add top-level
`dataset_name="OpenMathInstruct-2"` to the evaluation-only master config while
retaining the existing nested dataset configuration. No model, prompt, decoding,
or scientific design change is indicated. A successor requires a new package
identity and fresh authorization.
