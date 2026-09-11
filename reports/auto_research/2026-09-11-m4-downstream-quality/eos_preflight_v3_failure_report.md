# Downstream-quality v3 no-training preflight failure

Status: `FAILED_TERMINAL_NO_RETRY_AUTHORITY`

Parent pipeline `67350668`, child pipeline `67350774`, and workload job
`435301112` are terminal failed. The logs-before and logs-after jobs succeeded.
The preserved workload exit code is 1.

## What v3 established

The earlier resolver and checkpoint-hook repairs both worked. Authority,
safe-source, separately packaged Megatron-LM, and protocol gates passed.
`Qwen/Qwen3-0.6B` was accessed, imported, and loaded; the trace reports
596,049,920 parameters. The repaired zero-step export wrote and finalized an
iteration-0 checkpoint, passed its manifest assertions, and emitted
`M4_DOWNSTREAM_QUALITY_ZERO_STEP_EXPORT_PASS`.

The policy was constructed with `init_optimizer=False`. In the executed source,
that flag makes `setup_model_and_optimizer()` set optimizer and scheduler to
`None`. The trace's “Model, optimizer, and learning rate scheduler built” text is
an unconditional generic message, not evidence of optimizer initialization.
The export manifest asserted zero train steps, trainer version zero,
`optimizer_exported: false`, and a non-resumable training checkpoint. No
optimizer step or training step occurred.

## First causal failure

The next command invoked the Megatron-to-Hugging-Face converter with the main
`/opt/nemo_rl_venv/bin/python` interpreter. Its first import failed:

```text
from megatron.bridge import AutoBridge
ModuleNotFoundError: No module named 'megatron.bridge'
```

This is a dependency-packaging defect. Commit
`573dddce4cfca5fd615079c8045af05dddf80d03` records Megatron-Bridge as gitlink
`573e088c9c6740082c39744e03dc5b009e730ed4`, but the source archive contains
only the gitlink directory. The supplemental archive contains the pinned
Megatron-LM tree, which supplies `megatron.core`, but not the Megatron-Bridge
`src` tree that supplies `megatron.bridge`. The isolated policy worker could
load the model from the container environment; the later main-interpreter
converter could not import the missing package.

The container-fingerprint warning accurately reported the Bridge tree as
missing, although it was not itself fatal. The checkpoint export completed
before the converter import failed. Evaluation, OpenMath scoring, and scientific
acquisition never began.

## Evidence boundary and next repair

The zero-step export lived outside the JET assets directory and is not in the
downloaded archive, so its completion is evidenced by the authenticated trace
marker and checkpoint-finalization log rather than a preserved checkpoint.
No converted model, prompt manifest, evaluation data, or compact PASS result
exists. This run has no downstream-quality or M4 scientific interpretation.

A prospective v4 package should provide the exact pinned Megatron-Bridge source
tree to the converter environment. Before any model access, the assembled
package should run the actual converter interpreter and require both
`megatron.bridge.AutoBridge` and `megatron.core` imports to succeed. This repair
has not been implemented or launched.

The one-shot v3 authority is consumed. Any further execution requires a new
package identity, complete validation, and fresh explicit authorization.
