# M4 downstream-quality no-training EOS preflight v5 terminal failure

V5 is terminal failed without a scientific observation. Parent pipeline
`67386681`, child pipeline `67386833`, and workload job `435601161` all finished
failed. Both log-transfer jobs succeeded. The one authorized submission was
consumed; no retry or extension is authorized.

## Earliest causal failure

The v4 repair worked far enough to establish all of the following before model
access:

- the exact prefetched MegatronPolicyWorker mcore interpreter existed;
- packaged `megatron.bridge` imported;
- packaged `megatron.core` imported;
- `transformer_engine.pytorch` imported; and
- `AutoBridge` imported.

The embedded gate then failed at line 13:

```text
assert te_path.is_relative_to(venv)
AssertionError
```

Lines 11 and 12 had already established that the resolved Bridge and
Megatron-Core paths came from the packaged repository. Line 13 additionally
required the *resolved physical target* of Transformer Engine to remain under
the virtualenv directory. That condition was false. This is not the v4 missing
dependency failure: the Transformer Engine import itself succeeded.

The result is most consistent with a linked package whose physical target is
outside the virtualenv tree. That is a bounded inference, because the gate put
its diagnostic print after the assertion and therefore did not preserve the
raw or resolved Transformer Engine path.

## Execution boundary

The failure occurred before the converter `--help` invocation, Ray
initialization, model access, checkpoint conversion, vLLM reload, or evaluation.
The terminal archive contains only the JET exit code and output log; it contains
none of the registered terminal export, resolved configuration, converted model,
prompt manifest, evaluation data, or preflight result. No optimizer was
initialized, no optimizer or training step occurred, and no qualification,
pilot, acquisition, or causal estimate began.

The GitLab workload duration was 1,521.55 seconds, consisting primarily of a
20:54 Slurm queue and a 3:15 Slurm allocation. The declared workload exit code
was 1.

## Evidence-preserving repair boundary

A prospective successor should keep the exact interpreter and successful-import
requirements, keep resolved containment for the two packaged Megatron trees,
and replace only Transformer Engine's physical-containment assertion with a
symlink-aware logical provenance check. It should print both raw and resolved
paths before any assertion so a future failure remains diagnostic.

That repair has not been implemented. It requires a new package identity and
fresh authorization. V5 remains terminal and will not be retried.

Large raw artifacts remain outside Git and are referenced by SHA-256 in
`eos_preflight_v5_terminal_result.json`.
