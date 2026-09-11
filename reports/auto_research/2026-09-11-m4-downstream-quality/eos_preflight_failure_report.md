# Downstream-quality no-training preflight failure

Status: `FAILED_TERMINAL_NO_RETRY_AUTHORITY`

Parent pipeline `67340646`, child pipeline `67340771`, and workload job
`435223516` are terminal failed. Both log-transfer jobs succeeded, allowing the
failure to be authenticated from preserved terminal artifacts.

## What passed

The workload verified the image commit and every embedded source, dependency,
protocol, plan, and authorization hash. The EOS authority guard passed. Safe
source extraction found 2,238 members and 16 symlinks; the separately packaged
Megatron dependency and prospective protocol checks also passed.

## First causal failure

The first failure occurred while resolving the frozen anchor configuration:

```text
omegaconf.errors.UnsupportedInterpolationType: Unsupported interpolation type mul
full_key: policy.dynamic_batching.train_mb_tokens
```

The preflight called `load_config()` and immediately requested resolved
conversion. NeMo RL's normal entrypoints first call
`register_omegaconf_resolvers()`, which registers `mul`, `add`, `div`, and
`max`. The standalone preflight omitted that initialization step.

This is a package-level operational defect. It is not evidence of a GPU,
terminal-export, Megatron-to-HF conversion, vLLM, OpenMath, or scientific-design
failure: execution stopped before Ray initialization, tokenizer/model loading,
terminal export, conversion, or evaluation.

## Authenticated boundary

The workload exit code is 1. Output-log copies from the workload and logs-after
artifacts are byte-identical at SHA-256
`f71f7a771998a56c338e9aad0e312151089189d165c5a36b80800ffffbe98435`.
No terminal-export manifest, resolved configuration, converted model, prompt
manifest, evaluation data, or compact preflight result exists.

Therefore no model weights were accessed, no optimizer was initialized, and no
training, qualification, pilot, scientific acquisition, retry, or extension
occurred. The attempt has no scientific interpretation.

## Prospective repair boundary

The narrow repair is to import and call `register_omegaconf_resolvers()` before
loading and resolving the anchor configuration. Repository source confirms this
is the established entrypoint sequence. A local runtime proof is currently
blocked because the available local JET virtual environment does not contain
OmegaConf.

The consumed one-shot authority does not permit a retry. Any repaired execution
requires a new package identity, clean-room validation, and fresh explicit
authorization.
