# Downstream-quality repaired no-training preflight failure

Status: `FAILED_TERMINAL_NO_RETRY_AUTHORITY`

Parent pipeline `67345695`, child pipeline `67345877`, and workload job
`435264910` are terminal failed. The logs-before and logs-after jobs both
succeeded. The logs-after archive preserves the workload output and declared
exit code.

## What passed

The v1 repair is verified: configuration resolution completed after registering
the OmegaConf resolvers. Authority, safe-source, separately packaged Megatron
dependency, and protocol gates all passed. Ray and the Megatron policy worker
initialized; `Qwen/Qwen3-0.6B` weights were accessed, imported, and loaded. The
trace reports 596,049,920 parameters and completion of worker model setup.

The `python3-config` and container-fingerprint messages are warnings, not the
cause: the dataset helper compiled successfully after the former, and execution
continued through checkpoint load after the latter.

## First causal failure

The first terminal exception occurred at the beginning of the initial zero-step
terminal export:

```text
MegatronPolicyWorkerImpl.save_checkpoint()
  self.disable_forward_pre_hook()
MegatronPolicyWorkerImpl.disable_forward_pre_hook()
  assert isinstance(self.model, DistributedDataParallel)
AssertionError
```

The preflight deliberately constructed `TQPolicy` with
`init_optimizer=False`. Megatron setup therefore used `wrap_with_ddp=False`,
but `should_disable_forward_pre_hook` was still true because it is derived from
the distributed-optimizer overlap configuration. `save_checkpoint()` tested
only that configuration flag and entered a DDP-only hook operation even though
the inference-only model was not DDP-wrapped.

This is a narrow no-optimizer checkpoint-path defect, not an EOS resource
failure. It occurred before the terminal-export save call itself, conversion,
vLLM evaluation, OpenMath scoring, or any scientific acquisition.

## Authenticated boundary

The workload's preserved exit code is 1. The terminal metadata, workload trace,
and logs-after artifact have SHA-256 values recorded in
`eos_preflight_v2_terminal_result.json`. No terminal-export manifest, resolved
configuration, converted model, prompt manifest, evaluation data, or compact
preflight result exists in the preserved artifact.

Model weights were accessed, but the optimizer was not initialized and no
optimizer step, training step, learner-version advance, qualification, pilot,
or scientific acquisition occurred. The attempt has no scientific
interpretation.

## Prospective repair boundary

The code repair records whether a DDP forward pre-hook is actually enabled and
only disables/restores that hook when true. This preserves the existing DDP
training path and permits an unwrapped inference-only policy to reach the
checkpoint writer. A two-case focused regression test covers both an absent and
an enabled hook.

Python byte-compilation and `git diff --check` pass. Executing the focused test
locally is blocked because the shared environment lacks Ray and this worktree's
uv workspace lacks its declared `nemo-gym` member; this environmental blockage
is not reported as a passing test.

The one-shot v2 authority is consumed. No retry was launched. Any v3 execution
requires a new package identity, complete validation, and fresh explicit user
authorization.
