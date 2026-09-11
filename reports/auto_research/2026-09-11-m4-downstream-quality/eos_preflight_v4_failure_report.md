# Downstream-quality v4 no-training preflight failure

Status: `FAILED_TERMINAL_NO_RETRY_AUTHORITY`

Parent pipeline `67356618`, child pipeline `67356985`, and workload job
`435353105` are terminal failed. Logs-before and logs-after succeeded, and the
preserved workload exit code is 1.

## What v4 established

Authority and safe-source gates passed. Both exact dependency archives were
authenticated and safely extracted: Megatron-Bridge commit `573e088c...` and
its pinned Megatron-LM commit `6513e3e2...`. Python resolved and began importing
the packaged `megatron.bridge` and `megatron.core`, so the v3 missing-Bridge
packaging defect is repaired.

The fail-closed gate ran before Ray initialization or model access. Therefore
no model weights, optimizer, terminal export, conversion, evaluation, training,
qualification, pilot, or scientific acquisition began.

## First causal failure

The packaged Bridge import reached `megatron.bridge.peft.lora_layers`, which
imports Transformer Engine:

```text
import transformer_engine.pytorch as te
ModuleNotFoundError: No module named 'transformer_engine'
```

The error is an interpreter-tier mismatch. The gate and converter used the base
`/opt/nemo_rl_venv/bin/python`. NeMo-RL registers Megatron policy workers with
`PY_EXECUTABLES.MCORE` (`uv run --locked --extra mcore`), and v3 evidence shows
the resulting prefetched interpreter at
`/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python`.
That environment contains Transformer Engine; v3 logs show it loading and
patching `transformer_engine/pytorch/triton/permutation.py`. Repository converter
documentation likewise requires `uv run --extra mcore`.

## Evidence boundary and next repair

No terminal-export manifest, converted model, prompt manifest, evaluation data,
or compact PASS result exists. This run has no downstream-quality or M4
scientific interpretation.

A prospective v5 package should retain the exact packaged Bridge and
Megatron-LM roots but execute both the pre-model converter gate and the actual
post-export conversion with the prefetched Megatron policy-worker mcore
interpreter. It must first require that interpreter to exist and authenticate
imports of Transformer Engine, packaged Bridge, and packaged Megatron-LM before
model access. This repair has not been implemented or launched.

The one-shot v4 authority is consumed. Any v5 execution requires a new package
identity, full validation, and fresh explicit authorization.
