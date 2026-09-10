# Llama lifecycle-derived transport v2 operational amendment

Status: `FROZEN_LOCAL_REPAIR_AUTHORIZED_NO_EOS_LAUNCH_AUTHORIZED`.

## Separation from v1

The v1 execution remains terminal at commit `99540670e`. This amendment does
not reinterpret, erase, or retry that execution. It prospectively defines a v2
package after both v1 cells stopped before training and before any registered
qualification measurement.

## Evidence and diagnosis

Both v1 package traces fail at the embedded fingerprint/bootstrap check with
`ModuleNotFoundError: No module named 'megatron'`. The embedded code imported
`megatron` directly. `nemo_rl/__init__.py` appends the vendored Megatron-LM
workspace to `sys.path`, and the repository's virtual-cluster dependency check
explicitly imports `nemo_rl` before Megatron modules. The failure is therefore
fully explained by package initialization order.

## Sole permitted repair

The v2 package may import `nemo_rl` immediately before `megatron` and may add a
clean-room assertion for that exact order. Regenerated manifests must be named
as v2 artifacts and bound to `v2_operational_amendment.json`.

No scientific runtime code, configuration, instrument, gate, threshold, seed,
assignment domain, model, workload, estimand, or analysis may change. The
existing source archive remains the scientific payload.

## Execution gate

This amendment authorizes local repair and validation only. It authorizes no
EOS launch, qualification, or acquisition. Before any qualification, the exact
v2 package must pass a separately authorized credential-free no-training
preflight that executes the repaired bootstrap in the pinned container. Only
after that preflight is authenticated may a separately authorized fresh paired
qualification be considered. Acquisition remains locked unless both cells pass
every unchanged gate.
