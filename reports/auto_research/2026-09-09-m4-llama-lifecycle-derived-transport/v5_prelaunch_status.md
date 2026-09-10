# Llama M4 V5 pre-launch status

Status: `TERMINAL_MATERIAL_BOTH_WORKLOADS_REPLICATED`.

The V5 successor is preregistered, packaged, and submitted exactly once per
replicate. Parent pipelines are OpenMath r1 `67204364`, OpenMath r2 `67204399`,
GSM8K r1 `67204429`, and GSM8K r2 `67204363`.
The analyzer repair exactly replays both V4 terminal results. The primary V5
cell estimator analyzes each 400-version replicate independently, combines the
two estimates with equal replicate weight, uses block-diagonal HAC covariance,
and resamples circular eight-version blocks independently within each replicate.
Per-arm missingness must be at most 1% in each replicate.

Four fresh assignment identities are frozen: OpenMath r1/r2 use seeds 20261021
and 20261022; GSM8K r1/r2 use 20261023 and 20261024. All retain the accepted
448-step geometry and lifecycle-derived instrument. Their large, non-launchable
candidate manifests remain outside Git and are authenticated in
`v5_local_package_receipt.json`.

All four were submitted before any causal outcome was inspected. The joint
one-shot guard is consumed, all four parent/child chains succeeded, and their
authenticated artifacts support material effects on both workloads. There was
no automatic retry or extension. See `v5_terminal_report.md`.
