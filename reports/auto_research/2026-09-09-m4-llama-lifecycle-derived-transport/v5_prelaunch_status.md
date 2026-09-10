# Llama M4 V5 pre-launch status

Status: `FOUR_LOCAL_CANDIDATES_CLEAN_ROOM_GREEN_NO_LAUNCH_AUTHORITY`.

The V5 successor is preregistered and packaged, but it has not been launched.
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

The next step is a separate authorization decision for exactly four causal
acquisitions. If authorized, all four authorization-bound packages must be
rebuilt and clean-room checked, then submitted before inspecting any causal
outcome. There is no automatic retry or extension.
