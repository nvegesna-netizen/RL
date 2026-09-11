# Offline robustness analyses

All analyses in this document are retrospective and descriptive. They use only
the six authenticated terminal acquisitions and cannot replace preregistered
cell or interaction decisions.

## HAC versus circular-block bootstrap

| Cell | Adjusted estimate | HAC 95% interval | Bootstrap 95% interval | Conservative outer envelope |
| --- | ---: | ---: | ---: | ---: |
| 0.6B / OpenMath | 0.3054 | [0.2759, 0.3349] | [0.2763, 0.3339] | [0.2759, 0.3349] |
| 1.7B / OpenMath | 0.2982 | [0.2697, 0.3267] | [0.2703, 0.3267] | [0.2697, 0.3267] |
| 0.6B / GSM8K | 0.2333 | [0.2101, 0.2564] | [0.2085, 0.2579] | [0.2085, 0.2579] |
| 1.7B / GSM8K | 0.1431 | [0.1185, 0.1676] | [0.1172, 0.1676] | [0.1172, 0.1676] |
| 0.6B / NuminaMath | 0.3155 | [0.2881, 0.3429] | [0.2878, 0.3427] | [0.2878, 0.3429] |
| 1.7B / NuminaMath | 0.2979 | [0.2617, 0.3340] | [0.2615, 0.3334] | [0.2615, 0.3340] |

HAC and bootstrap intervals are close in every cell. Both registered interaction
interval families exclude zero. The dependency-aware simultaneous intervals
also exclude zero: [-0.1438, -0.0222] for GSM8K−OpenMath and
[-0.1378, -0.0073] for GSM8K−NuminaMath.

## Adjusted versus unadjusted estimates

| Cell | Adjusted common-window estimate | Unadjusted common-window estimate | Adjusted − unadjusted |
| --- | ---: | ---: | ---: |
| 0.6B / OpenMath | 0.3054 | 0.3335 | -0.0281 |
| 1.7B / OpenMath | 0.2982 | 0.2775 | +0.0208 |
| 0.6B / GSM8K | 0.2333 | 0.2484 | -0.0151 |
| 1.7B / GSM8K | 0.1431 | 0.1653 | -0.0222 |
| 0.6B / NuminaMath | 0.3155 | 0.3693 | -0.0538 |
| 1.7B / NuminaMath | 0.2979 | 0.3192 | -0.0213 |

All twelve point estimates are positive. Adjustment does not manufacture the
direction of the effect, though it materially changes magnitude in some cells.
The adjusted estimates remain authoritative because that estimator and its
pre-treatment covariates were frozen prospectively.

## Missingness endpoint sensitivity

Every common-window assignment has a known terminal disposition. Consequently,
the lower and upper sharp missingness endpoints are identical in all six cells,
their identification widths are zero, and the results require no complete-case
assumption.

## Materiality-threshold sensitivity

The registered threshold is 0.20. Using the conservative common-window outer
envelopes as descriptive decision regions:

| Threshold | Material cells | Inconclusive cells | Not-material cells |
| ---: | --- | --- | --- |
| 0.10 | all six | none | none |
| 0.15 | five cells | 1.7B/GSM8K | none |
| 0.20 | five cells | none | 1.7B/GSM8K |
| 0.25 | both NuminaMath cells; both OpenMath cells | 0.6B/GSM8K | 1.7B/GSM8K |
| 0.30 | none | both OpenMath and both NuminaMath cells | both GSM8K cells |

This table shows why “positive effect” and “material effect at 0.20” must remain
distinct. It is not a family of newly registered hypothesis tests.

## Common-window versus full-window results

| Cell | Common-window estimate | Registered full-window estimate | Difference |
| --- | ---: | ---: | ---: |
| 0.6B / OpenMath | 0.3054 | 0.3054 | 0.0000 |
| 1.7B / OpenMath | 0.2982 | 0.3025 | -0.0043 |
| 0.6B / GSM8K | 0.2333 | 0.2323 | +0.0009 |
| 1.7B / GSM8K | 0.1431 | 0.1378 | +0.0053 |
| 0.6B / NuminaMath | 0.3155 | 0.3155 | 0.0000 |
| 1.7B / NuminaMath | 0.2979 | 0.2979 | 0.0000 |

The maximum absolute difference is 0.0053. Harmonization therefore enables a
valid common-window interaction comparison without changing the substantive
cell pattern.

Machine-readable values are in `robustness_results.json`; its SHA-256 is
`4f12bafe8cca627a48b9ee21f9fb0d05a3964d2004994a5d28568f8d7c3b168a`.

## Prospective Llama 3.2 1B extension

These views use only the four frozen V5 acquisition ledgers. They supplement,
but do not redefine, the two registered equal-replicate workload endpoints.

| Workload | Adjusted estimate | HAC 95% interval | Bootstrap 95% interval | Unadjusted estimate |
| --- | ---: | ---: | ---: | ---: |
| OpenMath | 0.3991 | [0.3526, 0.4456] | [0.3504, 0.4467] | 0.3600 |
| GSM8K | 0.3461 | [0.3026, 0.3896] | [0.2952, 0.3898] | 0.4031 |

Every primary assignment is terminally scored, so sharp missingness endpoints
coincide. Common and registered-full windows are both versions 8–407 by design;
versions 408–447 are an excluded guard window, not an alternative endpoint.

| Threshold | OpenMath | GSM8K |
| ---: | --- | --- |
| 0.00 | above | above |
| 0.10 | above | above |
| 0.20 | above | above |
| 0.25 | above | above |
| 0.30 | above | inconclusive |
| 0.35 | above | inconclusive |
| 0.40 | inconclusive | below or equal |

Leave-one-replicate-out envelopes are [0.3233, 0.4634] and [0.3376, 0.4683]
for OpenMath, and [0.2634, 0.4298] and [0.2884, 0.3830] for GSM8K. Each remains
above 0.20. The secondary cross-workload contrast remains inconclusive:
0.0530, envelope [-0.0137, 0.1232]. Machine-readable values are in
`llama_v5_publication_extension.json`.

## Prospective Llama 3.2 3B extension

The four frozen 3B ledgers reproduce the registered endpoints exactly.

| Workload | Adjusted estimate | HAC 95% interval | Bootstrap 95% interval | Unadjusted estimate |
| --- | ---: | ---: | ---: | ---: |
| OpenMath | 0.2621 | [0.2248, 0.2994] | [0.2215, 0.2989] | 0.2358 |
| GSM8K | 0.2135 | [0.1878, 0.2393] | [0.1872, 0.2398] | 0.2635 |

All terminal-missingness endpoints coincide, and common and registered-full
windows are identically versions 8–407. OpenMath is above thresholds 0.00,
0.10, and 0.20; it crosses 0.25 and is below or equal at 0.30 and above. GSM8K
is above 0.00 and 0.10, crosses 0.20, and is below or equal at 0.25 and above.

Leave-one-replicate-out analysis exposes real replicate variation: OpenMath r1
is individually material while r2 crosses 0.20; GSM8K r1 crosses 0.20 and r2 is
individually material. The prospectively combined conclusions—not selected
replicates—remain authoritative. Both 3B-minus-1B secondary contrasts are
negative under HAC and bootstrap, and their simultaneous intervals exclude
zero. Machine-readable values are in `llama_3b_publication_extension.json`.
