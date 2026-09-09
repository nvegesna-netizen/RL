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

Machine-readable values are in `data/robustness.json`; its SHA-256 is
`4f12bafe8cca627a48b9ee21f9fb0d05a3964d2004994a5d28568f8d7c3b168a`.
