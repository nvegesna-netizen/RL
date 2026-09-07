# M4 model-by-workload grid harmonization audit

Status: `OFFLINE_HARMONIZATION_COMPLETE_PROCEED_TO_PREREGISTRATION`.

## Decision

The smallest informative next experiment is Qwen3-0.6B on GSM8K. It fills the
only missing cell in the existing two-model-by-two-workload grid and changes
only model scale relative to the completed Qwen3-1.7B/GSM8K acquisition. This
audit supports preregistration and compatibility work; it does not authorize an
acquisition or EOS launch.

| Model | OpenMath | GSM8K |
|---|---:|---:|
| Qwen3-0.6B | 0.305423 `[0.275935, 0.334912]` `MATERIAL` | missing |
| Qwen3-1.7B | 0.302493 `[0.276881, 0.328106]` `MATERIAL` | 0.137796 `[0.115233, 0.159916]` `NOT_MATERIAL` |

The OpenMath model-scale difference, Qwen3-1.7B minus Qwen3-0.6B, is only
`-0.0029299`. At Qwen3-1.7B, the OpenMath-minus-GSM8K difference is `0.1646978`.
This makes grid completion more informative than adding a third workload or a
new model family now.

## Common scientific contract

All three terminal results use the same normalized opportunity-loss estimand,
control:d5 allocation, zero- versus five-second intervention, cross-fitted
adjustment for pre-delay opportunity and its zero indicator, eight folds,
four-lag version HAC, eight-version circular blocks, 20,000 bootstrap draws,
outer confidence envelope, 0.20 materiality threshold, missingness rules,
terminal semantics, GRPO advantage estimator, batch size, and generation count.
Their opportunity headers independently confirm identical estimator and loss
settings.

The scientific members extracted from both OpenMath archives match their
embedded checksum ledgers: lifecycle, opportunity, observer duty, result, and
summary. Two log members were deliberately not extracted, so this audit claims
five verified scientific members rather than a new full-archive verification.
The GSM8K terminal archive and all seven of its embedded members were already
verified. Raw artifacts remain outside Git.

## Material differences

The Qwen3-0.6B/OpenMath result uses 400 primary start versions, while both
Qwen3-1.7B results use 500. GSM8K R2 also permits two epoch-specific exposures
to supply finite-dataset capacity. Dataset, model, randomization identity,
source commit, and bootstrap seed differ as they should across independent
studies.

The normalized estimand is comparable, but the unequal primary windows mean the
existing headline estimates target different training-phase populations. The
fourth-cell primary conclusion may use its own prospectively powered fixed
window. A formal grid interaction additionally requires a secondary analysis
frozen over common start versions 8–407 in all four cells, with independent
cross-study uncertainty. Until that exists, grid contrasts remain descriptive.

## Why the workload result is not a mechanism failure

At Qwen3-1.7B, GSM8K's adjusted numerator is only 19.50% of OpenMath's while its
pooled opportunity denominator is 42.80% as large; its normalized effect is
45.55% as large. GSM8K has substantially more zero-opportunity groups
(`0.8221` versus `0.6427`) and a higher mean sibling reward (`0.8889` versus
`0.4495`). These are exploratory distributional differences, not new causal
findings.

Meanwhile, the delay mechanism is stronger on GSM8K by the registered support
statistics: direct-chain rate is higher by `0.1118` and mean version advance by
`0.1850`. Thus the sub-threshold GSM8K result cannot reasonably be attributed to
failure to deliver the intervention.

## Required next gate

Before any training launch:

1. preregister Qwen3-0.6B/GSM8K as a new primary cell with the existing
   three-way materiality rule;
2. freeze the common 400-version secondary grid analysis and interaction
   uncertainty contract;
3. perform outcome-blind power and capacity simulations using preserved ledger
   structure;
4. derive a fresh config from GSM8K R2, changing only model-required fields and
   randomization/output identity;
5. pass local tests, a pinned no-training preflight, and a separately excluded
   neutral capacity qualification;
6. freeze and review the exact acquisition package before requesting a one-shot
   acquisition authorization.

Exact values and provenance are recorded in `harmonization_audit.json`.
