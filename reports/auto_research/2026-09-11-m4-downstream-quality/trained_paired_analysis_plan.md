# M4 downstream-quality analysis plan

Status: `FROZEN_BEFORE_TRAINED_OUTCOMES`

## Population and data lock

The analysis population is the 16 prospectively enumerated matched seed blocks.
The block is the independent causal unit. The data lock requires, for both runs
in every block, an authenticated 448/448 terminal export and exactly 1,024
scores matching the frozen prompt manifest.

No completed prompt, rollout group, checkpoint, or learner version is counted
as an independent training replicate. There is no per-protocol replacement for
the intention-to-treat population.

## Primary analysis

For run `r` in block `b`, define

`Y_br = sum(correct_brj for j=1..1024) / 1024`.

Define `D_b = Y_b,mixed_d5 - Y_b,immediate`. Report:

1. both run scores and `D_b` for every block;
2. the equal-weight mean `D_bar`;
3. its sample standard deviation;
4. the two-sided 95% paired Student interval with 15 degrees of freedom;
5. the frozen classification relative to zero and the ±0.02 margin.

This is the sole primary endpoint and contrast. All 16 blocks receive equal
weight regardless of group count, token count, duration, or realized mediator.

## Frozen sensitivity analyses

1. Enumerate all `2^16` within-block treatment-label sign flips. Report the
   two-sided randomization p-value for a zero effect and inverted randomization
   intervals where numerically identifiable.
2. Draw 20,000 paired block bootstraps with seed `20261401`; report percentile
   and studentized intervals.
3. Fit a descriptive prompt-level model with block fixed effects and prompt
   fixed effects. Cluster uncertainty by training block. This cannot override
   the primary interval.
4. Report unadjusted terminal scores and an ANCOVA sensitivity using only the
   shared V8 base-model prompt correctness as a pretreatment covariate.
5. Recompute the decision for absolute margins 0.01, 0.015, 0.025, and 0.03;
   the registered conclusion remains the 0.02 decision.
6. Report worst-case endpoint bounds if any prompt score is absent. The
   registered primary decision is nevertheless incomplete unless all 32 runs
   have exactly 1,024 scores.

## Secondary analyses

- Release-policy effect on realized normalized opportunity loss, direct-chain
  rate, version advance, throughput, wall time, tokens, response length, and
  collapse incidence.
- Scatterplot of block-level opportunity-loss contrast against `D_b`, with a
  descriptive slope and interval. This is a mediation diagnostic, not a causal
  mediator estimate.
- Equal-wall-clock quality and time-to-fixed-quality, if the frozen logs support
  them without selecting a post-treatment checkpoint. These do not replace the
  equal-update primary endpoint.

## Multiplicity and wording

There is one primary contrast, so no primary multiplicity adjustment is needed.
Every other p-value and interval is labeled secondary or sensitivity. A
material quality result supports a causal statement about the randomized
release policy in the tested setting. It does not prove that opportunity loss
is the exclusive cause, establish a general model-family effect, or imply a
production-wide delay prevalence.
