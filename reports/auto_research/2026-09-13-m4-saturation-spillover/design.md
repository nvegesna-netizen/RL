# Prospective M4 saturation and spillover study

Status: `FROZEN_LOCAL_DESIGN_NO_LAUNCH_AUTHORITY`

## Decision

The remaining high-value causal gap is shared-system interference, not another
model cell. Current within-acquisition results randomize groups under an
equal-mass control/d5 policy. Because groups share generation, buffer, and
learner resources, those results identify a direct assignment effect averaged
over approximately 50% delay saturation. They do not show whether delayed
groups alter the opportunities available to controls or whether the result is
stable when delay becomes rare or common.

The successor therefore compares two release policies: 25% and 75% of eligible
groups receive the same five-second hold. Sixteen matched training-seed blocks
each contain one low- and one high-saturation acquisition. This creates 32
independent scientific acquisitions while preserving the accepted model,
workload, optimizer, hardware topology, delay magnitude, learner budget, and
M4 instrument.

## Primary questions

1. **Control spillover:** Does raising the fraction of other delayed groups
   change lost opportunity among groups assigned to immediate release?
2. **Total policy effect:** Does high saturation change the fraction of all
   pre-release opportunity that fails to reach training?

Both estimates are paired over the 16 training-seed blocks. Groups and prompts
are not independent replicates. The two primary intervals are reported
individually and with a joint max-studentized paired-block bootstrap. A result
is scoped to 75% versus 25% saturation in this environment.

Within-policy d5-minus-control M4 effects, their interaction with saturation,
direct-chain behavior, version advance, throughput, tokens, wall time, and
buffer depth are secondary. A downstream accuracy endpoint is deliberately
excluded: it would multiply the cost and instability of the completed quality
study without being necessary to answer the interference question.

## Precision and stopping

The completed quality study supplies a planning SD of 0.01046 for the paired
policy contrast in normalized realized opportunity loss. At 16 blocks, the
projected Student half-width is 0.00557, below the frozen 0.006 planning target.
This projection is not an outcome guarantee, and spillover variance is unknown.
Exactly 16 authenticated blocks are analyzed; worse observed precision is
reported rather than repaired with an optional extension.

The cap is 128 H100 GPU-hours. Existing 32-run experience projects 104.2317
H100 GPU-hours at the same update budget, before savings from omitting terminal
model conversion and held-out evaluation. Every run retains a four-hour wall
cap and eight-H100-hour cap.

## Why this is preferable to a broad dose grid

A 0/25/50/75/100% grid would consume independent acquisition units faster than
it improves the primary spillover comparison. The two extreme nondegenerate
saturations retain both arms within every acquisition, maximize exposure
separation, and permit both direct and spillover estimation. The already
observed 50% studies may be shown only as historical descriptive context; they
do not enter the new primary estimator.

No configuration, package, preflight, qualification, training run, or EOS
submission is authorized by this design.
