# Downstream-quality prespecified secondary analysis

Status: `COMPLETE_OFFLINE_SECONDARY`

This analysis reconstructs the secondary endpoints named in the frozen
analysis plan from the 32 authenticated workload archives. It does not alter
the registered terminal-accuracy endpoint or its `INCONCLUSIVE` conclusion.
All mechanism summaries use the campaign's common start-version window 8--407.

## Release-policy effects

Across 16 matched training-seed blocks, mixed-d5 minus all-immediate release
increased normalized realized opportunity loss by 0.00739 (descriptive paired
95% Student interval [0.00181, 0.01296]). It increased the direct-chain rate by
0.12736 [0.12213, 0.13258] and mean version advance during release by 0.20699
[0.19484, 0.21913]. Thus the randomized run-level policy moved the registered
proximal mechanism in the expected direction during the downstream study.

The mixed policy took 395.7 additional seconds per 448-update run [258.1,
533.3], reducing learner-update throughput by 0.00513 updates/s [-0.00696,
-0.00330]. It generated 7.20 million additional valid actor tokens [4.67M,
9.73M], consistent with extra generation replacing groups lost before learner
consumption. Mean response length changed by -19.4 tokens [-68.2, 29.5]. These
are secondary systems effects, not independent primary outcomes.

Two immediate and four mixed-d5 runs had zero terminal accuracy. The paired
difference in zero-endpoint incidence is 0.125 with interval [-0.141, 0.391];
it is imprecise and does not establish a collapse effect.

## Opportunity loss versus terminal accuracy

The prespecified block-level diagnostic regresses the terminal-accuracy
contrast on the normalized realized-opportunity-loss contrast. The Pearson
correlation is -0.096. The descriptive slope is -1.53 with 95% interval
[-10.65, 7.58]. Consequently, the 16 blocks do not resolve whether variation
in realized M4 loss explains variation in terminal accuracy. This is not a
causal mediation estimate.

Equal-wall-clock quality and time-to-fixed-quality are not recoverable: the
protocol froze only terminal equal-update evaluation, prohibited in-loop
validation, and preserved no prospectively selected intermediate evaluations.
Inventing a checkpoint after observing training would create a post-treatment
selection rule, so those optional endpoints are reported as unsupported.

## Metric distinction within these runs

Across the 32 runs, 78,670 of 99,558 common-window assignments with positive
pre-release opportunity were not consumed. They are retained by M4 but absent
from any statistic defined only on consumed rollouts. At run level, normalized
realized loss correlates -0.252 with mean consumed-rollout version age and
-0.458 with ready-to-terminal latency. These descriptive correlations do not
define a new causal estimand; they demonstrate that the observables are not
interchangeable in these executions.

Canonical machine record:
`trained_paired_secondary_analysis.json`. Generated diagnostic:
`trained_paired_opportunity_accuracy_scatter.svg`.
