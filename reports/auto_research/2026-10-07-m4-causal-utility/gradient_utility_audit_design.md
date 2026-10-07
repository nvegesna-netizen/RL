# M4 gradient-utility construct audit

Status: **FROZEN BEFORE IMPLEMENTATION OR OUTCOME ACQUISITION**

## Why this audit is the next scientific step

The authenticated 14-acquisition transport analysis found that an M4-risk
ranking captured more *observed* lost M4 than a reward-variance-risk ranking in
all 14 leave-one-acquisition-out evaluations. That result is useful but partly
inherits M4's own definition: M4 is an absolute GRPO coefficient mass. The
secondary inverse-probability-weighted delay diagnostic did not show that M4
preferentially found groups whose randomized delay causally destroyed more M4.
Consequently, another scheduler-quality acquisition is not yet warranted.

The missing construct-validity question is narrower:

> At a fixed checkpoint, does pre-update M4 identify prompt groups with larger
> actual policy gradients, and—more importantly—gradients that align with an
> independently estimated task-consensus direction, beyond information already
> available from reward variance and service cost?

This audit answers that question without updating model parameters. It cannot
establish terminal-quality improvement.

## Frozen acquisition

- Model: `meta-llama/Llama-3.2-1B-Instruct`.
- Workload: GSM8K training split using the already validated native math
  rollout/reward path.
- Checkpoint: the unmodified pretrained checkpoint at learner version zero.
- Prompt groups: 256, each with eight sibling generations.
- Training updates: exactly zero.
- Optimizer steps and scheduler steps: exactly zero.
- Topology: one training GPU and one generation GPU on one EOS node.
- Group order and rollout seed are frozen in the machine-readable protocol.
- Every group is prepared with the production GRPO advantage and clipped-PG
  loss path, subjected to one forward/backward pass, measured, and then
  discarded through the existing abort path. `finish_train_step` is forbidden.

The collector records scalar metadata and compact gradient summaries only. It
does not retain prompts, completions, tokens, raw gradients, or checkpoints.

## Measurements

For group \(i\), let \(g_i\) be its actual token-normalized policy gradient at
the frozen checkpoint and let \(Q_i\) be its pre-update M4 opportunity.

The collector records:

1. exact \(\ell_2\) gradient norm \(\|g_i\|_2\), accumulated in float64;
2. two independently seeded 16,384-bin CountSketch summaries of \(g_i\);
3. M4 L1 and L2 coefficient masses, reward mean, reward variance, valid actor
   tokens, nonzero-advantage sibling count, and truncation count;
4. group identity, deterministic audit index, and fold assignment; and
5. pre- and post-acquisition parameter hashes plus counters proving no learner,
   optimizer, or scheduler advance.

CountSketch is used only for directional inner products. Exact gradient norm is
the magnitude endpoint. Each sketch is produced in one streaming pass over the
gradient buffers, with a source-bound deterministic bucket/sign mapping.

## Cross-fitted reference direction

Groups are assigned to eight folds before outcomes exist. For each held-out
fold \(f\), its reference direction is the mean sketch of groups outside
\(f\). The directional utility of held-out group \(i\) is

\[
U_i = \frac{g_i^\top \bar g_{-f(i)}}{\|\bar g_{-f(i)}\|_2}.
\]

This prevents a group's gradient from contributing to its own reference. Raw
signed \(U_i\), positive-part \(\max(U_i,0)\), and cosine alignment are
reported. The two independent sketches must yield the same scientific
decision; disagreement is a measurement failure, not a result to average away.

## Frozen comparisons

All primary comparisons are restricted to finite groups with positive M4 and a
nonzero exact gradient. Zero-M4 and zero-gradient rates remain mandatory
descriptive results.

The baseline feature set is fixed to information available without M4:

- log valid actor tokens;
- reward mean;
- reward variance; and
- truncation fraction.

The augmented set adds `log1p(M4 L1)`. Models are ridge regressions with the
penalty selected only inside each outer training fold. Predictions are generated
under the frozen eight-fold split.

Primary endpoints:

1. out-of-fold \(R^2\) improvement for `log(exact gradient norm)` from adding
   M4 to the baseline features;
2. out-of-fold mean-squared-error improvement for signed consensus-gradient
   utility from adding M4; and
3. the difference between M4 and reward variance in Spearman association with
   positive-part consensus-gradient utility.

Uncertainty uses 10,000 fold-stratified prompt-group bootstrap resamples. The
analysis is run independently for both sketch seeds.

Secondary analyses report rank correlations with exact gradient norm,
leave-one-fold-out influence, winsorization sensitivity, and results including
zero-M4 groups. They cannot override a primary failure.

## Decision ladder

Before scientific classification, all integrity gates must pass: 256 unique
groups, eight siblings per group, all finite summaries, exactly zero optimizer
and scheduler steps, learner version zero, identical pre/post parameter hashes,
all aborts acknowledged, no `finish_train_step` invocation, and both sketch
fidelity gates satisfied.

The scientific classification is then:

- `DIRECTIONAL_CONSTRUCT_PASS`: both sketch seeds show positive M4 incremental
  value for signed consensus utility and a positive M4-minus-reward-variance
  rank discriminant; the simultaneous 95% lower bounds exceed zero. The norm
  endpoint must also be positive, though it is not sufficient by itself.
- `MAGNITUDE_ONLY`: M4 improves prediction of exact gradient norm with a
  simultaneous 95% lower bound above zero, but the directional conditions do
  not all pass.
- `NO_CONSTRUCT_SUPPORT`: the magnitude condition does not pass.
- `MEASUREMENT_FAILURE`: any integrity, no-update, or sketch-fidelity gate
  fails.

Only `DIRECTIONAL_CONSTRUCT_PASS` authorizes design—not launch—of one matched
M4-Shield versus reward-variance outcome acquisition. `MAGNITUDE_ONLY` can
strengthen the paper's measurement interpretation but cannot justify a claim
that M4 prioritization improves learning. No result from this audit establishes
terminal quality, production readiness, or broad generalization.

