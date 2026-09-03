# M4 opportunity-loss prospective follow-up

Status: prospective design executed and complete. The follow-up concluded
`MATERIAL`, with mechanism `REPLICATED` and portability `SUPPORTED`. This does not
retroactively change the earlier confirmatory result, which remains
`INCONCLUSIVE`; see `result.md` for terminal evidence.

## Decision

Do not repeat the 224-step control:d5:d10 acquisition. The next study should be a
new, fixed-sample control:d5 experiment using prospectively specified pre-delay
covariate adjustment. The earlier d10 arm has served its positive-control purpose:
the M4 delay mechanism replicated with the registered dose ordering. Removing d10
from a future allocation places all compute on the primary randomized contrast.

The candidate planning target is 80% power at `Delta_L=0.25` against the material
null `Delta_L<=0.20`, with one-sided alpha 0.05. This chooses a 0.05 design margin;
it does not assert that the completed point estimate of 0.20981 is the truth.

## Evidence for redesign

The audit reconstructs all 3,298 primary assignments directly from the frozen EOS
artifact and exactly reproduces the registered estimate 0.20980682915795912 and
HAC standard error 0.04680227415847308. There were no missing terminals. Roughly
half of assignments had zero pre-delay opportunity: 49.7% in control, 53.7% in d5,
and 54.2% in d10.

Cross-fitted influence-score residualization was evaluated only as prospective
design evidence. Across 4, 8, and 16 contiguous start-version folds:

- pre-delay `Q` alone reduced estimated HAC variance by 50.7--50.9%;
- `Q` plus `1[Q=0]` reduced it by 73.3--73.6%;
- adding linear start version did not improve the conservative result.

These are exploratory estimates from the completed study and cannot reclassify its
primary endpoint. The follow-up power calculation therefore credits only 50%
variance reduction, below the observed cross-fitted reduction for the selected
two-covariate candidate.

The first formal generalized-regression implementation is now present. Applied
exploratorily to the frozen acquisition, its estimate was 0.3086--0.3090 with HAC
standard error 0.02365--0.02397 across 4, 8, and 16 folds. Restricting the
standardization population to control and d5 rather than using pre-treatment Q
from all three randomized arms changed the point estimate by less than 0.0002.
This stability supports testing the estimator prospectively, but the result is
post-selection evidence and is not a new confirmatory conclusion.

Removing d10 at fixed total assignment count provides a separate allocation gain:
only 82.78% of the completed assignments belonged to the control:d5 primary pair.
At otherwise equal information, reallocating d10 to a 1:1 control:d5 design lowers
the projected standard error from 0.04680 to 0.04258.

## Candidate fixed design

- Arms: control and d5 only, allocated 1:1 by a new counter-based randomization
  domain and seed fixed before acquisition.
- Runtime: retain the model, dataset, global batch, sibling count, queue geometry,
  staleness bound, and common opportunity instrumentation from the completed
  study.
- Window: 8 burn-in versions, 400 primary start versions, and 40 terminal-guard
  versions, for 448 successful trainer steps.
- Expected primary assignments: about 7,495 at the observed 18.74 assignments per
  primary version. The conservative normal calculation requires 7,395
  control-plus-d5 assignments for 80% power at `Delta_L=0.25` after crediting 50%
  variance reduction.
- Primary estimand: the same population opportunity-loss ratio, with a
  prospectively frozen generalized-regression estimator using only pre-delay `Q`
  and `1[Q=0]`.
- Continuity analysis: also report the original unadjusted registered estimator,
  but do not let it override the new primary estimator.
- Inference: cluster by start version and retain conservative HAC plus circular
  moving-block bootstrap intervals. Exact folds, coefficient fitting, influence
  function, bootstrap, and missing-terminal handling must be frozen in code and
  tested before any launch decision.
- Mechanism support: retain the d5-control direct-chain and version-advance checks.
  D10 dose-order checks are historical evidence, not part of the new acquisition.
- Acquisition count: one. No rerandomization, identical retry, or outcome-guided
  extension.

## Why this is the smallest useful follow-up

With the unchanged estimator, 80% power at the observed 0.00981 margin above the
threshold would require about 384,422 control-plus-d5 assignments. Even a 50%
variance reduction leaves roughly 192,000, so a study designed specifically around
the observed point estimate is not proportionate.

At a 0.05 margin (`Delta_L=0.25`), the conservative 50%-reduction design requires
7,395 primary assignments. That is close to two times the completed trainer-step
budget, rather than more than one hundred times its primary information. It is
large enough to answer a useful materiality question and small enough to justify a
formal implementation screen.

## Required implementation gate

Before any GPU launch or EOS submission:

1. Implement the exact cross-fitted generalized-regression point estimator rather
   than merely adjusting a reported standard error.
2. Prove on randomized synthetic fixtures that it controls type-I error and remains
   unbiased under null, heterogeneous, zero-inflated, and version-drift outcomes.
3. Run an empirical version-cluster resampling design simulation using the frozen
   acquisition, applying prospective effect shifts without reusing its observed
   treatment labels as a new confirmatory result.
4. Require at least 80% simulated power at `Delta_L=0.25` under the conservative
   planning rule and record sensitivity at 0.225, 0.30, and 0.40.
5. Freeze a new protocol, source commit, no-training preflight, analysis lock, and
   acquisition authorization. The current exploratory outputs grant no compute or
   launch authority.

The local implementation gate is complete. The point estimator was introduced at
`7403f11fb1a21f6f9b0e69c47460a1fca2db047f` and its inference contract was frozen
at `1bf0eb89666128139863ef09acd14dd8313d9b48`. The adjusted and registered
inference suites pass 16/16 together. They cover deterministic sharp-null
calibration, circular version-block bootstrap inference, row-order determinism,
sharp missing-terminal bounds, a 1% per-arm terminal-coverage gate, and invalid
bootstrap contracts.

The 20,000-draw empirical circular-block simulation uses seed `20260904`, block
size 8, 400 target primary versions, and the conservative 50% variance-reduction
credit. It estimates power of 0.36910 at `Delta_L=0.225`, 0.81555 at 0.25,
0.99925 at 0.30, and 1.00000 at 0.40. The 0.25 target therefore passes the
prospective 80% gate. Its conservative target standard error is 0.0199726, larger
than the raw adjusted resampling estimate of 0.0160425.

The protocol and no-training preflight were frozen successfully, followed by the
single registered acquisition. Parent pipeline `65985843`, downstream pipeline
`65986159`, and EOS compute job `423739790` completed successfully. No retry or
extension occurred.
