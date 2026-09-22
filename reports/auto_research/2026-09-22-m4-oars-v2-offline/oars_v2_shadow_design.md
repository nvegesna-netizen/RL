# OARS-v2 shadow design boundary

## Objective

Build a production-shaped observer that answers two separate questions before
any actuation:

1. When naturally available ready work exceeds one learner batch, how much
   value can opportunity-aware selection preserve over age-first scheduling?
2. Does absolute M4 materially change decisions beyond reward variance and
   token-normalized M4?

The observer must not create contention by waiting for an exact watermark. If
the natural ready set contains only one feasible batch, every scorer records
the same immediate-dispatch decision.

## Decision contract

For a ready candidate `g`, record:

- stable group identity;
- start learner version and current learner version;
- ready timestamp and current ready age;
- valid actor tokens;
- mean and variance of sibling rewards;
- L1 and L2 coefficient opportunity; and
- whether deferring the group across the next learner update would make it
  stale under the configured policy.

At each natural dispatch point:

1. Compute the existing FIFO batch.
2. Set a two-sided valid-token band of 0.98--1.02 times FIFO tokens. FIFO is
   therefore always feasible.
3. Evaluate the same feasible batches with each frozen scorer.
4. Record proposed identities, overlap, opportunity, tokens, age, reward
   composition, solver latency, and skip/fallback reason.
5. Delegate the actual selection to FIFO.

Risk-aware scorers maximize imminent-expiry value first, total value second,
and lower tokens third. The value is respectively ready/version age, reward
variance, token-normalized L1, or absolute L1.

## Adaptive candidate handling

- Do not require exactly eight ready groups.
- Do not discard the newest groups to force a fixed window.
- Evaluate all eligible ready groups up to a safety cap.
- For small sets, use exact fixed-cardinality enumeration.
- Above the exact-search cap, use deterministic top-value pruning that always
  retains the FIFO batch and every imminent-expiry group, followed by exact
  enumeration of the reduced set.
- If metadata is missing, the safety cap is exceeded, or the decision-time
  budget expires, record the reason and use FIFO.

The shadow must separately record whether a future acting policy would defer,
replace, or deliberately shed a candidate. Silent expiration is not a service
policy.

## Systems gates

Before an enacted qualification:

- disabled mode must be exactly equivalent to the existing sampler;
- shadow mode must match FIFO identities at every decision;
- all scorer proposals must satisfy cardinality and the two-sided token band;
- metadata coverage must be complete or produce an explicit FIFO fallback;
- no observer path may remove or mutate a candidate;
- p95 shadow-decision latency and total observer duty must each remain below
  1% of controller time;
- stress tests must cover four through the safety-cap candidate groups,
  cross-version candidates, stale eviction, missing metadata, solver timeout,
  shutdown, and sustained overload; and
- liveness must not depend on a replacement credit or exact candidate
  watermark.

## Scientific successor if shadow qualification passes

Use matched-seed whole-run arms with the same adaptive dispatch rule and
two-sided service contract:

1. age/FIFO baseline;
2. reward-variance-at-risk; and
3. absolute-M4-at-risk.

The primary practical contrast should be the opportunity-aware family versus
the baseline. The signal-specific contrast must be M4 versus reward variance,
not M4 versus shuffled scores alone. Fix cumulative valid-token exposure and
evaluate learning-curve area under the curve; report fixed-update and
fixed-wall-time views secondarily. Record explicit shed/replacement counts,
tail waiting time, prompt coverage, and score-dependent curriculum composition.

This document specifies local design work only. It does not authorize a
training run, qualification, EOS submission, retry, or acquisition.
