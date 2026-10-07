# M4-Shield causal utility at eligible live decisions

Date: 2026-10-07

Parent evidence commit: `d13692bbfca72b7f0566184697b5a13bf9308e06`

## Scientific question

At a live asynchronous training decision where the prospectively qualified
M4-Shield action differs from the reward-variance base action while preserving
the registered service and reward-variance contracts, does taking the Shield
batch cause a better immediate learning update on data that was not used to
choose either action?

This is a new successor study. It is not a retry or reinterpretation of the
closed version-zero conditional-M4 capsule qualification.

## Why this is the right decision regime

The authenticated M4-Shield live qualification observed five admissible Shield
interventions in 64 decisions. Each intervention changed the selected groups,
strictly increased imminent M4 L1, preserved both total and imminent
reward-variance utility exactly, and stayed within the 0.98--1.02 FIFO-relative
token band.

The later version-zero conditional-norm collector tested a different and more
restrictive regime. Exact reconstruction of its 16 frontiers reproduces the
runtime gains: fourteen were exactly zero and the two positive gains were
1.1847% and 3.6869%. None reached its frozen 10% gate. That result closes that
instrument but does not remove the already demonstrated live Shield
eligibility event.

The causal study therefore keys eligibility to the qualified Shield event,
not to predicted gradient-norm gain and not to learner version zero.

## Treatment, comparator, and eligibility

For one controlled eight-group frontier at learner version `v`:

- comparator `B`: the exact `reward_variance_risk` four-group proposal;
- treatment `S`: the exact `m4_shield` four-group proposal;
- eligible decision: `S != B`, both selectors complete exact search without
  fallback, `S` preserves total and imminent reward-variance sums to absolute
  tolerance `1e-12`, `S` retains at least two groups from `B`, both proposals
  satisfy the 0.98--1.02 FIFO token band, and `S` has strictly greater imminent
  M4 L1 than `B` by more than `1e-12`.

Only the first eligible decision in a trajectory may create a capsule. No
learning outcome, gradient, or post-update model quantity participates in
eligibility.

## Experimental unit and paired intervention

The independent unit is a preselected training seed/trajectory, not a scheduler
decision. Each valid trajectory contributes at most one capsule. From that
single captured pre-update state, two isolated replay arms are run:

1. restore the same policy, optimizer, scheduler, and RNG state and execute
   exactly one update on `B`;
2. restore the same state again and execute exactly one update on `S`.

The two arms use the exact retained tensors from the shared live frontier.
They do not regenerate rollouts, recompute action membership, or resample a
candidate set.

## Held-out endpoint

Four complete prompt groups generated at learner version `v`, disjoint from
the eight candidate groups, are retained by the deterministic rule "first four
ready current-version groups after the eligible frontier." Their prompts,
responses, old-policy log probabilities, reference-policy log probabilities,
rewards, masks, and pre-update GRPO advantages are frozen in the capsule.

Let `L(theta; H)` be the token-normalized clipped-PG loss evaluated without a
backward pass or optimizer/scheduler mutation on those frozen held-out tensors.
For arm `a` in `{B,S}`, define

`Delta_a = L(theta_after_a; H) - L(theta_before; H)`.

The primary paired endpoint is

`D = Delta_S - Delta_B`.

Lower is better, so `D < 0` favors M4-Shield. Pre-update loss equality across
arms is an integrity gate, not a result. Secondary mechanism endpoints are
update norm, train-batch gradient norm, held-out-gradient dot product, clipping,
and KL; they cannot replace the primary endpoint.

## Ordered sequence and outcome embargo

1. Freeze this design and its machine-readable protocol.
2. Implement a default-off first-eligible capture path that records the exact
   frontier, both proposal identities, all retained tensors, the pre-update
   training checkpoint, and recursive hashes.
3. Run one outcome-excluded systems qualification. It must capture a reachable
   eligible decision, authenticate every tensor and checkpoint file, preserve
   the pre-update parameter hash, and execute no update at the captured
   decision.
4. Independently qualify checkpoint restoration and loss evaluation without
   opening a treatment/comparator post-update contrast.
5. Predeclare a fixed acquisition set of 20 independent seeds. Proceed to
   outcome opening only if at least 16 yield valid authenticated capsules.
6. Run both one-update arms for every valid capsule. Authenticate all arms and
   pass the all-capsule completion gate before reading any post-update loss.
7. Analyze all valid predeclared capsules with a paired mean and 95% studentized
   bootstrap interval, a paired t interval, and an exact sign-flip test. Report
   every `D`; do not relabel numerical estimates as categorical decisions.

## Failure and stopping rules

- A qualification trajectory with no eligible decision by 64 completed updates
  is `NO_ELIGIBLE_DECISION`, not evidence about causal utility.
- Any selector fallback, incomplete frontier, state-hash mismatch, missing
  checkpoint/tensor file, non-disjoint held-out group, or update at the captured
  decision fails closed before outcomes.
- A failed seed is not replaced outside the fixed 20-seed acquisition set.
- Fewer than 16 valid capsules stops the acquisition unopened.
- No threshold, eligibility rule, endpoint, seed, or analysis method may be
  changed after an outcome-bearing arm is run.
- A one-update result does not establish terminal-quality improvement. A
  long-horizon randomized scheduler trial is a separate study and is considered
  only after this bridge is complete.

## Current authorization boundary

The current phase authorizes protocol freezing, implementation, local and
clean-room verification, and one outcome-excluded live capture qualification.
It does not authorize opening paired post-update outcomes as part of the same
qualification.
