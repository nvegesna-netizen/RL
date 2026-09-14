# Reviewer gate: from OARS shadow implementation to causal evaluation

## Decision

The local shadow implementation is ready for a representative Linux
FIFO-controlled shadow preflight. It is **not** yet ready to support an online
OARS-effect claim or to skip directly to a whole-run actuation acquisition.

This ordering matters scientifically. The retrospective screen establishes
that a fully observed OARS choice would have retained more coefficient
opportunity than the recorded FIFO choice under a frozen per-decision service
budget. The local implementation gate establishes that the proposed choice can
be computed deterministically without changing FIFO behavior. Neither result
establishes that all required metadata arrive at every live decision boundary,
that the observer is cheap enough in the real controller, or that enacting its
choice improves a whole training run.

## Evidence already in hand

1. The existing randomized M4 study establishes a material pre-release
   opportunity-loss effect in its tested setting. It does not identify the
   effect of deferring an already-ready group or of changing the scheduler.
2. The baseline-budgeted retrospective OARS screen passes across the preserved
   cells, with positive one-step L1 retention gains while respecting the frozen
   `1.02x` valid-actor-token ceiling.
3. The default-off shadow wrapper preserves FIFO selection and wait behavior in
   focused tests, fails closed, protects packed payloads, and flushes a distinct
   provenance ledger.
4. The exact 25-candidate selector is deterministic and completes in 8.04 ms
   local CPU p95. That is selector-only evidence, not end-to-end controller
   overhead.

## Reviewer-visible gaps

- **Live observability:** opportunity sidecars may be incomplete or late at
  real selection boundaries even though the replay-buffer contract is correct
  in focused tests.
- **Systems cost:** macOS selector timing does not determine Linux controller
  duty, queue interaction, or training throughput.
- **Counterfactual support:** retrospective one-step proposals hold the
  recorded arrival process fixed. An enacted scheduler changes removals,
  subsequent buffer composition, and possibly future arrivals.
- **Shared-system interference:** candidates compete in one buffer. A
  group-level comparison would not isolate a no-interference unit effect.
- **End-to-end utility:** retaining more coefficient opportunity need not
  improve valid tokens per unit time, time to a fixed update budget, or final
  evaluation quality.

## Required shadow-only Linux preflight

Use one representative, short qualification run with OARS observation enabled
and weight-FIFO retaining exclusive control. This is a systems/instrumentation
preflight, not an OARS outcome acquisition. Freeze its identity, duration, and
gates before launch. Do not inspect final-quality outcomes to decide whether to
proceed.

All of the following must pass:

1. Actual selected group IDs exactly equal an independently reconstructed FIFO
   choice at every selection boundary.
2. Opportunity metadata coverage is at least 99% of contended, eligible
   decisions; every exclusion has an enumerated skip reason.
3. No payload mutation, duplicated group, cardinality error, service-budget
   violation, or non-finite value occurs.
4. OARS observer duty is at most 1% of controller active time, reported with a
   common-window denominator.
5. Decision latency p95 stays below 10% of the observed median interval between
   selection boundaries, and maximum latency is reported.
6. The ledger flushes on normal termination and authenticates against the
   source commit, protocol hash, and artifact SHA-256.

Failure of a systems gate permits diagnosis and a separately versioned repair;
it does not permit changing scientific endpoints or interpreting training
outcomes.

## Whole-run causal study unlocked only after a pass

If the shadow preflight passes, freeze a paired, run-randomized comparison of
FIFO control versus enacted OARS. Pair runs by model, workload, seed, update
budget, and resource envelope, then randomize the policy label within each
pair. Analyze the total effect of the scheduling policy at the run level; do
not describe it as an isolated group effect or as mediation through M4.

The primary scientific endpoint should be retained registered L1 coefficient
opportunity per eligible assignment over a common update window. Co-primary or
strictly ordered utility endpoints should cover valid actor tokens per update
and wall time to the fixed update budget. Final task quality belongs as a
prospectively powered secondary endpoint with an interval, not as a success
condition for the instrumentation preflight. Report enacted-policy compliance,
fallback frequency, observer duty, and the frozen `1.02x` service constraint.

## Paper consequence

The current sequence strengthens the paper by providing a concrete,
auditable solution derived from the measured failure mode, but it should be
presented only as a validated shadow implementation and prospective
intervention until the randomized actuation study exists. A passing whole-run
study would support the substantially stronger contribution:

> randomized measurement of pre-release opportunity attrition, followed by a
> budget-constrained scheduler that causally reduces that attrition without
> sacrificing registered systems utility.

No EOS launch, training run, qualification, acquisition, retry, or extension
was performed for this reviewer gate.
