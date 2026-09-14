# OARS shadow implementation gate

Status: **PASS**

The frozen baseline-budgeted opportunity-at-risk scheduling (OARS) policy is
implemented as a default-off, observation-only wrapper around the existing
weight-FIFO sampler. When enabled, it records a counterfactual proposal and
delegates the actual choice to the unchanged baseline sampler.

## What passed

- Disabled, enabled-shadow, uncontended, and older-incomplete-version cases
  preserve the baseline sampler's choice and wait behavior.
- The proposal is deterministic, selects four groups, and stays within the
  frozen `1.02x` per-decision valid-actor-token budget.
- Missing opportunity metadata and choice sets larger than 25 groups fail
  closed: the baseline still runs and the observer records an explicit skip.
- Opportunity sidecars accept finite scalar fields only and do not mutate the
  packed training payload.
- Lifecycle, opportunity, observer-duty, and OARS ledgers must use distinct
  paths, and the OARS ledger is flushed during controller shutdown.
- Focused evidence comprises 6 selector/sampler contract tests, 24 replay
  buffer tests, 4 shadow-configuration tests, and 1 controller shutdown/flush
  test. Ruff and diff checks passed; Pyrefly reported 0 errors (1 suppression).

## Maximum-choice microbenchmark

The exact selector enumerated all `25 choose 4 = 12,650` combinations in each
of 250 measured trials after 20 warmups. It was deterministic and satisfied
the service budget in every trial. On the local arm64 macOS CPU, selector-only
latency was 6.34 ms median, 8.04 ms p95 (nearest rank), and 10.39 ms maximum.

The canonical benchmark record is `shadow_microbenchmark_result.json`. Its
implementation SHA-256 is
`1167c389adfb8754446bbb2647ccd9ff73875549836c661bfb0571730621e813`; the
benchmark-script SHA-256 is
`478be3c3540b085feebf842352c63728c12b2973fdbe5d4b49d032cb041e2dc7`.

## Claim boundary and next decision

This gate establishes implementation behavior and local CPU selector cost. It
does not establish end-to-end controller overhead, online policy effect,
training throughput, or final training quality. The repository lock is
Linux-only, so focused tests were run in an isolated macOS CPU environment.

The next justified step is a representative Linux shadow-only systems
preflight with FIFO still controlling selection. It should test live metadata
coverage, exact FIFO equivalence, skip reasons, decision latency, and observer
duty. Only a passing preflight should unlock a prospectively frozen,
randomized whole-run FIFO-versus-OARS causal study.
