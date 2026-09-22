# Reward-variance OARS-v2 actuation qualification

The authenticated, outcome-excluded reward-variance qualification passes the
corrected systems gate. Pipeline `69339424` / child `69339594` ended in an
operational failure because the embedded analyzer rejected a valid bounded
replenishment state; pipeline status is not a scientific endpoint.

## Preserved evidence

- All 64 learner updates and all 64 scheduler decisions completed.
- Every decision exposed exactly eight candidates and all four scorers searched
  exactly 70 four-of-eight combinations, with zero skips or fallbacks.
- The enacted selection exactly matched the reward-variance proposal on all 64
  decisions; every selection contained four unique ready groups.
- The two-sided token-service band passed on every proposal.
- There were 228 authenticated stale-group removals, 59 replacement batches
  earned, 58 consumed, and one bounded credit outstanding at the fixed terminal
  horizon. The invariant `earned - consumed = outstanding` holds, and the
  outstanding credit is bounded by one.
- Candidate excess was zero throughout. This is a valid bounded observation,
  not missing accounting: both the ledger and lifecycle stream record zero.
- Combined gradient-observer plus scheduler-decision duty was
  `0.002087856835559417` (0.2088%), below the preregistered 1% limit.
- Runtime was `380.242641458` seconds, below 14,400 seconds.

The original result hash is
`1ba43fee623f9d33f0140d09f4c698a89b4530335e349c322605adf31c2f927b`.
The corrected result was recomputed offline from two byte-identical,
manifest-covered terminal copies; runtime code, assignments, selections, and
artifacts were unchanged. No training-quality outcome was acquired or analyzed.

## Gate consequence

Reward-variance actuation is qualified. No rerun is justified. The absolute-M4
qualification remains unsubmitted until its credential-free package is rebuilt
from the analyzer-repair source commit and revalidated.
