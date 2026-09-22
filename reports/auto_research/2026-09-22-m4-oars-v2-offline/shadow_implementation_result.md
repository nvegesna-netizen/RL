# OARS-v2 local shadow implementation

## Result

The default-off, observe-only OARS-v2 shadow is implemented and passes both the
local implementation gate and the live safety/overhead gate. In the frozen
64-update live qualification, eager FIFO remained the sole acting selector at
every decision. The run did not provide natural policy-comparison support:
every decision exposed exactly four ready groups for a four-group batch.

The observer leaves eager weight-FIFO in control of admission, eviction, and
selection. At every natural dispatch point it evaluates four counterfactual
policies on the same ready set:

- learner-version and ready-time age;
- sibling reward variance at risk;
- token-normalized M4 at risk; and
- absolute M4 at risk.

Every proposal keeps four prompt groups and 0.98--1.02 times the valid actor
tokens in the FIFO batch. The implementation requires no fixed eight-candidate
watermark and does not drop work to manufacture a choice set. Candidate-cap,
metadata, mandatory-pruning, solver-time, and invariant failures are recorded
as explicit FIFO fallbacks.

## Why this is the correct successor

The authenticated 20-arm offline autopsy reconstructed all 1,280 decisions.
Two-sided M4 recovered 31.20% more L1 opportunity than age/FIFO in aggregate,
but only 0.633% more than reward-variance scheduling, with 95.72% mean selection
overlap. The new instrument therefore compares M4 directly with the simple
reward-variance proxy instead of using shuffled M4 as its only signal control.

The ledger records candidate identity, learner-version lag, ready age, L1/L2,
valid tokens, reward mean and variance, next-update expiry, FIFO membership,
scorer proposals, overlap, replacements, deferrals, hypothetical imminent
shedding, and decision latency. These fields are sufficient to decide whether
M4 adds operational information beyond age and reward variance before any
actuation study is considered.

## Verification

- The four frozen autopsy tests and two composition-extension tests pass.
- Both full analyses regenerate byte-for-byte from all 20 authenticated
  terminal archives.
- Fourteen focused OARS-v2 contracts cover reward moments, score-specific
  choices, two-sided token feasibility, natural and uncontended dispatch,
  candidate counts 4--64, cross-version expiry, safe pruning, refusal to drop
  mandatory candidates, stale eviction, missing metadata, safety-cap and
  timeout fallback, sustained overload, FIFO identity equivalence, and ledger
  shutdown.
- Ruff, formatting, Python 3.13 compilation, and `git diff --check` pass.
- The complete repository pytest suite could not be collected locally because
  the checked-in environment lacks `ray` and `torch`; this is recorded as an
  environment blocker, not converted into a test pass.

A local selector microbenchmark showed median per-scorer costs of 0.086 ms at
8 candidates, 0.563 ms at 12, 2.053 ms at 16, 5.529 ms at 20, and 14.469 ms at
25. This is not a production latency result. It motivated a default exact
frontier of 16 under the 5 ms per-scorer guard; larger sets are deterministically
pruned while retaining FIFO and every imminent candidate.

## Live qualification and claim boundary

The authenticated live result is
`PASS_SAFE_INSUFFICIENT_NATURAL_CONTENTION`. All 64 updates and shadow
decisions completed; actual identities matched eager FIFO in every decision;
proposal contracts were complete; there were no skips or fallbacks; combined
observer duty was 0.1032%; and p95 shadow latency was 0.212 ms, or 0.00595% of
the median learner-step interval. These observations qualify implementation
safety and overhead in this setting.

They do not qualify scheduler discrimination. Ready-set size was exactly four
at all 64 decisions, so each scorer had only the FIFO batch available and all
proposal overlaps were mechanically four of four. Repeating the same eager
FIFO design is not justified: it cannot reveal relative choices unless the
runtime happens to accumulate more than one feasible batch.

The next defensible study must explicitly target a contention regime and apply
the same admission rule to every randomized policy arm. That design change is
an estimand change, not a repair or retry of the natural-cadence qualification.
It should compare FIFO, reward variance, and absolute M4 under a common
prospectively declared candidate frontier, with fixed-token learning-curve area
primary and fixed-update and fixed-wall-time views secondary. This result does
not estimate queue feedback, cumulative training dose, learning quality,
convergence, or causal mediation.
