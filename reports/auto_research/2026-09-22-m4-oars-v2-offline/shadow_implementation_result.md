# OARS-v2 local shadow implementation

## Result

The default-off, observe-only OARS-v2 shadow is implemented and passes the
local implementation gate. It is not yet live-shadow qualified and it has not
been enabled for training or submitted to EOS.

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

## Claim boundary and next gate

This result establishes a locally verified instrument, not an improved live
scheduler. It does not estimate queue feedback, cumulative training dose,
learning quality, wall time, convergence, or causal mediation.

The next scientific step is one frozen live shadow qualification. It must use
a dependency-complete build and show:

1. actual selected identities match FIFO at every decision;
2. naturally occurring ready-set contention is sufficient to compare scorers;
3. proposal token bands and metadata coverage are complete;
4. fallback rates are acceptable and fully explained; and
5. p95 shadow latency plus combined observer duty remain below the prospective
   systems threshold.

Only after those gates pass should an enacted matched-seed comparison of FIFO,
reward variance, and absolute M4 be designed. The primary outcome of that later
study should be fixed-token learning-curve area, with fixed-update and
fixed-wall-time views secondary.
