# M4-Rescue live qualification diagnosis

## Terminal result

The repaired JET package executed correctly, all three preflight suites passed,
and training completed all 64 learner updates. The frozen qualification then
returned `FAIL` because `complete_liveness_accounting` was false. Both terminal
artifact copies authenticate byte-for-byte.

This is not a packaging, allocation, timeout, GPU, or training failure. It is
also not evidence about terminal training quality, which remained excluded.

## Immediate gate failure

Every decision contained a complete liveness-accounting record. All counters
were nondecreasing and mutually consistent, and lifecycle removals reconciled.
However, the analyzer additionally required at least one stale eviction. The
run had zero stale evictions, zero candidate-excess removals, and zero
replenishment credits, so that final positive-event requirement failed.

Removing that requirement after observing the run would not make this a useful
qualification pass, because the actuation path was never exercised.

## Reachability finding

M4-Rescue made 0 non-FIFO interventions in 64 decisions. This was not caused by
the token or reward-variance constraints:

- Decision 1 had zero imminent groups.
- Each of decisions 2--64 had exactly four imminent groups.
- FIFO selected every imminent group on every decision.
- No imminent group was ever deferred by FIFO.
- Even without the token, reward-variance, or one-swap restrictions, no
  decision admitted a strict increase in imminent L1 over FIFO.

The live queue was therefore an invariant FIFO trajectory: four lag-one groups
were consumed while four current-version groups replaced them. A policy that
returns FIFO unless it can improve imminent L1 cannot bootstrap away from this
state.

## Why the offline replay looked positive

The authenticated 1,280-decision development corpus separates cleanly by the
policy that generated each trajectory:

| Historical trajectory | Decisions | Decisions with deferred imminent groups | M4-Rescue replay interventions |
|---|---:|---:|---:|
| FIFO arms | 640 | 0 | 0 |
| OARS arms | 640 | 599 | 258 |

All 258 offline interventions came from OARS-generated queue states. In 580 of
640 OARS decisions all eight candidates were already imminent; by contrast,
each FIFO arm reproduced the same zero-opportunity structure seen live (one
initial decision with zero imminent groups, then 63 with exactly four, all
selected by FIFO).

The offline result therefore established conditional rescue feasibility on
backlogged states created by another active scheduler. It did not establish
that standalone M4-Rescue could reach those states from FIFO. The live run
falsified that standalone reachability assumption in the tested environment.

## Scientific disposition

The correct disposition is `AUTHENTICATED_QUALIFICATION_FAIL`, with no same-
design retry and no post hoc analyzer-only pass. The result is useful: it
identifies M4-Rescue as a potential safety layer for an aggressive scheduler,
not a standalone replacement for FIFO under the tested steady-state workload.

Any continuation should be a newly frozen hypothesis. The most defensible
next design is an outcome-excluded hybrid qualification in which a fixed,
previously validated non-FIFO proposer creates the candidate choice and an M4
safety layer can veto or repair choices that would shed high registered
opportunity. Its protocol must require prospectively:

1. reachable contention under the intended deployment mechanism;
2. at least one authenticated safety-layer intervention;
3. exact proposal-to-action identity;
4. complete liveness reconciliation without requiring an event that the design
   does not guarantee;
5. bounded service, reward-composition, overhead, and fallback behavior.

No such continuation is authorized by the completed qualification protocol.
