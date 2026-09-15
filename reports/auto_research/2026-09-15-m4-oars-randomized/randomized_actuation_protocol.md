# Randomized FIFO-versus-OARS actuation protocol

Status: **frozen before actuation qualification**.

## Scientific question

In the already tested Llama-3.2-1B/GSM8K asynchronous training setting, does
enacting baseline-budgeted Opportunity-at-Risk Scheduling (OARS) retain more
registered L1 coefficient opportunity than weight FIFO while preserving the
registered systems utility of the fixed 64-update run?

This is a whole-run policy intervention. It is not a group-level causal effect,
a mediation analysis, or a test of generalization. Final task quality is a
secondary estimate and is not powered as the success gate.

## Dependency-gated sequence

1. Implement a default-off `observe`/`act` mode without changing default FIFO
   behavior.
2. Pass unit, failure-atomicity, configuration, and authenticated-live-ledger
   replay tests locally.
3. Freeze and run one 64-update FIFO/OARS qualification pair using seed
   `20261421`. The pair is systems-only and excluded from every confirmatory
   estimate.
4. If and only if every qualification gate passes, execute the ten frozen
   matched pairs in `randomized_actuation_protocol.json`, serially and without
   interim outcome inspection.
5. Authenticate all terminal artifacts before opening outcomes, then evaluate
   the registered endpoints once.

## Confirmatory design

The analysis unit is a matched run pair. Both arms share the model, GSM8K
workload, training seed, 64-update horizon, two-GPU EOS envelope, controlled
release schedule, watermark of eight ready groups, four groups per update, and
maximum staleness one. Policy order within each pair was frozen using
`random.Random(20260915).getrandbits(1)`.

The FIFO arm observes but does not enact OARS. The OARS arm enacts the exact
baseline-budgeted OARS-v1 proposal. Both arms therefore carry the same
measurement path. The OARS token ceiling remains `floor(FIFO tokens * 1.02)`
at each decision.

The fixed sample is ten complete pairs. No replacement, retry, extension, or
interim outcome inspection is allowed by this protocol. If any arm is missing
or invalid, the primary result is `INCONCLUSIVE` unless a separately versioned
amendment is authorized before inspecting scientific outcomes.

## Endpoints and decision rule

The primary estimand is the mean paired OARS-minus-FIFO difference in retained
registered L1 coefficient opportunity per selected eligible prompt group over
updates 1--64. Superiority requires the lower endpoint of its two-sided 95%
paired-t interval to exceed zero. An exact paired sign-flip randomization
p-value is reported as robustness evidence.

The ordered systems gates are:

1. perfect enacted-policy compliance, complete metadata, zero fallbacks/skips,
   exact four-group cardinality, and the `1.02` decision-level token ceiling;
2. the upper endpoint of the two-sided 95% paired interval for
   `log(OARS/FIFO)` time to update 64 is below `log(1.10)`.

Valid actor tokens per update is reported with a paired ratio and 95% interval
as realized training dose; it is not mislabeled as throughput or suppressed if
it is lower under OARS. Terminal GSM8K quality is a secondary paired estimate
with a 95% interval and cannot rescue or veto the primary result.

The overall registered label is `MATERIAL` only when all ten pairs are valid,
the opportunity endpoint passes superiority, and both ordered systems gates
pass. Otherwise the result is `INCONCLUSIVE` or `NON-MATERIAL` as implied by
the primary interval, with every estimate still reported.

## Power and limitations

The retrospective 14-cell policy-development screen had mean fractional L1
gain `0.4078` (sample SD `0.1556`). That is not an online actuation effect or a
paired variance estimate, so it is used only for design sensitivity. A frozen
100,000-draw Gaussian calculation found that ten pairs have approximately
`0.803` power for paired standardized effect `1.0` and `0.957` power for
`1.31` when success means a two-sided 95% interval excludes zero. Final task
quality remains explicitly underpowered.

## Qualification gates

Both qualification arms must reach exactly 64 updates and exit zero. Their
lifecycle, opportunity, OARS, duty, result, and digest artifacts must
authenticate. Every decision must see exactly eight candidates, enumerate all
70 four-group combinations, select four unique ready groups, contain complete
metadata, satisfy the token ceiling, and match its registered policy. Fallback
and skip counts must be zero, and OARS decision duty must be at most 1% of
controller active time.

The qualification is interpreted only against these systems gates. Its arm
outcomes cannot change the confirmatory design.

The machine-readable protocol contains the frozen seed/order table, exact
claim boundary, stopping rules, and SHA-256 provenance.
