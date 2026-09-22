# OARS-v2 authenticated offline autopsy

## Decision

The frozen retrospective development gate passed every condition and supports a
local, default-off OARS-v2 shadow implementation. It does not support an EOS
launch, enacted scheduler comparison, training-quality claim, or causal
mediation claim.

The outcome-aware composition extension supplies an important qualification:
M4 clearly adds selection information beyond conventional version age and
ready age, but most of its choices in these traces can be reproduced by the
much simpler group reward-variance signal. OARS-v2 should therefore implement
both scorers behind the same constraints and compare them directly. A shuffled
M4 control alone is too weak for the next live scientific study.

## Authenticated scope

- All 20 main archives matched their terminal authentication receipts.
- Every lifecycle, opportunity, and OARS ledger matched its declared SHA-256.
- All 1,280 decisions were reconstructed from controller-ordered lifecycle
  events, yielding 10,240 candidate-decision rows.
- Every reconstructed candidate set, FIFO baseline, OARS proposal, selected
  identity, L1 sum, and token sum agreed with the recorded decision.
- Each arm used 2,000 deterministic within-decision shuffled-score replays.

## Frozen gate results

The two-sided-budget M4 risk policy constrained every batch to 0.98--1.02 times
the recorded FIFO token count. The observed decision-level range was
0.980026--1.019996, and the aggregate ratio was 0.994854.

| Comparison | Aggregate result | Positive arms |
| --- | ---: | ---: |
| M4-risk L1 versus age-only | +31.20% | 20/20 |
| M4-risk L2 versus age-only | +25.71% | 20/20 |
| M4-risk true L1 versus shuffled-M4 mean | +27.74% | 20/20 |

The age-only comparator used learner-version lag and ready age with the same
batch cardinality and two-sided token constraint. The shuffled comparator kept
the same choice sets, score distribution, optimizer, and service constraint but
permuted score identities.

The expiry term made a real allocation tradeoff. Relative to maximizing total
M4 under the same token band, the risk-first policy selected 5.11% less total
L1 but 11.73% more imminent-expiry L1. This supports retaining an explicit risk
term rather than scheduling on raw opportunity alone.

## Load shedding and composition

The enacted traces show that OARS-v1 behaved as an opportunity-weighted load
shedder, not merely as a reorderer:

- In the ten acting arms, 2,360 groups that were deferred at their first
  observed choice later became stale. They carried 1,040,455 L1 in total, or
  about 441 per group.
- Only 108 initially deferred groups were later selected in those arms. They
  carried 164,506 L1, or about 1,523 per group.
- In the ten FIFO-observe arms, 2,520 initially deferred groups were later
  selected and none of those first-deferred groups became stale.

These are descriptive policy-generated trajectories, not a randomized
comparison of individual group fates. They nevertheless explain why OARS-v1's
one-sided service ceiling is a scientific confound: its aggregate selected-token
ratio was 0.94857, and an individual decision could use only 0.32407 times the
FIFO tokens. The shorter whole-run time can therefore arise partly from
intentional data shedding.

M4 selection also changed curriculum composition. Across arms, the selected
mean within-group reward variance was 0.11401 for two-sided M4-risk versus
0.09208 for age-only selection. Because GRPO coefficient opportunity is built
from reward-derived advantages and token support, this association requires a
stronger comparator than shuffled scores.

## Outcome-aware composition extension

The extension was specified after the frozen gate result and does not modify
that gate.

| Comparison | Aggregate L1 difference | Positive arms | Selection overlap |
| --- | ---: | ---: | ---: |
| M4-risk versus reward-variance-risk | +0.633% | 19/20 | 95.72% |
| M4-risk versus token-normalized-M4-risk | +0.305% | 15/20 | 97.27% |

M4 and reward variance selected exactly the same four groups in 88.44% of all
decisions. M4 retained 0.534% more imminent L1 than reward variance. Thus M4 is
clearly distinct from staleness, but the present traces do not show a large
operational advantage over a simple reward-heterogeneity proxy.

## Scientific interpretation

The evidence supports three conclusions:

1. Conventional staleness-first scheduling leaves substantial coefficient
   opportunity on the table, even under nearly identical per-decision tokens.
2. Opportunity-aware selection can be implemented cheaply and can act as
   principled overload shedding: retain high-value groups and allow lower-value
   work to expire or be replaced.
3. The next experiment must distinguish M4 from reward-variance scheduling.
   Real M4 versus randomly shuffled M4 would demonstrate score identity, but it
   would not answer whether M4 improves on a much simpler available proxy.

The replay remains one-step and conditions on queues generated by FIFO or
OARS-v1. It cannot predict the queue trajectory, cumulative training dose,
wall time, learning curve, terminal accuracy, fairness, or convergence under
OARS-v2.

## Next boundary

Implement a default-off, score-pluggable shadow controller with identical
candidate sets and constraints for:

- age/ready-time scheduling;
- reward-variance-at-risk scheduling;
- token-normalized M4-at-risk scheduling; and
- absolute M4-at-risk scheduling.

The implementation must remove the exact-eight hard dependency, must not drop
still-valid candidates merely because they fall outside a fixed candidate
window, must enforce a two-sided service budget, and must record explicit
load-shedding and replacement accounting. Only shadow equivalence, latency,
liveness, and naturally occurring contention should be evaluated next.

No training acquisition or EOS submission is authorized by this result.

## Provenance

- Frozen protocol SHA-256:
  `eda6abc50bf3327c2994de522062f9cca0c706d0168e02a9e5745e89d1891eea`
- Frozen result SHA-256:
  `c7d1d48bf9f2f8268187301d840e74c46c48c1b936045473446658df55f44e86`
- Composition-extension protocol SHA-256:
  `199026fdfc92ccd758f921016af138275f5f3a4b33fcca0ae761a613a4034b25`
- Composition-extension result SHA-256:
  `987b5aedb970b1a4b4ec0ec3335a007f7e6b8564bf6a8560b4e22e391fe3017e`

Large raw archives remain outside Git. Their authenticated paths and SHA-256
digests are recorded per arm in `offline_autopsy_result.json`.
