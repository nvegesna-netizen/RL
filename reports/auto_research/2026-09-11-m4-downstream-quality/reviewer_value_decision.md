# Reviewer-value decision for a downstream-quality study

## Decision

Open a separate prospective downstream-quality study, but do not treat its GPU
acquisition as authorized yet. The completed M4 evidence is sufficient for the
bounded causal systems-measurement claim. A run-randomized final-quality result
is nevertheless the highest-value scientific addition because it addresses the
main remaining link between the measured proximal outcome and system utility.

The addition is worthwhile only if it is powered at the training-run level. A
few-seed or outcome-guided extension would not resolve the reviewer concern and
could weaken the paper by inviting pseudoreplication and low-power objections.

## Evidence

- Before this branch, the review PDF was five pages including references, which
  began on page four, despite a ten-page main-paper allowance. The present
  branch expands that account to six pages; further space should be used only
  where it improves the instrument, estimand, dose, practical context, or
  implications.
- M4 has strong within-scope evidence: 14 definitive acquisitions, 106,653
  terminally scored group assignments, two model families, two Llama sizes, and
  three math workloads. Another nearby model or workload cell has lower marginal
  value than an end-to-end outcome.
- The closest asynchronous LLM-RL systems papers pair systems claims with final
  performance, convergence, faster learning, or explicit throughput gains. M4 is
  a measurement paper rather than a scheduler paper, but the absence of either
  final quality or a scheduler-level utility result remains its clearest impact
  limitation.
- The existing acquisitions cannot be reanalyzed into a final-quality contrast:
  both group-level arms updated one shared policy, validation was disabled, and
  checkpointing was disabled. Final model quality requires independently
  randomized training runs.

## Claim boundary

The clean causal question is:

> Does the controlled-delay training regime that produces M4 opportunity loss
> change final held-out model quality relative to immediate release?

This identifies the total effect of the randomized delay regime. It does not by
itself identify the causal effect of the opportunity-loss mediator, because
delay can also change ordering, staleness, congestion, and other learning
dynamics. Publication wording must preserve that distinction.

## Ordered work

1. Expand the current paper without changing its existing claims.
2. Recover the provenance of the 0.20 M4 threshold and report honestly whether
   an operational rationale was recorded before outcomes.
3. Produce an offline trace/dose audit from authenticated lifecycle ledgers.
4. Freeze a run-randomized quality protocol, including a human-approved quality
   materiality or equivalence margin.
5. Implement and locally test a post-boundary terminal policy export plus an
   independent fixed-set evaluator. Do not add in-loop validation traffic or
   imply resumable-checkpoint support in SingleController.
6. Package a credential-free no-training preflight.
7. After separate launch authorization, run a nonconfirmatory blinded variance
   and endpoint-feasibility pilot.
8. Proceed to confirmatory acquisition only if the independent-run requirement
   fits a prospectively capped budget.

## Threshold provenance finding

The value `0.20` predates the first causal outcome and appears in the initial
opportunity-loss analysis implementation at commit `45edf5678`. It was carried
into the original prospective follow-up protocol at `3681e528b`. The recovered
records establish temporal prespecification, but they do not state an external
operational derivation for the value. The paper should therefore call it a
prespecified decision threshold, report continuous estimates and threshold
sensitivity prominently, and must not invent a retrospective business or
learning-quality rationale.
