# Post-M4-Shield reviewer gap audit

Date: 2026-10-07

## Verdict

The M4-Shield result materially improves the paper's solution-side
contribution. The paper now progresses from causal diagnosis, through
randomized scheduler tests, to a prospectively frozen live demonstration that
M4 can repair individual scheduling actions under explicit constraints. This
is credible evidence of **constrained actuation**. It is not evidence that the
repair improves terminal training quality or that M4-Shield is ready for
production deployment.

The result should remain a secondary systems contribution. The paper's primary
claim is still the causal measurement and cross-setting characterization of
opportunity loss.

## Evidence checked

- Frozen status: `PASS_M4_SHIELD_QUALIFIED`.
- Frozen protocol SHA-256:
  `a3c8f84297c2ce60c5cd5f0f0eb0a0286555ced63bbe5b169af0e3b008aca216`.
- Qualification-result SHA-256:
  `df6fb97686df03b8833cd9c5358588f574058dd5f46c1773d18376ee9e35e9bf`.
- Decision-ledger SHA-256:
  `4dfd6886238db11aecc6980f24b698c9578eb4c14a99139ca5751c080adac119`.
- Observer-duty SHA-256:
  `611b251dbbd292495a1af7b04f3d7e928fa0b9fb147525558da2252700eec7e7`.
- 64 decisions, five interventions, and 64/64 identity between proposed and
  enacted actions.
- Aggregate imminent-L1 gain of 685.0771 over the reward-variance base
  proposals.
- Exact zero difference for both registered reward-variance utilities.
- 64/64 exact searches without fallback and 64/64 decisions in the service
  band.
- Combined gradient-observer and decision duty of 0.00318175.
- Bounded shutdown, stale eviction, and replenishment accounting satisfied the
  frozen liveness gates.
- `training_quality_analyzed` and `scientific_outcome_acquisition` are false.

## Strongest reviewer objections

1. **The optimized objective is not independent validation.** The shield was
   designed to improve imminent M4 L1, so the positive gain demonstrates that
   its constrained optimization operated correctly; it does not validate M4
   against an external outcome.
2. **There is no terminal-quality endpoint.** The qualification intentionally
   excludes training quality. It cannot resolve the proxy--quality separation
   observed in the larger randomized follow-up.
3. **There is no randomized live comparator.** Counterfactual comparisons are
   to the logged reward-variance base proposal at the same decisions, not to a
   separately randomized end-to-end policy run.
4. **The intervention count is small.** Five interventions across 64 decisions
   are enough to establish nontrivial actuation but not a stable intervention
   rate or a deployment-level effect size.
5. **Utility preservation is scoped.** Exact preservation covers the two
   registered reward-variance utilities. Other quality-relevant properties of
   selected data may differ.
6. **External validity remains narrow.** The qualification uses one model,
   workload, horizon, and runtime environment.
7. **The policy was informed by earlier evidence.** Prospective freezing
   protects the live qualification itself, but M4-Shield is a developed
   solution, not a discovery-independent validation of the original signal.

## Required manuscript boundaries

- Call the result an outcome-excluded live qualification, not a quality study.
- Report the five interventions, +685.08 gain, exact utility preservation, and
  systems gates directly.
- Do not call it a randomized comparison, causal quality effect, replicated
  deployment benefit, or production-ready scheduler.
- Keep the two randomized policy studies and the live qualification distinct.
- State that the optimized imminent-L1 endpoint is evidence of successful
  constrained actuation, not an independently validated downstream benefit.

## Gap decision

No additional same-design qualification is warranted for this manuscript. A
repeat would add engineering replication but would not answer the missing
scientific question. A future scheduler paper would require a separately
preregistered, randomized, longer-horizon comparison with terminal quality and
systems outcomes, ideally across more than one model/workload environment.

For this paper, the correct action is to integrate the result, preserve the
scope boundaries above, rebuild the reviewer package, and proceed to human
scientific and release review.
