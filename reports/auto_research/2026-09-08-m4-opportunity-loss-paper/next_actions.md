# Next actions and compute decision

## Decision

Proceed toward an MLSys 2027 Research Track submission **without another M4
acquisition**. The present paper's claim is causal gradient-opportunity loss in
the tested asynchronous GRPO environments. The six-cell Qwen grid, eight
prospectively replicated Llama acquisitions across two sizes, replicated
mechanism, complete scoring, dependency-aware heterogeneity, and prespecified
size contrasts are sufficient for that bounded claim.

Do not add a 9B or 30B cell now. A seventh cell would improve external validity
incrementally. The separately preregistered 16-pair downstream-quality study
has now tested the paper's most important end-to-end limitation, but its
mixed-d5-minus-immediate estimate is -0.04492 with paired 95% interval
[-0.13394, 0.04410], so the registered conclusion is `INCONCLUSIVE`. The paper
must retain its proximal opportunity-loss claim. The 30B-A3B option additionally
changes architecture and topology, confounding a clean scale extension.

The separately preregistered 10-pair OARS/FIFO study and the larger 18-block
three-arm quality-primary follow-up are complete. Preserve both results. The
initial study reported secondary accuracy of 0.0998 [0.0275, 0.1720]. The
follow-up's primary absolute-M4-minus-FIFO estimate was -0.0331 [-0.1162,
0.0499], and reward variance minus FIFO was -0.0109 [-0.1062, 0.0844]. Reward
variance increased retained opportunity by 311.53 [125.40, 497.66], and both
active policies reduced wall time. Frame this as evidence that the signal is
actionable but proxy improvement is not sufficient for a stable terminal-
quality benefit—not as mediation or a production scheduler claim.

The prospectively frozen, outcome-excluded M4-Shield live qualification is
also complete. It changed 5 of 64 decisions and added 685.08 imminent L1
opportunity relative to reward-variance base proposals while preserving both
registered reward-variance utilities exactly and satisfying service, liveness,
and overhead gates. Include it as constrained actuation evidence, not as a
terminal-quality result or a production-scheduler evaluation.

## Ordered submission sequence

1. **Human scientific review.** Approve the claim ledger, title, abstract,
   author roster, and related-work characterization.
2. **Environment recovery.** Search authenticated logs for exact dependency and
   hardware metadata. Report what is found; mark the rest unavailable.
3. **PDF build and fit.** **Complete for review:** the official-style paper and
   supplement compile, fit, and pass visual QA. Convert figure fonts to
   Type-1-compatible outlines before final upload.
4. **Anonymous artifact build.** **Complete locally:** the compact reviewer
   bundle is scrubbed, deterministic, hash-validated, and replayed under a
   credential-stripped environment. External hosting/link testing remains a
   separate release gate.
5. **Final adversarial review.** **Complete:** claim/number verification, exact
   PDF/archive inspection, anonymity attacks, and artifact replay pass with the
   remaining release gates recorded separately.
6. **Submit once.** Upload only after every author and anonymity gate passes.

## Downstream-quality study

This prospective study is complete. It used the training run—not the rollout
group—as the independent unit, froze evaluation and stopping rules, and
completed all 16 matched seed blocks. The negative estimate is not precise
enough to determine the direction or practically relevant magnitude of the
accuracy difference. Report the estimate and interval as a bounded end-to-end
extension, and do not exclude zero-accuracy runs or add seeds under the frozen
protocol.

The current go/no-go is therefore:

- same-design M4 acquisition: **no-go**;
- 9B/30B external-validity cell: **defer**;
- present final-quality study: **complete, inconclusive, report with bounded
  wording**;
- another final-quality acquisition: **no-go unless separately preregistered
  with a stability intervention and variance-recalibrated power**;
- another same-design OARS acquisition: **no-go; both frozen policy studies are
  complete, and an outcome-guided extension would not answer a new question**;
- another same-design M4-Shield qualification: **no-go; the frozen live
  qualification answered its systems-feasibility question, while a repeat
  without a new endpoint would not establish terminal-quality benefit**;
- manuscript, reproducibility, and artifact preparation: **go now**.

## Post-manuscript novelty decision

The completed secondary analysis and identification audit leave one material
causal gap: shared-system interference. A prospective successor is frozen at
`../2026-09-13-m4-saturation-spillover/prospective_protocol.json` (SHA-256
`a51f182a95eb180a37a24bcb3401606e57a7a7acbcdf63b3b96ce7dfe7a16812`).
It pairs 25% and 75% delay-saturation policies in 16 training-seed blocks and
targets control spillover and total policy effects. This has higher scientific
value than another model cell.

The design is local only. No preflight, qualification, candidate package,
training acquisition, EOS submission, retry, or extension is authorized.
