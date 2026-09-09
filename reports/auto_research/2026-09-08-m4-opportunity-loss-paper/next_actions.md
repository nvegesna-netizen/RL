# Next actions and compute decision

## Decision

Proceed toward an MLSys 2027 Research Track submission **without another M4
acquisition**. The present paper's claim is causal gradient-opportunity loss in
the tested asynchronous GRPO environments. Six terminal cells, prospective
confirmation, replicated mechanism, complete scoring, and dependency-aware
heterogeneity are sufficient for that bounded claim.

Do not add a 9B or 30B cell now. A seventh cell would improve external validity
incrementally but would not address the paper's most important remaining
scientific limitation: whether opportunity loss changes final training quality.
The 30B-A3B option additionally changes architecture and topology, confounding a
clean scale extension.

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

## Optional downstream-quality study

Open this as a new study only if the human author review changes the target
claim from proximal systems opportunity loss to final learning outcome, or if a
venue decision explicitly requires that link. It must use the training run—not
the rollout group—as the independent unit, freeze evaluation tasks and stopping
rules, randomize scheduler condition across independent seeds, and power the
run-level effect before acquisition. Existing 49,153 group assignments cannot
substitute for independent training replicates.

This study would be valuable, but launching it before the main paper is compiled
would trade a known submission blocker (paper/artifact quality) for an expensive
and potentially underpowered new claim. The current go/no-go is therefore:

- same-design M4 acquisition: **no-go**;
- 9B/30B external-validity cell: **defer**;
- independently replicated final-quality study: **optional new hypothesis,
  design only after author review**;
- manuscript, reproducibility, and artifact preparation: **go now**.
