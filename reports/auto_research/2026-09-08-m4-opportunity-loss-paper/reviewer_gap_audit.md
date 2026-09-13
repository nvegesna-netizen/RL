# Reviewer-style gap audit

## Overall assessment

The current package is sufficient for a complete causal systems-measurement
paper. Its strongest contribution is the validated randomized instrument and
the prospective evidence sequence, not a new RL algorithm. MLSys 2027 Research
Track is the selected target, and a complete main-paper/appendix source now
exists. Scientific content is ready for human review; submission remains gated
on author metadata, PDF compilation/page fitting, dependency-version recovery,
and anonymous artifact preparation. The likely acceptance ceiling still
depends on how strongly reviewers demand downstream model-quality evidence from
a systems measurement paper.

## Likely major questions

### 1. Does opportunity loss affect final training quality?

**Reviewer concern:** The endpoint is proximal. A rollout can miss one gradient
opportunity without changing final reward if later data compensate.

**Current answer:** A separately preregistered 16-pair training-run study now
provides a final-quality endpoint. Mixed-d5 minus immediate terminal accuracy is
-0.04492 with paired 95% interval [-0.13394, 0.04410]; the exact sign-flip
`p`-value is 0.2982 and the registered conclusion is `INCONCLUSIVE`. The point
estimate is harmful, but the interval also includes null and benefit, so the
manuscript must continue to limit its headline claim to normalized
gradient-opportunity loss.

**Severity:** High for a broad learning-algorithm claim; reduced for a systems
measurement paper because the end-to-end question was tested prospectively and
reported transparently.

**Action completed:** The main paper and supplement now report the complete
inconclusive study, full interval, and instability caveat. They do not claim
equivalence, confirmed quality harm, or mediation. Do not extend the frozen
study post hoc. A future study would need a prospective stability intervention
and power recalibrated to the observed run-level variance.

### 2. Are the two interactions independent replications?

**Reviewer concern:** Both use the same GSM8K reference acquisitions.

**Current answer:** No. The paper now says so directly. The dependency-aware
synthesis reuses identical GSM8K bootstrap draws, estimates correlation
0.38–0.41, supplies simultaneous intervals, and performs a two-dimensional
global test. Both simultaneous intervals exclude zero.

**Severity:** Resolved if the wording remains precise.

**Action:** Keep “prospective same-direction extension with shared reference”;
never use “independent replication.”

The synthesis also assumes that distinct acquisition runs are independent, as
did the registered interaction analyses. State this assumption because an
unmeasured cluster-wide condition could induce residual cross-run correlation.

### 3. Why is five seconds the treatment, and is it comparable across cells?

**Reviewer concern:** Five seconds may represent different fractions of an
optimizer update at different model scales and workloads.

**Current answer:** Five seconds is a fixed controlled systems perturbation, and
heterogeneity is itself an observed result. The initial d10 positive control
supports the mechanism's dose ordering, but the definitive grid does not map a
full dose-response curve.

**Severity:** Low after offline descriptive audit.

**Action completed:** The authenticated lifecycle ledgers provide 400 common-
window inter-update intervals per cell. Median update times range from 7.769 to
13.201 seconds, so five seconds is 0.379--0.644 median update intervals. GSM8K
has the largest ratio at each model scale, but Qwen3-1.7B/GSM8K has the smallest
effect. Report this as descriptive context only; do not introduce a post hoc
normalized-dose causal claim. A future larger-model protocol should freeze its
dose rule prospectively.

### 4. Is the 0.20 materiality threshold arbitrary?

**Reviewer concern:** Five-of-six materiality depends on a chosen threshold.

**Current answer:** The threshold was frozen before the decisive acquisitions.
The threshold-sensitivity table separates positive effects from operational
materiality and shows the entire decision pattern from 0.10 to 0.30.

**Severity:** Low to medium.

**Action:** Add the original operational rationale for 0.20 to the final paper
if it exists in the design record. Do not retrofit a new rationale.

### 5. Could instrumentation perturb the measured system?

**Reviewer concern:** Logging and observation could cause the delay effect.

**Current answer:** Observer duty is measured with a corrected estimator and is
below the frozen 0.01 ceiling in every terminal cell. The randomized delay is
five seconds, orders of magnitude larger than recorded observer-duty fractions.

**Severity:** Low after the observer implementation and audit are described.

**Action:** Put the observer-duty table in the appendix and describe exactly
what time enters its numerator and denominator.

### 6. Are estimates sensitive to the estimator or window?

**Reviewer concern:** Cross-fitting or common-window trimming might produce the
pattern.

**Current answer:** All adjusted and unadjusted common-window estimates are
positive. HAC and bootstrap intervals agree closely. Common-window estimates
differ from registered full-window estimates by at most 0.0053. Missingness
bounds collapse to a point because terminal scoring is complete.

**Severity:** Resolved.

### 7. How broad is external validity?

**Reviewer concern:** Two small Qwen3 models and three math workloads may not
represent larger models, other families, other algorithms, or other clusters.

**Current answer:** The paper makes no family-wide claim. The six-cell design
demonstrates within-scope replication and heterogeneity, not population-wide
generalization.

**Severity:** Medium.

**Action:** A Qwen3.5-9B extension is preferable to Qwen3-30B-A3B if a larger
model is required: it has a simpler one-node dense recipe and introduces fewer
topology and MoE confounds. It remains secondary in value to a downstream
training-outcome experiment.

## Reproducibility attack

| Question | Evidence | Status |
| --- | --- | --- |
| Can a reviewer identify the assignment unit? | Epoch-specific group instance is stated in Methods. | Pass |
| Was opportunity measured before treatment? | Ledger contract and causal diagram state pre-release recording. | Pass |
| Are failed runs hidden? | Failures are retained in session evidence and excluded explicitly. | Pass |
| Were retries outcome-guided? | One-use guards and result records state no automatic retry or extension. | Pass |
| Are raw inputs immutable? | Each raw ledger and artifact is referenced by SHA-256. | Pass |
| Can the six-cell synthesis double-count GSM uncertainty? | Shared GSM draws and analytic covariance are explicit. | Pass |
| Can all compact-artifact tests run without project dependencies? | The archive's manifest, replay, figure renderer, and two standard-library tests pass after credential-stripped extraction. | Pass |
| Are large artifacts publicly available? | They are outside Git and presently referenced by hashes. | Open distribution question |

## Editorial and presentation status

1. **Complete:** MLSys 2027 Research Track selected; main and separate appendix
   LaTeX sources created against the official style contract.
2. **Open human gate:** final author list, institutional affiliations,
   acknowledgments, and conflicts.
3. **Complete review build / open submission-font gate:** main and supplement
   PDFs compile and pass text, metadata, pagination, and visual checks. Figure
   PDFs still introduce non-Type-1 fonts that require conversion for upload.
4. **Complete private and blind records:** the private record retains source and
   execution mappings; the blind supplement exposes only opaque acquisition IDs
   and protocol/artifact hash prefixes.
5. **Complete:** related work verified against primary records and a 16-entry
   all-author BibTeX database created.
6. **Complete local bundle / open release gate:** the deterministic anonymous
   compact bundle passes credential-stripped replay and adversarial scans.
   External hosting, link testing, and raw-ledger release remain pending.
7. **Open evidence-recovery gate:** recover exact driver, CUDA, PyTorch, vLLM,
   Megatron, CPU, and memory details from authenticated logs where available;
   do not infer missing versions.

## Go/no-go decision

**Go for manuscript and submission preparation now.** No additional same-design
GPU acquisition is needed. The only new experiment with a clearly higher
scientific return is an independently replicated final-training-outcome study;
run it only if the target claim or venue requires that downstream link. A 9B
external-validity cell is optional and lower priority. A 30B-A3B acquisition is
not recommended for the present paper because it changes scale, MoE
architecture, cluster topology, and effective delay dose simultaneously.
