# Post-gradient-utility reviewer gap audit

Date: 2026-10-07

## Verdict

The exact-gradient audit materially improves the paper's construct-validity
contribution. It connects M4 to an independently computed optimization quantity
and shows a large held-out increment in gradient-magnitude prediction beyond
ordinary covariates. The result is stable across the tested post-primary
specifications and is not reproduced by within-fold permutations.

The defensible contribution is **conditional gradient-magnitude information**.
The audit does not support M4 as a standalone ranker, a gradient-direction
signal, a successful scheduler, or evidence of terminal-quality improvement.
Keeping those boundaries makes the result useful: it explains why M4 can add
information that reward and token summaries miss, while also explaining why
policies that optimize M4 alone need not improve final quality.

## Evidence checked

- Frozen terminal classification: `MAGNITUDE_ONLY`.
- 256 measured groups; 214 positive-M4, positive-gradient primary groups.
- Baseline cross-fitted R²: 0.757642; augmented R²: 0.936437.
- Incremental R²: +0.178795; simultaneous 95% interval
  [0.133276, 0.240928].
- M4-only cross-fitted R²: -0.006450.
- Both frozen directional MSE-gain estimates are negative and all directional
  simultaneous intervals include zero.
- No optimizer, scheduler, parameter, or learner-version update occurred.
- Median sketch-relative errors are approximately 0.00355; 95th-percentile
  errors are approximately 0.010.
- Corrected observer duty is 0.000348.
- Every leave-one-fold-out, alternative-fold, winsorized, fixed-penalty, and
  include-zero incremental R² is positive.
- None of 1,000 within-fold M4 permutations matched the observed gain; the
  largest permuted gain is 0.008692.
- Detached replay matches the terminal classification and all values within
  relative tolerance 1e-12.
- The rebuilt anonymous artifact passes manifest verification, forbidden-
  content scanning, replay, and standard-library unit tests; a second build is
  byte-identical.

## Strongest reviewer objections

1. **M4 is not useful by itself.** The M4-only R² is negative. The manuscript
   must describe complementarity conditional on the baseline covariates, not a
   generic magnitude correlation.
2. **Direction is unsupported.** Both seeded directional analyses fail. Any
   claim that M4 identifies update direction, alignment, or gradient sign is
   prohibited.
3. **The magnitude result may be setting-specific.** There is one no-update
   Llama-3.2-1B/GSM8K acquisition. The post-primary checks reuse those groups
   and are not independent replication.
4. **Predictive increment is not scheduler efficacy.** The audit contains no
   policy comparison, parameter update, or terminal-quality endpoint.
5. **The baseline already predicts much of the magnitude.** The contribution
   is the additional held-out information, not a claim that M4 replaces token,
   reward, variance, or truncation features.
6. **Post-primary robustness is descriptive.** It strengthens stability of the
   frozen magnitude interpretation but cannot modify the registered
   `MAGNITUDE_ONLY` classification.
7. **The scientific next step is prospective.** A scheduler that combines M4
   with the cheap baseline features must be frozen and evaluated end-to-end;
   this audit alone cannot justify another outcome claim.

## Manuscript decision

Integrate the result as a secondary construct-validity contribution. Report the
baseline and augmented held-out R², incremental simultaneous interval, M4-only
R², and directional estimates together. Use the result to motivate
feature-combined scheduler design, but do not claim that such a scheduler has
already succeeded.

No additional analysis of this same ledger is needed for the current paper.
The unresolved question requires independent prospective compute: whether a
frozen policy using the conditionally informative signal improves a registered
downstream objective. That is a new study, not another robustness retry.
